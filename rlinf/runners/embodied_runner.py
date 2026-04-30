# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import time

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        # Controller-injected resume support:
        # these fields are only used for walltime-safe same-task resume and can be
        # removed if the workflow goes back to task-boundary-only checkpointing.
        self.global_step = int(self.cfg.runner.get("resume_global_step", 0) or 0)
        self.stop_unix_time = int(self.cfg.runner.get("stop_unix_time", 0) or 0)
        self.resume_save_grace_seconds = int(
            self.cfg.runner.get("resume_save_grace_seconds", 0) or 0
        )
        self.partial_resume_exit_code = int(
            self.cfg.runner.get("partial_resume_exit_code", 0) or 0
        )
        rolling_cfg = self.cfg.runner.get("rolling_checkpoint", {})
        self.rolling_checkpoint_enabled = bool(rolling_cfg.get("enabled", False))
        self.rolling_latest_name = rolling_cfg.get("latest_name", "latest_partial")
        self.rolling_previous_name = rolling_cfg.get(
            "previous_name", "previous_partial"
        )
        self.rolling_tmp_name = rolling_cfg.get("tmp_name", "latest_partial_tmp")
        self.rolling_keep_previous = bool(rolling_cfg.get("keep_previous", True))
        self._last_epoch_duration_seconds = None

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()
        self.actor.preallocate_memory()

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        env_futures.wait()
        actor_futures.wait()
        rollout_futures.wait()

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_futures.wait()
        rollout_results = rollout_futures.wait()
        eval_metrics_list = [
            results for results in rollout_results if results is not None
        ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        for _step in tqdm(range(start_step, self.max_steps), ncols=120):
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            epoch_start_time = time.time()
            with self.timer("step"):
                with self.timer("rollout"):
                    self.update_rollout_weights()
                    self.generate_rollouts()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                # actor training.
                with self.timer("actor_training"):
                    is_last_step = (_step == self.max_steps - 1)
                    actor_training_futures = self.actor.run_training(is_last_step=is_last_step)
                    actor_training_metrics = actor_training_futures.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                # Controller-injected rolling partial checkpoint:
                # keep only the latest/previous resumable task-local checkpoints
                # instead of accumulating one directory per outer epoch.
                if self.rolling_checkpoint_enabled and not is_train_end:
                    self._save_rolling_partial_checkpoint()

            time_metrics = self.timer.consume_durations()
            epoch_duration_seconds = max(time.time() - epoch_start_time, 0.0)
            self._last_epoch_duration_seconds = epoch_duration_seconds

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            # Controller-injected walltime stop:
            # stop only after a completed outer epoch so the controller can resume
            # from an exact task-local epoch boundary.
            if self._should_exit_for_walltime(epoch_duration_seconds, is_train_end):
                if self.global_step < self.max_steps and self._rank0():
                    remaining = int(self.stop_unix_time - time.time())
                    print(
                        "Stopping after a completed outer epoch to preserve a resumable checkpoint. "
                        f"global_step={self.global_step} max_steps={self.max_steps} "
                        f"remaining_seconds={remaining}"
                    )
                self.metric_logger.finish()
                raise SystemExit(self.partial_resume_exit_code)

        self.metric_logger.finish()
        
        # Compute and save EWC data if enabled
        if self.cfg.algorithm.get("use_ewc", False):
            ewc_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                "ewc_data.pt"
            )

            # Delegate EWC saving to the actor worker group (runs on training ranks)
            if hasattr(self.actor, "compute_and_save_ewc_data"):
                futures = self.actor.compute_and_save_ewc_data(ewc_save_path)
                futures.wait()

    def _save_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_name = f"global_step_{self.global_step}"
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            f"checkpoints/{checkpoint_name}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        save_futures = self.actor.save_checkpoint(actor_save_path, self.global_step)
        save_futures.wait()
        return base_output_dir

    def _save_rolling_partial_checkpoint(self):
        # Controller-injected rolling checkpoint layout:
        #   latest_partial
        #   previous_partial
        # This avoids unbounded checkpoint growth while preserving one rollback copy.
        checkpoint_root = os.path.join(self.cfg.runner.logger.log_path, "checkpoints")
        tmp_dir = os.path.join(checkpoint_root, self.rolling_tmp_name)
        latest_dir = os.path.join(checkpoint_root, self.rolling_latest_name)
        previous_dir = os.path.join(checkpoint_root, self.rolling_previous_name)

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        self._save_checkpoint(self.rolling_tmp_name)

        if self.rolling_keep_previous:
            if os.path.exists(previous_dir):
                shutil.rmtree(previous_dir)
            if os.path.exists(latest_dir):
                os.replace(latest_dir, previous_dir)
        elif os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)

        os.replace(tmp_dir, latest_dir)
        self._write_resume_metadata(latest_dir)

    def _write_resume_metadata(self, checkpoint_dir):
        # Controller-injected resume metadata consumed by TEST_CONTROLLER.sh when it
        # resubmits the same task after a walltime-driven stop.
        metadata = {
            "global_step": self.global_step,
            "max_steps": self.max_steps,
            "epoch": self.epoch,
            "actor_checkpoint_path": os.path.join(checkpoint_dir, "actor"),
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_path = os.path.join(checkpoint_dir, "resume_state.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def _should_exit_for_walltime(self, epoch_duration_seconds, is_train_end):
        # Controller-injected stop heuristic:
        # do not start another outer epoch if there is not enough time left to
        # finish it and still save a resumable checkpoint cleanly.
        if is_train_end or self.stop_unix_time <= 0 or self.partial_resume_exit_code <= 0:
            return False

        projected_next_epoch_seconds = epoch_duration_seconds
        if self._last_epoch_duration_seconds is not None:
            projected_next_epoch_seconds = max(
                projected_next_epoch_seconds,
                self._last_epoch_duration_seconds,
            )

        required_remaining_seconds = projected_next_epoch_seconds + self.resume_save_grace_seconds
        remaining_seconds = self.stop_unix_time - time.time()
        return remaining_seconds <= required_remaining_seconds

    def _rank0(self):
        return int(os.environ.get("RANK", "0")) == 0

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
