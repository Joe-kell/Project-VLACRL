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

import gc
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model, get_model_config_and_processor
from rlinf.models.embodiment.model_utils import (
    default_logits_processor,
    prepare_observations,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


def _bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def _value_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_value_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_nbytes(item) for item in value)
    return 0


def _first_array_like(value: Any) -> Any:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        for item in value.values():
            first = _first_array_like(item)
            if first is not None:
                return first
    if isinstance(value, (list, tuple)):
        for item in value:
            first = _first_array_like(item)
            if first is not None:
                return first
    return None


def _value_entry_count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 1


def _describe_array_like(value: Any) -> str:
    first = _first_array_like(value)
    if first is None:
        return "first=<none>"
    if torch.is_tensor(first):
        return (
            f"first_shape={tuple(first.shape)} first_dtype={first.dtype} "
            f"first_device={first.device}"
        )
    return f"first_shape={first.shape} first_dtype={first.dtype}"


def _process_memory_line() -> str:
    try:
        import psutil

        mem = psutil.Process(os.getpid()).memory_info()
        return (
            f"pid={os.getpid()} rss={_bytes_to_mib(mem.rss):.1f}MiB "
            f"vms={_bytes_to_mib(mem.vms):.1f}MiB"
        )
    except Exception:
        status = {}
        try:
            with open(f"/proc/{os.getpid()}/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                        key, value = line.split(":", 1)
                        status[key] = int(value.strip().split()[0])
        except OSError:
            return f"pid={os.getpid()} rss=<unavailable>"

        def kib_to_mib(key: str) -> str:
            return f"{status[key] / 1024:.1f}MiB" if key in status else "<unavailable>"

        return (
            f"pid={os.getpid()} rss={kib_to_mib('VmRSS')} "
            f"hwm={kib_to_mib('VmHWM')} vms={kib_to_mib('VmSize')}"
        )


def _log_replay_memory(logger: Any, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message, flush=True)


def _replay_payload_stats(data: dict[str, Any]) -> tuple[int, list[tuple[str, int, int, str]]]:
    stats = []
    total_bytes = 0
    for key, value in data.items():
        num_bytes = _value_nbytes(value)
        total_bytes += num_bytes
        stats.append((key, num_bytes, _value_entry_count(value), _describe_array_like(value)))
    stats.sort(key=lambda item: item[1], reverse=True)
    return total_bytes, stats


def _log_replay_payload_summary(
    logger: Any,
    label: str,
    data: dict[str, Any],
    top_k: int = 8,
) -> None:
    total_bytes, stats = _replay_payload_stats(data)
    top_items = "; ".join(
        f"{key}={_bytes_to_mib(num_bytes):.1f}MiB entries={entries} {description}"
        for key, num_bytes, entries, description in stats[:top_k]
    )
    _log_replay_memory(
        logger,
        "[SmolVLA replay memory] "
        f"{label}: total_tensor_payload={_bytes_to_mib(total_bytes):.1f}MiB "
        f"keys={len(stats)} {_process_memory_line()} top=[{top_items}]",
    )


def create_rollout_batch(data, debug_label: str | None = None, logger: Any = None):
    ret_data = {}
    for key, value in data.items():
        if debug_label is not None:
            _log_replay_memory(
                logger,
                "[SmolVLA replay stack] "
                f"{debug_label} key={key} pre_stack_payload="
                f"{_bytes_to_mib(_value_nbytes(value)):.1f}MiB "
                f"entries={_value_entry_count(value)} {_describe_array_like(value)} "
                f"{_process_memory_line()}",
            )
        if "env_info/" not in key:
            ret_data[key] = torch.stack(value, dim=0).contiguous().cpu()
        else:
            ret_data[key] = torch.cat(value, dim=0).contiguous().cpu()
        if debug_label is not None:
            _log_replay_memory(
                logger,
                "[SmolVLA replay stack] "
                f"{debug_label} key={key} post_stack_payload="
                f"{_bytes_to_mib(_value_nbytes(ret_data[key])):.1f}MiB "
                f"{_describe_array_like(ret_data[key])} {_process_memory_line()}",
            )
    return ret_data


def _check_actor_memory(device_index, threshold_gb=4.0):
    """
    Check if 'EmbodiedFSDPActor' process on the given device is using less than threshold_gb.
    Returns True if safe (memory low or process not found), False if memory high.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        actor_mem_bytes = 0
        found_actor = False
        for p in procs:
                # p.pid is available, but name might require looking up via psutil or nvmlSystemGetProcessName
                # nvmlSystemGetProcessName is available in newer pynvml/drivers
                name = pynvml.nvmlSystemGetProcessName(p.pid)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                    
                # We look for the Ray process name pattern
                if "EmbodiedFSDPActor" in name:
                    actor_mem_bytes += p.usedGpuMemory
                    found_actor = True
                
        pynvml.nvmlShutdown()
        
        if not found_actor:
            return True # Actor not on this GPU or not found, assume safe
            
        actor_mem_gb = actor_mem_bytes / (1024**3)
        print(f"Actor memory usage: {actor_mem_gb}, threshold: {threshold_gb}", flush=True)
        return actor_mem_gb < threshold_gb

    except Exception as e:
        print(f"Warning: Failed to check actor memory: {e}")
        return True # Fail open to avoid deadlock if NVML issues


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.model_config, self.input_processor = get_model_config_and_processor(
            cfg.actor
        )
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = self._component_placement.get_world_size("actor")
        self._weight_src_rank_in_actor = self._rank % actor_world_size
        self.channel = self.connect_channel(cfg.rollout.channel.name)
        for i in range(self._component_placement.get_world_size("rollout")):
            self.channel.create_queue(
                f"{self._action_queue_name}_{i}", maxsize=cfg.rollout.channel.queue_size
            )

        self.use_proprio = self.cfg.actor.model.get("use_proprio", False)

        # Debug logging setup
        self.enable_action_logging = cfg.rollout.get("enable_action_logging", False)
        if self.enable_action_logging:
            self.action_log_dir = cfg.rollout.get(
                "action_log_dir",
                os.path.join(cfg.runner.logger.log_path, "action_logs"),
            )
            os.makedirs(self.action_log_dir, exist_ok=True)
            self.action_log_data = defaultdict(list)
            if self._rank == 0:
                print(
                    f"[DEBUG] Action logging enabled. Saving to: {self.action_log_dir}"
                )
        self._smolvla_replay_schema_logged = False

    def init_worker(self):
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        self.hf_model.setup_params(self.model_config, self.cfg)
        if self.cfg.actor.model.get("model_name") == "smolvla":
            # SmolVLA is natively mixed precision. A global dtype cast here
            # breaks LeRobot's fp32 action/state projection path, especially
            # after PEFT wraps the model and bypasses the wrapper's custom to().
            device = (
                torch.device("cuda", self.device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.hf_model.to(device=device)
        else:
            self.hf_model.to(self.precision)
        self.hf_model.eval()
        self.setup_sample_params()
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": not self._sampling_params["use_greedy"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": not self._sampling_params["use_greedy"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, processed_obs, mode="train"):
        action_token_len = self.hf_model.action_dim * self.hf_model.num_action_chunks

        sample_kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        with torch.no_grad():
            actions, action_tokens, action_logits, last_hidden_state = (
                self.hf_model.predict_action_batch(
                    input_ids=processed_obs["input_ids"],
                    attention_mask=processed_obs["attention_mask"],
                    pixel_values=processed_obs["pixel_values"],
                    **sample_kwargs,
                )
            )

        chunk_logprobs = default_logits_processor(
            action_logits,
            action_tokens,
            self.hf_model.vocab_size,
            self.hf_model.config.n_action_bins,
        )["logprobs"]

        chunk_values = None
        if self.cfg.algorithm.require_values:
            if self.cfg.actor.model.vh_mode == "a0":
                hidden_features = last_hidden_state[
                    :, -action_token_len
                ]  # [batch_size, hidden_dim]
                with torch.no_grad():
                    chunk_values = self.hf_model.value_head(
                        hidden_features
                    )  # [batch_size, 1]

        if chunk_values is None:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )
        chunk_action_tokens = action_tokens.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )

        return chunk_actions, chunk_action_tokens, chunk_logprobs, chunk_values

    def log_actions_and_tokens(
        self, chunk_actions, chunk_action_tokens, step, stage_id, rollout_epoch
    ):
        """
        Log actions and action tokens to disk for debugging.

        Args:
            chunk_actions: Tensor of shape [batch, num_chunks, action_dim] with continuous actions
            chunk_action_tokens: Tensor of shape [batch, num_chunks, action_dim] with token IDs
            step: Current step in rollout
            stage_id: Stage ID in pipeline
            rollout_epoch: Current rollout epoch
        """
        if not self.enable_action_logging:
            return

        # Convert to numpy and store
        actions_np = chunk_actions  # [batch, num_chunks, action_dim]
        tokens_np = chunk_action_tokens.cpu().numpy()  # [batch, num_chunks, action_dim]

        # Store in memory (will save to disk at end of generate)
        self.action_log_data["actions"].append(actions_np)
        self.action_log_data["action_tokens"].append(tokens_np)
        self.action_log_data["step"].append(step)
        self.action_log_data["stage_id"].append(stage_id)
        self.action_log_data["rollout_epoch"].append(rollout_epoch)

    def save_action_logs(self, global_step):
        """Save accumulated action logs to disk."""
        if not self.enable_action_logging or not self.action_log_data["actions"]:
            return

        # Concatenate all logged data
        all_actions = np.concatenate(self.action_log_data["actions"], axis=0)
        all_tokens = np.concatenate(self.action_log_data["action_tokens"], axis=0)
        all_steps = np.array(self.action_log_data["step"])
        all_stage_ids = np.array(self.action_log_data["stage_id"])
        all_epochs = np.array(self.action_log_data["rollout_epoch"])

        # Save to npz file
        save_path = os.path.join(
            self.action_log_dir, f"rank_{self._rank}_step_{global_step}.npz"
        )

        np.savez_compressed(
            save_path,
            actions=all_actions,
            action_tokens=all_tokens,
            steps=all_steps,
            stage_ids=all_stage_ids,
            rollout_epochs=all_epochs,
            vocab_size=getattr(self.hf_model, "vocab_size", -1),
            n_action_bins=getattr(getattr(self.hf_model, "config", None), "n_action_bins", -1),
            action_dim=self.hf_model.action_dim,
            num_action_chunks=self.hf_model.num_action_chunks,
        )

        if self._rank == 0:
            print(f"[DEBUG] Saved action logs to: {save_path}")
            print(f"  Total samples: {all_actions.shape[0]}")
            print(f"  Actions shape: {all_actions.shape}")
            print(f"  Tokens shape: {all_tokens.shape}")
            print(f"  Action range: [{all_actions.min():.4f}, {all_actions.max():.4f}]")
            print(f"  Token range: [{all_tokens.min()}, {all_tokens.max()}]")

        # Clear logged data
        self.action_log_data.clear()

    def update_env_batch(self, i, env_batch):
        # first step for env_batch
        if env_batch["rews"] is None:
            self.buffer_list[i]["dones"].append(env_batch["dones"].contiguous().cpu())
            return

        self.buffer_list[i]["rewards"].append(env_batch["rews"].cpu().contiguous())
        self.buffer_list[i]["dones"].append(
            env_batch["dones"].bool().cpu().contiguous()
        )

        if self.cfg.env.train.auto_reset or self.cfg.env.train.ignore_terminations:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                self.buffer_list[i][f"env_info/{key}"].append(value)

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_batch["dones"].any() and self.cfg.env.train.auto_reset:
            if self.cfg.algorithm.require_values:
                if self.cfg.actor.model.model_name == "smolvla":
                    # Current SmolVLA online path is GRPO-oriented and does not use
                    # bootstrap values here.
                    return
                dones = env_batch["dones"]
                # if self.require_values:
                final_obs = env_batch["infos"]["final_observation"]
                with torch.no_grad():
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=final_obs,
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                    )
                    _, _, _, _final_values = self.predict(processed_obs)
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i]["rewards"][-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    async def generate(self, global_step=0):
        is_smolvla = self.cfg.actor.model.model_name == "smolvla"
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = []
        for i in range(self.stage_num):
            self.buffer_list.append(defaultdict(list))
        if is_smolvla:
            _log_replay_memory(
                self._logger,
                "[SmolVLA replay memory] "
                f"rollout_start rank={self._rank} global_step={global_step} "
                f"rollout_epoch={self.cfg.algorithm.rollout_epoch} "
                f"n_chunk_steps={self.cfg.algorithm.n_chunk_steps} "
                f"num_group_envs={self.cfg.algorithm.num_group_envs} "
                f"stage_num={self.stage_num} {_process_memory_line()}",
            )

        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            self._logger.info(f"Now epoch is={rollout_epoch}")
            for step in tqdm(
                range(self.cfg.algorithm.n_chunk_steps),
                desc=f"Rollout ID {self._rank} Epoch {rollout_epoch} in Generate Step",
            ):
                for i in range(self.stage_num):
                    env_batch = await self.recv_env_batch()
                    self.update_env_batch(i, env_batch)
                    if is_smolvla:
                        policy_batch = self.hf_model.prepare_policy_batch(env_batch["obs"])
                        (
                            chunk_actions,
                            chunk_action_token,
                            chunk_logprobs,
                            chunk_values,
                        ) = self.hf_model.rollout_train_step(policy_batch)
                        await self.send_chunk_actions(chunk_actions)

                        # Log actions and sampled action tensors for debugging.
                        self.log_actions_and_tokens(
                            chunk_actions, chunk_action_token, step, i, rollout_epoch
                        )

                        packed_policy_batch = self.hf_model.pack_policy_batch_for_replay(
                            policy_batch
                        )
                        if not self._smolvla_replay_schema_logged:
                            schema_tensors = dict(packed_policy_batch)
                            schema_tensors["action_tokens"] = chunk_action_token
                            schema_tensors["prev_logprobs"] = chunk_logprobs
                            schema_tensors["prev_values"] = chunk_values
                            _log_replay_payload_summary(
                                self._logger,
                                (
                                    f"first_batch rank={self._rank} stage={i} "
                                    f"epoch={rollout_epoch} step={step}"
                                ),
                                schema_tensors,
                                top_k=32,
                            )
                            _, schema_stats = _replay_payload_stats(schema_tensors)
                            for key, num_bytes, entries, description in schema_stats:
                                _log_replay_memory(
                                    self._logger,
                                    "[SmolVLA replay schema] "
                                    f"rank={self._rank} key={key} "
                                    f"payload={_bytes_to_mib(num_bytes):.3f}MiB "
                                    f"entries={entries} {description}",
                                )
                            self._smolvla_replay_schema_logged = True
                        for key, value in packed_policy_batch.items():
                            self.buffer_list[i][key].append(value.cpu().contiguous())

                        batch_size = chunk_action_token.shape[0]
                        self.buffer_list[i]["input_ids"].append(
                            torch.zeros(batch_size, 1, dtype=torch.long)
                        )
                        self.buffer_list[i]["pixel_values"].append(
                            torch.zeros(batch_size, 1, dtype=torch.float32)
                        )
                        self.buffer_list[i]["attention_mask"].append(
                            torch.ones(batch_size, 1, dtype=torch.bool)
                        )
                        self.buffer_list[i]["action_tokens"].append(
                            chunk_action_token.cpu().contiguous()
                        )
                        self.buffer_list[i]["prev_logprobs"].append(
                            chunk_logprobs.cpu().contiguous()
                        )
                        self.buffer_list[i]["prev_values"].append(
                            chunk_values.cpu().contiguous()
                        )
                    else:
                        processed_obs = prepare_observations(
                            simulator_type=self.cfg.env.train.simulator_type,
                            model_name=self.cfg.actor.model.model_name,
                            raw_obs=env_batch["obs"],
                            use_proprio=self.use_proprio,
                            max_length=self.hf_model.max_prompt_length,
                            processor=self.input_processor,
                            precision=self.precision,
                        )
                        chunk_actions, chunk_action_token, chunk_logprobs, chunk_values = (
                            self.predict(processed_obs)
                        )
                        await self.send_chunk_actions(chunk_actions)

                        # Log actions and tokens for debugging
                        self.log_actions_and_tokens(
                            chunk_actions, chunk_action_token, step, i, rollout_epoch
                        )

                        self.buffer_list[i]["input_ids"].append(
                            processed_obs["input_ids"].cpu().contiguous()
                        )
                        self.buffer_list[i]["pixel_values"].append(
                            processed_obs["pixel_values"].cpu().contiguous()
                        )
                        self.buffer_list[i]["attention_mask"].append(
                            processed_obs["attention_mask"].bool().cpu().contiguous()
                        )
                        self.buffer_list[i]["action_tokens"].append(
                            chunk_action_token.cpu().contiguous()
                        )
                        self.buffer_list[i]["prev_logprobs"].append(
                            chunk_logprobs.cpu().contiguous()
                        )
                        self.buffer_list[i]["prev_values"].append(
                            chunk_values.cpu().contiguous()
                        )

            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                self.update_env_batch(i, env_batch)
                if is_smolvla:
                    batch_size = len(env_batch["obs"]["task_descriptions"])
                    final_chunk_values = torch.zeros(
                        batch_size,
                        self.hf_model.num_action_chunks,
                        1,
                        dtype=torch.float32,
                    )
                else:
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=env_batch["obs"],
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                    )
                    _, _, _, final_chunk_values = self.predict(processed_obs)
                self.buffer_list[i]["prev_values"].append(
                    final_chunk_values.cpu().contiguous()
                )

                if (
                    not self.cfg.env.train.auto_reset
                    and not self.cfg.env.train.ignore_terminations
                ):
                    infos = env_batch["infos"]
                    if "episode" in infos:
                        for key, value in infos["episode"].items():
                            self.buffer_list[i][f"env_info/{key}"].append(value.cpu())

            if is_smolvla:
                for stage_id in range(self.stage_num):
                    _log_replay_payload_summary(
                        self._logger,
                        (
                            f"buffered_after_epoch rank={self._rank} "
                            f"stage={stage_id} epoch={rollout_epoch}"
                        ),
                        self.buffer_list[stage_id],
                    )

        # Save action logs to disk
        self.save_action_logs(global_step)

        for i in range(self.stage_num):
            if is_smolvla:
                _log_replay_payload_summary(
                    self._logger,
                    f"before_send_rollout_batch rank={self._rank} stage={i}",
                    self.buffer_list[i],
                )
            await self.send_rollout_batch(i)
            self.buffer_list[i].clear()

        gc.collect()

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    async def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        eval_info = defaultdict(list)

        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps), desc="Rollout in Eval Step"
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                if self.cfg.actor.model.model_name == "smolvla":
                    chunk_actions = self.hf_model.predict_action_step_batch(
                        raw_obs=env_batch["obs"],
                        done_mask=env_batch.get("dones", None),
                        mode="eval",
                    )
                else:
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.eval.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=env_batch["obs"],
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                    )
                    chunk_actions, _, _, _ = self.predict(
                        processed_obs,
                        mode="eval",
                    )
                await self.send_chunk_actions(chunk_actions)

                if "meta" in env_batch:
                    env_info_list = env_batch["meta"]
                    for key, value in env_info_list.items():
                        eval_info[f"env_info/{key}"].append(value)

        env_batch = await self.recv_env_batch()
        if "meta" in env_batch:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                eval_info[f"env_info/{key}"].append(value)
        eval_metrics = create_rollout_batch(eval_info)
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
        return eval_metrics

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    def _wait_for_actor_offload(self, threshold_gb=2.0):
        """Wait until Actor process on this GPU uses less than threshold_gb memory."""
        if self._rank == 0:
            print(f"[Rollout] Waiting for Actor to offload GPU memory (Threshold: {threshold_gb}GB)...")
            
        while True:
            is_safe = _check_actor_memory(self.device, threshold_gb)
            if is_safe:
                break
                
            time.sleep(1.0)

    def sync_model_from_actor(self):
        if self.cfg.actor.model.get('use_fsdp2', False):
            print("Waiting for actor to offload memory...", self._rank)
            self._wait_for_actor_offload(threshold_gb=5.0)
        
        param_state_dict = self.recv(
            self._actor_group_name, src_rank=self._weight_src_rank_in_actor
        )
        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def recv_env_batch(self):
        env_batch = await self.channel.get(
            queue_name=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_batch

    async def send_chunk_actions(self, chunk_actions):
        await self.channel.put(
            item=chunk_actions,
            queue_name=f"{self._action_queue_name}_{self._rank}",
            async_op=True,
        ).async_wait()

    async def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        is_smolvla = self.cfg.actor.model.model_name == "smolvla"
        debug_label = (
            f"rank={self._rank} stage={stage_id} create_rollout_batch"
            if is_smolvla
            else None
        )
        rollout_batch = create_rollout_batch(
            self.buffer_list[stage_id],
            debug_label=debug_label,
            logger=self._logger if is_smolvla else None,
        )
        if is_smolvla:
            _log_replay_payload_summary(
                self._logger,
                f"after_create_rollout_batch rank={self._rank} stage={stage_id}",
                rollout_batch,
            )
        for i in range(split_num):
            rollout_batch_i = {}
            for key in rollout_batch.keys():
                if "env_info/" not in key:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=1
                    )[i].contiguous()
                else:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=0
                    )[i].contiguous()
            if is_smolvla:
                _log_replay_payload_summary(
                    self._logger,
                    (
                        f"before_replay_put rank={self._rank} stage={stage_id} "
                        f"split={i}/{split_num}"
                    ),
                    rollout_batch_i,
                    top_k=6,
                )
            await self.channel.put(
                item=rollout_batch_i, queue_name=self._replay_buffer_name, async_op=True
            ).async_wait()
            if is_smolvla:
                _log_replay_memory(
                    self._logger,
                    "[SmolVLA replay memory] "
                    f"after_replay_put rank={self._rank} stage={stage_id} "
                    f"split={i}/{split_num} {_process_memory_line()}",
                )
