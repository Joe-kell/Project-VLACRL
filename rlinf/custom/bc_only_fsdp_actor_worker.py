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
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.custom.libero_trajectory_dataset import LiberoSFTDataset
from rlinf.custom.loss import behavior_cloning_ce_loss
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import compute_action_tokens_from_actions
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict


def bc_actor_forward(model, bc_batch, action_token_len, temperature=1.0, top_k=-1):
    """Forward pass for pure behavior cloning (no RL batch, no concatenation).

    Args:
        model: The actor model.
        bc_batch: Dict with keys input_ids, attention_mask, pixel_values.
        action_token_len: Number of action tokens (action_dim * num_action_chunks).
        temperature: Temperature applied to logits (unused for CE loss, kept for consistency).
        top_k: Top-k filtering (unused for CE loss, kept for consistency).

    Returns:
        raw_logits: Tensor of shape [B, action_token_len, vocab_size] — raw logits for
                    the action token positions, used directly by the CE loss.
    """
    outputs = model(
        input_ids=bc_batch["input_ids"],
        attention_mask=bc_batch["attention_mask"],
        pixel_values=bc_batch["pixel_values"],
        output_hidden_states=False,
    )
    # Extract logits at the action token positions: [B, action_token_len, vocab_size]
    raw_logits = outputs.logits[:, -action_token_len - 1 : -1]
    return raw_logits.clone()


class BCOnlyFSDPActorWorker(FSDPModelManager, Worker):
    """Pure behavior cloning actor worker using FSDP.

    Removes all RL components (PPO loss, advantages, rollout buffer, etc.)
    and trains exclusively on the SFT/demonstration dataset.
    """

    def __init__(self, cfg):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        # BC dataset / dataloader
        self._init_sft_replay_buffer()

        self._preallocated_memory = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _init_sft_replay_buffer(self):
        dataset_path = os.environ.get("LIBERO_REPO_PATH")
        if self._rank == 0:
            print(f"Initializing SFT dataset on rank {self._rank}")

        self.sft_dataset = LiberoSFTDataset(
            cfg=self.cfg,
            root_dir=dataset_path,
            demos_per_task=self.cfg.algorithm.get("demos_per_task", None),
            rank=self._rank,
            world_size=self._world_size,
            use_cached_logits=False,
            logits_type="",
            use_preprocessed=True,
            task_ids=self.cfg.env.get("fixed_task_ids", None)
        )

        self.sft_dataloader = cycle(
            DataLoader(
                self.sft_dataset,
                batch_size=self.cfg.actor.micro_batch_size,
                shuffle=True,
                num_workers=self.cfg.actor.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
        )
        self.sft_iterator = iter(self.sft_dataloader)

        if self._rank == 0:
            print(f"SFT dataset initialized: {len(self.sft_dataset)} samples")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_worker(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        else:
            if torch.distributed.get_backend() != "nccl":
                torch.distributed.destroy_process_group()
                torch.distributed.init_process_group(backend="nccl")

        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Trainability check
        if self._rank == 0:
            print("\n[BCActorWorker] Model Trainability Check:")
            print(f"    is_lora: {self.cfg.actor.model.get('is_lora', 'N/A')}")
            print(f"    partial_finetune: {self.cfg.actor.model.get('partial_finetune', 'N/A')}")
            print(f"    gradient_checkpointing: {self.cfg.actor.model.get('gradient_checkpointing', 'N/A')}")

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"    Total Params:     {total_params:,}")
            print(f"    Trainable Params: {trainable_params:,}")
            print(f"    Trainable %:      {trainable_params / total_params * 100:.2f}%")

            if trainable_params == 0:
                print("  [WARNING] NO TRAINABLE PARAMETERS DETECTED!")
            elif trainable_params == total_params:
                print("  [INFO] Model is FULLY trainable.")
            else:
                print("  [INFO] Model is partially frozen (expected for LoRA / partial finetune).")

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def preallocate_memory(self):
        return
        preallocate_gb = self.cfg.actor.get("preallocate", 0)
        if preallocate_gb == 0:
            return
        size_bytes = int(float(preallocate_gb) * 1024 ** 3)
        num_elements = size_bytes // 4  # float32
        if self._rank == 0:
            print(f"[INFO] Preallocating {preallocate_gb} GB of GPU memory...")
        try:
            self._preallocated_memory = torch.empty(
                num_elements, dtype=torch.float32, device=self.device
            )
            torch.cuda.synchronize()
        except RuntimeError as e:
            if self._rank == 0:
                print(f"[ERROR] Failed to preallocate memory: {e}")
            raise

    def _deallocate_preallocated_memory(self):
        return
        if self._preallocated_memory is not None:
            if self._rank == 0:
                size_gb = self._preallocated_memory.numel() * 4 / (1024 ** 3)
                print(f"[INFO] Deallocating {size_gb:.2f} GB of preallocated memory...")
            del self._preallocated_memory
            self._preallocated_memory = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            if self._rank == 0:
                print("[WARNING] No preallocated memory to free.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_training(self):
        """Run one epoch of pure behavior cloning.

        Pulls `steps_per_epoch` micro-batches from the SFT iterator and
        performs gradient accumulation, mirroring the structure of the RL
        actor's run_training for easy drop-in replacement.

        Returns:
            mean_metric_dict: Dict of averaged metrics across all ranks.
        """
        self._deallocate_preallocated_memory()

        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()

        # How many gradient steps to take per call to run_training.
        steps_per_epoch = self.cfg.actor.get("bc_steps_per_epoch", 8)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        bc_coeff = 1.0 
        action_token_len = self.model.action_dim * self.model.num_action_chunks

        metrics = {}

        for step_idx in tqdm(range(steps_per_epoch), desc="BC training steps"):
            self.optimizer.zero_grad()

            for _ in range(gradient_accumulation):
                bc_batch = next(self.sft_iterator)
                for k, v in bc_batch.items():
                    bc_batch[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                # Forward pass — raw logits at action positions
                raw_logits = bc_actor_forward(
                    self.model,
                    bc_batch=bc_batch,
                    action_token_len=action_token_len,
                    temperature=self.cfg.algorithm.sampling_params.temperature_train,
                    top_k=self.cfg.algorithm.sampling_params.top_k,
                )

                # Convert continuous actions → discrete token ids for CE target
                expert_action_tokens = torch.tensor(
                    compute_action_tokens_from_actions(
                        self.model, bc_batch["actions"]
                    ),
                    device=f"cuda:{int(os.environ['LOCAL_RANK'])}",
                )

                loss, bc_metrics_data = behavior_cloning_ce_loss(
                    intermediate_logits=raw_logits,
                    expert_actions_tokens=expert_action_tokens,
                    bc_coeff=bc_coeff,
                    vocab_size=self.model.vocab_size,
                    n_action_bins=self.model.config.n_action_bins,
                )

                (loss / gradient_accumulation).backward()
                append_to_dict(metrics, bc_metrics_data)

            torch.cuda.empty_cache()
            # grad_norm = self.model.clip_grad_norm_(
            #     max_norm=self.cfg.actor.optim.clip_grad
            # )
            self.optimizer.step()
            self.optimizer.zero_grad()

            step_data = {
                # "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            # if self._rank == 0 and grad_norm.detach().item() == 0.0:
                # print(f"[WARNING] Step {step_idx}: grad_norm is 0.0!")
            append_to_dict(metrics, step_data)

        mean_metric_dict = {k: np.mean(v) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        return mean_metric_dict

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        is_lora = self.cfg.actor.model.get("is_lora", False)
        model = self.model

        optim_state = self.get_optimizer_state_dict()

        if is_lora:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = get_peft_model_state_dict(model, model.state_dict())

                if self._rank == 0:
                    os.makedirs(save_base_path, exist_ok=True)
                    print(f"Saving checkpoint to {save_base_path}")
                    torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
                    torch.save(
                        cpu_state, os.path.join(save_base_path, "adapter_model.bin")
                    )
                    model.peft_config["default"].save_pretrained(save_base_path)
        else:
            model_state = self.get_model_state_dict()
            if self._rank == 0:
                os.makedirs(save_base_path, exist_ok=True)
                torch.save(model_state, os.path.join(save_base_path, "model.pt"))
                torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))

        torch.distributed.barrier()
