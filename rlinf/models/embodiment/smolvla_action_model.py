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

from __future__ import annotations

import os
import sys
import types
import json
import tempfile
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal

from rlinf.config import torch_dtype_from_precision


def _ensure_lerobot_src_on_path() -> None:
    """Allow CRL eval jobs to use the LeRobot source checkout used for SFT."""
    candidates = [
        os.environ.get("LEROBOT_SRC"),
        "/home/s2758621/Octo_RL/lerobot/src",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            return


def _find_lerobot_src() -> Path | None:
    for candidate in [
        os.environ.get("LEROBOT_SRC"),
        "/home/s2758621/Octo_RL/lerobot/src",
    ]:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "lerobot").is_dir():
            return path
    return None


def _install_lerobot_inference_import_shims() -> None:
    """Avoid robot-hardware optional deps when LeRobot is used only for policy inference.

    The current LeRobot package imports robot config modules from several policy
    paths. Those modules import physical-robot/video helper deps (`serial`,
    `deepdiff`, `av`) even though CRL evaluation only needs SmolVLA model
    inference. These small shims let those imports complete while still keeping
    real robot IO/video decoding unavailable.
    """
    lerobot_src = _find_lerobot_src()
    if lerobot_src is not None:
        import lerobot

        if "lerobot.policies" not in sys.modules:
            policies_module = types.ModuleType("lerobot.policies")
            policies_module.__path__ = [str(lerobot_src / "lerobot" / "policies")]
            policies_module.__package__ = "lerobot.policies"
            sys.modules["lerobot.policies"] = policies_module
            setattr(lerobot, "policies", policies_module)

        if "lerobot.envs" not in sys.modules:
            envs_module = types.ModuleType("lerobot.envs")
            envs_module.__path__ = [str(lerobot_src / "lerobot" / "envs")]
            envs_module.__package__ = "lerobot.envs"

            class EnvConfig:
                pass

            envs_module.EnvConfig = EnvConfig
            sys.modules["lerobot.envs"] = envs_module
            setattr(lerobot, "envs", envs_module)

    try:
        import serial  # noqa: F401
    except ModuleNotFoundError:
        serial_module = types.ModuleType("serial")
        tools_module = types.ModuleType("serial.tools")
        list_ports_module = types.ModuleType("serial.tools.list_ports")

        class SerialException(Exception):
            pass

        class Serial:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError(
                    "pyserial is not installed. CRL SmolVLA eval only shims this "
                    "module for LeRobot policy imports; real robot serial IO is unavailable."
                )

        list_ports_module.comports = lambda *args, **kwargs: []
        tools_module.list_ports = list_ports_module
        serial_module.Serial = Serial
        serial_module.SerialException = SerialException
        serial_module.tools = tools_module

        sys.modules["serial"] = serial_module
        sys.modules["serial.tools"] = tools_module
        sys.modules["serial.tools.list_ports"] = list_ports_module

    try:
        import deepdiff  # noqa: F401
    except ModuleNotFoundError:
        deepdiff_module = types.ModuleType("deepdiff")

        class DeepDiff(dict):
            pass

        deepdiff_module.DeepDiff = DeepDiff
        sys.modules["deepdiff"] = deepdiff_module

    try:
        import av  # noqa: F401
    except ModuleNotFoundError:
        av_module = types.ModuleType("av")
        video_module = types.ModuleType("av.video")
        frame_module = types.ModuleType("av.video.frame")

        class VideoFrame:
            pict_type = None

            @staticmethod
            def from_image(*args, **kwargs):
                del args, kwargs
                raise RuntimeError(
                    "PyAV is not installed. CRL SmolVLA eval only shims this "
                    "module for LeRobot policy imports; video IO is unavailable."
                )

        class _AvLogging:
            ERROR = 40

            @staticmethod
            def set_level(*args, **kwargs):
                del args, kwargs
                return None

            @staticmethod
            def restore_default_callback():
                return None

        def _missing_av_open(*args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "PyAV is not installed. CRL SmolVLA eval only shims this "
                "module for LeRobot policy imports; video IO is unavailable."
            )

        av_module.VideoFrame = VideoFrame
        av_module.AVError = RuntimeError
        av_module.FFmpegError = RuntimeError
        frame_module.VideoFrame = VideoFrame
        video_module.frame = frame_module
        av_module.video = video_module
        av_module.logging = _AvLogging
        av_module.open = _missing_av_open
        av_module.time_base = 1
        sys.modules["av"] = av_module
        sys.modules["av.video"] = video_module
        sys.modules["av.video.frame"] = frame_module


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


class SmolVLAForEvalActionPrediction(torch.nn.Module):
    """CRL wrapper around a LeRobot-format SmolVLA checkpoint.

    This intentionally mirrors the LeRobot-style evaluator used in Octo_RL:
    load `PreTrainedConfig`, build the policy through LeRobot's factory, load
    the saved pre/post processors, then return postprocessed LIBERO actions.
    For online RL, CRL wraps SmolVLA chunk means with a Gaussian policy head
    to produce PPO/GRPO-compatible logprobs.
    """

    _NATIVE_FP32_MODULE_NAMES = (
        "state_proj",
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    )

    def __init__(self, model_path: str | os.PathLike[str], cfg: Any):
        super().__init__()
        _ensure_lerobot_src_on_path()
        _install_lerobot_inference_import_shims()

        # Direct imports keep LeRobot scoped to SmolVLA inference. The broader
        # factory/env imports pull in robot hardware dependencies that CRL eval
        # does not need.
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )
        self._obs_language_tokens_key = OBS_LANGUAGE_TOKENS
        self._obs_language_attention_mask_key = OBS_LANGUAGE_ATTENTION_MASK

        self.model_path = self._resolve_policy_path(Path(model_path).expanduser())
        self.cfg = cfg
        self.action_dim = int(_get_cfg(cfg, "action_dim", 7))
        self.num_action_chunks = int(_get_cfg(cfg, "num_action_chunks", 8))
        self.local_files_only = _as_bool(_get_cfg(cfg, "local_files_only", True), True)
        self.use_amp = _as_bool(_get_cfg(cfg, "use_amp", False), False)
        self.zero_action_noise = _as_bool(
            _get_cfg(cfg, "zero_action_noise", False), False
        )
        self.log_action_stats = _as_bool(
            _get_cfg(cfg, "log_action_stats", False), False
        )
        self.noise_method = str(_get_cfg(cfg, "noise_method", "reinflow"))
        self.min_action_std = float(_get_cfg(cfg, "min_action_std", 1e-3))
        self.train_action_std = float(
            _get_cfg(cfg, "reinflow_std", _get_cfg(cfg, "train_action_std", 0.10))
        )
        if self.train_action_std <= 0:
            raise ValueError(
                "SmolVLA train_action_std/reinflow_std must be > 0. "
                f"Got {self.train_action_std}."
            )
        if self.min_action_std <= 0:
            raise ValueError(
                "SmolVLA min_action_std must be > 0. "
                f"Got {self.min_action_std}."
            )
        self.policy_batch_prefix = str(
            _get_cfg(cfg, "policy_batch_prefix", "smolvla_pb__")
        )
        self.compact_replay_images = _as_bool(
            _get_cfg(cfg, "compact_replay_images", True), True
        )
        replay_image_precision = _get_cfg(cfg, "replay_image_dtype", "bf16")
        self.replay_image_dtype = torch_dtype_from_precision(replay_image_precision)
        self._logged_action_stats = False
        self._logged_step_action_stats = False
        self._action_queues: list[deque[np.ndarray]] = []

        device = _get_cfg(cfg, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        requested_precision = _get_cfg(cfg, "precision", None)
        self.target_dtype = (
            torch_dtype_from_precision(requested_precision)
            if requested_precision is not None
            else None
        )

        empty_cameras = int(_get_cfg(cfg, "empty_cameras", 1))

        policy_cfg = self._load_smolvla_config(self.model_path, SmolVLAConfig)
        policy_cfg.pretrained_path = self.model_path
        policy_cfg.device = str(self.device)
        policy_cfg.use_amp = self.use_amp
        policy_cfg.push_to_hub = False
        if hasattr(policy_cfg, "empty_cameras"):
            policy_cfg.empty_cameras = empty_cameras

        policy_n_action_steps = _get_cfg(cfg, "policy_n_action_steps", None)
        if policy_n_action_steps is not None and hasattr(policy_cfg, "n_action_steps"):
            policy_cfg.n_action_steps = int(policy_n_action_steps)

        init_log_std = float(np.log(max(self.train_action_std, 1e-8)))
        self.log_std = torch.nn.Parameter(
            torch.full(
                (self.action_dim,),
                init_log_std,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.policy_cfg = policy_cfg
        self.config = policy_cfg
        self.base_env_cfg = None
        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_path,
            config=policy_cfg,
            local_files_only=self.local_files_only,
        )
        self.restore_native_mixed_precision()
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(policy_cfg.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=policy_cfg.pretrained_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=self.local_files_only,
            overrides=preprocessor_overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=policy_cfg.pretrained_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=self.local_files_only,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.max_prompt_length = int(getattr(policy_cfg, "tokenizer_max_length", 48))
        self._print_load_summary()

    @classmethod
    def from_pretrained_crl(
        cls,
        model_path: str | os.PathLike[str],
        cfg: Any,
    ) -> "SmolVLAForEvalActionPrediction":
        return cls(model_path=model_path, cfg=cfg)

    @staticmethod
    def _resolve_policy_path(path: Path) -> Path:
        candidates = [
            path,
            path / "pretrained_model",
            path / "checkpoints" / "last" / "pretrained_model",
            path / "last" / "pretrained_model",
        ]
        for candidate in candidates:
            if (
                candidate.is_dir()
                and (candidate / "config.json").is_file()
                and (candidate / "model.safetensors").is_file()
            ):
                return candidate
        checked = "\n  ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Could not resolve a LeRobot SmolVLA pretrained_model directory. "
            "Checked:\n  "
            f"{checked}"
        )

    @staticmethod
    def _load_smolvla_config(model_path: Path, smolvla_config_cls):
        """Load SmolVLA config while tolerating LeRobot's top-level policy selector.

        LeRobot checkpoints store `{"type": "smolvla", ...}` at top level. That
        key is valid for generic policy selection, but invalid when directly
        parsing `SmolVLAConfig`. We remove only the top-level `type` key in
        memory and keep all nested `type` keys (e.g. in feature specs) intact.
        """
        import draccus

        config_path = model_path / "config.json"
        raw_cfg = json.loads(config_path.read_text())

        if isinstance(raw_cfg, dict) and "type" in raw_cfg:
            raw_cfg = dict(raw_cfg)
            raw_cfg.pop("type", None)

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(raw_cfg, f)
            tmp_cfg_path = f.name
        try:
            with draccus.config_type("json"):
                return draccus.parse(smolvla_config_cls, tmp_cfg_path, args=[])
        finally:
            try:
                os.remove(tmp_cfg_path)
            except OSError:
                pass

    def _print_load_summary(self) -> None:
        print("Loaded SmolVLA policy")
        print(f"  policy_path={self.model_path}")
        print(f"  action_dim={self.action_dim}")
        print(f"  num_action_chunks={self.num_action_chunks}")
        print(f"  policy_chunk_size={getattr(self.policy.config, 'chunk_size', '<unknown>')}")
        print(f"  policy_n_action_steps={getattr(self.policy.config, 'n_action_steps', '<unknown>')}")
        print(f"  empty_cameras={getattr(self.policy.config, 'empty_cameras', '<unknown>')}")
        print(f"  requested_runtime_dtype={self.target_dtype}")
        print(f"  state_proj_dtype={self._module_param_dtype(self.policy.model.state_proj)}")
        print(f"  action_in_proj_dtype={self._module_param_dtype(self.policy.model.action_in_proj)}")
        print(f"  action_out_proj_dtype={self._module_param_dtype(self.policy.model.action_out_proj)}")
        print(f"  action_time_mlp_in_dtype={self._module_param_dtype(self.policy.model.action_time_mlp_in)}")
        print(f"  action_time_mlp_out_dtype={self._module_param_dtype(self.policy.model.action_time_mlp_out)}")
        print(f"  image_features={list(self.policy.config.image_features.keys())}")
        print(f"  input_features={list(self.policy.config.input_features.keys())}")
        print(f"  output_features={list(self.policy.config.output_features.keys())}")

    def restore_native_mixed_precision(self):
        """Keep LeRobot SmolVLA's fp32 action/state heads after external casts.

        Native SmolVLA explicitly feeds fp32 state/action tensors and upcasts
        denoising suffix outputs to fp32 before action_out_proj. PEFT/FSDP
        wrappers can still call Module._apply(dtype=bf16) around this wrapper,
        so pin these small projection heads back to fp32.
        """
        if not hasattr(self, "policy") or not hasattr(self.policy, "model"):
            return self

        smolvla_model = self.policy.model
        for module_name in self._NATIVE_FP32_MODULE_NAMES:
            module = getattr(smolvla_model, module_name, None)
            if module is not None:
                module.to(dtype=torch.float32)
        return self

    def _apply(self, fn):
        result = super()._apply(fn)
        self.restore_native_mixed_precision()
        return result

    def setup_params(self, model_config, cfg) -> None:
        del model_config, cfg
        return None

    def preprocess_for_train(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # SmolVLA replay tensors are already in train-ready shape.
        return data

    def prepare_policy_batch(self, raw_obs: dict) -> dict[str, Any]:
        batch = self._raw_crl_obs_to_lerobot_batch(raw_obs)
        return self.preprocessor(batch)

    def pack_policy_batch_for_replay(
        self, policy_batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        packed: dict[str, torch.Tensor] = {}
        for key, value in policy_batch.items():
            if torch.is_tensor(value):
                packed[f"{self.policy_batch_prefix}{key}"] = (
                    self._pack_replay_tensor(key, value)
                )
        if not packed:
            raise RuntimeError(
                "SmolVLA policy batch had no tensor fields to pack for replay."
            )
        return packed

    def extract_policy_batch_from_replay(
        self, replay_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        policy_batch = {
            key[len(self.policy_batch_prefix) :]: value
            for key, value in replay_batch.items()
            if key.startswith(self.policy_batch_prefix)
        }
        if not policy_batch:
            raise KeyError(
                "No SmolVLA packed policy batch keys found in replay batch. "
                f"Expected keys prefixed with '{self.policy_batch_prefix}'."
            )
        for key, value in list(policy_batch.items()):
            if torch.is_tensor(value):
                policy_batch[key] = self._unpack_replay_tensor(key, value)
        return policy_batch

    def _pack_replay_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self._should_compact_replay_image(key, value):
            return value.to(dtype=self.replay_image_dtype)
        return value

    def _unpack_replay_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self._should_compact_replay_image(key, value):
            return value.to(dtype=self._input_dtype()).contiguous()
        return value

    def _should_compact_replay_image(self, key: str, value: torch.Tensor) -> bool:
        return (
            self.compact_replay_images
            and value.is_floating_point()
            and key in self._smolvla_image_feature_keys()
        )

    def _smolvla_image_feature_keys(self) -> set[str]:
        return set(getattr(self.policy.config, "image_features", {}).keys())

    def rollout_train_step(
        self, policy_batch: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample train-time actions and return PPO-compatible replay statistics."""
        self.policy.eval()
        with torch.no_grad():
            action_mean = self._predict_policy_action_chunk(
                policy_batch, noise=self._zero_noise_like_policy_batch(policy_batch)
            )
            action_mean = action_mean[:, : self.num_action_chunks, : self.action_dim]
            dist = self._action_distribution(action_mean)

            if self.zero_action_noise:
                sampled_actions = action_mean
            else:
                sampled_actions = dist.rsample()
            sampled_actions = sampled_actions.clamp(-1.0, 1.0)

            logprobs = dist.log_prob(sampled_actions)
            values = torch.zeros(
                sampled_actions.shape[0],
                sampled_actions.shape[1],
                1,
                dtype=sampled_actions.dtype,
                device=sampled_actions.device,
            )

            flat_actions = sampled_actions.reshape(-1, self.action_dim)
            env_flat_actions = self.postprocessor(flat_actions)

        if torch.is_tensor(env_flat_actions):
            env_flat_actions = env_flat_actions.detach().cpu().to(torch.float32).numpy()
        env_flat_actions = np.asarray(env_flat_actions, dtype=np.float32)
        env_actions = env_flat_actions.reshape(
            sampled_actions.shape[0],
            sampled_actions.shape[1],
            -1,
        )
        return (
            env_actions,
            sampled_actions.detach(),
            logprobs.detach(),
            values.detach(),
        )

    def actor_forward_from_replay(
        self,
        policy_batch: dict[str, torch.Tensor],
        sampled_policy_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recompute logprobs for PPO/GRPO from replayed observations/actions."""
        self.policy.eval()
        action_mean = self._predict_policy_action_chunk(
            policy_batch, noise=self._zero_noise_like_policy_batch(policy_batch)
        )
        action_mean = action_mean[:, : self.num_action_chunks, : self.action_dim]
        sampled_policy_actions = sampled_policy_actions[
            :, : self.num_action_chunks, : self.action_dim
        ].to(dtype=action_mean.dtype)

        dist = self._action_distribution(action_mean)
        logprobs = dist.log_prob(sampled_policy_actions)
        entropy = dist.entropy()
        values = torch.zeros(
            sampled_policy_actions.shape[0],
            sampled_policy_actions.shape[1],
            1,
            dtype=sampled_policy_actions.dtype,
            device=sampled_policy_actions.device,
        )
        return {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

    def default_forward(self, **kwargs):
        forward_type = kwargs.pop("forward_type", "actor_replay")
        if forward_type == "actor_replay":
            return self.actor_forward_from_replay(
                policy_batch=kwargs["policy_batch"],
                sampled_policy_actions=kwargs["sampled_policy_actions"],
            )
        raise ValueError(f"Unsupported SmolVLA forward_type: {forward_type}")

    def forward(self, *args, **kwargs):
        if args:
            raise ValueError(
                "SmolVLA forward expects keyword arguments only in CRL."
            )
        return self.default_forward(**kwargs)

    def to(self, *args, **kwargs):  # noqa: D401
        device, dtype = self._parse_to_device_dtype(*args, **kwargs)
        if dtype is not None:
            self.target_dtype = dtype
        if device is not None:
            super().to(device=device)
            self.device = torch.device(device)
            self.policy_cfg.device = str(self.device)
        self.restore_native_mixed_precision()
        return self

    def predict_action_batch(self, raw_obs: dict, mode: str = "eval") -> np.ndarray:
        if mode != "eval":
            raise NotImplementedError(
                "SmolVLA CRL backend currently supports eval-only action prediction."
            )

        batch = self._raw_crl_obs_to_lerobot_batch(raw_obs)
        autocast_ctx = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            policy_batch = self.preprocessor(batch)
            noise = self._zero_noise_like_policy_batch(policy_batch) if self.zero_action_noise else None
            action_chunk = self.policy.predict_action_chunk(policy_batch, noise=noise)
            action_chunk = action_chunk[:, : self.num_action_chunks, : self.action_dim]

            flat_actions = action_chunk.reshape(-1, self.action_dim)
            env_flat_actions = self.postprocessor(flat_actions)

        if torch.is_tensor(env_flat_actions):
            env_flat_actions = env_flat_actions.detach().cpu().to(torch.float32).numpy()
        env_flat_actions = np.asarray(env_flat_actions, dtype=np.float32)
        env_actions = env_flat_actions.reshape(
            action_chunk.shape[0],
            action_chunk.shape[1],
            -1,
        )

        if self.log_action_stats and not self._logged_action_stats:
            self._logged_action_stats = True
            gripper = env_actions[..., -1]
            print(
                "[SmolVLA] postprocessed action stats before CRL env adapter: "
                f"shape={env_actions.shape}, min={env_actions.min():.4f}, "
                f"max={env_actions.max():.4f}, gripper_min={gripper.min():.4f}, "
                f"gripper_max={gripper.max():.4f}"
            )

        return env_actions

    def predict_action_step_batch(
        self,
        raw_obs: dict,
        done_mask: Any = None,
        mode: str = "eval",
    ) -> np.ndarray:
        """LeRobot-equivalent step-wise inference with per-env action queues.

        This mirrors `policy.select_action(...)` behavior at rollout level:
        chunk prediction is only used to refill per-env queues, and exactly one
        action per env is emitted each call.
        """
        if mode != "eval":
            raise NotImplementedError(
                "SmolVLA CRL backend currently supports eval-only action prediction."
            )

        batch = self._raw_crl_obs_to_lerobot_batch(raw_obs)
        autocast_ctx = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            policy_batch = self.preprocessor(batch)
            batch_size = self._infer_batch_size(policy_batch)
            self._ensure_action_queues(batch_size)

            done_rows = self._coerce_done_rows(done_mask, batch_size)
            for env_idx in done_rows:
                self._action_queues[env_idx].clear()

            refill_envs = [
                env_idx
                for env_idx, queue in enumerate(self._action_queues)
                if len(queue) == 0
            ]
            if refill_envs:
                refill_policy_batch = self._slice_policy_batch(policy_batch, refill_envs)
                noise = (
                    self._zero_noise_like_policy_batch(refill_policy_batch)
                    if self.zero_action_noise
                    else None
                )
                refill_chunk = self.policy.predict_action_chunk(
                    refill_policy_batch, noise=noise
                )
                refill_chunk = refill_chunk[
                    :, : int(getattr(self.policy.config, "n_action_steps", 1)), : self.action_dim
                ]

                refill_flat = refill_chunk.reshape(-1, self.action_dim)
                refill_env_flat = self.postprocessor(refill_flat)
                if torch.is_tensor(refill_env_flat):
                    refill_env_flat = (
                        refill_env_flat.detach().cpu().to(torch.float32).numpy()
                    )
                refill_env_chunk = np.asarray(refill_env_flat, dtype=np.float32).reshape(
                    refill_chunk.shape[0], refill_chunk.shape[1], -1
                )

                for local_idx, env_idx in enumerate(refill_envs):
                    for step_action in refill_env_chunk[local_idx]:
                        self._action_queues[env_idx].append(step_action.copy())

            step_actions = []
            for env_idx, queue in enumerate(self._action_queues):
                if len(queue) == 0:
                    raise RuntimeError(
                        f"SmolVLA queue refill failed for env index {env_idx}."
                    )
                step_actions.append(queue.popleft())

        step_actions = np.stack(step_actions, axis=0).astype(np.float32, copy=False)
        if self.log_action_stats and not self._logged_step_action_stats:
            self._logged_step_action_stats = True
            gripper = step_actions[:, -1]
            print(
                "[SmolVLA] step-wise action stats before CRL env adapter: "
                f"shape={step_actions.shape}, min={step_actions.min():.4f}, "
                f"max={step_actions.max():.4f}, gripper_min={gripper.min():.4f}, "
                f"gripper_max={gripper.max():.4f}"
            )

        return step_actions[:, None, :]

    def _zero_noise_like_policy_batch(self, policy_batch: dict[str, Any]) -> torch.Tensor:
        device, _, batch_size = self._first_tensor_device_dtype(policy_batch)
        dtype = self._policy_action_dtype()
        action_dim = int(self.policy.config.action_feature.shape[0])
        chunk_size = int(getattr(self.policy.config, "chunk_size", self.num_action_chunks))
        max_action_dim = int(getattr(self.policy.config, "max_action_dim", action_dim))
        return torch.zeros(
            (batch_size, chunk_size, max_action_dim),
            device=device,
            dtype=dtype,
        )

    def _ensure_action_queues(self, batch_size: int) -> None:
        if len(self._action_queues) == batch_size:
            return
        n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        self._action_queues = [deque(maxlen=n_action_steps) for _ in range(batch_size)]

    @staticmethod
    def _infer_batch_size(batch: dict[str, Any]) -> int:
        for value in batch.values():
            if torch.is_tensor(value):
                return int(value.shape[0])
        raise ValueError("Could not infer SmolVLA policy batch size.")

    @staticmethod
    def _slice_policy_batch(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        if not indices:
            return batch

        sliced: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=value.device)
                sliced[key] = value.index_select(0, index_tensor)
            elif isinstance(value, dict):
                sliced[key] = SmolVLAForEvalActionPrediction._slice_policy_batch(
                    value, indices
                )
            elif isinstance(value, list):
                sliced[key] = [value[i] for i in indices]
            elif isinstance(value, tuple):
                sliced[key] = tuple(value[i] for i in indices)
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _coerce_done_rows(done_mask: Any, batch_size: int) -> set[int]:
        if done_mask is None:
            return set()

        if torch.is_tensor(done_mask):
            done_arr = done_mask.detach().cpu().numpy()
        else:
            done_arr = np.asarray(done_mask)

        if done_arr.size == 0:
            return set()

        if done_arr.ndim == 1:
            row_done = done_arr.astype(bool)
        else:
            row_done = np.any(done_arr.astype(bool), axis=tuple(range(1, done_arr.ndim)))

        if row_done.shape[0] != batch_size:
            raise ValueError(
                "SmolVLA done mask batch size mismatch: "
                f"got {row_done.shape[0]}, expected {batch_size}."
            )
        return {idx for idx, is_done in enumerate(row_done.tolist()) if is_done}

    @staticmethod
    def _first_tensor_device_dtype(batch: dict[str, Any]) -> tuple[torch.device, torch.dtype, int]:
        for value in batch.values():
            if torch.is_tensor(value):
                dtype = value.dtype if value.is_floating_point() else torch.float32
                return value.device, dtype, int(value.shape[0])
        raise ValueError("Could not infer SmolVLA batch size/device from policy batch.")

    def _action_distribution(self, action_mean: torch.Tensor) -> Normal:
        std = torch.exp(self.log_std).clamp_min(self.min_action_std)
        std = std.to(device=action_mean.device, dtype=action_mean.dtype)
        std = std.view(1, 1, -1).expand_as(action_mean)
        return Normal(loc=action_mean, scale=std)

    def _predict_policy_action_chunk(
        self,
        policy_batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use the underlying policy model directly so gradients can flow in actor training.
        batch = self.policy._prepare_batch(policy_batch)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[self._obs_language_tokens_key]
        lang_masks = batch[self._obs_language_attention_mask_key]
        actions = self.policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        original_action_dim = int(self.policy.config.action_feature.shape[0])
        actions = actions[:, :, :original_action_dim]
        if getattr(self.policy.config, "adapt_to_pi_aloha", False):
            actions = self.policy._pi_aloha_encode_actions(actions)
        return actions

    def _raw_crl_obs_to_lerobot_batch(self, raw_obs: dict) -> dict[str, Any]:
        if "images_and_states" not in raw_obs:
            raise KeyError("Expected raw_obs['images_and_states'] for SmolVLA LIBERO eval.")
        images_and_states = raw_obs["images_and_states"]

        batch: dict[str, Any] = {}
        self._add_images(batch, images_and_states)
        batch["observation.state"] = self._state_to_tensor(images_and_states["state"])
        batch["task"] = list(raw_obs["task_descriptions"])
        return batch

    def _input_dtype(self) -> torch.dtype:
        return self._module_param_dtype(self.policy.model.state_proj)

    @staticmethod
    def _module_param_dtype(module: torch.nn.Module) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _policy_action_dtype(self) -> torch.dtype:
        return self._module_param_dtype(self.policy.model.action_in_proj)

    @staticmethod
    def _parse_to_device_dtype(*args, **kwargs) -> tuple[torch.device | None, torch.dtype | None]:
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                device = first.device
                dtype = first.dtype if first.is_floating_point() else dtype
            elif isinstance(first, torch.dtype):
                dtype = first
            elif isinstance(first, (str, torch.device)):
                device = first

        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]

        if device is not None:
            device = torch.device(device)
        return device, dtype

    def _add_images(self, batch: dict[str, Any], images_and_states: dict[str, Any]) -> None:
        expected_keys = [
            key
            for key in self.policy.config.image_features.keys()
            if "empty_camera" not in key
        ]
        if not expected_keys:
            raise ValueError("SmolVLA checkpoint has no non-empty image features.")

        full_image = images_and_states.get("full_image")
        wrist_image = images_and_states.get("wrist_image")
        if full_image is None:
            raise KeyError("Expected raw_obs['images_and_states']['full_image'].")
        if len(expected_keys) > 1 and wrist_image is None:
            raise KeyError(
                "SmolVLA checkpoint expects multiple image streams "
                f"({expected_keys}), but CRL observation is missing wrist_image. "
                "Set LIBERO env num_images_in_input=2 for SmolVLA eval parity."
            )

        used_full = False
        used_wrist = False
        for idx, key in enumerate(expected_keys):
            key_lower = key.lower()
            wants_wrist = "wrist" in key_lower or "image2" in key_lower or idx > 0
            if wants_wrist and wrist_image is not None and not used_wrist:
                batch[key] = self._image_to_tensor(wrist_image)
                used_wrist = True
            elif not used_full:
                batch[key] = self._image_to_tensor(full_image)
                used_full = True
            elif wrist_image is not None and not used_wrist:
                batch[key] = self._image_to_tensor(wrist_image)
                used_wrist = True
            else:
                batch[key] = self._image_to_tensor(full_image)

        if not any(key in batch for key in expected_keys):
            raise ValueError(
                "Could not map CRL LIBERO images to SmolVLA image features: "
                f"{expected_keys}"
            )

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, list):
            if value and torch.is_tensor(value[0]):
                return torch.stack([item.detach().clone() for item in value], dim=0)
            return torch.as_tensor(np.asarray(value))
        return torch.as_tensor(value)

    def _state_to_tensor(self, value: Any) -> torch.Tensor:
        state = self._to_tensor(value).to(dtype=self._input_dtype())
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"Expected state shape [B, D], got {tuple(state.shape)}.")
        return state

    def _image_to_tensor(self, value: Any) -> torch.Tensor:
        image = self._to_tensor(value)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim == 4 and image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2).contiguous()
        elif image.ndim == 5 and image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 1, 4, 2, 3).contiguous()
        if image.ndim not in (4, 5):
            raise ValueError(
                "Expected image shape [B,H,W,C], [B,C,H,W], [B,T,H,W,C], "
                f"or [B,T,C,H,W], got {tuple(image.shape)}."
            )

        image = image.to(dtype=self._input_dtype())
        if image.numel() > 0 and image.max() > 1.5:
            image = image / 255.0
        return image
