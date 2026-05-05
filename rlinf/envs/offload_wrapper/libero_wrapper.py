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

import io

import numpy as np
import torch

from rlinf.envs.libero.libero_env import LiberoEnv as BaseLiberoEnv
from rlinf.envs.offload_wrapper.base import EnvOffloadMixin


class LiberoEnv(BaseLiberoEnv, EnvOffloadMixin):
    def get_state(self) -> bytes:
        simulator_state = {
            "generator_state": self._generator.bit_generator.state,
            "generator_ordered_state": self._generator_ordered.bit_generator.state,
            "start_idx": self.start_idx,
            "reset_state_ids_all": self.reset_state_ids_all,
            "reset_state_ids": self.reset_state_ids,
            "task_ids": self.task_ids,
            "trial_ids": self.trial_ids,
            "task_descriptions": self.task_descriptions,
            "prev_step_reward": self.prev_step_reward,
            "success_once": self.success_once,
            "fail_once": self.fail_once,
            "returns": self.returns,
            "_elapsed_steps": self._elapsed_steps,
            "video_cnt": self.video_cnt,
        }

        buffer = io.BytesIO()
        torch.save(simulator_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        self._generator.bit_generator.state = state["generator_state"]
        self._generator_ordered.bit_generator.state = state["generator_ordered_state"]
        self.start_idx = state["start_idx"]
        self.reset_state_ids_all = state["reset_state_ids_all"]
        self.reset_state_ids = state["reset_state_ids"]
        self.task_descriptions = state["task_descriptions"]
        self.prev_step_reward = state["prev_step_reward"]
        self.success_once = state["success_once"]
        self.fail_once = state["fail_once"]
        self.returns = state["returns"]
        self._elapsed_steps = state["_elapsed_steps"]
        self.video_cnt = state["video_cnt"]

        # A fresh BaseLiberoEnv may have subprocesses configured for different tasks.
        # Mark every slot stale so the next reset reconfigures all vector envs.
        self.task_ids = np.full(self.num_envs, -1, dtype=np.int64)
        self.trial_ids = np.full(self.num_envs, -1, dtype=np.int64)
        self._is_start = True

    def close(self):
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self.env = None


__all__ = ["LiberoEnv"]
