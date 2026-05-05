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

__all__ = ["ManiskillEnv", "LiberoEnv"]


def __getattr__(name):
    if name == "ManiskillEnv":
        from rlinf.envs.offload_wrapper.maniskill_wrapper import ManiskillEnv

        globals()[name] = ManiskillEnv
        return ManiskillEnv

    if name == "LiberoEnv":
        from rlinf.envs.offload_wrapper.libero_wrapper import LiberoEnv

        globals()[name] = LiberoEnv
        return LiberoEnv

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
