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

import copyreg
import gc
import time
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from rlinf.scheduler.worker.worker import Worker


COLOR_END = "\033[0m"


def color_print(rank, *args, **kwargs):
    if "end" in kwargs:
        print(f"\033[{31 + rank}m[DP rank {rank}]", *args, **kwargs)
    else:
        print(f"\033[{31 + rank}m[DP rank {rank}]", *args, **kwargs, end="")
    print("\033[0m")


def green(text: str):
    return f"\033[32m{text}\033[0m"


@contextmanager
def sharp_cover(header_text: str, prelen: int = 30, color="\033[32m"):
    len(header_text)
    print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)

    try:
        yield
    finally:
        print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)


def get_module_from_name(module: torch.nn.Module, name: str):
    """
    Args:
        name: str, the name of the module, e.g. model.layers.0.self_attn.qkv_proj
    """
    parts = name.split(".")

    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module


def rebind_param_attr(
    model: torch.nn.Module,
    name: str,
    reserved_attr: Dict[str, Dict],
    materialize: bool = False,
):
    """
    here name is already converted to the vLLM format.
    """
    name_paths = name.split(".")
    last_name = name_paths[-1]
    assert last_name in ["weight", "bias"]

    module = get_module_from_name(model, ".".join(name_paths[:-1]))

    param = getattr(module, last_name)

    if materialize and param.device.index != torch.cuda.current_device():
        module.to_empty(device=torch.cuda.current_device())

    param.__dict__.update(reserved_attr[name])
    del reserved_attr[name]


def swap_tensor_pointer(t1: torch.Tensor, t2: torch.Tensor):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
    # Ensure there are no weakrefs
    if weakref.getweakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")
    if weakref.getweakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
    t1_slots = set(copyreg._slotnames(t1.__class__))  # type: ignore[attr-defined]
    t2_slots = set(copyreg._slotnames(t2.__class__))  # type: ignore[attr-defined]
    if t1_slots != t2_slots:
        raise RuntimeError("Cannot swap t1 and t2 if they have different slots")

    def swap_attr(name):
        tmp = getattr(t1, name)
        setattr(t1, name, (getattr(t2, name)))
        setattr(t2, name, tmp)

    def error_pre_hook(grad_outputs):
        raise RuntimeError(
            "Trying to execute AccumulateGrad node that was poisoned by swap_tensors "
            "this can happen when you try to run backward on a tensor that was swapped. "
            "For a module m with `torch.__future__.set_swap_module_params_on_conversion(True)` "
            "you should not change the device or dtype of the module (e.g. `m.cpu()` or `m.half()`) "
            "between running forward and backward. To resolve this, please only change the "
            "device/dtype before running forward (or after both forward and backward)."
        )

    def check_use_count(t, name="t1"):
        use_count = t._use_count()
        error_str = (
            f"Expected use_count of {name} to be 1 or 2 with an AccumulateGrad node but got {use_count} "
            f"make sure you are not holding references to the tensor in other places."
        )
        if use_count > 1:
            if use_count == 2 and t.is_leaf:
                accum_grad_node = torch.autograd.graph.get_gradient_edge(t).node
                # Make sure that the accumulate_grad node was not lazy_init-ed by get_gradient_edge
                if t._use_count() == 2:
                    accum_grad_node.register_prehook(error_pre_hook)
                else:
                    raise RuntimeError(error_str)
            else:
                raise RuntimeError(error_str)

    check_use_count(t1, "t1")
    check_use_count(t2, "t2")

    # Swap the types
    # Note that this will fail if there are mismatched slots
    # swap_attr("__class__")

    # Swap the dynamic attributes
    # swap_attr("__dict__")

    # Swap the slots
    # for slot in t1_slots:
    #     if hasattr(t1, slot) and hasattr(t2, slot):
    #         swap_attr(slot)
    #     elif hasattr(t1, slot):
    #         setattr(t2, slot, (getattr(t1, slot)))
    #         delattr(t1, slot)
    #     elif hasattr(t2, slot):
    #         setattr(t1, slot, (getattr(t2, slot)))
    #         delattr(t2, slot)

    # Swap the at::Tensor they point to
    torch._C._swap_tensor_impl(t1, t2)


class CudaMemoryProfiler:
    def __init__(self, device: Optional[torch.types.Device] = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            mem = torch.cuda.max_memory_allocated(self.device)
        return mem

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


class CudaTimeProfiler:
    def __init__(
        self,
        device: Optional[torch.types.Device] = None,
        name: str = "",
        do_print: bool = True,
    ):
        self.device = device
        self.name = name
        self.do_print = do_print

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        self.time_cost = self.start_event.elapsed_time(self.end_event)
        if self.do_print:
            print(green(f"Event {self.name} cost: {self.time_cost:.3f} ms"))


class TimeProfiler:
    def __init__(
        self,
        name: str = "",
        do_print: bool = True,
        do_tb=False,
        writer: SummaryWriter = None,
        tag="",
        step=0,
    ):
        self.name = name
        self.do_print = do_print
        self.do_tb = do_tb
        self.writer = writer
        self.tag = tag
        self.step = step
        if do_tb:
            assert writer is not None

    def __enter__(self):
        self.start_time = time.time()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.time_cost = self.end_time - self.start_time
        if self.do_print:
            print(green(f"Event {self.name} cost: {self.time_cost * 1000:.3f} ms"))
        if self.do_tb:
            self.writer.add_scalar(self.tag, self.time_cost, self.step)
