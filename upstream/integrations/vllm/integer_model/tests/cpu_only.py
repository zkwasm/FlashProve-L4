# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Lets vLLM's CPU worker start from a vLLM wheel that was built for GPUs. For tests on a machine
without a GPU only.

With VLLM_TARGET_DEVICE=cpu, vLLM 0.29.0 uses its CPU worker, which calls one compiled function that
only the CPU build of vLLM contains (init_cpu_memory_env, called in vllm/v1/worker/cpu_worker.py).
That function binds the process's memory to NUMA nodes. If torch.ops._C has no such function,
register() defines one that does nothing, with a signature that fits the call. The model class
(IntegerModelForCausalLM) needs no other compiled function of vLLM. On a machine without a GPU,
vllm_tools.install() makes register() a vLLM plug-in, so that it runs in each of vLLM's processes.
"""

import torch

_library = None


def register():
    global _library
    if _library is None and not hasattr(torch.ops._C, "init_cpu_memory_env"):
        _library = torch.library.Library("_C", "FRAGMENT")
        _library.define("init_cpu_memory_env(int[] nodes) -> ()")
        _library.impl("init_cpu_memory_env", lambda nodes: None, "CompositeExplicitAutograd")
