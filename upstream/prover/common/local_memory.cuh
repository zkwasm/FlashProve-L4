// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A run-time check that the compiler gave a kernel no local memory. Some kernels hold hidden values
// in registers only. If the compiler spills such a register, or places an array in a stack frame,
// the values are in local memory: GPU memory owned by the CUDA driver, which no caller can address
// and so none can wipe. The build (compiler, flags, architecture) decides whether that happens; the
// source does not show it. A test that is not part of this repository looks for local memory in
// built programs; this check runs in the program itself, so it also covers an untested build. Three
// kernels are checked this way: k_zfold_turn() (prover/common/zfold_turned.cuh), a byte-matrix
// transpose, and the two kernels of prover/hash_commitment/hf_pads.cuh.
// The idea is public: querying a kernel's attributes:
// [attributes] NVIDIA, CUDA Runtime API, "Execution Control", cudaFuncGetAttributes(),
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html (as of 2026-07-01)
// and "cudaFuncAttributes Struct Reference",
// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html (as of 2025-09-21).
// The pages do not say whether localSizeBytes, each thread's local memory in bytes, counts a stack
// frame; another test outside this repository requires that a control kernel with an array in a
// stack frame is rejected. Spilled registers count too: a build of k_zfold_turn() that spilled
// three pairs was rejected with "24 bytes". No test repeats that.
#pragma once
#include "common/common.h" // CK
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
namespace local_memory {
// Exits with code 1 if the kernel has local memory, or on a CUDA error. Call it before the
// kernel's first launch. It first checks the calling thread's CUDA status, as CK_ATTRIBUTE
// (prover/common/common.h) does, in case cudaFuncGetAttributes() resets it and loses an earlier
// launch's error.
// The answer holds only for the calling thread's current GPU.
template <class Kernel> static inline void refuse_local_memory(Kernel *kernel, const char *name) {
  CK(cudaGetLastError());
  cudaFuncAttributes is{};
  CK(cudaFuncGetAttributes(&is, kernel));
  if (is.localSizeBytes) {
    printf("internal error: the kernel %s has %zu bytes of local memory for each thread and must have none: the "
           "program stops\n",
           name, is.localSizeBytes);
    exit(1);
  }
}
} // namespace local_memory
