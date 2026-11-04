// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "common/common.h"
#include "field/field_lazy.cuh"
// Returns x * 2^s mod p. Expects a canonical x (below p = 2^64 - 2^32 + 1) and s in [0, 192);
// checks neither. Uses shifts and one reduction (blk::lazy::times_2_to() in
// prover/field/lazy_gl64.cuh).
__host__ __device__ static inline u64 gf_times_2_to(u64 x, int s) {
  return blk::lazy::times_2_to(blk::F{x}, s).v;
}
