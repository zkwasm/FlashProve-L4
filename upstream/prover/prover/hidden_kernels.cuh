// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CUDA kernels for the layer and output-head sub-proofs. T is an element of the extension field E,
// F its arithmetic (prover/hiding/transcript.cuh), and GSL(i, n) a grid-stride loop over i < n.
#pragma once
#include "common/element_from_hash.h"
#include "prover/base.cuh"
// Prover::plan_sections() (header.cuh) counts arrays of T in 32-bit words, EWORDS for each element,
// and a sub-proof's messages are copied by these counts: padding in T would cut a copy short.
static_assert(sizeof(T) == 8 * EDEG && sizeof(T) == 4 * EWORDS, "an element must be EDEG 64-bit coordinates, unpadded");
// Builds n elements from a row that is stored as EDEG planes, one array for each coordinate,
// `stride` 64-bit words apart. Coordinates are not reduced. Tree::slot_pads()
// (prover/hiding/trees.cuh) samples a sub-proof's mask entries, one random element for each message
// slot, into such planes. The sub-proofs launch k_pl2e() to copy the entries into their message
// buffer.
__global__ void k_pl2e(const u64 *plane0, long long stride, size_t n, T *o) {
  GSL(i, n) {
    u64 c[EDEG];
    for (int q = 0; q < EDEG; q++) {
      c[q] = plane0[q * stride + (long long)i];
    }
    o[i] = ext_from(c);
  }
}
// Launch with one block.
__global__ void k_times(const T *a, const T *s, int n, T *o) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = F::mul(a[i], s[0]);
}
__global__ void k_vmul(const T *a, const T *b, size_t n, T *o) {
  GSL(i, n) o[i] = F::mul(a[i], b[i]);
}

