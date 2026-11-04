// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Kernels that overwrite words of a table in GPU memory, or spread one row over a block of rows,
// without copying the table to the host. They do not validate arguments: their callers in
// EngineGpu (engine_gpu.cuh) do.
#pragma once
#include "engine/device.cuh"
#include "engine/encode.cuh"
#include "field/field.h"

namespace blk {

// The indices at[i] must be distinct.
static __global__ void k_replace_words(u64 *words, const u64 *at, const u64 *value, size_t count) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) words[at[i]] = value[i];
}
// Copies entry j of `row` to row first_row + (j mod 2^bits), column j div 2^bits, of `table`.
// Entries are elements of the extension field E (E_DEGREE words); `table` has len columns.
// bits must be below 64.
static __global__ void k_lay_row(u64 *table, const u64 *row, u64 count, u64 first_row, unsigned bits, u64 len) {
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= count) return;
  const u64 to = ((first_row + (j & (((u64)1 << bits) - 1))) * len + (j >> bits)) * E_DEGREE;
  for (unsigned d = 0; d < E_DEGREE; d++) table[to + d] = row[j * E_DEGREE + d];
}

} // namespace blk
