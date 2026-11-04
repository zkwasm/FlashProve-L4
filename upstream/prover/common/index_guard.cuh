// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Bounds checks for GPU kernels, made where a number that comes from a request's values is used as
// an index: a hardware fault in GPU memory or a test can have changed the number. An out-of-range
// index can fault a kernel, and on a shared machine that can bring down other users' work too.
#pragma once
#include "common/slot_flags.h"
// What the checked number is (the parameter `use`), one bit each. "Position" means token position.
enum : unsigned {
  INDEX_EXPERT = 1,       // a position's expert in a mixture-of-experts (MoE) layer
  INDEX_FUNCTION_KEY = 2, // a lookup key of a function table (one stored value per key)
  INDEX_ARG_MAX = 4,      // the argmax of a row of attention scores
  INDEX_WINNER = 8,       // a position's checked token (the token after it) in the output head
  // a request's token
  INDEX_TOKEN = 16,
  INDEX_ROUTER_LOGIT = 32, // a position's largest router logit
  // an entry of the tables that order the tokens by expert (MoeTables, prover/prover/forward_moe.cuh)
  // or of the host's copy of them, or an index computed from such entries
  INDEX_MOE_TABLE = 64,
  INDEX_HISTOGRAM = 128 // a bin of a position's histogram of logit distances in the output head
};
// Reports a failed check. Nothing stops: the caller continues with a harmless index. `INDEX` is a
// failure bit of the request that the kernel works for (prover/common/slot_flags.h): its proof and
// token file are withheld; other requests keep theirs. Set between requests, it fails the batch.
// The cause need not be a hardware fault: out-of-range forward-pass values can give such numbers.
__device__ __forceinline__ void index_is_outside(unsigned use) {
  atomicOr(&g_live_flags_device[1], use);
  raise_flags(INDEX);
}
// Returns i if 0 <= i < n. If not, reports the failure and returns 0, valid in any non-empty table.
__device__ __forceinline__ unsigned index_below(long long i, unsigned n, unsigned use) {
  if (i >= 0 && i < (long long)n) return (unsigned)i;
  index_is_outside(use);
  return 0;
}
// For the indexes first, first + 1, ..., first + count - 1: returns count if 0 <= count <= most,
// first >= 0 and first + count <= n. If not, reports the failure, sets `first` to 0 and returns 0.
__device__ __forceinline__ int indexes_below(int &first, long long count, int most, unsigned n, unsigned use) {
  if (count >= 0 && count <= most && first >= 0 && first + count <= (long long)n) return (int)count;
  index_is_outside(use);
  first = 0;
  return 0;
}
// For tests: sets *word to the `use` bits in g_live_flags_device, words 1 and 3 (3: set between two
// requests). It misses bits saved with a request's state. Call it after the checking kernels end.
// Public source for comparison: see prover/common/slot_flags.h, g_live_flags_device.
static inline cudaError_t indexes_outside(unsigned *word) {
  unsigned words[4] = {0, 0, 0, 0};
  const cudaError_t e = cudaMemcpyFromSymbol(words, g_live_flags_device, sizeof words);
  *word = words[1] | words[3];
  return e;
}
