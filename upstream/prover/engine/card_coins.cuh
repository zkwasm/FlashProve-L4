// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// `Coins` (private randomness, blocks/engine/table.h) that also fill GPU memory (`card` in
// identifiers). Every Merkle leaf needs a salt and every main row a random tail (its encoding
// randomness); generating them on the host and copying would take longer than the hashing.
// EngineGpu::commit() (prover/engine/engine_gpu.cuh) therefore calls the functions below if given a
// `CardCoins`. Implementations: prover/engine/stream_coins.cuh;
// prover/engine/tests/expand_coins.cuh (tests and benchmarks).
#pragma once
#include "engine/device.cuh"
#include "engine/table.h"
#include "hash/hash.h"

namespace blk {

class CardCoins : public Coins {
public:
  // Like fill() and bytes(), but into GPU memory, queued on `stream`. bytes_card(): 8 divides n,
  // and `out` must be 8-byte aligned (written as u64 words; not checked).
  virtual void fill_card(F *out, size_t n, cudaStream_t stream) = 0;
  virtual void bytes_card(u8 *out, size_t n, cudaStream_t stream) = 0;
};

} // namespace blk
