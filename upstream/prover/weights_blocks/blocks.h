// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// How a layer's weight matrix is represented in GPU memory for the forward pass, if the model is
// stored as quantized blocks (GGUF): as the file's own blocks (Blocks) or two byte planes (Planes).
// Neither struct owns its memory; BlockWeights::fill() (block_weights.cuh) fills in both.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// `wb`, after prover/weights_blocks/: types and kernels for weights stored as quantized blocks.
namespace wb {

// How the file stores a tensor (`ty` in fold_rule.h). A row is divided into blocks of 256 elements.
// For Q8_0, whose own blocks have 32 elements and 34 bytes, a block here is eight of them, one after
// the other. For F16 (binary16) and BF16 (bfloat16) it is 256 values of two bytes.
enum BlockType : int { Q4_K = 0, Q5_K = 1, Q6_K = 2, F32 = 3, Q8_0 = 4, F16 = 5, BF16 = 6 };
inline size_t block_bytes(int type) {
  if (type < Q4_K || type > BF16) {
    fprintf(stderr, "internal error: %d is not the number of a storage type (weights_blocks/blocks.h)\n", type);
    exit(4);
  }
  return type == Q4_K ? 144 : type == Q5_K ? 176 : type == Q6_K ? 210 : type == F32 ? 1024 : type == Q8_0 ? 272 : 512;
}
// Whether the kernels of weights_blocks/decode.cuh read this type, so that a tensor of it can stay as blocks.
inline bool decoded_from_blocks(int type) {
  return type == Q4_K || type == Q5_K || type == Q8_0 || type == F16 || type == BF16;
}

// A tensor kept as the file's blocks. Kernels convert a block to committed integers (the integer
// model's weights; rule and exponent e: fold_rule.h) with `gain`, one multiplier per input column
// or null, and `scale` = 2^e. `inputs` must be a multiple of 256 and `bytes` 16-byte aligned:
// decode.cuh loads 128-bit words. A routed tensor's experts are expert_bytes() apart.
struct Blocks {
  const uint8_t *bytes = nullptr;
  int type = Q4_K; // a type for which decoded_from_blocks() holds
  size_t rows = 0; // of one expert, if routed
  int inputs = 0;
  const float *gain = nullptr;
  float scale = 4096.f;
  size_t expert_bytes() const { return rows * (size_t)(inputs / 256) * block_bytes(type); }
};

// Each committed integer w is split into balanced limbs: w = w0 + 256 * w1, w0 from -128 to 127.
// A plane holds one limb of every w, indexed [row][input]; a row has `inputs` bytes.
struct Planes {
  const int8_t *w0 = nullptr, *w1 = nullptr;
  size_t rows = 0;
  int inputs = 0;
};

} // namespace wb
