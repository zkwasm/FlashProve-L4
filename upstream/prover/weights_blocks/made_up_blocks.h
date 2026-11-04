// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The quantized tensors (ggml's formats Q4_K and Q5_K) and the norms' gains of the synthetic model
// ("made_up" in identifiers) that cp_commit commits to under `-weights made-up-blocks`. Each byte
// is a function of a seed and its position in the model: nothing is stored. Only block()'s type argument is checked.
//
// MADE_UP_WIDTH chooses at compile time how the tensors that are given as blocks are stored: 0, the default, is the mix
// of Q4_K and Q5_K; 8 is Q8_0; 16 is F16; 17 is BF16. The weights differ from width to width, so each width has its own
// commitment. No target of ./build.sh sets the macro. A test hands it to nvcc through nvcc's own environment variable:
//   NVCC_APPEND_FLAGS=-DMADE_UP_WIDTH=8 OUT=<a folder for this width> ./build.sh <target>
// A serving target's compiler does not see that variable. The prover and the committer must be built with the same
// width, so give each width a build folder of its own (OUT). make_model, which g++ builds, never gets the macro: the
// types in the model files that it writes follow its own arguments.
#pragma once
#include <stdint.h>
#ifndef MADE_UP_WIDTH
#define MADE_UP_WIDTH 0
#endif
static_assert(MADE_UP_WIDTH == 0 || MADE_UP_WIDTH == 8 || MADE_UP_WIDTH == 16 || MADE_UP_WIDTH == 17,
              "MADE_UP_WIDTH must be 0 (Q4_K and Q5_K), 8 (Q8_0), 16 (F16) or 17 (BF16)");
#if defined(__CUDACC__)
#define MADE_UP_HD __host__ __device__
#else
#define MADE_UP_HD
#endif

namespace made_up {

// Scrambles a 64-bit word with SplitMix64's increment and output mix. Different inputs give different outputs.
MADE_UP_HD inline uint64_t mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// Returns pseudo-random word i of a block. stream is the tensor's number (2 to 5: q, k, v, o; 6 to 8: gate, up, down;
// 10 to 12: routed gate, up, down). Requires layer, expert < 2^10, row < 2^16, block < 2^8 and i < 32.
// word() does not check this: a larger expert, row, block or i repeats another position's words.
// Public source: Counter-based generation: Salmon, Moraes, Dror and Shaw, "Parallel Random Numbers:
// As Easy as 1, 2, 3" (SC 2011, section 5.1).
// Public source: the nesting hash(X + hash(Y)): Jarzynski and Olano, "Hash Functions for GPU
// Rendering" (JCGT 9(3), 2020, section 4.2).
MADE_UP_HD inline uint64_t word(uint64_t seed, int stream, uint64_t layer, uint64_t expert, uint64_t row,
                                uint64_t block, int i) {
  const uint64_t place = ((layer * 1024 + expert) * 65536 + row) * 256 + block;
  return mix(mix(seed + 0x51ull * (uint64_t)stream) + place * 32 + (uint64_t)i);
}

// Returns the type of a stream's tensor in a layer (0: Q4_K, 1: Q5_K), for streams 2 to 8 and 10 to
// 12. Streams 10 and 11 are always Q4_K. With 48 layers, of 24 routed down projections (stream 12)
// 9 are Q5_K: that share follows the counts in the public model's file headers
// (THIRD_PARTY.md, "ggml", says where it comes from). Which layers have the larger type is made up.
// With MADE_UP_WIDTH 8, 16 or 17 every tensor has the type 4 (Q8_0), 5 (F16) or 6 (BF16). The numbers
// are those of wb::BlockType (prover/weights_blocks/blocks.h).
MADE_UP_HD inline int type_of(int stream, uint64_t layer) {
  if (MADE_UP_WIDTH) return MADE_UP_WIDTH == 8 ? 4 : MADE_UP_WIDTH == 16 ? 5 : 6;
  const uint64_t m = (layer >> 1) & 7;
  if (stream == 12) return m == 0 || m == 3 || m == 6;
  return stream >= 2 && stream <= 8 && (layer * 5 + (uint64_t)stream) % 4 == 1;
}

// Returns the bytes of a block of 256 weights of this type (the sizes of wb::block_bytes), or 0 for a number that
// type_of() never returns.
MADE_UP_HD inline int block_bytes(int type) {
  return type == 0 ? 144 : type == 1 ? 176 : type == 4 ? 272 : type == 5 || type == 6 ? 512 : 0;
}

// Returns the 16-bit pattern of n * 2^p, truncated towards zero: binary16, or bfloat16 if bf16 is set. The result must
// be below 2^15. A binary16 result below 2^-14 is subnormal. A bfloat16 result below 2^-126 is given as zero.
// Public source: normalizing a product on the unpacked fields is Algorithm N in D. E. Knuth, The Art of Computer
// Programming, vol. 2, section 4.2.1 (see half_range() below); here the fraction is an integer of up to 32 bits.
MADE_UP_HD inline uint32_t pattern16(uint32_t n, int p, bool bf16) {
  if (!n) return 0;
  const int keep = bf16 ? 8 : 11; // significant bits
  while (n >= (1u << keep)) n >>= 1, p++;
  while (n < (1u << (keep - 1))) n <<= 1, p--;
  const int field = p + keep - 1 + (bf16 ? 127 : 15);
  if (field <= 0) return bf16 || 1 - field > 11 ? 0 : n >> (1 - field);
  return ((uint32_t)field << (keep - 1)) | (n - (1u << (keep - 1)));
}

// Writes a block of 256 weights of a wider type to out: 272 bytes for type 4 (eight Q8_0 blocks), 512 for type 5 (F16)
// or 6 (BF16). 'first' is the block's word 0 (word()). A wider block needs more than word()'s 32 words, so its word i
// is mix(first + 1 + i), one more nesting of the same kind. The weights have the spread of the Q4_K blocks of block(),
// so that the quantization scales that fit those also fit these. As there, the block has a scale d, which is composed
// from 'first' in the same way, and each sub-block of 32 weights has a scale sc from 0 to 63. A weight is spread evenly
// from -7.5 d sc to 7.5 d sc, where a Q4_K weight takes one of 16 values in about that range. So 4096 * |v| <= 4096
// * 7.5 * 63 * (1 + 959 / 1024) * 2^-10 < 3661.
// - Q8_0: a sub-block's scale is 15 d sc / 256, truncated to binary16, and its codes are random bytes.
// - F16, BF16: v = d sc t / 16, truncated, with t from 0 to 120 and a random sign.
// Public sources for values that are composed field by field: see block().
MADE_UP_HD inline void wide_block(int type, uint64_t first, uint8_t *out) {
  const uint32_t negative = ((first >> 20) & 15) == 0 ? 0x8000u : 0u;
  const int exponent = 4 + (int)(first & 1) - 25; // d = fraction * 2^exponent
  const uint32_t fraction = ((first >> 28) & 255) == 0 ? 0 : 1024u + (uint32_t)((first >> 1) % 960);
  const uint64_t scales = mix(first + 1);
  for (int j = 0; j < 8; j++) {
    const uint32_t ds = fraction * (uint32_t)((scales >> (6 * j)) & 63);
    if (type == 4) {
      const uint32_t d = negative | pattern16(ds * 15u, exponent - 8, false);
      out[0] = (uint8_t)d, out[1] = (uint8_t)(d >> 8);
      for (int g = 0; g < 4; g++) {
        const uint64_t codes = mix(first + 2 + (uint64_t)(4 * j + g));
        for (int i = 0; i < 8; i++) out[2 + 8 * g + i] = (uint8_t)(codes >> (8 * i));
      }
      out += 34;
      continue;
    }
    for (int g = 0; g < 8; g++) {
      const uint64_t four = mix(first + 2 + (uint64_t)(8 * j + g));
      for (int i = 0; i < 4; i++) {
        const uint32_t r = (uint32_t)(four >> (16 * i)) & 65535u, t = ((r & 127u) * 121u) >> 7;
        const uint32_t h = (r & 0x8000u) | pattern16(ds * t, exponent - 4, type == 6);
        out[8 * g + 2 * i] = (uint8_t)h, out[8 * g + 2 * i + 1] = (uint8_t)(h >> 8);
      }
    }
    out += 64;
  }
}

// Returns the binary16 pattern of c * d / 2, truncated towards zero. d16 is d's pattern; c is 15 or
// 31. d must be normal: zero, subnormal numbers and overflow are not handled.
// Public source: Multiplying on the unpacked fields is Algorithm M with Algorithm N in D. E. Knuth,
// The Art of Computer Programming, vol. 2, section 4.2.1, stated there for two normalized operands
// with rounding to nearest; here one operand is a plain integer, the halving is 1 taken from the
// exponent, and the result is truncated.
MADE_UP_HD inline uint32_t half_range(uint32_t d16, uint32_t c) {
  uint32_t v = (1024u + (d16 & 1023u)) * c, e = ((d16 >> 10) & 31u) - 1u;
  while (v >= 2048u) {
    v >>= 1;
    e++;
  }
  return (d16 & 0x8000u) | (e << 10) | (v - 1024u);
}

// Writes a block (256 weights) to out, field by field, without a quantizer. Its 144 (Q4_K) or 176
// (Q5_K) bytes: d and dmin (binary16, little-endian); 12 bytes that pack a 6-bit scale sc and
// minimum m for each of 8 sub-blocks; for Q5_K, 32 bytes of the codes' fifth bits; 128 bytes of
// the codes' low 4 bits. A code q decodes to v = d * sc * q - dmin * m. Here m = sc and dmin is d
// times half the largest code (truncated), so the mean of v is close to zero.
// Every v has 4096 * |v| < 3673, and gains are below 1.09375 (gain_bits()). So the committed
// integer, 4096 * gain * v rounded, stays below 4095 in magnitude, the limit of
// BlockWeights::fill() (block_weights.cuh). A block of a wider type is written by wide_block(), which
// has its own bound.
// Public source: vLLM's 'dummy' load format, random weights: vLLM documentation v0.4.1, Engine
// Arguments, --load-format, https://docs.vllm.ai/en/v0.4.1/models/engine_args.html.
// Public source: a valid value for each field: R. Dutra, R. Gopinath, A. Zeller, "FormatFuzzer:
// Effective Fuzzing of Binary File Formats", 2021, section 3, https://arxiv.org/abs/2109.11277.
// Public source: codes centred by half their range: Modular, MAX/Mojo kernels API documentation,
// quantization.per_channel_grouped_4bit, struct Q4sym,
// https://docs.modular.com/mojo/kernels/quantization/per_channel_grouped_4bit/Q4sym.
// Public source: the minimum beside each scale in Hugging Face's table: Hugging Face Hub
// documentation, GGUF, table "Quantization Types", https://huggingface.co/docs/hub/gguf.
MADE_UP_HD inline void block(int type, uint64_t seed, int stream, uint64_t layer, uint64_t expert, uint64_t row,
                             uint64_t blk, uint8_t *out) {
  if (!block_bytes(type)) return; // a number that type_of() never returns: nothing is written
  auto w = [&](int i) {
    return word(seed, stream, layer, expert, row, blk, i);
  };
  const uint64_t r = w(0);
  if (type >= 4) {
    wide_block(type, r, out);
    return;
  }
  // d, the block's scale, is a binary16 pattern composed from one random word r. It is zero for one
  // r in 256, otherwise negative for one in 16. Its exponent field is 4 or 5 and its mantissa field
  // is below 960 (Q4_K), or 3 or 4 and below 900 (Q5_K); the bound 3673 rests on these limits.
  // Public source: a float composed field by field: M. Frojd, A. Khan, J. Makela, "Constraining the
  // Real Problem of Floating Point Number Distribution", DVCon 2014, section IV (A, B, D).
  uint32_t d = ((((r >> 20) & 15) == 0 ? 1u : 0u) << 15) |
               (type ? ((3u + (uint32_t)(r & 1)) << 10) | (uint32_t)((r >> 1) % 900)
                     : ((4u + (uint32_t)(r & 1)) << 10) | (uint32_t)((r >> 1) % 960));
  if (((r >> 28) & 255) == 0) d = 0;
  const uint32_t dmin = d ? half_range(d, type ? 31u : 15u) : 0;
  out[0] = (uint8_t)d, out[1] = (uint8_t)(d >> 8), out[2] = (uint8_t)dmin, out[3] = (uint8_t)(dmin >> 8);
  const uint64_t s1 = w(1), s2 = w(2);
  for (int i = 0; i < 4; i++) {
    out[4 + i] = out[8 + i] = (uint8_t)(s1 >> (8 * i));
    const uint8_t n = (uint8_t)(s2 >> (8 * i)) & 15u;
    out[12 + i] = (uint8_t)(n | (n << 4));
  }
  uint8_t *codes = out + 16;
  if (type) {
    for (int g = 0; g < 4; g++)
      for (int i = 0; i < 8; i++) codes[8 * g + i] = (uint8_t)(w(19 + g) >> (8 * i));
    codes += 32;
  }
  for (int g = 0; g < 16; g++)
    for (int i = 0; i < 8; i++) codes[8 * g + i] = (uint8_t)(w(3 + g) >> (8 * i));
}

// Returns the norm whose gains multiply a stream's tensor: 0 attention, 1 feed-forward, -1 none.
MADE_UP_HD inline int norm_of(int stream) {
  return stream >= 2 && stream <= 4 ? 0 : (stream == 6 || stream == 7 || stream == 10 || stream == 11) ? 1 : -1;
}
// Returns the gain of input column j as binary32 bits, a fixed exponent and a random fraction: in
// [0.5, 1) seven times in eight, otherwise in [1, 1.09375), as 786432 = 0.09375 * 2^23. norm is
// norm_of()'s 0 or 1. Requires layer, j < 2^20 (not checked): beyond, two positions share a gain.
// Public source: The word is made by the method that Saito and Matsumoto recall in section 2 of
// "A PRNG specialized in double precision floating point numbers using an affine transition"
// (Monte Carlo and Quasi-Monte Carlo Methods 2008, Springer, 2009), there for binary64 in [1, 2).
MADE_UP_HD inline uint32_t gain_bits(uint64_t seed, int norm, uint64_t layer, uint64_t j) {
  const uint64_t r = mix(seed + ((uint64_t)(90 + norm) << 40) + (layer << 20) + j);
  return (r & 7) ? (0x3F000000u | ((uint32_t)(r >> 8) & 0x7FFFFFu)) : (0x3F800000u | (uint32_t)((r >> 8) % 786432u));
}

} // namespace made_up
