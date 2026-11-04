// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The base field F under names that do not contain the prime, for the prover's sources:
// p = 2^64 - 2^32 + 1 (Goldilocks). Its constants and arithmetic come from blocks/field/field.h, so
// blocks/ must be on the include path.
// gf_add() and similar names in prover/common/common.h are defined in terms of this file's
// names: they work modulo the p above. A constant or rule that depends on p belongs here.
// Arguments must be canonical (below p) unless noted otherwise; nothing checks this. Results are
// canonical.
#pragma once
#include <cstdint>

#include "base.h"
#include "field/field.h"

// Under nvcc every function below is compiled for host and device.
#if defined(__CUDACC__)
#define FIELD_HD __host__ __device__ static inline
#else
#define FIELD_HD static inline
#endif

// p and its length in bits. x^3 = FIELD_W = 2 in the cubic extension of F; 2 is not a cube in F.
constexpr uint64_t FIELD_P = blk::P;
constexpr int FIELD_BITS = (int)blk::F_BITS;
constexpr uint64_t FIELD_W = blk::NR;
constexpr uint64_t FIELD_WORD_MASK = ~0ull >> (64 - FIELD_BITS); // the low FIELD_BITS bits of a word
constexpr uint64_t FIELD_MULTIPLES = ~0ull / FIELD_P;            // how many multiples of p fit into 64 bits: 1
// The largest multiple of p below 2^64: a uniform word below it is uniform modulo p.
constexpr uint64_t FIELD_WORD_LIMIT = FIELD_MULTIPLES * FIELD_P;
static_assert(0 - FIELD_WORD_LIMIT < FIELD_P, "no further multiple of p fits below 2^64");
constexpr uint64_t FIELD_HALF = FIELD_P / 2 + 1; // 1 / 2 in F

// field_times_w() returns FIELD_W times v. field_from_u64() takes any word.
FIELD_HD uint64_t field_add(uint64_t a, uint64_t b) { return blk::BaseField::add(a, b); }
FIELD_HD uint64_t field_sub(uint64_t a, uint64_t b) { return blk::BaseField::sub(a, b); }
FIELD_HD uint64_t field_mul(uint64_t a, uint64_t b) { return blk::BaseField::mul(a, b); }
FIELD_HD uint64_t field_times_w(uint64_t v) { return blk::BaseField::times_nr(v); }
FIELD_HD uint64_t field_from_u64(uint64_t v) { return blk::BaseField::from_u64(v); }
// The residue of a signed integer.
FIELD_HD uint64_t field_from_signed(long long y) {
  const uint64_t a = field_from_u64(y < 0 ? 0 - (uint64_t)y : (uint64_t)y);
  return y < 0 ? (a ? FIELD_P - a : 0) : a;
}
// One try at a uniform element from a uniform word: the word's low FIELD_BITS bits, rejected
// (false, out = 0) if they are not below p. What is not rejected is exactly uniform: no word is
// reduced modulo p. About one word in 2^32 is rejected, so second tries do occur in practice.
FIELD_HD bool field_from_word(uint64_t word, uint64_t &out) {
  const uint64_t low = word & FIELD_WORD_MASK;
  out = low < FIELD_P ? low : 0;
  return low < FIELD_P;
}
