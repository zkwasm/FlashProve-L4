// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Integer arithmetic for the weight commitment's parameters (hf_table_params.h,
// hf_opening_params.h): logarithms, the number of codeword positions to open for a target of 2^-bits,
// given one position's error, a comparison with a fixed bound below the size of the extension field
// E, and the height of a Merkle tree's truncated top layer. No floating point, so no result depends
// on rounding. Host code only.
// Terms: docs/glossary.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hf {

// ceil(log2 x); 0 for x = 0. For x > 2^63 it returns 63, not 64, and power_of_two_at_least(x) < x.
inline int log2_ceil(uint64_t x) {
  int l = 0;
  while (l < 63 && ((uint64_t)1 << l) < x) l++;
  return l;
}
inline uint64_t power_of_two_at_least(uint64_t x) { return (uint64_t)1 << log2_ceil(x); }

// Returns the least t >= 1 with (NUM / 2^LD)^t <= 2^-bits, tested as NUM^t <= 2^(t LD - bits): the
// number of positions to open if one position's error is NUM / 2^LD. Returns 0 if bits <= 0;
// otherwise -1 if NUM is 0, NUM >= 2^LD, LD is not in 1 .. 40, or no t <= `most` satisfies it.
// tcol() in prover/hiding/trees.cuh has the same rule on narrower ranges; this function is meant
// for NUM up to 2^34.
inline long least_count(uint64_t NUM, int LD, int bits, long most = 1 << 20) {
  if (bits <= 0) return 0;
  if (!NUM || LD < 1 || LD > 40 || NUM >> LD) return -1;
  std::vector<uint32_t> p{1}; // NUM^t, least limb first
  for (long t = 1; t <= most; t++) {
    unsigned __int128 carry = 0;
    for (auto &x : p) {
      carry += (unsigned __int128)x * NUM;
      x = (uint32_t)carry;
      carry >>= 32;
    }
    while (carry) p.push_back((uint32_t)carry), carry >>= 32;
    const long E = t * LD - bits, len = 32L * (long)(p.size() - 1) + (32 - __builtin_clz(p.back()));
    if (len <= E) return t; // NUM^t < 2^E
    if (len == E + 1) {     // NUM^t = 2^E exactly?
      bool pow2 = (p.back() & (p.back() - 1)) == 0;
      for (size_t i = 0; i + 1 < p.size(); i++) pow2 = pow2 && !p[i];
      if (pow2) return t;
    }
  }
  return -1;
}

// Returns the height c of the truncated top layer when `count` leaves of a Merkle tree of depth ln
// are opened: the 2^c digests at depth c are given once, and each leaf comes with its ln - c lowest
// siblings. c is the least number in 0 .. ln that minimizes their total, 2^c + count (ln - c).
// Not checked: 0 <= ln <= 63, count ln < 2^63.
inline int top_layer(int ln, uint64_t count) {
  int best = 0;
  uint64_t least = 1 + count * (uint64_t)ln;
  for (int c = 1; c <= ln; c++) {
    const uint64_t digests = ((uint64_t)1 << c) + count * (uint64_t)(ln - c);
    if (digests < least) least = digests, best = c;
  }
  return best;
}

} // namespace hf
