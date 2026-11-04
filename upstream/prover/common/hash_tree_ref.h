// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A CPU reference of the linear code that encodes a matrix's rows. The code is plain Reed-Solomon on
// a coset modulo 2^64 - 2^32 + 1. tables() in prover/common/hash_tree.cuh must give the same twiddle
// factors (tw). GPU sources also call the field arithmetic.
#pragma once
#include "common/hash.h"
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
namespace row_commit_ref {
typedef unsigned __int128 u128;
static const u64 P = FIELD_P;
// add() and sub() expect operands below P; mul() takes any.
static inline u64 add(u64 a, u64 b) {
  const u128 s = (u128)a + b;
  return (u64)(s >= P ? s - P : s);
}
static inline u64 sub(u64 a, u64 b) {
  return a >= b ? a - b : (u64)((u128)a + P - b);
}
static inline u64 mul(u64 a, u64 b) {
  return (u64)((u128)a * b % P);
}
static inline u64 power(u64 a, u64 e) {
  u64 r = 1;
  for (; e; e >>= 1, a = mul(a, a))
    if (e & 1) r = mul(r, a);
  return r;
}
// The code of length n = 2^ln <= 2^32: evaluation of a polynomial at the points of the coset 7 H,
// H the subgroup of order n. 7 generates the multiplicative group, of order 2^32 (2^32 - 1);
// g = 7^(5 (2^32 - 1) 2^(32 - ln)) generates H. Position I has the point 7 g^(I's ln bits in
// reversed order), so that positions 2 m and 2 m + 1 have z and -z. Coefficient j belongs to z^j.
// tw: encode()'s twiddle factors, f_i = z^(2^i) in layer i.
struct Code {
  int ln = 0;
  size_t n = 0;
  u64 g = 0;
  std::vector<u64> tw;
  // Call before any other member function: it sets n, g and tw. It exits with code 1 unless
  // 1 <= ln <= 32 and g has order n.
  void init(int ln_) {
    ln = ln_;
    n = (size_t)1 << ln;
    if (ln < 1 || ln > 32) {
      printf("row_commit_ref: this reference has one coset: the code's length must be from 2^1 to 2^32\n");
      exit(1);
    }
    g = power(power(7, 5 * 4294967295ull), 1ull << (32 - ln));
    if (power(g, n / 2) != P - 1) {
      printf("row_commit_ref: the transform's generator does not have the expected order\n");
      exit(1);
    }
    tw.assign(n, 0);
    for (int i = 0; i < ln; i++)
      for (size_t hh = 0; hh < (n >> (i + 1)); hh++)
        tw[((size_t)1 << (ln - 1 - i)) + hh] = power(point(hh << (i + 1)), 1ull << i); // f_i where bit i is 0
  }
  // Replaces the n coefficients in A, each below P, by their codeword.
  void encode(u64 *A) const {
    for (int i = ln - 1; i >= 0; i--) {
      size_t st = (size_t)1 << i;
      for (size_t J = 0; J < n; J++)
        if (!(J & st)) {
          u64 t = tw[((size_t)1 << (ln - 1 - i)) + (J >> (i + 1))], a = A[J], b = mul(A[J + st], t);
          A[J] = add(a, b);
          A[J + st] = sub(a, b);
        }
    }
  }
  u64 point(size_t I) const {
    u64 e = 0;
    for (int i = 0; i < ln; i++)
      if ((I >> i) & 1) e |= 1ull << (ln - 1 - i);
    return mul(7, power(g, e));
  }
  // Entry I of the codeword of the K coefficients c, computed directly from the definition.
  u64 enc_at(const u64 *c, size_t K, size_t I) const {
    const u64 z = point(I);
    u64 acc = 0, zj = 1;
    for (size_t j = 0; j < K; j++, zj = mul(zj, z)) acc = add(acc, mul(c[j] % P, zj));
    return acc;
  }
};
} // namespace row_commit_ref
