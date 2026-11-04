// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A slow CPU reference, for tests, of prover/hash_commitment/hf_code.cuh: the code that encodes the
// rows of the weight table (all the weights as a matrix). It evaluates the defining sums,
// precomputes nothing and checks no argument.
// prover/hash_commitment/tests/hf_code_test.cu compares the two, at lengths up to 2^34.
// Position I of a codeword of length 2^ln has the point z_I; the basis value bas_i(z_I) is the
// product of the factors f_a(z_I) over the set bits a of i.
//
// The code is plain Reed-Solomon on cosets modulo p = 2^64 - 2^32 + 1: bas_i(z) = z^i. Points and
// factors come from blocks/code/cosets.h, which prover/hash_commitment/hf_code.cuh does not use.
// The field's arithmetic comes from prover/common/hash_tree_ref.h (namespace row_commit_ref), but not its
// code, row_commit_ref::Code.
#pragma once
#include "common/hash_tree_ref.h"
#include "code/cosets.h"
#include <array>
#include <vector>
namespace hfr {
using row_commit_ref::add;
using row_commit_ref::mul;
using row_commit_ref::sub;
// An element of E = F[x] / (x^3 - FIELD_W), F the integers modulo p: the coefficients of 1, x, x^2.
// e_add(), e_sub() and e_mul() compute in E; e_scale() multiplies by s, an element of F.
typedef std::array<u64, 3> E;
static inline E e_add(const E &a, const E &b) {
  return {add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])};
}
static inline E e_sub(const E &a, const E &b) {
  return {sub(a[0], b[0]), sub(a[1], b[1]), sub(a[2], b[2])};
}
static inline E e_scale(const E &a, u64 s) {
  return {mul(a[0], s), mul(a[1], s), mul(a[2], s)};
}
static inline E e_mul(const E &a, const E &b) {
  return {add(mul(a[0], b[0]), mul(FIELD_W, add(mul(a[1], b[2]), mul(a[2], b[1])))),
          add(add(mul(a[0], b[1]), mul(a[1], b[0])), mul(FIELD_W, mul(a[2], b[2]))),
          add(add(mul(a[0], b[2]), mul(a[1], b[1])), mul(a[2], b[0]))};
}
// f_0(z_I), .., f_(count - 1)(z_I): f_i = z_I^(2^i).
static inline std::vector<u64> factors(int ln, u64 I, int count) {
  std::vector<u64> f(count);
  blk::gl64_factors((unsigned)ln, I, (unsigned)count, f.data());
  return f;
}
// The product of f[first + a] over the set bits a of j.
static inline u64 product(const std::vector<u64> &f, int first, u64 j) {
  u64 v = 1;
  for (int a = 0; (j >> a) != 0; a++)
    if ((j >> a) & 1) v = mul(v, f[first + a]);
  return v;
}
// Symbol I of c's codeword: the sum over i < dim of c[i] bas_i(z_I). dim, the number of
// coefficients (HfTableParams::KD for a weight table's row:
// prover/hash_commitment/hf_table_params.h), is at most 2^ln.
static inline u64 enc_at(int ln, const u64 *c, u64 dim, u64 I) {
  const std::vector<u64> f = factors(ln, I, ln);
  u64 sum = 0;
  for (u64 i = 0; i < dim; i++) sum = add(sum, mul(c[i] % row_commit_ref::P, product(f, 0, i)));
  return sum;
}
// Run h of height m: positions h 2^m + a, a < 2^m. For m <= min(ln, 32) the factors from f_m on
// are constant on it.
// The twiddle factors of an FFT on run h: entry 0 is 0; entry 2^(m - 1 - L) + tt, which layer L
// reads, is f_L at position h 2^m + tt 2^(L + 1).
static inline std::vector<u64> run_table(int ln, int m, u64 h) {
  std::vector<u64> tw((size_t)1 << m, 0);
  for (int L = 0; L < m; L++)
    for (u64 tt = 0; tt < ((u64)1 << (m - 1 - L)); tt++)
      tw[((size_t)1 << (m - 1 - L)) + tt] = factors(ln, (h << m) + (tt << (L + 1)), L + 1)[L];
  return tw;
}
// high[x] for x < count <= 2^(ln - m): the product of f_(m + c)(z_at) over the set bits c of x.
// `at`: any position of the run.
static inline std::vector<u64> run_high(int ln, int m, u64 at, u64 count) {
  const std::vector<u64> f = factors(ln, at, ln);
  std::vector<u64> high(count);
  for (u64 x = 0; x < count; x++) high[x] = product(f, m, x);
  return high;
}
// The fold of one row of dim coefficients for run h: V[i'] = the sum over x of
// high[x] c[x 2^m + i'].
static inline std::vector<u64> run_fold(int ln, int m, u64 h, const u64 *c, u64 dim) {
  const u64 A = (u64)1 << m;
  const std::vector<u64> high = run_high(ln, m, h << m, (dim + A - 1) >> m);
  std::vector<u64> V(A, 0);
  for (u64 i = 0; i < dim; i++) V[i & (A - 1)] = add(V[i & (A - 1)], mul(high[i >> m], c[i] % row_commit_ref::P));
  return V;
}
// From a row's fold V for run h, the row's code symbols on the run: out[a] is symbol h 2^m + a of
// V's own codeword (2^m coefficients).
static inline std::vector<u64> run_finish(int ln, int m, u64 h, const std::vector<u64> &V) {
  const u64 A = (u64)1 << m;
  std::vector<u64> out(A);
  for (u64 a = 0; a < A; a++) out[a] = enc_at(ln, V.data(), A, (h << m) + a);
  return out;
}
} // namespace hfr
