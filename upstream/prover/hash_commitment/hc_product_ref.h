// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Arithmetic and index rules shared by the GPU and host code of
// prover/hash_commitment/hf_passes.cuh, whose passes over the weights multiply the committed weight
// matrix W by vectors of field elements; their slow references are in
// prover/hash_commitment/hf_passes_ref.h. A cell is one signed byte of W. F is the prime field of
// size p (`P`); a canonical element is an integer in 0 .. p - 1. Only the functions marked `HD`
// (__host__ __device__ under nvcc) also compile for the GPU.
//
// The idea of balanced digits is public: drouyang, "Signed Bucket Indexes for Multi-Scalar
// Multiplication (MSM)" (https://hackmd.io/@drouyang/signed-bucket-index): a digit s_i >= L/2
// becomes s_i - L and adds 1 to the next digit.
// The idea of residues cut into bytes, multiplied on tensor cores into 32-bit sums and joined
// again, is public: S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, M. Zhang, "TensorFHE: Achieving
// Practical Computation on Encrypted Data Using GPGPU", arXiv:2212.14191, the "segment-fusion
// scheme".
// The steps of limbs() and join() that depend on the prime are in blk::lazy::signed_digits() and
// join_digit() (prover/field/field_lazy.cuh).
#pragma once
#include "common/field_host.h"         // HD
#include "field/field_names.h"
#include "field/field_lazy.cuh"        // blk::F, blk::lazy::signed_digits, join_digit
#include "hash_commitment/hc_format.h" // HC_P
#include <cstddef>
#include <cstdint>
namespace hcp {
constexpr uint64_t P = blk::P; // p
static_assert(P == HC_P, "one prime here and in hc_format.h");
constexpr int LIMBS = blk::lazy::DIGITS; // signed bytes to an element of F
static_assert(LIMBS == 8, "the products' shapes are written for eight digits");
// How many products of two signed bytes a caller may add into one 32-bit sum before join(). A
// product is at most 2^14 in absolute value, so such a sum is at most 2^30.
constexpr int CUT = 65536;

// Writes a canonical x as eight balanced base-256 digits (`limbs` in identifiers):
// x = sum over b of 256^b * d[b] modulo p, with -128 <= d[b] <= 127; as an integer the sum is x or
// x - p. The digits are signed because the cells are, and the cuBLAS calls in prover/common/blas.h
// take both operands as signed bytes (CUDA_R_8I). Unsigned digits would fit 32-bit sums too
// (128 * 255 * CUT < 2^31) but need the mixed instruction mma...s8.u8, which cuBLAS does not offer.
HD void limbs(uint64_t x, int8_t d[LIMBS]) {
  const uint64_t w = blk::lazy::signed_digits(blk::F{x});
  for (int b = 0; b < LIMBS; b++) d[b] = (int8_t)(w >> (8 * b));
}
HD uint64_t add(uint64_t a, uint64_t b) { // a, b canonical
  return (blk::F{a} + blk::F{b}).v;
}
// Returns the sum over b < 8 of 256^b * s_b modulo p, canonical, for any eight 32-bit integers. The
// four low and the four high terms are summed separately in 64 bits. In absolute value each sum is
// at most 2^31 * (1 + 2^8 + 2^16 + 2^24) < 2^56; join_digit() requires less than 2^JOIN_BITS (61).
HD uint64_t join(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
  const long long lo = (long long)s0 + (long long)s1 * 256 + (long long)s2 * 65536 + (long long)s3 * 16777216;
  const long long hi = (long long)s4 + (long long)s5 * 256 + (long long)s6 * 65536 + (long long)s7 * 16777216;
  return (blk::lazy::join_digit(lo, 0) + blk::lazy::join_digit(hi, 32)).v;
}
HD uint64_t join(const int s[LIMBS]) {
  return join(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
}

// Whether routed_calls() can handle unit u. The unit must store a tensor of the routed experts
// (kind X = 7, 8 or 10 in layoutW(), prover/common/weight_layout.h) with two bytes per weight. Its
// groups must be input ('S'), expert ('S') and output ('P'), on adjacent column bits in this order
// from bit 0. The input's extent and its column window 2^take must be multiples of 256, the length
// of a block (see RoutedCall).
template <class Unit> inline bool routed_fits(const Unit &u) {
  if ((u.X != 7 && u.X != 8 && u.X != 10) || u.g.size() != 3) return false;
  const auto &in = u.g[0], &expert = u.g[1], &out = u.g[2];
  return in.cls == 'S' && expert.cls == 'S' && out.cls == 'P' && in.take >= 8 && in.cb0 == 0 && expert.cb0 == in.take &&
         out.cb0 == in.take + expert.take && u.nlw == 2 && in.R % 256 == 0;
}
// One rectangle of a routed tensor [expert][output index][block of 256 inputs], as W stores it.
// Rows: output index x0 + i, for i < L, has its low bytes in row y_lo + i * ys of W and its high
// bytes in row y_hi + i * ys.
// Columns: for s < experts * in_window and el < 256, column c0 + 256 * s + el stores element el of
// block sb0 + s % in_window of expert expert0 + s / in_window.
// `outputs` and per_row are the tensor's sizes: output indices per expert, blocks per output index.
struct RoutedCall {
  size_t expert0;
  int experts, L, x0, outputs, per_row, sb0, in_window;
  size_t c0, y_lo, y_hi;
  int ys;
};
// Calls f(c) for each rectangle c of unit u; together they cover every weight byte of u once.
// Requires routed_fits(u), which is not checked. An output index is
// (high << out.z) | (top << rest) | low: `top` selects the columns, (high, low) the row.
// A test that is not part of this repository checks the rectangles cell by
// cell at toy shapes; at full size (Llama 4 Maverick's shape) it checks only the number of cells
// and a sample of them.
template <class Unit, class F> inline void routed_calls(const Unit &u, F f) {
  const auto &in = u.g[0], &expert = u.g[1], &out = u.g[2];
  const int rest = out.z - out.take;
  const size_t highs = out.take ? out.R >> out.z : 1;
  RoutedCall c;
  c.experts = 1 << expert.take, c.L = out.take ? 1 << rest : (int)out.R, c.outputs = (int)out.R;
  c.per_row = (int)(in.R / 256), c.in_window = (1 << in.take) / 256, c.ys = (int)(expert.left * in.left);
  for (size_t high = 0; high < highs; high++)
    for (size_t eh = 0; eh < expert.left; eh++)
      for (size_t kh = 0; kh < in.left; kh++)
        for (size_t top = 0; top < ((size_t)1 << out.take); top++) {
          c.expert0 = eh << expert.take;
          c.x0 = (int)((high << out.z) + (top << rest));
          c.sb0 = (int)(kh * c.in_window);
          c.c0 = top << out.cb0;
          c.y_lo = u.off + ((high << rest) * expert.left + eh) * in.left + kh;
          c.y_hi = u.off + ((out.left + (high << rest)) * expert.left + eh) * in.left + kh;
          f(c);
        }
}

} // namespace hcp
