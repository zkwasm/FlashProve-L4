// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Slow host-side reference versions of the GPU code in prover/hash_commitment/hf_passes.cuh: the
// two products with the committed weight matrix W (u and g below), the masks' part of each and,
// for a table's record, g with coefficients given one by one (ref_value_sums()).
// That header itself calls none of them. A GPU test that is not part of this repository compares
// that header's results with the results of these functions, in every entry.
// W has k = 2^lk columns of signed bytes. The weight table encodes them as R = 2^b rows, each with
// nm masks. Row r's message and random tail, with lo = lk - b and cl < 2^lo:
//     message[y * 2^lo + cl] = W[y][r * 2^lo + cl],   tail[j] = mask[r][j].
// An element of the prime field F is a canonical uint64_t (below hcp::P). An element of
// E = F[x]/(x^3 - blk::NR) (blocks/field/field.h) is three of them, lowest degree first.
#pragma once
#include "hash_commitment/hc_product_ref.h" // hcp::P, add
#include <cstddef>
#include <cstdint>
namespace hfp {
inline uint64_t mul(uint64_t a, uint64_t b) {
  return (uint64_t)((unsigned __int128)a * b % hcp::P);
}
inline uint64_t sub(uint64_t a, uint64_t b) {
  return a >= b ? a - b : a + hcp::P - b;
}
// x times the signed byte w, in F.
inline uint64_t times_byte(int w, uint64_t x) {
  return w >= 0 ? mul((uint64_t)w, x) : sub(0, mul((uint64_t)(-w), x));
}
inline void mul_E(const uint64_t x[3], const uint64_t y[3], uint64_t z[3]) {
  const uint64_t r0 = hcp::add(mul(x[0], y[0]), mul(blk::NR, hcp::add(mul(x[1], y[2]), mul(x[2], y[1]))));
  const uint64_t r1 = hcp::add(hcp::add(mul(x[0], y[1]), mul(x[1], y[0])), mul(blk::NR, mul(x[2], y[2])));
  const uint64_t r2 = hcp::add(hcp::add(mul(x[0], y[2]), mul(x[2], y[0])), mul(x[1], y[1]));
  z[0] = r0, z[1] = r1, z[2] = r2;
}
// hi_a(idx): the product of hv[a * nb + j] over the set bits j of idx.
// Not checked: idx < 2^nb; higher bits are ignored.
inline uint64_t hi_of(const uint64_t *hv, int nb, int a, uint64_t idx) {
  uint64_t v = 1;
  for (int j = 0; j < nb; j++)
    if ((idx >> j) & 1) v = mul(v, hv[(size_t)a * nb + j]);
  return v;
}
// For y0 <= y < y0 + rows, in E (omega is not written if el is null):
//     u[y * 2^lo + cl] = sum over r < 2^b of eg[r] * W[y][r * 2^lo + cl]
//     omega[y] = sum over cl < 2^lo of el[cl] * u[y * 2^lo + cl]
// cell(y, c) returns W[y][c].
template <class Cell>
inline void ref_fold(Cell cell, size_t y0, size_t rows, int lk, int b, const uint64_t *eg, const uint64_t *el,
                     uint64_t *u, uint64_t *omega) {
  const int lo = lk - b;
  const size_t R = (size_t)1 << b, C = (size_t)1 << lo;
  for (size_t y = y0; y < y0 + rows; y++) {
    uint64_t om[3] = {0, 0, 0};
    for (size_t cl = 0; cl < C; cl++) {
      uint64_t s[3] = {0, 0, 0};
      for (size_t r = 0; r < R; r++) {
        const int w = cell(y, (r << lo) + cl);
        if (w)
          for (int e = 0; e < 3; e++) s[e] = hcp::add(s[e], times_byte(w, eg[r * 3 + e]));
      }
      for (int e = 0; e < 3; e++) u[((y << lo) + cl) * 3 + e] = s[e];
      if (el) {
        uint64_t z[3];
        mul_E(s, el + cl * 3, z);
        for (int e = 0; e < 3; e++) om[e] = hcp::add(om[e], z[e]);
      }
    }
    if (el)
      for (int e = 0; e < 3; e++) omega[y * 3 + e] = om[e];
  }
}
// For each fibre a < t (hf_open_side.cuh calls it a run: 2^lo codeword positions), in F:
//     g[a * k + c] += sum over y0 <= y < y0 + rows of hi_a(y) * W[y][c].
template <class Cell>
inline void ref_fibre_sums(Cell cell, size_t y0, size_t rows, size_t k, const uint64_t *hv, int nb, int t,
                           uint64_t *g) {
  for (int a = 0; a < t; a++)
    for (size_t y = y0; y < y0 + rows; y++) {
      const uint64_t h = hi_of(hv, nb, a, y);
      for (size_t c = 0; c < k; c++) {
        const int w = cell(y, c);
        if (w) g[(size_t)a * k + c] = hcp::add(g[(size_t)a * k + c], times_byte(w, h));
      }
    }
}
// The same sums with coefficients given one by one, for a < t, in F:
//     g[a * k + c] += sum over y0 <= y < y0 + rows of values[a * m0 + y] * W[y][c].
// values: t rows of m0 canonical elements.
template <class Cell>
inline void ref_value_sums(Cell cell, size_t y0, size_t rows, size_t k, const uint64_t *values, size_t m0, int t,
                           uint64_t *g) {
  for (int a = 0; a < t; a++)
    for (size_t y = y0; y < y0 + rows; y++) {
      const uint64_t h = values[(size_t)a * m0 + y];
      for (size_t c = 0; c < k; c++) {
        const int w = cell(y, c);
        if (w) g[(size_t)a * k + c] = hcp::add(g[(size_t)a * k + c], times_byte(w, h));
      }
    }
}
// um[j] = sum over r < R of eg[r] * mask[r][j] in E, for j < nm: the masks' part of u.
// mask(r, j) returns mask[r][j], canonical.
template <class Mask>
inline void ref_mask_fold(Mask mask, uint32_t R, uint64_t nm, const uint64_t *eg, uint64_t *um) {
  for (uint64_t j = 0; j < nm; j++)
    for (int e = 0; e < 3; e++) {
      uint64_t s = 0;
      for (uint32_t r = 0; r < R; r++) s = hcp::add(s, mul(mask(r, j), eg[(size_t)r * 3 + e]));
      um[j * 3 + e] = s;
    }
}
// For j = q * 2^lo + cl < nm: g[a * k + r * 2^lo + cl] += hi_a(base + q) * mask[r][j].
// `base` * 2^lo is the i0 of hf_mask_columns(): the coefficient index assigned to mask 0.
template <class Mask>
inline void ref_mask_columns(Mask mask, uint32_t R, int lo, uint64_t nm, const uint64_t *hv, int nb, uint64_t base,
                             int t, uint64_t *g) {
  const size_t k = (size_t)R << lo;
  for (int a = 0; a < t; a++)
    for (uint64_t j = 0; j < nm; j++) {
      const uint64_t h = hi_of(hv, nb, a, base + (j >> lo));
      for (uint32_t r = 0; r < R; r++) {
        uint64_t &o = g[(size_t)a * k + ((size_t)r << lo) + (j & (((uint64_t)1 << lo) - 1))];
        o = hcp::add(o, mul(mask(r, j), h));
      }
    }
}
} // namespace hfp
