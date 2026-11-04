// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests, on a small run: writes out the data with which the verifier's test code
// (verifier/tests/claims_report.h) finds which claims of a malicious prover's sub-proof are false. A
// sub-proof is the argument for one part of the forward pass; a `Door` (prover/hiding/door.cuh) is
// what it passes to the code that closes its part of the proof: the terms of one linear equation.
// A claim states that a linear form is 0. The form combines entries of "virtual rows" of committed
// arrays, hidden values numbered by message slot (one per masked element sent), and a public
// constant. The verifier builds every form but has no data; no claim is formed here. The data are
// the mask row (a slot's hidden value is its sent value minus a public factor times its mask
// entry) and, for each `Pair` (row coefficients `cv`, column coefficients `L`) of each `Term` (one
// committed array; prover/hiding/door_terms.cuh), the virtual row
//     D[c] = sum over the array's rows r of cv[r] * entry(r, c),   for each of its K columns c.
// For tests on synthetic data only: the values are derived from the committed arrays, weights
// among them, and the mask row is private randomness. Nothing in this file checks that the data
// are synthetic.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hiding/door_terms.cuh"
namespace hd {
// The loop form is public: the grid-stride loop; see k_wcol() (prover/hiding/walk.cuh).
__global__ void k_row_of_cells16(const T *cv, const uint16_t *cells, size_t ld, size_t R, size_t K, T *out) {
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < K; c += (size_t)gridDim.x * blockDim.x) {
    T s = F::zero();
    for (size_t r = 0; r < R; r++) s = F::add(s, F::mulb(cv[r], cells[r * ld + c]));
    out[c] = s;
  }
}
// A symbol is three signed cells: symbol (i, c) = sum over t < 3 of B^t cell(3 i + t, c),
// B = 2^radix_bits; cell (y, c) is at c * m + y, and a cell in a row y >= m counts as 0.
// Not checked: cv has at least ceil(m / 3) entries; 128 * B^2 is below the field's prime, because
// F::mulb() needs a reduced second factor.
__global__ void k_row_of_symbols(const T *cv, const int8_t *cells, size_t m, int radix_bits, size_t K, T *out) {
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < K; c += (size_t)gridDim.x * blockDim.x) {
    T s = F::zero();
    for (size_t y = 0; y < m; y++) {
      const int cell = cells[c * m + y];
      const T term = F::mulb(cv[y / 3], (unsigned long long)(cell < 0 ? -cell : cell) << (radix_bits * (int)(y % 3)));
      s = cell < 0 ? F::sub(s, term) : F::add(s, term);
    }
    out[c] = s;
  }
}
// Not checked: Term::R = 1; only cv[0] is read.
__global__ void k_row_of_elements(const T *cv, const T *vector, size_t K, T *out) {
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < K; c += (size_t)gridDim.x * blockDim.x) out[c] = F::mul(cv[0], vector[c]);
}
struct DoorValues {
  std::string array; // `Term::nm`, or the name given for it in `names`
  unsigned pair = 0;
  std::vector<T> cv; // R row coefficients; the verifier matches virtual rows by them, not by `pair`
  std::vector<T> D;  // K entries
};
// terms: one sub-proof's terms. radix_bits: the prover passes ERAD (prover/common/setting.h); read
// only for kind SYMBOLS. names: null, or one name for each term, written in place of Term::nm;
// claimsLoad() (verifier/tests/claims_report.h) finds arrays by name and rejects names over 64 bytes.
// Call door_values() while Term::data still points to the array and after other streams' work that
// fills a cv or an array has finished: the kernels run on the default stream.
inline std::vector<DoorValues> door_values(const std::vector<Term> &terms, int radix_bits, const std::vector<std::string> *names = nullptr) {
  std::vector<DoorValues> out;
  if (names && names->size() != terms.size()) {
    printf("door_values: one name for each term\n");
    exit(2);
  }
  for (const Term &t : terms) {
    if (!t.data) {
      printf("door_values: the term of array %s does not say where its cells lie\n", t.nm.c_str());
      exit(2);
    }
    T *row;
    CK(cudaMalloc(&row, t.K * sizeof(T)));
    const unsigned blocks = (unsigned)std::min<size_t>((t.K + 127) / 128, 4096);
    for (size_t i = 0; i < t.v.size(); i++) {
      if (t.kind == CELLS16)
        k_row_of_cells16<<<blocks, 128>>>(t.v[i].cv, (const uint16_t *)t.data, t.ld, t.R, t.K, row);
      else if (t.kind == SYMBOLS)
        k_row_of_symbols<<<blocks, 128>>>(t.v[i].cv, (const int8_t *)t.data, t.m, radix_bits, t.K, row);
      else
        k_row_of_elements<<<blocks, 128>>>(t.v[i].cv, (const T *)t.data, t.K, row);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      DoorValues x;
      x.array = names ? (*names)[&t - terms.data()] : t.nm, x.pair = (unsigned)i, x.D.resize(t.K), x.cv.resize(t.R);
      CK(cudaMemcpy(x.D.data(), row, t.K * sizeof(T), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(x.cv.data(), t.v[i].cv, t.R * sizeof(T), cudaMemcpyDeviceToHost));
      out.push_back(std::move(x));
    }
    cudaFree(row);
  }
  return out;
}
// File format. Each number is 8 bytes in the host's byte order, as claimsLoad() reads it (low
// byte first on a little-endian host); each field element is EDEG numbers, its coordinates. The
// number of virtual rows; for each: the length of the array's name, the name, `pair` (index in
// Term::v), R, K, the R elements of cv, the K of D; then the mask row's length and elements. The
// results of fwrite() are not checked.
inline void put_door_values(FILE *o, const std::vector<DoorValues> &v, const std::vector<T> &mask_row) {
  auto word = [o](unsigned long long x) { fwrite(&x, 8, 1, o); };
  auto element = [&](const T &x) {
    for (int k = 0; k < EDEG; k++) word(coord(x, k));
  };
  word(v.size());
  for (const DoorValues &x : v) {
    word(x.array.size());
    fwrite(x.array.data(), 1, x.array.size(), o);
    word(x.pair), word(x.cv.size()), word(x.D.size());
    for (const T &e : x.cv) element(e);
    for (const T &e : x.D) element(e);
  }
  word(mask_row.size());
  for (const T &e : mask_row) element(e);
}
} // namespace hd
