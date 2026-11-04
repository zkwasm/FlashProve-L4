// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The tie of one matrix M in the approximate range check (`tail` in identifiers) of a sub-proof
// (`FORK` in identifiers): the linear claim that relates the combinations Z = M Cf to M's
// committed symbols. M has m rows and k columns of cells (small integers). Cf holds challenges in
// {-1, 0, 1}. Z[y][u] + 2^ELBZ is committed as two digits. A symbol is a field element
// e0 + B e1 + B^2 e2 that packs up to three adjacent cells of a column, B = 2^ERAD
// (prover/common/setting.h), so M has m' = ceil(m / 3) symbol rows.
//
// Include this file after prover/hiding/transcript.cuh (T, F), prover/hiding/tail_gam.cuh
// (kf_tie_powers()) and prover/hiding/place.cuh (hz::alloc()). Only setup() allocates. Launches
// use the default stream.
#pragma once
namespace hc {
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_tie_left_h(const T *Er, u32 m, T *o) {
  for (u32 y = blockIdx.x * blockDim.x + threadIdx.x; y < m; y += gridDim.x * blockDim.x) {
    const u32 ip = y / 3;
    o[y] = F::mulb(Er[ip], 1ull << (ERAD * (y - 3 * ip)));
  }
}
// The tie's vectors for matrix number mu; mp is m', nb = ceil(log2(m')).
// In the claim (walk_tail_f(), prover/hiding/walk_calls.cuh), the low digit for Z[y][u] has the
// coefficient Er[y div 3] B^(y mod 3) theta^u, and the high digit 2^EZD0 times that. Symbol (i', c)
// has -Er[i'] g[c]. The claim's batching coefficient multiplies all three. theta is the sub-proof's
// batching challenge. Er[i'] = step^i' for i' < m'; step = theta^NP, NP the number of combinations
// rounded up to a power of two; g[c] = sum_u theta^u Cf[c][u] (computed in
// HiddenTail::make_powers(), prover/hiding/tail_f.cuh). The symbols' coefficients are a product of
// two vectors, so no m' x k array is stored: add_matrix() (prover/hiding/door.cuh) takes r and Er.
// Both are scratch memory of one run of the sub-proof (hz::FORK).
// m <= 2^32 - 2^20 is assumed, not checked (32-bit row index in k_tie_left_h()).
struct TieH {
  u32 mu = 0, m = 0, mp = 0, k = 0;
  int nb = 0;
  T *r = nullptr, *Er = nullptr;
  void setup(u32 number_in_fold_list, u32 m_, u32 k_) {
    mu = number_in_fold_list, m = m_, k = k_, mp = (m + 2) / 3;
    nb = 0;
    while (((u32)1 << nb) < mp) nb++;
    if (!m || nb > 256 || mu >= (1u << 24)) { // loose sanity bounds on mu and nb
      fprintf(stderr, "tie_h: matrix number %u with %u rows is out of range: the number must be below "
                      "2^24, and the matrix needs at least one row\n", mu, m);
      exit(2);
    }
    hz::alloc(r, (size_t)(nb + 1) * sizeof(T), hz::FORK);
    hz::alloc(Er, (size_t)mp * sizeof(T) + 64, hz::FORK);
  }
  // Fills Er and r from *step (GPU memory): r[j] = step^(2^(nb - 1 - j)), j < nb, so Er[i'] is the
  // product of r[j] over the set bits nb - 1 - j of i'. Call it in each run, once *step is written.
  void powers(const T *step) {
    kf_tie_powers<<<(unsigned)std::min<u32>((mp + 255) / 256, 4096), 256>>>(step, nb, mp, r, Er); // hiding/tail_gam.cuh; mp >= nb always
  }
  // out (GPU memory) receives m elements: the tie's coefficients on the rows of Z.
  // Call it after powers().
  void left(T *out) const { k_tie_left_h<<<(unsigned)std::min<u32>((m + 255) / 256, 4096), 256>>>(Er, m, out); }
};
} // namespace hc
