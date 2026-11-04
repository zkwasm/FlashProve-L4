// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// mask_door() runs a product layer's first three sumcheck rounds on a lookup's table side, from
// packed multiplicity bits. prover/hiding/table_side.cuh describes them and includes this file.
// Limit on hiding: addresses (k_mask) and branches (k_mask, k_leaf) depend on multiplicity bits.
// Public sources: see prover/hiding/table_side.cuh
namespace tside {
// Each of a layer's two tables, L and R, has 8 slabs of columns. In column c = tl nlo + x,
// tl < 2^lsl is the low bits of a lookup-table row number, and the slab a < 8 is the next three
// bits. x < nlo is the pair (identity, plane): an identity is a class of lookup keys with its own
// multiplicities, and a plane is a bit position of them. The column's weight is w = eqT[tl] eqL[x].
// Bit a of a mask byte is slab a's multiplicity bit: b_a(c) or bL_a(c) in ML[c], bR_a(c) in MR[c].
// LAYER 0, leaf layer:   L(a, c) = b_a(c),  R(a, c) = g(a 2^lsl + tl), where g(t) = lut[t] - 1, or
//                        alpha - 1 - t if lut is null. lut[t] is alpha minus row t's value.
// LAYER 1, bit layer:    L(a, c) = b_a(c),  R(a, c) = b_a(c) - 1.
// LAYER 2, bottom layer: L(a, c) = 1 + bL_a(c) (lut[a 2^lsl + tl] - 1),
//                        R(a, c) = 1 + bR_a(c) (lut[(8 + a) 2^lsl + tl] - 1). lut must not be null.
// Planes that the masks do not store have all bits 0 and add nothing in layers 0 and 1. In layer 2,
// L = R = 1 there, so it sums w (L R - 1) over stored columns and adds 1, the sum of all weights.
// Public source for the mask bytes: one-bit weights that meet the same inputs, grouped as a table
// index: Y. Jeon, B. Park, S. J. Kwon, B. Kim, J. Yun, D. Lee, "BiQGEMM: Matrix Multiplication with
// Lookup Table For Binary-Coding-based Quantized DNNs", arXiv:2005.09904 (2020), section III-B.
// Public source for the mask bytes: the same grouping as BiQGEMM's, for each bit plane: J. Wei,
// S. Cao, T. Cao, L. Ma, L. Wang, Y. Zhang, M. Yang, "T-MAC: CPU Renaissance via Table Lookup for
// Low-Bit LLM Deployment on Edge", arXiv:2407.00088 (2024), section 3.1.
// BiQGEMM and T-MAC look up partial sums of a matrix product; here a group is the eight entries,
// one per slab, that an 8-to-1 fold combines, and the sums are of field elements.
struct Masks {
  const u8 *ML, *MR;
  const T *lut, *alpha, *eqT, *eqL;
  int lsl, nlo;
};
constexpr int WPB = TB / 32;
// Round RD (0, 1 or 2) of layers 1 and 2: both tables depend on x. After RD rounds, X (L or R) has
// 8 >> RD slabs, cur(s) = sum_h E[h] X(h (8 >> RD) + s), with E[h] = eq(the earlier challenges, h)
// (Rs::E). For t < NT = 4 >> RD, with lo = cur(t) and hi = cur(t + NT), the kernel writes to `part`
// partial sums of v0_t = sum w lo_L lo_R and v1_t = sum w (hi_L - lo_L)(hi_R - lo_R) for k_end
// (table_side_common.inc). In two cases shared memory holds both products, indexed by mask bits:
//   bit layer, RD > 0: P0[i] = c_i (c_i - 1), P1[j][i] = (c_j - c_i)^2, c_i = sum_h E[h] bit_h(i);
//   bottom layer, RD = 0: Q0[t][bL bR] = L R - 1 and Q1[t][bL bL' bR bR'] = (L' - L)(R' - R), for
//     the current tl, where a prime marks slab t + 4.
// The loop form and the compiler hints are public: the block-stride loop: shown in a listing in
// NVIDIA, "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html;
// __launch_bounds__ (Best Practices Guide) and #pragma unroll, cited at k_leaf(); static indexing:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int RD, int LAYER> __global__ void __launch_bounds__(TB) k_mask(const Masks m, const Rs *rs, T *part) {
  constexpr int NS = 8 >> RD, NT = 4 >> RD, NH = 1 << RD, NC = 1 << NH;
  static_assert(LAYER == 1 || LAYER == 2, "use k_leaf for layer 0");
  constexpr bool BLOCK_TABLE = LAYER == 1 && RD > 0, WARP_TABLE = LAYER == 2 && RD == 0;
  // Layer 2: tv[.][side][a] = E[a >> (3 - RD)] (lut[(8 side + a) 2^lsl + tl] - 1); E = 1 if RD = 0.
  __shared__ T tv[LAYER == 2 ? WPB : 1][2][8];
  __shared__ T P0[BLOCK_TABLE ? NC : 1], P1[BLOCK_TABLE ? NC * NC : 1];
  __shared__ T Q[WARP_TABLE ? WPB : 1][WARP_TABLE ? 80 : 1]; // Q0[t][.] at 4 t, Q1[t][.] at 16 + 16 t
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
  const size_t ntl = (size_t)1 << m.lsl, nlo = (size_t)m.nlo;
  const T one = F::one();
  T v[2 * NT];
#pragma unroll
  for (int q = 0; q < 2 * NT; q++) v[q] = zero();
  if (LAYER == 2 && blockIdx.x == 0 && threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < NT; t++) v[2 * t] = one;
  }
  if constexpr (BLOCK_TABLE) {
    for (int u = threadIdx.x; u < NC * NC; u += TB) {
      T ci = zero(), cj = zero();
#pragma unroll
      for (int h = 0; h < NH; h++) {
        if (((u % NC) >> h) & 1) ci = PF::add(ci, rs->E[h]);
        if (((u / NC) >> h) & 1) cj = PF::add(cj, rs->E[h]);
      }
      const T d = PF::sub(cj, ci);
      P1[u] = PF::mul(d, d);
      if (u < NC) P0[u] = PF::mul(ci, PF::sub(ci, one));
    }
    __syncthreads();
  }
  for (size_t tl = blockIdx.x * (size_t)WPB + wb; tl < ntl; tl += (size_t)gridDim.x * WPB) {
    if constexpr (LAYER == 2) {
      __syncwarp();
      if (lane < 16) {
        const T x = PF::sub(m.lut[((size_t)lane << m.lsl) + tl], one);
        tv[wb][lane >> 3][lane & 7] = RD ? PF::mul(rs->E[(lane & 7) >> (3 - RD)], x) : x;
      }
      __syncwarp();
    }
    const T(&xl)[8] = tv[LAYER == 2 ? wb : 0][0], (&xr)[8] = tv[LAYER == 2 ? wb : 0][1];
    if constexpr (WARP_TABLE) {
      for (int u = lane; u < 80; u += 32) {
        if (u < 16) {
          const int t = u >> 2;
          const T l = (u & 1) ? PF::add(one, xl[t]) : one, r = (u & 2) ? PF::add(one, xr[t]) : one;
          Q[wb][u] = PF::sub(PF::mul(l, r), one);
        } else {
          const int t = (u - 16) >> 4;
          const T dl = PF::sub((u & 2) ? xl[t + 4] : zero(), (u & 1) ? xl[t] : zero()),
                  dr = PF::sub((u & 8) ? xr[t + 4] : zero(), (u & 4) ? xr[t] : zero());
          Q[wb][u] = PF::mul(dl, dr);
        }
      }
      __syncwarp();
    }
    const T e = m.eqT[tl];
    const u8 *ml = m.ML + tl * nlo, *mr = LAYER == 2 ? m.MR + tl * nlo : nullptr;
    for (size_t x = lane; x < nlo; x += 32) {
      const u32 bl = ml[x], br = LAYER == 2 ? mr[x] : 0;
      const T w = PF::mul(e, m.eqL[x]);
#pragma unroll
      for (int t = 0; t < NT; t++) {
        if constexpr (LAYER == 1 && RD == 0) {
          // L is 0 or 1 and R = L - 1, so both products are 0 or 1 and muli() gets no negative y.
          const int lo = (bl >> t) & 1, hi = (bl >> (t + NT)) & 1;
          v[2 * t] = PF::add(v[2 * t], muli(w, (long long)lo * (lo - 1)));
          v[2 * t + 1] = PF::add(v[2 * t + 1], muli(w, (long long)(hi - lo) * (hi - lo)));
        } else if constexpr (BLOCK_TABLE) {
          u32 i = 0, j = 0;
#pragma unroll
          for (int h = 0; h < NH; h++) {
            i |= ((bl >> (h * NS + t)) & 1) << h;
            j |= ((bl >> (h * NS + NT + t)) & 1) << h;
          }
          v[2 * t] = PF::add(v[2 * t], PF::mul(w, P0[i]));
          v[2 * t + 1] = PF::add(v[2 * t + 1], PF::mul(w, P1[j * NC + i]));
        } else if constexpr (WARP_TABLE) {
          const u32 l0 = (bl >> t) & 1, l1 = (bl >> (t + 4)) & 1, r0 = (br >> t) & 1, r1 = (br >> (t + 4)) & 1;
          v[2 * t] = PF::add(v[2 * t], PF::mul(w, Q[wb][4 * t + (l0 | r0 << 1)]));
          v[2 * t + 1] = PF::add(v[2 * t + 1], PF::mul(w, Q[wb][16 + 16 * t + (l0 | l1 << 1 | r0 << 2 | r1 << 3)]));
        } else {
          T c[2][2]; // [side][low or high half]
#pragma unroll
          for (int hl = 0; hl < 2; hl++) {
            T l = one, r = one;
#pragma unroll
            for (int h = 0; h < NH; h++) {
              const int a = h * NS + hl * NT + t;
              if ((bl >> a) & 1) l = PF::add(l, xl[a]);
              if ((br >> a) & 1) r = PF::add(r, xr[a]);
            }
            c[0][hl] = l;
            c[1][hl] = r;
          }
          v[2 * t] = PF::add(v[2 * t], PF::mul(w, PF::sub(PF::mul(c[0][0], c[1][0]), one)));
          v[2 * t + 1] =
              PF::add(v[2 * t + 1], PF::mul(w, PF::mul(PF::sub(c[0][1], c[0][0]), PF::sub(c[1][1], c[1][0]))));
        }
      }
    }
  }
  block_sums<2 * NT>(v, part);
}
// Round RD of the leaf layer. R does not depend on x, so the columns of one tl enter only through
// W[a] = sum_x eqL[x] b_a(tl nlo + x). Both products (see k_mask) are linear in W, so LPT threads
// form them from partial sums of W, one per residue x mod LPT. LPT must be 8, the number of slabs.
constexpr int LPT = 8;
// The compiler hints are public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; #pragma unroll: NVIDIA,
// "CUDA Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
template <int RD> __global__ void __launch_bounds__(TB) k_leaf(const Masks m, const Rs *rs, T *part) {
  constexpr int NS = 8 >> RD, NT = 4 >> RD, NH = 1 << RD;
  __shared__ T tr_[TB / LPT][8]; // tr_[.][a] = E[h(a)] g(a 2^lsl + tl), h(a) = a >> (3 - RD). For RD = 0, E[0] = 1.
  const int sub = threadIdx.x % LPT, grp = threadIdx.x / LPT;
  const size_t ntl = (size_t)1 << m.lsl, nlo = (size_t)m.nlo, step = (size_t)gridDim.x * (TB / LPT);
  const T one = F::one();
  T v[2 * NT], E[NH];
#pragma unroll
  for (int q = 0; q < 2 * NT; q++) v[q] = zero();
#pragma unroll
  for (int h = 0; h < NH; h++) E[h] = RD ? rs->E[h] : one;
  // A block's threads share t0, so all reach both __syncwarp() calls; then a tl >= ntl is skipped.
  for (size_t t0 = blockIdx.x * (size_t)(TB / LPT); t0 < ntl; t0 += step) {
    const size_t tl = t0 + grp;
    const bool on = tl < ntl;
    __syncwarp();
    if (on) {
      static_assert(LPT == 8, "LPT must be 8");
      const T g = m.lut ? PF::sub(m.lut[((size_t)sub << m.lsl) + tl], one)
                        : PF::sub(PF::sub(*m.alpha, one), F::small(((u64)sub << m.lsl) + tl));
      tr_[grp][sub] = RD ? PF::mul(rs->E[sub >> (3 - RD)], g) : g;
    }
    __syncwarp();
    if (!on) continue;
    T W[8];
#pragma unroll
    for (int a = 0; a < 8; a++) W[a] = zero();
    const u8 *ml = m.ML + tl * nlo;
    for (size_t x = sub; x < nlo; x += LPT) {
      const u32 b = ml[x];
      const T w = m.eqL[x];
#pragma unroll
      for (int a = 0; a < 8; a++)
        if ((b >> a) & 1) W[a] = PF::add(W[a], w);
    }
    const T e = m.eqT[tl];
    T cl[NS], cr[NS];
#pragma unroll
    for (int s = 0; s < NS; s++) {
      cl[s] = cr[s] = zero();
#pragma unroll
      for (int h = 0; h < NH; h++) {
        cl[s] = PF::add(cl[s], RD ? PF::mul(E[h], W[h * NS + s]) : W[h * NS + s]);
        cr[s] = PF::add(cr[s], tr_[grp][h * NS + s]);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
      v[2 * t] = PF::add(v[2 * t], PF::mul(e, PF::mul(cl[t], cr[t])));
      v[2 * t + 1] = PF::add(v[2 * t + 1], PF::mul(e, PF::mul(PF::sub(cl[t + NT], cl[t]), PF::sub(cr[t + NT], cr[t]))));
    }
  }
  block_sums<2 * NT>(v, part);
}
// Binds the slab's bits, coordinates 0 to 2 of the point Chal::rho. Sets k8[a] = eq(challenges, a).
template <int LAYER>
static inline void mask_rounds(const Masks &m, T *room, St *st, Chal<F> *cd, T *out, T *k8, cudaStream_t stream) {
  const Ctx x = ctx_at(room);
  // nb blocks per launch: at most MAXB, the number of blocks whose sums fit in Ctx::part.
  constexpr int per_sm = 2;
  constexpr int per_block = LAYER == 0 ? TB / LPT : WPB; // the kernel's loop advances tl by nb * per_block
  const int nb =
      (int)std::min<size_t>((((size_t)1 << m.lsl) + per_block - 1) / per_block, (size_t)std::min(MAXB, g_sms * per_sm));
  if constexpr (LAYER == 0) {
    k_leaf<0><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 0, st, cd, 0, 1, 2, out, k8, stream);
    k_leaf<1><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 1, st, cd, 0, 1, 2, out, k8, stream);
    k_leaf<2><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 2, st, cd, 0, 1, 2, out, k8, stream);
  } else {
    k_mask<0, LAYER><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 0, st, cd, 0, 1, 2, out, k8, stream);
    k_mask<1, LAYER><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 1, st, cd, 0, 1, 2, out, k8, stream);
    k_mask<2, LAYER><<<nb, TB, 0, stream>>>(m, x.rs, x.part);
    CK(cudaGetLastError());
    end_round(x, nb, 2, st, cd, 0, 1, 2, out, k8, stream);
  }
}
// Runs a layer's first three rounds on `stream`. `layer`: 0 is the leaf layer, 1 the bit layer, any
// other value the bottom layer. `room` is scratch memory of at least ctx_elements() elements.
static inline void mask_door(int layer, const u8 *ML, const u8 *MR, const T *lut, const T *alpha, int lsl, int nlo,
                             const T *eqT, const T *eqL, T *room, St *st, Chal<F> *cd, T *out, T *k8,
                             cudaStream_t stream) {
  const Masks m{ML, MR, lut, alpha, eqT, eqL, lsl, nlo};
  if (layer == 0)
    mask_rounds<0>(m, room, st, cd, out, k8, stream);
  else if (layer == 1)
    mask_rounds<1>(m, room, st, cd, out, k8, stream);
  else
    mask_rounds<2>(m, room, st, cd, out, k8, stream);
}
} // namespace tside
