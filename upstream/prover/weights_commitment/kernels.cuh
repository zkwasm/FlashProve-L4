// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

/* CUDA kernels and host helpers for W, the committed weight matrix. k_genWm writes W's cells (see
 * UM) from the synthetic model (`made_up` in identifiers: weights generated from a seed), k_cellsT
 * from a tensor's integers.
 * Nothing in this repository uses this file's k_keys or Tm; other files define a k_keys and a Tm
 * of their own.
 * This file has no includes: prover/weights_commitment/common.cuh includes it after the headers it
 * needs.
 */
#pragma once
// g5 holds kernels and small helpers of the weight commitment, here and in unit_folds.cuh.
namespace g5 {
typedef Ext T;
/* Scales Rp x Cp 16-bit cells by cf into lookup keys:
 * key[off + r2 * CP + c2] = cf * cells[(r0 + r2) * ld + c0 + c2] for r2 < Rp, c2 < Cp.
 * A product, taken modulo 2^32, that does not fit in 16 bits sets the failure flag NO_ELEMENT,
 * through `raise_the_refused_draw_flag` (prover/common/element_from_hash.h); its low 16 bits are
 * stored. */
// The loop form is public: the grid-stride loop (cited at k_genWm()).
__global__ void k_keys(const u16 *cells, size_t ld, size_t r0, size_t Rp, size_t c0, size_t Cp, size_t CP, size_t off,
                       u32 cf, u16 *key) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < Rp * Cp; i += (size_t)gridDim.x * blockDim.x) {
    const size_t r2 = i / Cp, c2 = i % Cp;
    const u32 v = cf * (u32)cells[(r0 + r2) * ld + c0 + c2];
    if (v >> 16) ::raise_the_refused_draw_flag();
    key[off + r2 * CP + c2] = (u16)v;
  }
}
// An entry of the unit table. A unit is the rows [off, off + rows) of W, which store the tensor of
// kind X of layer l. A cell of W is a signed byte: one limb (base-256 digit) of a weight. The other
// fields invert layoutW() (prover/common/weight_layout.h).
// A row's offset in the unit is divided by left[0], left[1] and so on, ng times. Let lf be
// remainder g, and col the take[g] bits of the column from bit cb0[g]. Tensor index g, by class
// cls[g]:
//   'A': col if take[g] > 0, else lf (the column has the whole index or none of it);
//   'S': lf * 2^take[g] + col (the column has the index's lowest bits);
//   'P': lf with col inserted as the highest take[g] of the index's z[g] low bits (in the code:
//        every class but 'A' and 'S').
// The last quotient, below nlw (limbs per weight), is the limb's number (0: least significant). A
// cell is 0 if an index is >= R[g] or the column is >= 2^ncb.
// Kind X: 0 q and k; 1 v; 2 o; 3 to 6 and 9 feed-forward; 7, 8 and 10 routed experts; 11 router;
// 12 embedding; 13 output head.
struct UM {
  unsigned long long off, rows;
  int X, l, nlw, ncb, ng;
  char cls[3];
  int z[3], take[3], cb0[3];
  unsigned long long left[3], R[3];
};
// The synthetic model's seed and dimensions: width d; H query and Gk key heads of dh entries
// (dq = H * dh, dkv = Gk * dh); feed-forward widths f and, with experts, fe; NE experts; V
// tokens. moe: nonzero if odd layers have experts.
struct SM {
  u64 seed;
  long long d, H, Gk, dh, dq, dkv, f, fe, NE, V;
  int moe;
  int cw;   // 14 would widen the synthetic model's draw to 16 bits. No code sets it, so it is 0 in every SM
};
// Synthetic weight i of stream st: a signed integer v with |v| < 2^(bits - 1).
__device__ __forceinline__ long long v13m(u64 seed, int st, u64 i, int bits = 13) {
  long long v = (long long)(mix64(seed + ((u64)st << 40) + i) & ((1u << bits) - 1)) - (1 << (bits - 1));
  v = v == -(1 << (bits - 1)) ? 0 : v;
  return v;
}
// Weight i of stream st of the synthetic model, for streams 4 .. 12 (v, o, gate, up,
// down, router and the experts' three): 13 bits, as from v13m(), unless S.cw is 14 (see SM): then
// 16 bits, and 0 where the magnitude exceeds 32,639, the largest value that two limbs hold.
__device__ __forceinline__ long long made_up(const SM &S, int st, u64 i) {
  if (S.cw != 14) return v13m(S.seed, st, i);
  const long long w = (long long)(mix64(S.seed + ((u64)st << 40) + i) & 65535) - 32768;
  return w < -32639 || w > 32639 ? 0 : w;
}
// Returns W's cell (row y, column c) for the synthetic model, by the rule at UM; 0 outside every
// unit. U is the unit table: NU >= 1 entries, sorted by off.
__device__ __forceinline__ int cellWm(const UM *U, int NU, const SM &S, u64 y, u64 c) {
  int lo = 0, hi = NU - 1;
  while (lo < hi) {
    const int m = (lo + hi + 1) / 2;
    if (U[m].off <= y)
      lo = m;
    else
      hi = m - 1;
  }
  const UM &u = U[lo];
  u64 r = y - u.off;
  if (r >= u.rows || (c >> u.ncb)) return 0;
  long long ix[3] = {0, 0, 0};
  for (int i = 0; i < u.ng; i++) {
    const u64 lf = r % u.left[i];
    r /= u.left[i];
    const u64 col = (c >> u.cb0[i]) & ((1ull << u.take[i]) - 1);
    u64 v;
    if (u.cls[i] == 'A')
      v = u.take[i] ? col : lf;
    else if (u.cls[i] == 'S')
      v = (lf << u.take[i]) | col;
    else {
      const int rem = u.z[i] - u.take[i];
      v = ((lf >> rem) << u.z[i]) | (col << rem) | (lf & ((1ull << rem) - 1));
    }
    if (v >= u.R[i]) return 0;
    ix[i] = (long long)v;
  }
  const int limb = (int)r;
  const long long l = u.l, a = ix[0], b = ix[1], cc = ix[2], d = S.d;
  const bool mo = S.moe && (l & 1);
  long long w;
  switch (u.X) {
  case 0:
    if (a < S.H)
      w = v13m(S.seed, 2, (l * d + cc) * S.dq + a * S.dh + b);
    else if (a < S.H + S.Gk)
      w = v13m(S.seed, 3, (l * d + cc) * S.dkv + (a - S.H) * S.dh + b);
    else
      return 0;
    break;
  case 1:
    w = made_up(S, 4, (l * d + a) * S.dkv + b);
    break;
  case 2:
    w = made_up(S, 5, (l * S.dq + a) * d + b);
    break;
  case 3:
  case 5:
    w = made_up(S, 6, (l * d + a) * (mo ? S.fe : S.f) + b);
    break;
  case 4:
  case 6:
    w = made_up(S, 7, (l * d + a) * (mo ? S.fe : S.f) + b);
    break;
  case 7:
    w = made_up(S, 10, ((l * S.NE + b) * d + a) * S.fe + cc);
    break;
  case 8:
    w = made_up(S, 11, ((l * S.NE + b) * d + a) * S.fe + cc);
    break;
  case 9:
    w = made_up(S, 8, (l * S.f + a) * d + b);
    break;
  case 10:
    w = made_up(S, 12, ((l * S.NE + b) * S.fe + a) * d + cc);
    break;
  case 11:
    w = made_up(S, 9, (l * d + a) * S.NE + b);
    break;
  case 12:
    w = v13m(S.seed, 13, a * d + b, 20);
    break;
  default:
    w = v13m(S.seed, 14, b * S.V + a);
  }
  int bb = 0;
  for (int q = 0; q <= limb; q++) {
    bb = (int)(((w + 128) & 255) - 128);
    w = (w - bb) >> 8;
  }
  return bb;
}
/* Fills o (k columns of ld bytes) with rows y0 .. y0 + rows - 1 of W. Rows below m0 come from
 * cellWm, rows in [m0, m) from the bytes RB (pitch D2 per column), and everything else is 0. The
 * callers pass m = m0 and a null RB. */
// The loop form is public: the grid-stride loop: M. Harris, "CUDA Pro Tip: Write Flexible Kernels
// with Grid-Stride Loops", NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_genWm(const UM *U, int NU, SM S, u64 k, u64 y0, u64 rows, u64 ld, const i8 *RB, u64 m0, u64 m, int D2,
                        i8 *o) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < k * ld; i += (u64)gridDim.x * blockDim.x) {
    const u64 c = i / ld, yy = i % ld, y = y0 + yy;
    int v = 0;
    if (yy < rows) {
      if (y >= m0)
        v = y < m ? RB[c * D2 + (y - m0)] : 0;
      else
        v = cellWm(U, NU, S, y, c);
    }
    o[i] = (i8)v;
  }
}
// Tensor index g (see UM) is the bitwise OR of two parts with no bit in common: one from lf,
// remainder g of the row offset, and one from the column c.
__device__ __forceinline__ u64 index_bits_of_row(const UM &u, int g, u64 lf) {
  if (u.cls[g] == 'A') return u.take[g] ? 0 : lf;
  if (u.cls[g] == 'S') return lf << u.take[g];
  const int rem = u.z[g] - u.take[g];
  return ((lf >> rem) << u.z[g]) | (lf & ((1ull << rem) - 1));
}
__device__ __forceinline__ u64 index_bits_of_column(const UM &u, int g, u64 c) {
  const u64 col = (c >> u.cb0[g]) & ((1ull << u.take[g]) - 1);
  return u.cls[g] == 'A' || u.cls[g] == 'S' ? col : col << (u.z[g] - u.take[g]);
}
/* Writes rows y0 .. y0 + nr - 1 of W (o: k columns of ld bytes) as far as they belong to the
 * unit u, by the rule at UM. Every other byte of o is set to 0.
 *
 * p0, p1 and emb must contain the unit's tensors as integers in the model file's element order
 * (input index fastest, then output row, then expert; the embedding's rows are the tokens): emb for
 * kind u.X = 12, p0 for every other kind; for kind 0, p0 = q (H heads) and p1 = k (Gk heads), with
 * dh rows to a head. Required and not checked: nr <= ld, k >= 1, blockDim.x is a multiple of 32,
 * u.ng <= 3, and that each weight fits in the unit's limbs (two limbs hold -32896 .. 32639): higher
 * digits are dropped. With u.ng >= 4 the loops over g would run past the three-element arrays;
 * `layoutW` makes only units with u.ng = 2 or 3.
 * A test that is not part of this repository compares the output byte for byte with a plain
 * kernel's. */
// Public sources for the techniques of this routine.
// Tiled transposition through shared memory, with a padded tile. The idea is public: M. Harris, "An
// Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013, "Coalesced Transpose Via
// Shared Memory" and "Shared Memory Bank Conflicts",
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// One read of a number serves all of its digits. The idea is public: D. Snider, R. Liang, "Operator
// Fusion in XLA: Analysis and Evaluation", arXiv:2301.13062, section III.B, "Multi-Output Fusion",
// https://arxiv.org/pdf/2301.13062. The paper is about the operations of a compiler's graph, not
// about digits.
// The loop forms are public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html;
// the same form for the lanes of a warp: N. Bell, M. Garland, "Efficient Sparse Matrix-Vector
// Multiplication on CUDA", NVIDIA Technical Report NVR-2008-004, 2008, Figure 22,
// https://www.nvidia.com/docs/io/66889/nvr-2008-004.pdf.
__global__ void k_cellsT(UM u, long long H, long long Gk, long long dh, const short *p0, const short *p1,
                         const int *emb, u64 k, u64 y0, u64 nr, u64 ld, i8 *o) {
  // A weight row is a row offset below the product of u.left[]. row_ix and
  // col_ix hold a weight row's and a column's part of the tensor indices. row_at
  // and col_at hold what that part adds to the element's position in its tensor.
  constexpr int CELL_ROWS = 64;
  __shared__ u64 row_ix[3][CELL_ROWS], row_at[CELL_ROWS], col_ix[3][32], col_at[32], col[32];
  __shared__ int tile[CELL_ROWS][33];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  u64 step[3] = {1, u.R[0], 0}; // how far one step of index g moves in the tensor
  if (u.X == 0) step[0] = (u64)dh * u.R[2], step[1] = u.R[2], step[2] = 1;
  if (u.X == 7 || u.X == 8 || u.X == 10) step[1] = u.R[2] * u.R[0], step[2] = u.R[0];
  if (u.X == 12 || u.X == 13) step[0] = u.R[1], step[1] = 1;
  u64 weight_rows = 1;
  for (int g = 0; g < u.ng; g++) weight_rows *= u.left[g];
  // If 32 columns 2^shift apart hold consecutive elements of the tensor (`turn`), a warp reads them
  // into `tile`; otherwise each lane reads the weight that it writes. `shift` and `turn` only
  // choose the order of reads and writes: every choice gives the same bytes of o.
  int shift = 0;
  bool turn = false;
  for (int g = 0; g < u.ng; g++)
    if (step[g] == 1 && u.take[g] >= 5 && index_bits_of_column(u, g, 1ull << u.cb0[g]) == 1)
      shift = u.cb0[g], turn = true;
  const u64 row_tiles = (weight_rows + CELL_ROWS - 1) / CELL_ROWS, col_tiles = (((k - 1) >> (shift + 5)) + 1) << shift;
  // Rows lo .. hi - 1 of o are the unit's.
  const u64 end = u.off + u.rows, lo = u.off <= y0 ? 0 : u.off - y0 < nr ? u.off - y0 : nr;
  const u64 hi = end <= y0 ? 0 : end - y0 < nr ? end - y0 : nr;
  // The weight at weight row row0 + i and column col[j]; 0 where the tensor has none.
  auto weight = [&](int i, int j) -> long long {
    if (col[j] >> u.ncb) return 0;
    for (int g = 0; g < u.ng; g++)
      if ((row_ix[g][i] | col_ix[g][j]) >= u.R[g]) return 0;
    const u64 at = row_at[i] + col_at[j];
    if (u.X == 12) return emb[at];
    if (u.X != 0) return p0[at];
    const long long head = (long long)(row_ix[0][i] | col_ix[0][j]);
    return head < H ? p0[at] : head < H + Gk ? p1[at - (u64)H * step[0]] : 0;
  };
  for (u64 t = blockIdx.x; t < row_tiles * col_tiles; t += gridDim.x) {
    const u64 row0 = t % row_tiles * CELL_ROWS, ct = t / row_tiles;
    __syncthreads();
    for (int i = threadIdx.x; i < CELL_ROWS; i += blockDim.x) {
      u64 r = row0 + i;
      row_at[i] = 0;
      for (int g = 0; g < u.ng; g++) {
        row_ix[g][i] = index_bits_of_row(u, g, r % u.left[g]);
        row_at[i] += row_ix[g][i] * step[g];
        r /= u.left[g];
      }
    }
    for (int j = threadIdx.x; j < 32; j += blockDim.x) {
      col[j] = (((ct >> shift) * 32 + j) << shift) | (ct & ((1ull << shift) - 1));
      col_at[j] = 0;
      for (int g = 0; g < u.ng; g++) {
        col_ix[g][j] = index_bits_of_column(u, g, col[j]);
        col_at[j] += col_ix[g][j] * step[g];
      }
    }
    __syncthreads();
    if (turn) {
      for (int i = warp; i < CELL_ROWS; i += warps) tile[i][lane] = row0 + i < weight_rows ? (int)weight(i, lane) : 0;
      __syncthreads();
    }
    for (int j = warp; j < 32; j += warps) {
      if (col[j] >= k) continue;
      i8 *column = o + col[j] * ld;
      for (int i = lane; i < CELL_ROWS && row0 + i < weight_rows; i += 32) {
        long long w = turn ? tile[i][j] : weight(i, j);
        for (u64 y = u.off + row0 + i; y < end; y += weight_rows) {
          const int digit = (int)(((w + 128) & 255) - 128);
          w = (w - digit) >> 8;
          if (y >= y0 && y - y0 < nr) column[y - y0] = (i8)digit;
        }
      }
      if (row0) continue;
      for (u64 y = lane; y < lo; y += 32) column[y] = 0;
      for (u64 y = hi + lane; y < ld; y += 32) column[y] = 0;
    }
  }
}
// Copies h[0 .. n) into a new device buffer, which the caller owns.
template <class X> static X *up(const X *h, size_t n) {
  X *d = nullptr;
  CK(cudaMalloc(&d, n * sizeof(X) + 64));
  if (n) CK(cudaMemcpy(d, h, n * sizeof(X), cudaMemcpyHostToDevice));
  return d;
}
// Accumulates named timings: GPU steps by CUDA events, host steps by wall clock.
struct Tm {
  std::map<std::string, double> ms;
  std::vector<std::string> ord;
  cudaEvent_t a, b;
  std::chrono::steady_clock::time_point h0;
  Tm() {
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
  }
  ~Tm() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  Tm(const Tm &) = delete;
  void add(const std::string &n, double v) {
    if (!ms.count(n)) ord.push_back(n);
    ms[n] += v;
  }
  void g0() { CK(cudaEventRecord(a, cudaStreamPerThread)); }
  void g1(const char *n) {
    CK(cudaEventRecord(b, cudaStreamPerThread));
    CK(cudaEventSynchronize(b));
    float x;
    CK(cudaEventElapsedTime(&x, a, b));
    add(std::string("GPU  ") + n, x);
  }
  void c0() {
    h0 = std::chrono::steady_clock::now();
    skip = 0;
  }
  void c1(const char *n) {
    add(std::string("HOST ") + n,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count() - skip);
    skip = 0;
  }
  /* The time between `a0` and `a1` is subtracted from the enclosing host step and reported on
     a separate line. */
  double skip = 0;
  std::chrono::steady_clock::time_point ha;
  void a0() {
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    ha = std::chrono::steady_clock::now();
  }
  void a1() {
    const double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ha).count();
    skip += d;
    add("SETUP buffers allocated and cleared inside the run (cudaMalloc, cudaMemset: shapes only)", d);
  }
  void print() {
    double g = 0, h = 0, s = 0;
    for (auto &n : ord) {
      printf("   %-86s %10.3f ms\n", n.c_str(), ms[n]);
      (n[0] == 'G' ? g : n[0] == 'H' ? h : s) += ms[n];
    }
    printf("   SUM of the GPU stages %.3f ms; SUM of the HOST stages (serial; copies between the two inside them) %.3f "
           "ms; set-up inside the run, in neither sum: %.3f ms\n",
           g, h, s);
  }
};
} // namespace g5
