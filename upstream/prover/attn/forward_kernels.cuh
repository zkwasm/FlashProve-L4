// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Attention's two matrix products in exact integer arithmetic, by cuBLAS: the scores s = q k^T,
// with each score's softmax weight e and its record, and the weighted sums e v, with their division
// by the row's sum Z of weights. A record is what the attention sub-proof (the proof's argument for
// attention) stores for a score. Row i is query token i of a head; j is a key token.
//
// q, k, v and e are each kept as two signed bytes (limbs), x = 256 x1 + x0 with x0 from -128 to
// 127. A product of two such matrices is a sum of four products of matrices of signed bytes, which
// cuBLAS computes exactly into 32-bit integers. Nothing is rounded except where the arithmetic of
// the proof requires it: the exponential is a table of integers, and the divisions are integer
// divisions with their remainders kept (in k_records and k_divide).
//
// Included by prover/attn/forward.cuh, whose declarations it uses. Not checked here: a_forward_ok()
// in that file must accept the APlan that contains p (struct AP, prover/attn/plan.cuh), and |q|,
// |k| <= 2,047 must hold, which gives |s| <= Smax. In p: T tokens, H query heads (HA per key-value
// head), dh channels per head, Lam layers. The lie:: functions are fault-injection hooks for
// tests (prover/common/lie.cuh), without effect in this repository.
//
// Public source: NVIDIA, cuBLAS documentation (https://docs.nvidia.com/cuda/cublas/),
// "cublasGemmEx()" (prover/common/blas.h has the details).
// Public source: a grid-stride loop (k_divide): M. Harris, "CUDA Pro Tip: Write Flexible Kernels
// with Grid-Stride Loops" (NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
// Public source: a warp's all-reduce by XOR shuffles: J. Luitjens,
// "Faster Parallel Reductions on Kepler" (NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) (k_row_max, k_records).
// Public source: __shfl_xor_sync() (k_row_max, k_records): NVIDIA, CUDA C++ Programming Guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/), "Warp Shuffle Functions".
// See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions", "Warp Shuffle Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// Further public works, cited below for ideas that are public:
// Wen-mei W. Hwu, David Kirk, "Advanced Algorithmic Techniques for GPUs, Lecture 4: Thread
// Coarsening and more on Tiling/Blocking", PASI Summer School, Chile, January 5-7, 2011, slides
// "Thread Coarsening" and "Outline of Technique",
// https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf (k_records).
// The page "C/C++ Language Extensions" of the CUDA Programming Guide, named above,
// "#pragma unroll" (k_records).
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
// (k_records).
//
// k_records reads the exponential of a softmax from a table after the low part of the non-negative
// gap D has been dropped, and gives the weight 0 from a fixed gap on. A public work that states
// this technique: Sun, Li, Zhang, "zkLLM", arXiv:2404.16109, sections 5.1 and 5.1.1. In zkLLM the
// gap is measured from the logarithm of the sum of the row's exponentials, scaled to the scores'
// units and rounded (equation (17)), whose computation is not verified directly, and no division is
// left. Here it is measured from the row maximum m, which the sub-proof verifies, and the division
// by Z is proved.
// Source for a softmax in a proof (maximum subtracted, exponential from a table, division rounded to nearest): Chen,
// Waiwitlikhit, Stoica, Kang, "ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs", EuroSys 2024,
// sections 5.1 and 6.1.
// The same, and source for the causal form (row i has the entries j <= i) and for one rounded division of the
// weighted sum by the sum of the weights: Qu, Sun, Liu, Lu, Guo, Chen, Zhang, "zkGPT: An Efficient Non-interactive
// Zero-knowledge Proof Framework for LLM Inference", USENIX Security 2025, sections 4.1 (eq. (1)), 4.3 and 5.3 (eq.
// (5)). There the score is rounded before the maximum is taken, and the scales of the quantization enter the
// quotient; here the score is not rounded, and the distance to the maximum is divided by 2^sh and rounded down.
#pragma once
#include "common/lie.cuh"
#include "common/blas.h"
namespace gfk {
using gf::a_pos;
using gf::limb2;
constexpr int ROWS = 1024;                    // maximum rows of one head in a chunk
constexpr size_t WORKSPACE = (size_t)4 << 20; // cuBLAS's workspace
// The scratch memory of scores_values(), for one chunk (see Piece). Index a is head h0 + a.
struct Room {
  int *S;            // [a][row][key]: the scores
  i8 *E0, *E1;       // [a][row][key]: the weights' low and high signed bytes
  int *C0, *C1, *C2; // [a][row][channel]: sums of e v's byte products, of weight 1, 256, 65536
  char *work;
};
static inline size_t up256(size_t n) {
  return (n + 255) / 256 * 256;
}
// Bytes of scratch that scores_values() needs and does not erase.
static inline size_t room_bytes(const AP &p) {
  const size_t rows = (size_t)p.HA * std::min(p.T, ROWS);
  return up256(rows * p.T * 4) + 2 * up256(rows * p.T) + 3 * up256(rows * p.dh * 4) + WORKSPACE + 256;
}
static inline Room room_at(const AP &p, void *at) {
  const size_t rows = (size_t)p.HA * std::min(p.T, ROWS);
  char *x = (char *)(((uintptr_t)at + 255) & ~(uintptr_t)255);
  Room r;
  r.S = (int *)x, x += up256(rows * p.T * 4);
  r.E0 = (i8 *)x, x += up256(rows * p.T);
  r.E1 = (i8 *)x, x += up256(rows * p.T);
  r.C0 = (int *)x, x += up256(rows * p.dh * 4);
  r.C1 = (int *)x, x += up256(rows * p.dh * 4);
  r.C2 = (int *)x, x += up256(rows * p.dh * 4);
  r.work = x;
  return r;
}
// A chunk has rows i0 .. i0 + n - 1 of query heads h0 .. h0 + heads - 1 of layer lam, which share
// key-value head g: all T rows of g's HA query heads if T <= ROWS, otherwise ROWS rows of one head.
// Source for query heads that share one key head and one value head: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy,
// Lebrón, Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
// arXiv:2305.13245, section 2.2.
struct Piece {
  int lam, g, h0, heads, i0, n;
};
// For each row i of the chunk, writes the largest score m over j <= i to mo and the smallest j that
// attains it to io. Launch: ceil(heads n / 8) blocks of 256 threads, one warp per row.
__global__ void k_row_max(AP p, Piece c, const int *__restrict__ S, int *mo, int *io) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, n = c.n, i0 = c.i0;
  if (w >= c.heads * n) return;
  const int i = i0 + w % n;
  const int *s = S + (size_t)w * p.T;
  int m = INT32_MIN, at = 0;
  for (int j = lane; j <= i; j += 32)
    if (s[j] > m) m = s[j], at = j;
  for (int o = 1; o < 32; o <<= 1) {
    const int m2 = __shfl_xor_sync(~0u, m, o), at2 = __shfl_xor_sync(~0u, at, o);
    if (m2 > m || (m2 == m && at2 < at)) m = m2, at = at2;
  }
  if (lane == 0) {
    const size_t R = ((size_t)c.lam * p.H + c.h0 + w / n) * p.T + i;
    mo[R] = lie::at(lie::MAXIMUM, c.h0 + w / n, i, m); // m: lie::at() is a test hook, the identity
    io[R] = lie::at(lie::INDEX, c.h0 + w / n, i, at);
  }
}
// Computes each score's weight e and record (r, flag d, table row f), for j up to the end of i's
// tile of B key tokens (the diagonal tile). Tt, the exponential's table, has u0 rows;
// sh = shl[lam]. D = m - s, plus Om = 2 Smax + u0 2^sh if j > i (causal mask: then D >= u0 2^sh).
// If D < u0 2^sh: f = D >> sh, e = Tt[f], d = 0, r = D's low sh bits. Otherwise e = 0, d = 1, and
// D - u0 2^sh is divided by 65536 - 2^sh: f = u0 + quotient, r = 2^sh + remainder. Zo gets Z.
// E0, E1 get e's two signed bytes (e must be below 32,640 to fit), 0 at the other j. With KEYS it
// also writes the sub-proof's arrays, at index a_pos(). `as` gets five planes of one byte per
// score: (r & 255) - 128, (r >> 8) - 128, d, (e & 255) - 128, e >> 8 (not the bytes in E0, E1). kM
// (diagonal tile: kD) gets r, f as 16-bit lookup keys, at offsets oRM, oFM (oRD, oFD). Not checked:
// `as` must be 4-byte aligned, the key arrays at these offsets 8-byte. Launch: as for k_row_max().
// Public sources for the techniques of this routine (the works are at the file's head).
// The idea is public: thread coarsening: Hwu, Kirk.
// The compiler hint is public: #pragma unroll: CUDA Programming Guide, "C/C++ Language Extensions".
// The compiler hint is public: static indexing of private arrays: Milakov.
template <bool KEYS>
__global__ void k_records(AP p, Piece c, const int *__restrict__ shl, const int *__restrict__ Tt,
                          const int *__restrict__ S, const int *__restrict__ mo, int *Zo, i8 *E0, i8 *E1, i8 *as,
                          AOff kM, AOff kD, const size_t *oRM, const size_t *oFM, const size_t *oRD,
                          const size_t *oFD) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, n = c.n, lam = c.lam, g = c.g;
  if (w >= c.heads * n) return;
  const int h = c.h0 + w / n, i = c.i0 + w % n, in = lam * p.HA + h % p.HA, sh = shl[lam], T = p.T;
  const size_t R = ((size_t)lam * p.H + h) * T + i;
  const int m = mo[R], with_record = ((i >> p.lB) + 1) << p.lB;
  const u32 C = (u32)p.u0 << sh, rho = 65536u - (1u << sh), Om = (u32)(2 * p.Smax) + C, msk = (1u << sh) - 1;
  const size_t NI = (size_t)p.Lam * p.HA, psM = NI << p.nM, psD = NI << p.nD;
  i8 *aM = as + ((size_t)in << p.nM), *aD = as + ((size_t)5 * NI << p.nM) + ((size_t)in << p.nD);
  u16 *rM = nullptr, *fM = nullptr, *rD = nullptr, *fD = nullptr;
  if (KEYS)
    rM = kM.kr[lam] + oRM[in], fM = kM.kf[lam] + oFM[in], rD = kD.kr[lam] + oRD[in], fD = kD.kf[lam] + oFD[in];
  int Z = 0;
  for (int j = 4 * lane; j < T; j += 128) {
    const size_t x = (size_t)w * T + j;
    if (j >= with_record) {
      *(u32 *)(E0 + x) = 0;
      *(u32 *)(E1 + x) = 0;
      continue;
    }
    const int4 s4 = *(const int4 *)(S + x);
    const int s[4] = {s4.x, s4.y, s4.z, s4.w};
    u32 r_[4], d_[4], e_[4], f_[4];
    int lo[4], hi[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const u32 Dp = (u32)(m - s[q]) + (j + q <= i ? 0u : Om);
      if (Dp < C) {
        r_[q] = Dp & msk;
        d_[q] = 0;
        f_[q] = Dp >> sh;
        e_[q] = (u32)Tt[f_[q]];
      } else {
        const u32 y = Dp - C, hh = y / rho;
        r_[q] = (1u << sh) + y % rho;
        d_[q] = 1;
        e_[q] = 0;
        f_[q] = p.u0 + hh;
      }
      // lie::RECORD lowers f by `steps`. D = f 2^sh + r holds; only flag d = 0 is false: r >= 2^sh.
      if (const u32 steps = lie::at(lie::RECORD, h, i, 0u, j + q))
        if (Dp < C && f_[q] >= steps) f_[q] -= steps, r_[q] += steps << sh, e_[q] = (u32)Tt[f_[q]];
      e_[q] = lie::at(lie::WEIGHT, h, i, e_[q], j + q); // test hook: a changed e makes (f, e) false
      Z += (int)e_[q];
      limb2((int)e_[q], lo[q], hi[q]);
    }
    *(uchar4 *)(E0 + x) = make_uchar4((u8)lo[0], (u8)lo[1], (u8)lo[2], (u8)lo[3]);
    *(uchar4 *)(E1 + x) = make_uchar4((u8)hi[0], (u8)hi[1], (u8)hi[2], (u8)hi[3]);
    if (KEYS) {
      bool dg;
      const size_t pos = a_pos(p, g, i, j, dg);
      i8 *A = (dg ? aD : aM) + pos;
      const size_t ps = dg ? psD : psM;
      *(uchar4 *)A = make_uchar4((u8)((r_[0] & 255) ^ 128), (u8)((r_[1] & 255) ^ 128), (u8)((r_[2] & 255) ^ 128),
                                 (u8)((r_[3] & 255) ^ 128));
      *(uchar4 *)(A + ps) = make_uchar4((u8)((r_[0] >> 8) - 128), (u8)((r_[1] >> 8) - 128), (u8)((r_[2] >> 8) - 128),
                                        (u8)((r_[3] >> 8) - 128));
      *(uchar4 *)(A + 2 * ps) = make_uchar4((u8)d_[0], (u8)d_[1], (u8)d_[2], (u8)d_[3]);
      *(uchar4 *)(A + 3 * ps) = make_uchar4((u8)((e_[0] & 255) ^ 128), (u8)((e_[1] & 255) ^ 128),
                                            (u8)((e_[2] & 255) ^ 128), (u8)((e_[3] & 255) ^ 128));
      *(uchar4 *)(A + 4 * ps) = make_uchar4((u8)(e_[0] >> 8), (u8)(e_[1] >> 8), (u8)(e_[2] >> 8), (u8)(e_[3] >> 8));
      *(ushort4 *)((dg ? rD : rM) + pos) = make_ushort4((u16)r_[0], (u16)r_[1], (u16)r_[2], (u16)r_[3]);
      *(ushort4 *)((dg ? fD : fM) + pos) = make_ushort4((u16)f_[0], (u16)f_[1], (u16)f_[2], (u16)f_[3]);
    }
  }
  for (int o = 1; o < 32; o <<= 1) Z += __shfl_xor_sync(~0u, Z, o);
  if (lane == 0) Zo[R] = lie::at(lie::SUM, h, i, Z);
}
// With acc = C0 + 256 C1 + 65536 C2 = sum over j of e[i][j] v[j][c], for the chunk's rows i and
// channels c: o = floor((2 acc + Z) / (2 Z)), acc / Z rounded to the nearest integer, and rem =
// 2 acc + Z - 2 Z o. Sets *err = 1, never 0, if rem may not fit in wd bytes (2 Z - 1 >= 2^(8 wd)).
// An ordinary store by any thread: public sources cited at kf_zz_plain(), prover/hiding/tail_f.cuh.
__global__ void k_divide(AP p, Piece pc, const int *__restrict__ C0, const int *__restrict__ C1,
                         const int *__restrict__ C2, const int *__restrict__ Zo, i16 *o, u32 *rem, u32 *err) {
  GSA(x, (size_t)pc.heads * pc.n * p.dh) {
    const int c = (int)(x % p.dh), w = (int)(x / p.dh);
    const size_t R = ((size_t)pc.lam * p.H + pc.h0 + w / pc.n) * p.T + pc.i0 + w % pc.n;
    const I64 Z = Zo[R], den = 2 * Z, acc = (I64)C0[x] + (I64)C1[x] * 256 + (I64)C2[x] * 65536, num = 2 * acc + Z,
              y = lie::at(lie::OUTPUT, pc.h0 + w / pc.n, pc.i0 + w % pc.n, num >= 0 ? num / den : -((-num + den - 1) / den), c);
    if (den - 1 >= ((I64)1 << (8 * p.wd)) && !lie::pass(lie::ATTENTION_REMAINDER)) *err = 1;
    o[R * p.dh + c] = (i16)y;
    rem[R * p.dh + c] = (u32)(num - den * y);
  }
}
// Computes all chunks, on stream st. Q0, Q1: q's bytes, [layer][head][token][channel]. K0, K1: k's,
// [layer][token][channel]. vt: v's, [byte][layer][channel][token]. `room`: room_bytes(p) bytes.
// Others: see a_scores_values(). The thread's cuBLAS handle gets its stream and workspace back.
static inline void scores_values(const AP &p, bool keys, const int *shl, const i8 *Q0, const i8 *Q1, const i8 *K0,
                                 const i8 *K1, const i8 *vt, const int *Tt, int *mo, int *io, int *Zo, i16 *o,
                                 u32 *rem, i8 *as, const AOff &kM, const AOff &kD, const size_t *oRM,
                                 const size_t *oFM, const size_t *oRD, const size_t *oFD, u32 *err, void *room,
                                 cudaStream_t st) {
  const Room r = room_at(p, room);
  const blas::Borrowed lent(st, r.work, WORKSPACE);
  const int T = p.T, dh = p.dh, heads = T <= ROWS ? p.HA : 1; // heads are contiguous in Q0, Q1
  const i8 *Q[2] = {Q0, Q1}, *K[2] = {K0, K1}, *E[2] = {r.E0, r.E1};
  int *C[3] = {r.C0, r.C1, r.C2};
  const int first_layer = lie::layer_now(); // for the test hooks below alone; 0 in this repository
  for (int lam = 0; lam < p.Lam; lam++)
    for (int h0 = 0; h0 < p.H; h0 += heads)
      for (int i0 = 0; i0 < T; i0 += ROWS) {
        const Piece c{lam, h0 / p.HA, h0, heads, i0, std::min(T - i0, ROWS)};
        const int rows = c.heads * c.n, warps = (rows + 7) / 8;
        lie::here(first_layer + lam, lie::ATTENTION, st);
        // The chunk's row index of (head, query token), or -1 if the chunk does not have it.
        [[maybe_unused]] auto row_of = [&c](long long head, long long i) {
          return head >= c.h0 && head < c.h0 + c.heads && i >= c.i0 && i < c.i0 + c.n ? (head - c.h0) * c.n + (i - c.i0) : -1ll;
        };
        // Test hook (returns c.g), for reading another head's k and v; all else remains head g's.
        const long long named = lie::on_the_host(first_layer + lam, lie::ATTENTION, lie::GROUP, c.g, c.g, st);
        const int g_read = named >= 0 && named < p.G ? (int)named : c.g;
        // S = sum over x, y of 256^(x + y) (byte x of q) (byte y of k); the sum fits in 32 bits.
        for (int x = 0; x < 2; x++)
          for (int y = 0; y < 2; y++)
            blas::gemm_tn(lent.handle, T, rows, dh, K[y] + (size_t)lam * T * p.dkv + g_read * dh, p.dkv,
                          Q[x] + (((size_t)lam * p.H + h0) * T + i0) * dh, dh, r.S, T, x + y > 0, 1 << (8 * (x + y)));
        lie::in_memory<int>(first_layer + lam, lie::ATTENTION, lie::SCORE, st, [&](long long head, long long i, long long j) -> int * {
          const long long w = row_of(head, i);
          return w < 0 || j < 0 || j >= T ? nullptr : r.S + w * T + j;
        });
        k_row_max<<<warps, 256, 0, st>>>(p, c, r.S, mo, io);
        if (keys)
          k_records<true>
              <<<warps, 256, 0, st>>>(p, c, shl, Tt, r.S, mo, Zo, r.E0, r.E1, as, kM, kD, oRM, oFM, oRD, oFD);
        else
          k_records<false>
              <<<warps, 256, 0, st>>>(p, c, shl, Tt, r.S, mo, Zo, r.E0, r.E1, as, kM, kD, oRM, oFM, oRD, oFD);
        CK(cudaGetLastError());
        // C[z] = sum over j and x + y = z of (byte x of e) (byte y of v); acc can exceed 32 bits.
        for (int x = 0; x < 2; x++)
          for (int y = 0; y < 2; y++)
            blas::gemm_tn(lent.handle, dh, rows, T, vt + (((size_t)y * p.Lam + lam) * p.dkv + g_read * dh) * T, T, E[x], T,
                          C[x + y], dh, x == 1 && y == 0);
        // Test hook, empty in this repository: it is given C0, whose weight in acc is 1.
        lie::in_memory<int>(first_layer + lam, lie::ATTENTION, lie::ACCUMULATOR, st, [&](long long head, long long i, long long channel) -> int * {
          const long long w = row_of(head, i);
          return w < 0 || channel < 0 || channel >= dh ? nullptr : r.C0 + w * dh + channel;
        });
        k_divide<<<nbk((size_t)rows * dh, 256, 4096), 256, 0, st>>>(p, c, r.C0, r.C1, r.C2, Zo, o, rem, err);
        CK(cudaGetLastError());
      }
}
} // namespace gfk
