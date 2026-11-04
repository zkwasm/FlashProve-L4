// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Attention's forward pass in exact integers, with its sub-proof's lookup keys and cells (committed
// bytes). cuBLAS multiplies q, k, v as signed byte limbs, x = lo + 256 hi, with exact 32-bit sums.
// Callers must first check the shape with a_forward_ok(). No function here waits for the stream st.
#pragma once
#include "attn/plan.cuh"
// Key arrays of the lookup calls (ACall) with layer lam's lookup segment (ASeg) RM or RD, kr[lam],
// and FM or FD, kf[lam]. A piece (APiece) of RM starts at oRM[lam * HA + a]; likewise oFM .. oFD.
struct AOff {
  u16 *kr[16] = {}, *kf[16] = {}; // null where unused: a launch copies all entries to the GPU
};
// Balanced byte limb l of v: v = sum_l 256^l limb_l, limbs in [-128, 127].
__device__ __forceinline__ int a_limb(I64 v, int l) {
  int b = 0;
  for (int q = 0; q <= l; q++) {
    b = (int)(((v + 128) & 255) - 128);
    v = (v - b) >> 8;
  }
  return b;
}
__global__ void kA_fill16(u16 *k, size_t n, u16 v) {
  GSA(i, n) k[i] = v;
}
namespace gf {
__device__ __forceinline__ void limb2(int x, int &lo, int &hi) {
  lo = ((x + 128) & 255) - 128;
  hi = (x - lo) >> 8;
}
__global__ void k_split(const i16 *__restrict__ x, size_t n, i8 *__restrict__ lo, i8 *__restrict__ hi) {
  GSA(i, n) {
    int a, b;
    limb2(x[i], a, b);
    lo[i] = (i8)a;
    hi[i] = (i8)b;
  }
}
// The index of q[lam][t][ch]'s byte in a plane stored head by head: [layer][head][token][channel].
__device__ __forceinline__ size_t by_heads(const AP &p, int lam, int t, int ch) {
  return (((size_t)lam * p.H + ch / p.dh) * p.T + t) * p.dh + ch % p.dh;
}
// Writes the two limbs of q to lo and hi, head by head (by_heads()).
__global__ void k_split_heads(AP p, const i16 *__restrict__ q, i8 *__restrict__ lo, i8 *__restrict__ hi) {
  GSA(i, (size_t)p.Lam * p.T * p.dq) {
    int a, b;
    limb2(q[i], a, b);
    const size_t y = by_heads(p, (int)(i / p.dq / p.T), (int)(i / p.dq % p.T), (int)(i % p.dq));
    lo[y] = (i8)a;
    hi[y] = (i8)b;
  }
}
// Splits v into two limb planes, transposed: vt[((l Lam + lam) dkv + ch) T + t] = limb l of
// v[lam][t][ch]. Launch: grid (T / 64, ceil(dkv / 64), Lam), blocks of 256 threads. 64 must divide
// T, which a_forward_ok() tests (lt >= 6). Rows of s are padded to 66 against bank conflicts.
// Public sources for the techniques of this routine.
// The idea is public: a tiled transpose through shared memory: M. Harris, "An Efficient Matrix
// Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013, "Coalesced Transpose Via Shared Memory",
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The loop form is public: a tile covered in several loop passes: Harris's post, cited above.
__global__ void __launch_bounds__(256) k_splitT(AP p, const i16 *__restrict__ v, i8 *__restrict__ vt) {
  __shared__ i16 s[64][66];
  const int t0 = blockIdx.x * 64, c0 = blockIdx.y * 64, lam = blockIdx.z, tid = threadIdx.x;
  for (int x = tid; x < 4096; x += 256) {
    const int t = x >> 6, c = x & 63;
    s[t][c] = c0 + c < p.dkv ? v[((size_t)lam * p.T + t0 + t) * p.dkv + c0 + c] : (i16)0;
  }
  __syncthreads();
  for (int x = tid; x < 4096; x += 256) {
    const int c = x >> 6, t = x & 63;
    if (c0 + c < p.dkv) {
      int a, b;
      limb2(s[t][c], a, b);
      vt[((size_t)lam * p.dkv + c0 + c) * p.T + t0 + t] = (i8)a;
      vt[((size_t)(p.Lam + lam) * p.dkv + c0 + c) * p.T + t0 + t] = (i8)b;
    }
  }
}
// The index, within a piece, of the score of token i against token j for key-value head g, in tile
// (I, J) of B x B scores. Requires J <= I. dg = (I == J): region D, the diagonal; else M, below it.
// Source for pairing block i of a sequence with block 2 n - 1 - i (below: tile rows I and nT - 1 - I): Llama Team,
// "The Llama 3 Herd of Models", arXiv:2407.21783, section 3.3.2 ("Context parallelism for long sequences"). There the
// pairing spreads a sequence over ranks; here it is an index layout.
// A related format, a triangle stored as a full rectangle: Gustavson, Waśniewski, Dongarra, Langou, "Rectangular Full
// Packed Format for Cholesky's Algorithm: Factorization, Solution and Inversion", arXiv:0901.1696, section 2. There
// two smaller triangles are joined along their diagonals; here whole tile rows are joined end to end, and one slot of
// each slot row stays empty.
__device__ __forceinline__ size_t a_pos(const AP &p, int g, int i, int j, bool &dg) {
  const int I = i >> p.lB, J = j >> p.lB, il = i & (p.B - 1), jl = j & (p.B - 1);
  dg = I == J;
  size_t sl;
  if (dg)
    sl = (size_t)g * p.nT + I;
  else {
    const int Pp = I < p.hP ? I : p.nT - 1 - I;
    sl = ((size_t)g * p.hP + Pp) * p.nT + (I < p.hP ? J : J + Pp);
  }
  return (sl << (2 * p.lB)) + ((size_t)il << p.lB) + jl;
}
// Region M packs tile rows I and nT - 1 - I into one slot row of nT slots; the last has no tile.
// Fills these empty slots with what gfk::k_records writes for a gap (row maximum minus score) of
// u0 2^sh, sh = shl[lam]: r = 2^sh, d = 1, e = 0 in the byte planes of `as`; r, f = u0 in kr, kf.
__global__ void k_empty(AP p, const int *shl, i8 *as, AOff kM, const size_t *oRM, const size_t *oFM) {
  const int BB = p.B * p.B;
  const size_t NI = (size_t)p.Lam * p.HA, n = NI * p.G * p.hP * BB, ps = NI << p.nM;
  GSA(x, n) {
    const int y = (int)(x % BB);
    size_t u = x / BB;
    const int Pp = (int)(u % p.hP);
    u /= p.hP;
    const int g = (int)(u % p.G), in = (int)(u / p.G), lam = in / p.HA, r = 1 << shl[lam];
    const size_t pos = ((((size_t)g * p.hP + Pp) * p.nT + p.nT - 1) << (2 * p.lB)) + y;
    i8 *A = as + ((size_t)in << p.nM) + pos;
    A[0] = (i8)((r & 255) - 128);
    A[ps] = (i8)((r >> 8) - 128);
    A[2 * ps] = 1;
    A[3 * ps] = -128;
    A[4 * ps] = 0;
    kM.kr[lam][oRM[in] + pos] = (u16)r;
    kM.kf[lam][oFM[in] + pos] = (u16)p.u0;
  }
}
} // namespace gf
#include "attn/forward_kernels.cuh"
// Whether a_forward() and a_forward_all() support P. Both assume |q|, |k| <= 2,047 and v < 32,640.
static inline bool a_forward_ok(const APlan &P) {
  const AP &p = P.p;
  int shm = 0;
  for (int s : P.sh) shm = std::max(shm, s);
  return (p.dh == 32 || p.dh == 64 || p.dh == 128) && p.lt >= 6 && p.lt <= 15 && p.lB >= 4 && p.S <= 14 &&
         4 * p.Smax + ((I64)p.u0 << shm) < ((I64)1 << 32) && p.Lam <= 16;
}
// Q0, Q1: q's limb planes, head by head (gf::by_heads()); K0, K1: k's, [lam][t][channel]; vt: v's,
// transposed to [limb][lam][channel][t]. shl[lam]: layer lam's shift, at most max(P.sh), unchecked;
// Tt: the exponential table P.Tt; both on the GPU. `room`: gfk::room_bytes(p) bytes. Only with keys
// true are `as` and the lookup keys of RM, FM (kM) and RD, FD (kD) written and oRM .. oFD read.
static inline void a_scores_values(const APlan &P, const int *shl, const i8 *Q0, const i8 *Q1, const i8 *K0,
                                   const i8 *K1, const i8 *vt, const int *Tt, int *mo, int *io, int *Zo, i16 *o,
                                   u32 *rem, i8 *as, const AOff &kM, const AOff &kD, const size_t *oRM,
                                   const size_t *oFM, const size_t *oRD, const size_t *oFD, u32 *err, void *room,
                                   cudaStream_t st, bool keys = true) {
  const AP &p = P.p;
  if (keys) // without keys nothing reads the empty slots
    gf::k_empty<<<nbk((size_t)p.Lam * p.HA * p.G * p.hP * p.B * p.B, 256, 4096), 256, 0, st>>>(p, shl, as, kM, oRM,
                                                                                               oFM);
  gfk::scores_values(p, keys, shl, Q0, Q1, K0, K1, vt, Tt, mo, io, Zo, o, rem, as, kM, kD, oRM, oFM, oRD, oFD, err,
                     room, st);
}
// Bytes that a_forward() needs at ws: the limb planes of q, k and v, then gfk::room_bytes(p).
static inline size_t a_forward_scratch(const AP &p) {
  return (size_t)2 * p.Lam * p.T * (p.dq + 2 * p.dkv) + 1024 + gfk::room_bytes(p);
}
// q is [lam][t][H dh]; k, v are [lam][t][G dh]; shl, Tt, keys: see a_scores_values(). Per query row
// [lam][h][t], writes mo (the largest score over j <= t), io (the first j with it), Zo (the sum Z
// of the softmax weights e); per channel c, o (sum_j e v / Z, rounded) and its remainder rem. Sets
// *err = 1, and never clears it, if some 2 Z - 1 >= 2^(8 wd). All but shl must be 16-byte aligned.
static inline void a_forward(const APlan &P, const int *shl, const i16 *q, const i16 *k, const i16 *v, const int *Tt,
                             int *mo, int *io, int *Zo, i16 *o, u32 *rem, i8 *as, const AOff &kM, const AOff &kD,
                             const size_t *oRM, const size_t *oFM, const size_t *oRD, const size_t *oFD, u32 *err,
                             void *ws, cudaStream_t st = 0, bool keys = true) {
  const AP &p = P.p;
  const size_t nq = (size_t)p.Lam * p.T * p.dq, nk = (size_t)p.Lam * p.T * p.dkv;
  i8 *Q0 = (i8 *)ws, *Q1 = Q0 + nq, *K0 = Q1 + nq, *K1 = K0 + nk, *vt = K1 + nk;
  gf::k_split_heads<<<nbk(nq, 256, 4096), 256, 0, st>>>(p, q, Q0, Q1);
  gf::k_split<<<nbk(nk, 256, 4096), 256, 0, st>>>(k, nk, K0, K1);
  gf::k_splitT<<<dim3(p.T / 64, (p.dkv + 63) / 64, p.Lam), 256, 0, st>>>(p, v, vt);
  a_scores_values(P, shl, Q0, Q1, K0, K1, vt, Tt, mo, io, Zo, o, rem, as, kM, kD, oRM, oFM, oRD, oFD, err,
                  vt + 2 * nk, st, keys);
}
namespace gf {
// Copies np byte planes of a tile (channels ch0 .. ch0 + 63, tokens t0 .. t0 + 63) from s to cells
// indexed (channel, token): plane pl goes to dst[pl][ch T + t]. In s, the 64 bytes of plane pl and
// channel ch0 + cl start at (64 pl + cl) 68. Channels at or past nch and tokens at or past T are
// skipped. Requires: 4 divides T and t0; s and dst[pl] are 4-byte aligned; blocks of 256 threads.
// The loop form is public: a tile covered in several loop passes: Harris (cited at k_splitT()).
__device__ __forceinline__ void put_planes(const u8 *s /* the planes' bytes */, int np,
                                           i8 *const *dst /* plane bases: + ch T + t */, int ch0, int nch, int t0,
                                           int T) {
  for (int x = threadIdx.x; x < np * 1024; x += 256) {
    const int w4 = x & 15, cl = (x >> 4) & 63, pl = x >> 10, ch = ch0 + cl, t = t0 + 4 * w4;
    if (ch < nch && t < T)
      *(u32 *)(dst[pl] + (size_t)ch * T + t) = *(const u32 *)(s + ((size_t)pl * 64 + cl) * 68 + 4 * w4);
  }
}
// Splits q into two limb planes: head by head into N0, N1 (skipped when N0 is null), transposed
// into qa's q cells. qa has the cells of q, k, v, o in this order, each [limb][lam][channel][t].
// The compiler hint is public (cited at k_splitT()).
__global__ void __launch_bounds__(256) k_pre_q(AP p, const i16 *__restrict__ q, i8 *N0, i8 *N1, i8 *qa) {
  __shared__ __align__(16) u8 s[2 * 64 * 68];
  const int t0 = blockIdx.x * 64, c0 = blockIdx.y * 64, lam = blockIdx.z, T = p.T;
  for (int x = threadIdx.x; x < 4096; x += 256) {
    const int tl = x >> 6, cl = x & 63, t = t0 + tl, ch = c0 + cl;
    if (t < T && ch < p.dq) {
      const size_t y = ((size_t)lam * T + t) * p.dq + ch;
      int a, b;
      limb2(q[y], a, b);
      if (N0) {
        N0[by_heads(p, lam, t, ch)] = (i8)a;
        N1[by_heads(p, lam, t, ch)] = (i8)b;
      }
      s[cl * 68 + tl] = (u8)a;
      s[(64 + cl) * 68 + tl] = (u8)b;
    }
  }
  __syncthreads();
  i8 *d[2] = {qa + (size_t)lam * p.dq * T, qa + (size_t)(p.Lam + lam) * p.dq * T};
  put_planes(s, 2, d, c0, p.dq, t0, T);
}
// Splits k into limb planes N0, N1 in k's own order, [lam][t][channel] (skipped when N0 is null),
// and writes the transposed limb planes of k and v into qa's k and v cells.
// The compiler hint is public (cited at k_splitT()).
__global__ void __launch_bounds__(256)
    k_pre_kv(AP p, const i16 *__restrict__ k, const i16 *__restrict__ v, i8 *N0, i8 *N1, i8 *qa) {
  __shared__ __align__(16) u8 s[4 * 64 * 68];
  const int t0 = blockIdx.x * 64, c0 = blockIdx.y * 64, lam = blockIdx.z, T = p.T;
  for (int x = threadIdx.x; x < 4096; x += 256) {
    const int tl = x >> 6, cl = x & 63, t = t0 + tl, ch = c0 + cl;
    if (t < T && ch < p.dkv) {
      const size_t y = ((size_t)lam * T + t) * p.dkv + ch;
      int a, b;
      limb2(k[y], a, b);
      if (N0) {
        N0[y] = (i8)a;
        N1[y] = (i8)b;
      }
      s[cl * 68 + tl] = (u8)a;
      s[(64 + cl) * 68 + tl] = (u8)b;
      limb2(v[y], a, b);
      s[(128 + cl) * 68 + tl] = (u8)a;
      s[(192 + cl) * 68 + tl] = (u8)b;
    }
  }
  __syncthreads();
  const size_t oK = (size_t)2 * p.Lam * p.dq, oV = oK + (size_t)2 * p.Lam * p.dkv;
  i8 *d[4] = {qa + (oK + (size_t)lam * p.dkv) * T, qa + (oK + (size_t)(p.Lam + lam) * p.dkv) * T,
              qa + (oV + (size_t)lam * p.dkv) * T, qa + (oV + (size_t)(p.Lam + lam) * p.dkv) * T};
  put_planes(s, 4, d, c0, p.dkv, t0, T);
}
// From rem and 2 Zo - 1 - rem, fills the lookup keys of WL (low 16 bits) and WH and ao's cells (the
// higher bytes); from o's limbs, qa's o cells. The compiler hint is public (cited at k_splitT()).
__global__ void __launch_bounds__(256)
    k_post_o(AP p, const int *__restrict__ Zo, const i16 *__restrict__ o, const u32 *__restrict__ rem, i8 *ao, i8 *qa,
             u16 *kW, u16 *kH, const size_t *offWL, const size_t *offWH) {
  // Source for a rounded quotient through a range-checked remainder: Chen, Waiwitlikhit, Stoica, Kang, "ZKML: An
  // Optimizing System for ML Inference in Zero-Knowledge Proofs", EuroSys 2024, section 5.1.
  // Source for the same as one range relation, the distances to both ends of the range looked up (rem and 2 Zo - 1 -
  // rem): Qu, Sun, Liu, Lu, Guo, Chen, Zhang, "zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for
  // LLM Inference", USENIX Security 2025, sections 4.3 and 2.3.
  __shared__ __align__(16) u8 s[6 * 64 * 68];
  const int t0 = blockIdx.x * 64, c0 = blockIdx.y * 64, lam = blockIdx.z, T = p.T, dh = p.dh;
  for (int x = threadIdx.x; x < 4096; x += 256) {
    const int tl = x >> 6, cl = x & 63, t = t0 + tl, ch = c0 + cl;
    if (t < T && ch < p.dq) {
      const int h = ch >> p.ldh, c = ch & (dh - 1), g = h / p.HA, in = lam * p.HA + h % p.HA;
      const size_t r = ((size_t)lam * p.H + h) * T + t, xx = r * dh + c, y = ((size_t)g * T + t) * dh + c;
      const u32 rm = rem[xx], r2 = 2 * (u32)Zo[r] - 1 - rm;
      kW[offWL[in] + y] = (u16)rm;
      kW[offWL[in] + (size_t)p.G * T * dh + y] = (u16)r2;
      kH[offWH[in] + y] = (u16)(((rm >> 16) & 255) + 256 * ((r2 >> 16) & 255));
      if (p.wd == 4) {
        kH[offWH[in] + (size_t)p.G * T * dh + y] = (u16)((rm >> 24) + 256 * (r2 >> 24));
        s[(256 + cl) * 68 + tl] = (u8)((int)(rm >> 24) - 128);
        s[(320 + cl) * 68 + tl] = (u8)((int)(r2 >> 24) - 128);
      }
      int a, b;
      limb2(o[xx], a, b);
      s[cl * 68 + tl] = (u8)((int)((rm >> 16) & 255) - 128);
      s[(64 + cl) * 68 + tl] = (u8)((int)((r2 >> 16) & 255) - 128);
      s[(128 + cl) * 68 + tl] = (u8)a;
      s[(192 + cl) * 68 + tl] = (u8)b;
    }
  }
  __syncthreads();
  const size_t oO = (size_t)2 * p.Lam * (p.dq + 2 * p.dkv);
  i8 *d[6] = {ao + (size_t)lam * p.dq * T,
              ao + (size_t)(p.Lam + lam) * p.dq * T,
              qa + (oO + (size_t)lam * p.dq) * T,
              qa + (oO + (size_t)(p.Lam + lam) * p.dq) * T,
              ao + (size_t)(2 * p.Lam + lam) * p.dq * T,
              ao + (size_t)(3 * p.Lam + lam) * p.dq * T};
  put_planes(s, p.wd == 4 ? 6 : 4, d, c0, p.dq, t0, T);
}
// Writes each query row's limbs of mo and Zo and low lt bits of io to ah, and its IX key to kW.
__global__ void k_post_rows(AP p, const int *mo, const int *io, const int *Zo, i8 *ah, u16 *kW, const size_t *offIX) {
  const int T = p.T, H = p.H, Lam = p.Lam, HA = p.HA;
  const size_t nr = (size_t)Lam * H * T;
  GSA(x, nr) {
    int t = x % T, h = (int)(x / T) % H, lam = (int)(x / T / H), g = h / HA, a = h % HA;
    I64 m = mo[x], Z = Zo[x];
    int id = io[x];
    for (int l = 0; l < 4; l++) {
      ah[((size_t)(l * Lam + lam) * H + h) * T + t] = (i8)a_limb(m, l);
      ah[((size_t)4 * Lam * H + (size_t)(l * Lam + lam) * H + h) * T + t] = (i8)a_limb(Z, l);
    }
    for (int kp = 0; kp < p.lt; kp++)
      ah[((size_t)8 * Lam * H + (size_t)lam * p.lt * H + kp * H + h) * T + t] = (i8)((id >> kp) & 1);
    kW[offIX[lam * HA + a] + (size_t)g * T + t] = (u16)((t - id) << (16 - p.lt));
  }
}
} // namespace gf
// Bytes that a_forward_all() needs at ws: the limb planes of q and k, then gfk::room_bytes(p).
// The limbs of v are read from qa's cells.
// Public source: several readers of one repacked copy of the weights: XNNPack's weights cache
// (Ng and Dukhan, TensorFlow Blog, June 2022,
// https://blog.tensorflow.org/2022/06/memory-efficient-inference-with-xnnpack.html).
// Public source: reading the preceding step's format: oneDNN's guide (Intel, "oneAPI Deep Neural
// Network Developer Guide and Reference", 2025.0, "Memory Format Propagation"). Here the data is v.
static inline size_t a_forward_all_scratch(const AP &p) {
  return (size_t)2 * p.Lam * p.T * (p.dq + p.dkv) + 1024 + gfk::room_bytes(p);
}
// Does what a_forward() does with keys true, and writes the cells of the matrices ah, ao, qa (AMat)
// and the lookup keys of WL, WH, IX. kW, kH: key arrays of the lookup calls with WL, WH; oIX is
// relative to kW. ws: a_forward_all_scratch(p) bytes. Aligned as for a_forward(); qa, ao: 4 bytes.
static inline void a_forward_all(const APlan &P, const int *shl, const i16 *q, const i16 *k, const i16 *v,
                                 const int *Tt, int *mo, int *io, int *Zo, i16 *o, u32 *rem, i8 *as, const AOff &kM,
                                 const AOff &kD, const size_t *oRM, const size_t *oFM, const size_t *oRD,
                                 const size_t *oFD, i8 *ah, i8 *ao, i8 *qa, u16 *kW, u16 *kH, const size_t *oWL,
                                 const size_t *oWH, const size_t *oIX, u32 *err, void *ws, cudaStream_t st = 0) {
  const AP &p = P.p;
  const int T = p.T, Lam = p.Lam, nt = (T + 63) / 64;
  const size_t nq = (size_t)Lam * T * p.dq, nk = (size_t)Lam * T * p.dkv;
  i8 *Q0 = (i8 *)ws, *Q1 = Q0 + nq, *K0 = Q1 + nq, *K1 = K0 + nk;
  [[maybe_unused]] const int first_layer = lie::layer_now(); // 0; only the test hooks below get it
  gf::k_pre_q<<<dim3(nt, (p.dq + 63) / 64, Lam), 256, 0, st>>>(p, q, Q0, Q1, qa);
  gf::k_pre_kv<<<dim3(nt, (p.dkv + 63) / 64, Lam), 256, 0, st>>>(p, k, v, K0, K1, qa);
  // Public sources for reading v's limbs from qa's cells (cited at a_forward_all_scratch()).
  const i8 *vt = qa + ((size_t)2 * Lam * p.dq + (size_t)2 * Lam * p.dkv) * T;
  a_scores_values(P, shl, Q0, Q1, K0, K1, vt, Tt, mo, io, Zo, o, rem, as, kM, kD, oRM, oFM, oRD, oFD, err, K1 + nk,
                  st);
  gf::k_post_o<<<dim3(nt, (p.dq + 63) / 64, Lam), 256, 0, st>>>(p, Zo, o, rem, ao, qa, kW, kH, oWL, oWH);
  gf::k_post_rows<<<nbk((size_t)Lam * p.H * T), 256, 0, st>>>(p, mo, io, Zo, ah, kW, oIX);
  // Empty test hooks (prover/common/lie.cuh), on cells that this pass never reads.
  for (int lam = 0; lam < Lam; lam++) {
    lie::in_memory<i8>(first_layer + lam, lie::ATTENTION, lie::BIT, st, [&](long long head, long long t, long long bit) -> i8 * {
      if (head < 0 || head >= p.H || t < 0 || t >= T || bit < 0 || bit >= p.lt) return nullptr;
      return ah + ((size_t)8 * Lam * p.H + (size_t)lam * p.lt * p.H + (size_t)bit * p.H + (size_t)head) * T + t;
    });
    lie::in_memory<i8>(first_layer + lam, lie::ATTENTION, lie::EMPTY, st, [&](long long head, long long slot_row, long long place) -> i8 * {
      if (head < 0 || head >= p.H || slot_row < 0 || slot_row >= p.hP || place < 0 || place >= (long long)p.B * p.B) return nullptr;
      const size_t NI = (size_t)Lam * p.HA, in = (size_t)lam * p.HA + (size_t)(head % p.HA), g = (size_t)(head / p.HA);
      return as + ((3 * NI + in) << p.nM) + ((((g * p.hP + (size_t)slot_row) * p.nT + p.nT - 1) << (2 * p.lB)) + (size_t)place);
    });
  }
}
