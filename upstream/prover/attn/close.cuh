// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Kernels, host helpers and sumcheck operand sources of the prover's attention sub-proof. Include
// after engines.cuh and forward.cuh. In formulas T (tokens), H, dh etc. are AP's fields (plan.cuh);
// layer lam; key-value head g; query head h = g HA + a; piece (APiece) `in` or in[x] = lam HA + a.
#pragma once
#include "common/index_guard.cuh"
#define FA F::add
#define FM_ F::mul
#define FS F::sub
// a times the small signed integer v.
__device__ __forceinline__ T f_mi(T a, I64 v) {
  return F::mulb(a, F::fromi(v));
}
// Sum of v over the block's 256 threads, returned to every thread. Requires blockDim.x == 256, and
// every thread of the block must call it: it has barriers, the last so that it can be called again.
// The idea is public: a tree reduction in shared memory with sequential addressing: M. Harris,
// "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology, slides "Parallel Reduction:
// Sequential Addressing" and "Reduction #3: Sequential Addressing",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
__device__ __forceinline__ T blk_sum(T v) {
  __shared__ T sh_[256];
  const int t = threadIdx.x;
  sh_[t] = v;
  __syncthreads();
  for (int w = 128; w; w >>= 1) {
    if (t < w) sh_[t] = FA(sh_[t], sh_[t + w]);
    __syncthreads();
  }
  T r = sh_[0];
  __syncthreads();
  return r;
}
// v[pl] = sum_x wp[x] sum_hi eh[hi] tab[base + pl pstr + (in[x] << nh) + hi], x < cnt, hi < 2^nh.
// One block of 256 threads per plane pl. tab: a table (see hval()), one entry per column of `as`,
// which has region M, then D, each of 5 planes. base: columns before the region; pstr: per plane.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html; such a
// loop before a tree reduction: Harris's slides, "Algorithm Cascading" (cited at blk_sum()).
__global__ void k_planes(const T *tab, const T *eh, int nh, const T *wp, const int *in, int cnt, size_t base,
                         size_t pstr, T *v) {
  const int pl = blockIdx.x;
  T s = F::zero();
  for (int y = threadIdx.x; y < cnt << nh; y += 256) {
    int x = y >> nh, hi = y & ((1 << nh) - 1);
    s = FA(s, FM_(FM_(wp[x], eh[hi]), tab[base + (size_t)pl * pstr + ((size_t)in[x] << nh) + hi]));
  }
  s = blk_sum(s);
  if (!threadIdx.x) v[pl] = s;
}
// A table on a request matrix (AMat) with weights phi: U[e] = sum_y phi[y] cell[e L + y], y < L.
// U is a short table on ah: L = T. Returns sum_l 256^l U[((4 X + l) Lam + lam) H + h], l < 4: the
// limbs of m, a row's largest score (X = 0), or of Z, the sum of the row's softmax weights (X = 1).
__device__ __forceinline__ T hval(const AP &p, const T *U, int X, int lam, int h) {
  T v = F::zero();
  for (int l = 3; l >= 0; l--)
    v = FA(f_mi(v, 256), U[(size_t)X * 4 * p.Lam * p.H + (size_t)(l * p.Lam + lam) * p.H + h]);
  return v;
}
// o = sum_x wp[x] sum_g eg[g] hval(p, U, X, lam, h), x < cnt. Launch with one block of 256 threads.
// The loop form is public: the block-stride loop before a tree reduction: CUDA Programming Guide,
// "Writing SIMT Kernels" (cited at k_planes()); Harris, "Algorithm Cascading" (cited at blk_sum()).
__global__ void k_rowval(AP p, const T *U, int X, const T *wp, const int *in, int cnt, const T *eg, T *o) {
  T s = F::zero();
  for (int y = threadIdx.x; y < cnt * p.G; y += 256) {
    int x = y / p.G, g = y % p.G;
    s = FA(s, FM_(FM_(wp[x], eg[g]), hval(p, U, X, in[x] / p.HA, g * p.HA + in[x] % p.HA)));
  }
  s = blk_sum(s);
  if (!threadIdx.x) *o = s;
}
// UF is a long table: L = 2^ls T, one entry per 2^ls consecutive features of T cells each. Returns
//   sum_ch hi[ch] sum_l 256^l UF[((f0 + l dl) >> ls) + ch], ch < dh >> ls, l < nl.
// f0 and dl count features, not entries of UF.
__device__ __forceinline__ T xl(const AP &p, const T *UF, size_t f0, size_t dl, int nl, const T *hi) {
  T s = F::zero();
  for (int ch = 0; ch < (p.dh >> p.ls); ch++) {
    T v = F::zero();
    for (int l = nl - 1; l >= 0; l--) v = FA(f_mi(v, 256), UF[((f0 + l * dl) >> p.ls) + ch]);
    s = FA(s, FM_(hi[ch], v));
  }
  return s;
}
// o[w] = sum_x wp[x] sum_g eg[g] xl(p, UF, (w Lam + lam) dq + h dh, 0, 1, hi). UF is a long table
// on ao, whose tensors store bytes minus 128: bits 16 to 23 of the division remainder rem (w = 0)
// or of 2 Z - 1 - rem (w = 1); if p.wd == 4, also bits 24 to 31 (w = 2, 3). One block, 256 threads.
// The loop form is public: the block-stride loop before a tree reduction: CUDA Programming Guide,
// "Writing SIMT Kernels" (cited at k_planes()); Harris, "Algorithm Cascading" (cited at blk_sum()).
__global__ void k_rhval(AP p, const T *UF, const T *hi, const T *wp, const int *in, int cnt, const T *eg, T *o) {
  for (int w = 0; w < (p.wd == 4 ? 4 : 2); w++) {
    T s = F::zero();
    for (int y = threadIdx.x; y < cnt * p.G; y += 256) {
      int x = y / p.G, g = y % p.G;
      s = FA(s, FM_(FM_(wp[x], eg[g]),
                    xl(p, UF, ((size_t)(w * p.Lam + in[x] / p.HA) * p.dq + (size_t)(g * p.HA + in[x] % p.HA) * p.dh), 0,
                       1, hi)));
    }
    s = blk_sum(s);
    if (!threadIdx.x) o[w] = s;
  }
}
// o[kappa] = sum_x wp[x] sum_g eg[g] U[8 Lam H + (lam lt + kappa) H + h]. U: a short table on ah.
// After the 8 Lam H features that hval() reads, ah stores bit kappa < lt of io, the number of the
// first token with the row's largest score. Launch: one block of 256 threads per kappa.
// The loop form is public: the block-stride loop before a tree reduction: CUDA Programming Guide,
// "Writing SIMT Kernels" (cited at k_planes()); Harris, "Algorithm Cascading" (cited at blk_sum()).
__global__ void k_ibval(AP p, const T *U, const T *wp, const int *in, int cnt, const T *eg, T *o) {
  const int kp = blockIdx.x;
  T s = F::zero();
  for (int y = threadIdx.x; y < cnt * p.G; y += 256) {
    int x = y / p.G, g = y % p.G;
    s = FA(s,
           FM_(FM_(wp[x], eg[g]),
               U[(size_t)8 * p.Lam * p.H + (size_t)(in[x] / p.HA) * p.lt * p.H + kp * p.H + g * p.HA + in[x] % p.HA]));
  }
  s = blk_sum(s);
  if (!threadIdx.x) o[kp] = s;
}
// o[in lt + kappa] = sum_g eg[g] U[8 Lam H + (lam lt + kappa) H + h], in < Lam HA, kappa < lt:
// the entries that k_ibval() reads, summed over g only.
__global__ void k_ibs(AP p, const T *U, const T *eg, T *o) {
  GSA(y, (size_t)p.Lam * p.HA * p.lt) {
    int kp = y % p.lt, in = (int)(y / p.lt);
    T s = F::zero();
    for (int g = 0; g < p.G; g++)
      s = FA(s, FM_(eg[g],
                    U[(size_t)8 * p.Lam * p.H + (size_t)(in / p.HA) * p.lt * p.H + kp * p.H + g * p.HA + in % p.HA]));
    o[y] = s;
  }
}
// o[in] = sum_g eg[g] hval(p, U, 0, lam, h), in < Lam HA.
__global__ void k_m6(AP p, const T *U, const T *eg, T *o) {
  GSA(in, (size_t)p.Lam * p.HA) {
    T s = F::zero();
    for (int g = 0; g < p.G; g++) s = FA(s, FM_(eg[g], hval(p, U, 0, (int)in / p.HA, g * p.HA + (int)in % p.HA)));
    o[in] = s;
  }
}
// d = a followed by b. Launch with a single block.
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_planes()).
__global__ void k_cat(T *d, const T *a, int na, const T *b, int nb) {
  for (int i = threadIdx.x; i < na + nb; i += blockDim.x) d[i] = i < na ? a[i] : b[i - na];
}
namespace gc {
// Up to 32 eq tables in one launch: t[x] = eq(z, x), x < 2^n. eq(z, x) is the product over i < n of
// z[i] if bit n-1-i of x is set, else of 1 - z[i]: coordinate 0 of a point goes with x's top bit.
struct EqJob {
  const T *z;
  int n;
  T *t;
};
struct EqJobs {
  EqJob j[32] = {}; // 0 where unused: a launch copies all 32 to the GPU
};
// Builds the eq table of job blockIdx.y. It requires blocks of 256 threads.
// The compiler hint and the loop form are public: __launch_bounds__: NVIDIA, "CUDA C++ Best
// Practices Guide", "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; the
// block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels" (cited at k_planes()).
__global__ void __launch_bounds__(256) k_eqs(EqJobs J) {
  constexpr int LLO = 10;
  __shared__ T lo[1 << LLO];
  const EqJob jb = J.j[blockIdx.y];
  const T *z = jb.z;
  T *t = jb.t;
  const int n = jb.n, nl = n < LLO ? n : LLO, nh = n - nl;
  if (blockIdx.x >= ((size_t)1 << nh)) return;
  if (threadIdx.x == 0) lo[0] = F::one();
  __syncthreads();
  for (int v = 0; v < nl; v++) {
    const int s = 1 << v;
    const T zz = z[nh + v];
    T h[4], c[4];
    int q = 0;
    for (int x = threadIdx.x; x < s; x += 256, q++) {
      c[q] = lo[x];
      h[q] = F::mul(c[q], zz);
    }
    __syncthreads();
    q = 0;
    for (int x = threadIdx.x; x < s; x += 256, q++) {
      lo[2 * x + 1] = h[q];
      lo[2 * x] = F::sub(c[q], h[q]);
    }
    __syncthreads();
  }
  for (size_t hi = blockIdx.x; hi < ((size_t)1 << nh); hi += gridDim.x) {
    T p = F::one();
    for (int v = 0; v < nh; v++) {
      const T zz = z[v];
      p = F::mul(p, ((hi >> (nh - 1 - v)) & 1) ? zz : F::sub(F::one(), zz));
    }
    const F::P pp = F::prep(p);
    for (int x = threadIdx.x; x < (1 << nl); x += 256) st16(t + (hi << nl) + x, F::mulp(pp, lo[x]));
  }
}
// Collects eq-table jobs; run() launches them in batches of up to 32 and clears the list.
struct EqList {
  std::vector<EqJob> v;
  void add(const T *z, int n, T *t) { v.push_back({z, n, t}); }
  void run(cudaStream_t st = 0) {
    for (size_t i = 0; i < v.size(); i += 32) {
      EqJobs J;
      int m = (int)std::min<size_t>(32, v.size() - i), mx = 0;
      for (int k = 0; k < m; k++) {
        J.j[k] = v[i + k];
        mx = std::max(mx, v[i + k].n);
      }
      k_eqs<<<dim3(1u << std::min(std::max(mx - 10, 0), 7), m), 256, 0, st>>>(J);
    }
    v.clear();
  }
};
// Up to 24 outer products in one launch: o[(i << (lb + lc)) | (j << lc) | k] = a[i] b[j] c[k];
// c may be null with lc = 0.
struct OuJob {
  const T *a, *b, *c;
  T *o;
  int la, lb, lc;
};
struct OuJobs {
  OuJob j[24] = {}; // 0 where unused, as in EqJobs
};
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_outers(OuJobs J) {
  const OuJob jb = J.j[blockIdx.y];
  const size_t n = (size_t)1 << (jb.la + jb.lb + jb.lc);
  for (size_t y = blockIdx.x * (size_t)blockDim.x + threadIdx.x; y < n; y += (size_t)gridDim.x * blockDim.x) {
    T v = F::mul(jb.a[y >> (jb.lb + jb.lc)], jb.b[(y >> jb.lc) & (((size_t)1 << jb.lb) - 1)]);
    if (jb.c) v = F::mul(v, jb.c[y & (((size_t)1 << jb.lc) - 1)]);
    st16(jb.o + y, v);
  }
}
// Collects outer-product jobs; run() launches them in batches of up to 24 and clears the list.
struct OuList {
  std::vector<OuJob> v;
  void add(const T *a, int la, const T *b, int lb, T *o) { v.push_back({a, b, nullptr, o, la, lb, 0}); }
  void add3(const T *a, int la, const T *b, int lb, const T *c, int lc, T *o) { v.push_back({a, b, c, o, la, lb, lc}); }
  void run(cudaStream_t st = 0) {
    for (size_t i = 0; i < v.size(); i += 24) {
      OuJobs J;
      int m = (int)std::min<size_t>(24, v.size() - i), mx = 0;
      for (int k = 0; k < m; k++) {
        J.j[k] = v[i + k];
        mx = std::max(mx, v[i + k].la + v[i + k].lb + v[i + k].lc);
      }
      k_outers<<<dim3(1u << std::min(std::max(mx - 8, 0), 9), m), 256, 0, st>>>(J);
    }
    v.clear();
  }
};
struct WpJob {
  const T *zk;
  int n, b, cnt;
  const size_t *off;
  T *wp;
};
struct WpJobs {
  WpJob j[32] = {}; // 0 until first used: a launch copies all 32 to the GPU
};
// The largest number of pieces of one segment kind (ASeg) that a single lookup call may contain,
// and so the largest cnt of a job of k_wps().
constexpr int WP_PIECES = 128;
// A job covers the cnt pieces of one segment kind in a lookup call of 2^n keys. Piece x has 2^b
// keys, the first at off[x], and the weight wp[x] = eq(first n - b coordinates of zk, off[x] >> b);
// wp[cnt] is the weights' sum. Requires cnt <= WP_PIECES and a block of WP_PIECES threads per job.
__global__ void k_wps(WpJobs J) {
  const WpJob jb = J.j[blockIdx.x];
  __shared__ T sh[WP_PIECES];
  const int x = threadIdx.x;
  if (x < jb.cnt) {
    const size_t a = jb.off[x] >> jb.b;
    T w = F::one();
    for (int v = 0; v < jb.n - jb.b; v++) {
      const int bit = (a >> (jb.n - jb.b - 1 - v)) & 1;
      w = F::mul(w, bit ? jb.zk[v] : F::sub(F::one(), jb.zk[v]));
    }
    jb.wp[x] = w;
    sh[x] = w;
  }
  __syncthreads();
  if (x == 0) {
    T s = F::zero();
    for (int i = 0; i < jb.cnt; i++) s = F::add(s, sh[i]);
    jb.wp[jb.cnt] = s;
  }
}
// k_sends() (one thread) sends each job's a, then b (may be empty), all masked: sc_door::claim()
// (prover/hiding/sumcheck_door.cuh) stores each value plus its message slot's mask entry in out.
struct SdJob {
  const T *a;
  int na;
  const T *b;
  int nb;
};
struct SdJobs {
  SdJob j[16] = {}; // 0 where unused, as in EqJobs
  int n = 0;
};
__global__ void k_sends(Tr *tr, SdJobs J, T *out) {
  Tr t = *tr;
  for (int s = 0; s < J.n; s++) {
    const SdJob jb = J.j[s];
    for (int i = 0; i < jb.na; i++) sc_door::claim(sc_door::Site::ATTENTION_SENT, jb.a[i], out + i);
    for (int i = 0; i < jb.nb; i++) sc_door::claim(sc_door::Site::ATTENTION_SENT, jb.b[i], out + jb.na + i);
    sc_door::absorb(t, out, jb.na + jb.nb);
    out += jb.na + jb.nb;
  }
  *tr = t;
}
// Sets the tile (I, J) of slot sl of region D (dg nonzero: I = J) or M (J < I). False: empty slot.
__device__ __forceinline__ bool slotIJ(const AP &p, int dg, int sl, int &I, int &J) {
  if (dg) {
    I = J = sl;
    return true;
  }
  const int Pp = sl / p.nT, sg = sl % p.nT;
  if (sg == p.nT - 1) {
    I = J = 0;
    return false;
  }
  if (sg < Pp) {
    I = Pp;
    J = sg;
  } else {
    I = p.nT - 1 - Pp;
    J = sg - Pp;
  }
  return true;
}
// Returns eq(z, a) for the n-bit index a: the product over a's bits, top bit first, of z[v] where
// the bit is 1 and 1 - z[v] where it is 0.
__device__ __forceinline__ T eqv(const T *z, int n, int a) {
  T w = F::one();
  for (int v = 0; v < n; v++) w = F::mul(w, ((a >> (n - 1 - v)) & 1) ? z[v] : F::sub(F::one(), z[v]));
  return w;
}
// The tile slots' weights: wt[sl] = eq(zP, sl), or 0 on an empty slot. WI[I] and WJ[J] are their
// sums over tile row I and tile column J. zP has lnT coordinates in region D (dg nonzero), else
// 2 lnT - 1. Launch with one block and, if dg is 0, dynamic shared memory for hP + nT elements.
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_planes()).
__global__ void k_slots(AP p, int dg, const T *zP, T *wt, T *WI, T *WJ) {
  extern __shared__ __align__(16) char sm_[];
  T *eP = (T *)sm_, *eS = eP + p.hP;
  const int nT = p.nT, lnT = p.lnT, tid = threadIdx.x;
  if (dg) {
    for (int I = tid; I < nT; I += blockDim.x) {
      const T w = eqv(zP, lnT, I);
      wt[I] = w;
      WI[I] = w;
      WJ[I] = w;
    }
    return;
  }
  for (int x = tid; x < p.hP + nT; x += blockDim.x) {
    if (x < p.hP)
      eP[x] = eqv(zP, lnT - 1, x);
    else
      eS[x - p.hP] = eqv(zP + lnT - 1, lnT, x - p.hP);
  }
  __syncthreads();
  for (int sl = tid; sl < p.hP * nT; sl += blockDim.x)
    wt[sl] = (sl % nT == nT - 1) ? F::zero() : F::mul(eP[sl / nT], eS[sl % nT]);
  __syncthreads();
  for (int x = tid; x < 2 * nT; x += blockDim.x) {
    const int X = x >> 1;
    T s = F::zero();
    for (int sl = 0; sl < p.hP * nT; sl++) {
      int I, J;
      if (slotIJ(p, 0, sl, I, J) && ((x & 1) ? J : I) == X) s = F::add(s, wt[sl]);
    }
    if (x & 1)
      WJ[X] = s;
    else
      WI[X] = s;
  }
}
// wv[sl] = eih[I] for slot sl's tile (I, J), or 0 on an empty slot. Launch with one block.
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at k_planes()).
__global__ void k_slotI(AP p, int dg, const T *eih, T *wv) {
  const int nsl = dg ? p.nT : p.hP * p.nT;
  for (int sl = threadIdx.x; sl < nsl; sl += blockDim.x) {
    int I, J;
    wv[sl] = slotIJ(p, dg, sl, I, J) ? eih[I] : F::zero();
  }
}
// S4: an exact sum of (field element, split into limbs: SL) x (small signed integer), reduced once;
// see engines.cuh. Capacity, not checked: (s4mac() calls on a sum) x (largest |v|) <= 2^29.
typedef ga::I4 S4;
typedef ga::Limbs SL;
__device__ __forceinline__ void s4mac(S4 &s, const SL &l, int v) {
  ga::i4mac(s, l, v);
}
__device__ __forceinline__ T s4red(const S4 &s) {
  return ga::i4red(s);
}
// One step of a butterfly sum within a warp: lane i adds the sums of lane i xor w.
// All 32 lanes must call it, so k_dotc() and k_rowsE() round their loops' bounds up to a multiple
// of 256. Two lanes' sums add as integers, limb by limb.
// S4's capacity counts the s4mac() calls of all lanes so added. s4xor() uses __shfl_xor_sync()
// (CUDA C++ Programming Guide, "Warp Shuffle Functions"). See also, for a listing of such a sum:
// NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions", "Warp Shuffle Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// The compiler hint is public: #pragma unroll: the same page of the CUDA Programming Guide as
// above, "#pragma unroll"; M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA
// Technical Blog, 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
__device__ __forceinline__ void s4xor(S4 &s, int w) {
#pragma unroll
  for (int i = 0; i < EDEG; i++) s.l[i] += __shfl_xor_sync(~0u, s.l[i], w), s.h[i] += __shfl_xor_sync(~0u, s.h[i], w);
}
// o[((lam nh_ + h) nT + I) dh + c] = sum_l el[l] x[lam][I B + l][h dh + c], l < B (qf, kf of Src4).
// x is q (nh_ = H heads) or k (nh_ = G), indexed [layer][token][head][channel].
// Requires B <= 2^14 (S4's capacity, B int16 terms), 256 threads, dynamic shared memory for B SL.
// The idea is public: a block's threads share data staged once in shared memory: NVIDIA's Best
// Practices Guide (cited at k_eqs()), "Shared Memory in Matrix Multiplication (C=AB)"; M. Harris,
// "Optimizing CUDA", NVIDIA Developer Technology, slides headed "S05: High Performance Computing
// with CUDA", slide "Take Advantage of Shared Memory" (the address runs over two lines),
// https://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/Cours09/CUDA/
// SC07_CUDA_5_Optimization_Harris.pdf.
// The compiler hint is public: __launch_bounds__: NVIDIA's Best Practices Guide (cited at k_eqs()).
__global__ void __launch_bounds__(256) k_foldtile(AP p, const i16 *__restrict__ x, int nh_, const T *el, T *o) {
  extern __shared__ __align__(16) char sm_[];
  SL *le = (SL *)sm_;
  for (int l = threadIdx.x; l < p.B; l += 256) le[l] = ga::limbs_of(el[l]);
  __syncthreads();
  const size_t n = (size_t)p.Lam * nh_ * p.nT * p.dh, rs = (size_t)nh_ * p.dh;
  GSA(y, n) {
    const int c = y % p.dh, I = (int)(y / p.dh) % p.nT, h = (int)(y / p.dh / p.nT) % nh_,
              lam = (int)(y / p.dh / p.nT / nh_);
    const i16 *s = x + ((size_t)lam * p.T + (size_t)I * p.B) * rs + (size_t)h * p.dh + c;
    S4 a = ga::i4zero();
    for (int l = 0; l < p.B; l++) s4mac(a, le[l], s[(size_t)l * rs]);
    st16(o + y, s4red(a));
  }
}
// o[(lam G + g) T + j] = sum_c ec[c] x[lam][j][g dh + c]. Launch: 256 threads, dynamic shared
// memory for dh SL. Eight threads share a sum (S4): dh int16 terms in all, within its capacity if
// dh <= 2^14.
// The idea is public: staging in shared memory: NVIDIA's Best Practices Guide, "Shared Memory in
// Matrix Multiplication (C=AB)" (cited at k_eqs()); Harris, "Optimizing CUDA" (at k_foldtile()).
// The compiler hints and the loop forms are public: __launch_bounds__: NVIDIA's Best Practices
// Guide (cited at k_eqs()); #pragma unroll: CUDA Programming Guide (cited at s4xor()); the
// grid-stride loop (cited at k_outers()); the block-stride loop (cited at k_planes()).
__global__ void __launch_bounds__(256) k_dotc(AP p, const i16 *__restrict__ x, const T *ec, T *o) {
  extern __shared__ __align__(16) char sm_[];
  SL *le = (SL *)sm_;
  for (int c = threadIdx.x; c < p.dh; c += 256) le[c] = ga::limbs_of(ec[c]);
  __syncthreads();
  const size_t n = (size_t)p.Lam * p.G * p.T * 8;
  for (size_t q = blockIdx.x * (size_t)256 + threadIdx.x; q < ((n + 255) / 256) * 256; q += (size_t)gridDim.x * 256) {
    const size_t y = q >> 3;
    const int u = q & 7;
    S4 a = ga::i4zero();
    if (q < n) {
      const int j = y % p.T, g = (int)(y / p.T) % p.G, lam = (int)(y / p.T / p.G);
      const i16 *s = x + ((size_t)lam * p.T + j) * p.dkv + (size_t)g * p.dh;
      for (int c = u; c < p.dh; c += 8) s4mac(a, le[c], s[c]);
    }
#pragma unroll
    for (int w = 1; w < 8; w <<= 1) s4xor(a, w);
    if (q < n && u == 0) st16(o + y, s4red(a));
  }
}
// Output y = (x G + g) T + i reads row r = (lam H + h) T + i of o, rem, Zo. Launch like k_dotc().
// With Z = Zo[r]: BE[y] = sum_c ec[c] o[r][c] and
// AC[y] = (sum_c ec[c] rem[r][c] - Z sum_c ec[c] + 2 Z BE[y]) / 2.
// AC[y] = sum_c ec[c] acc[r][c]; acc is the output before division by Z: 2 acc + Z = 2 Z o + rem.
// rem has up to 32 bits, so it is summed in two 16-bit halves to keep the 64-bit sums exact.
// Eight threads share a sum (S4) of dh terms, |v| < 2^16: requires dh <= 2^13.
// The idea is public: staging in shared memory: NVIDIA's Best Practices Guide, "Shared Memory in
// Matrix Multiplication (C=AB)" (cited at k_eqs()); Harris, "Optimizing CUDA" (at k_foldtile()).
// The compiler hints and the loop forms are public: __launch_bounds__: NVIDIA's Best Practices
// Guide (cited at k_eqs()); #pragma unroll: CUDA Programming Guide (cited at s4xor()); the
// grid-stride loop (cited at k_outers()); the block-stride loop (cited at k_planes()).
__global__ void __launch_bounds__(256) k_rowsE(AP p, const int *in, int cnt, const T *ec, const int *__restrict__ Zo,
                                               const i16 *__restrict__ o, const u32 *__restrict__ rem, T *BE, T *AC) {
  extern __shared__ __align__(16) char sm_[];
  SL *le = (SL *)sm_;
  __shared__ T sec;
  for (int c = threadIdx.x; c < p.dh; c += 256) le[c] = ga::limbs_of(ec[c]);
  if (threadIdx.x == 0) {
    T s = F::zero();
    for (int c = 0; c < p.dh; c++) s = F::add(s, ec[c]);
    sec = s;
  }
  __syncthreads();
  const size_t n = (size_t)cnt * p.G * p.T * 8;
  const T inv2 = F::emb(FIELD_HALF); // 1 / 2
  for (size_t q = blockIdx.x * (size_t)256 + threadIdx.x; q < ((n + 255) / 256) * 256; q += (size_t)gridDim.x * 256) {
    const size_t y = q >> 3;
    const int u = q & 7;
    S4 a = ga::i4zero(), b = a, b2 = a;
    size_t r = 0;
    if (q < n) {
      const int i = y % p.T, g = (int)(y / p.T) % p.G, x = (int)(y / p.T / p.G), lam = in[x] / p.HA,
                h = g * p.HA + in[x] % p.HA;
      r = ((size_t)lam * p.H + h) * p.T + i;
      const i16 *so = o + r * p.dh;
      const u32 *sr = rem + r * p.dh;
      for (int c = u; c < p.dh; c += 8) {
        s4mac(a, le[c], so[c]);
        s4mac(b, le[c], (int)(sr[c] & 0xFFFF));
        s4mac(b2, le[c], (int)(sr[c] >> 16));
      }
    }
#pragma unroll
    for (int w = 1; w < 8; w <<= 1) {
      s4xor(a, w);
      s4xor(b, w);
      s4xor(b2, w);
    }
    if (q < n && u == 0) {
      const T be = s4red(a), re = F::add(s4red(b), F::mulb(s4red(b2), 65536));
      const u64 Z = F::fromi(Zo[r]);
      const T zb = F::mulb(be, Z);
      st16(BE + y, be);
      st16(AC + y, F::mul(inv2, F::add(F::sub(re, F::mulb(sec, Z)), F::add(zb, zb))));
    }
  }
}
// AV[((bi G + g) nsl + sl) B + jl] = sum_il eil[il] e(il, jl). e = e0 + 128 + 256 e1 is the softmax
// weight at (il, jl) in slot sl's tile for head g of piece in[bi]; e0, e1: planes 3, 4 of `as`.
// Launch: 256 threads, shared memory for B F::SP and 256 elements. B divides 256 or 256 divides B.
// Not checked: e1 >= 0 and B e <= 2^32 (the capacity of F::SAcc).
// Public source: CUTLASS's documentation ("CUTLASS Convolution", "Implicit GEMM Algorithm")
// describes computing a matrix operand from the stored tensor while loading it, so that no second
// array holds the operand; here the stored form is the committed byte planes.
// The idea is public: staging in shared memory: NVIDIA's Best Practices Guide, "Shared Memory in
// Matrix Multiplication (C=AB)" (cited at k_eqs()); Harris, "Optimizing CUDA" (at k_foldtile()).
// The compiler hint is public: __launch_bounds__: NVIDIA's Best Practices Guide (cited at k_eqs()).
__global__ void __launch_bounds__(256)
    k_arrV(AP p, int dg, const int *in, int cnt, const T *eil, const i8 *__restrict__ as, T *AV) {
  extern __shared__ __align__(16) char sm_[];
  F::SP *le = (F::SP *)sm_;
  T *red = (T *)(le + p.B);
  const int nsl = dg ? p.nT : p.hP * p.nT, B = p.B, n0 = dg ? p.nD : p.nM, tid = threadIdx.x;
  for (int l = tid; l < B; l += 256) le[l] = F::sprep(eil[l]);
  __syncthreads();
  const size_t pstr = (size_t)p.Lam * p.HA << n0, b0 = dg ? ((size_t)(5 * p.Lam * p.HA) << p.nM) : 0;
  const int ng = B >= 256 ? 1 : 256 / B;
  for (size_t tl = blockIdx.x; tl < (size_t)cnt * p.G * nsl; tl += gridDim.x) {
    const int bi = (int)(tl / ((size_t)p.G * nsl));
    const size_t gs = tl % ((size_t)p.G * nsl);
    const i8 *e0 = as + b0 + 3 * pstr + ((size_t)in[bi] << n0) + gs * B * B, *e1 = e0 + pstr;
    for (int j0 = 0; j0 < B; j0 += 256) {
      const int jl = j0 + tid % (B < 256 ? B : 256), ig = B < 256 ? tid / B : 0;
      F::SAcc a = F::szero();
      for (int il = ig; il < B; il += ng) {
        const u32 e = (u32)((int)e0[(size_t)il * B + jl] + 128 + 256 * (int)e1[(size_t)il * B + jl]);
        F::smac(a, le[il], e);
      }
      red[tid] = F::sred(a);
      __syncthreads();
      if (ig == 0) {
        T s = red[tid];
        for (int w = 1; w < ng; w++) s = F::add(s, red[tid + w * B]);
        st16(AV + tl * B + jl, s);
      }
      __syncthreads();
    }
  }
}
// Operand sources of SC3, the sumcheck of sum_b sum_x W_b(x) A_b(x) B_b(x) (see engines.cuh).
struct Src4 {
  static constexpr bool INT = false, ROW = false;
  AP p;
  int dg, lsl;
  const int *in;
  const T *qf, *kf; // from `k_foldtile`; x = (g | slot | c), high bits first
  __device__ __forceinline__ T A(int b, size_t x) const {
    const int c = (int)x & (p.dh - 1), sl = (int)(x >> p.ldh) & ((1 << lsl) - 1), g = (int)(x >> (p.ldh + lsl));
    int I, J;
    if (!slotIJ(p, dg, sl, I, J)) return F::zero();
    const int n = in[b];
    return ld16(qf + (((size_t)(n / p.HA) * p.H + g * p.HA + n % p.HA) * p.nT + I) * p.dh + c);
  }
  __device__ __forceinline__ T B(int b, size_t x) const {
    const int c = (int)x & (p.dh - 1), sl = (int)(x >> p.ldh) & ((1 << lsl) - 1), g = (int)(x >> (p.ldh + lsl));
    int I, J;
    if (!slotIJ(p, dg, sl, I, J)) return F::zero();
    return ld16(kf + (((size_t)(in[b] / p.HA) * p.G + g) * p.nT + J) * p.dh + c);
  }
};
struct SrcE {
  static constexpr bool INT = false, ROW = false;
  AP p;
  const int *in;
  const int *Zo;
  const T *BE; // from `k_rowsE`; x = (g | i), high bits first
  __device__ __forceinline__ T A(int b, size_t x) const {
    const int i = (int)x & (p.T - 1), g = (int)(x >> p.lt), n = in[b];
    return F::emb(F::fromi(Zo[((size_t)(n / p.HA) * p.H + g * p.HA + n % p.HA) * p.T + i]));
  }
  __device__ __forceinline__ T B(int b, size_t x) const { return ld16(BE + ((size_t)b << (p.lgk + p.lt)) + x); }
};
struct SrcV {
  static constexpr bool INT = false, ROW = false;
  AP p;
  int dg, lsl;
  const int *in;
  const T *AV, *n2b, *gam; // x = (g | slot | jl), high bits first; B = gam + n2b; empty slot: gam
  __device__ __forceinline__ T A(int b, size_t x) const { return ld16(AV + ((size_t)b << (p.lgk + lsl + p.lB)) + x); }
  __device__ __forceinline__ T B(int b, size_t x) const {
    const int jl = (int)x & (p.B - 1), sl = (int)(x >> p.lB) & ((1 << lsl) - 1), g = (int)(x >> (p.lB + lsl));
    int I, J;
    const T gm = *gam;
    if (!slotIJ(p, dg, sl, I, J)) return gm;
    return F::add(gm, ld16(n2b + ((size_t)(in[b] / p.HA) * p.G + g) * p.T + J * p.B + jl));
  }
};
struct SrcF {
  static constexpr bool INT = true, ROW = true;
  AP p;
  const i16 *q, *k;
  // x = (i | g | c), high bits first; xt = (i | g). A row: the dh values of one xt. b = lam HA + a.
  const int *io;
  __device__ __forceinline__ const i16 *Ar(int b, size_t xt) const {
    const int g = (int)xt & (p.G - 1), i = (int)(xt >> p.lgk);
    return q + ((size_t)(b / p.HA) * p.T + i) * p.dq + (size_t)(g * p.HA + b % p.HA) * p.dh;
  }
  __device__ __forceinline__ const i16 *Br(int b, size_t xt) const {
    const int g = (int)xt & (p.G - 1), i = (int)(xt >> p.lgk), lam = b / p.HA;
    const int row = io[((size_t)lam * p.H + g * p.HA + b % p.HA) * p.T + i];
    return k + ((size_t)lam * p.T + index_below(row, p.T, INDEX_ARG_MAX)) * p.dkv + (size_t)g * p.dh;
  }
  __device__ __forceinline__ int Ai(int b, size_t x) const {
    const int c = (int)x & (p.dh - 1), g = (int)(x >> p.ldh) & (p.G - 1), i = (int)(x >> (p.ldh + p.lgk));
    return q[((size_t)(b / p.HA) * p.T + i) * p.dq + (size_t)(g * p.HA + b % p.HA) * p.dh + c];
  }
  __device__ __forceinline__ int Bi(int b, size_t x) const {
    const int c = (int)x & (p.dh - 1), g = (int)(x >> p.ldh) & (p.G - 1), i = (int)(x >> (p.ldh + p.lgk)),
              lam = b / p.HA;
    const int row = io[((size_t)lam * p.H + g * p.HA + b % p.HA) * p.T + i];
    return k[((size_t)lam * p.T + index_below(row, p.T, INDEX_ARG_MAX)) * p.dkv + (size_t)g * p.dh + c];
  }
};
struct SrcH {
  static constexpr bool INT = true, ROW = false;
  AP p;
  // x = (i | g | kappa), high bits first, kappa of 4 bits. Ai: bit kappa of io, 0 if kappa >= lt.
  const int *io;
  __device__ __forceinline__ int Ai(int b, size_t x) const {
    const int kp = (int)x & 15, g = (int)(x >> 4) & (p.G - 1), i = (int)(x >> (4 + p.lgk));
    return kp < p.lt ? (io[((size_t)(b / p.HA) * p.H + g * p.HA + b % p.HA) * p.T + i] >> kp) & 1 : 0;
  }
  __device__ __forceinline__ int Bi(int b, size_t x) const { return Ai(b, x) - 1; }
};
} // namespace gc
