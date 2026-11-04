// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* `Walk` records the linear claims of one sub-proof (`fork` in identifiers), the argument for one
   part of the forward pass. A linear claim states that an affine form (AF) in committed values and
   mask entries (random elements of the committed mask row) has a public value. A form is a sum of
   range terms (Tm): scalar * vector[i] * variable[x0 + i * xs] over i. Set-up turns the terms into
   jobs (HJob); in each proof, kernels fill the scalars and vectors, and `finish` batches the claims
   into one equation. Only its coefficients are computed here, its public value by the verifier.
   Variables are grouped in spaces (Space). Variable j of space 0 is the hidden value of message
   slot j, element j of the sub-proof's message array: (what was sent in slot j) - factor * (mask
   entry j). The factor is 1 for a value (`slot`) sent outside the masked sumchecks (PSC; `psc` in
   identifiers) and beta, a challenge of PSC, for a sumcheck's end value (`Walk::shown`). */
#pragma once
#include "common/element_from_hash.h"
#include "hiding/slot_count.cuh"
#include <array>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
namespace hw { // a sub-proof's linear claims: `Walk`, its parts and the functions that add claims
// T (= Ext, an extension-field element) and F, its arithmetic, are typedefs of transcript.cuh.
// HE: also Ext, for values computed on the host. p: the base field's prime, FIELD_P in the code.
typedef Ext HE;
static inline HE hE(u64 a) {
  return ext_base(a % FIELD_P);
}
static inline HE hadd(HE a, HE b) {
  return F::add(a, b);
}
static inline HE hneg(HE a) {
  return F::sub(F::zero(), a);
}
static inline HE hmul(HE a, HE b) {
  return F::mul(a, b);
}
// 2^k, not reduced: it must be below p.
static inline HE h2(int k) {
  return ext_base(1ull << k);
}
static inline bool hz0(HE a) {
  return F::eq(a, F::zero());
}
// Symbolic scalar: the host constant c times the values at p[0 .. n), at most four. k_scat reads
// these from GPU memory when it runs: a proof's kernels write them before that.
// Public source: scalars given by device pointers: NVIDIA, cuBLAS documentation, section 2.1.5
// "Scalar Parameters", https://docs.nvidia.com/cuda/cublas/index.html.
// Public source: such scalars are read at execution: NVIDIA, cuBLAS documentation, section 2.1.12
// "CUDA Graphs Support". Here what is built once is a table of records, not a CUDA graph.
struct Sy {
  HE c = ext_base(1);
  const T *p[4] = {nullptr, nullptr, nullptr, nullptr};
  int n = 0;
  Sy() {}
  Sy(HE c_) : c(c_) {}
  Sy(const T *q) {
    p[0] = q;
    n = 1;
  }
};
static inline Sy operator*(Sy a, const Sy &b) {
  a.c = hmul(a.c, b.c);
  for (int i = 0; i < b.n; i++) {
    if (a.n >= 4) {
      printf("internal error: walk: a scalar has more than four factors\n");
      exit(2);
    }
    a.p[a.n++] = b.p[i];
  }
  return a;
}
// Public sources: runs of equally spaced entries, one record per run: the CSX sparse-matrix format,
// as described in the diploma thesis on the SparseX library, National Technical University of
// Athens, 2014, section 2.2, core.ac.uk/download/38465372.pdf; two operands given by start and
// stride, with one factor: the BLAS routine axpy, cuBLAS documentation, cublas<t>axpy. Here the
// runs are written directly and not detected in a matrix.
// A null vec means that every vec[i * vs] is 1.
struct Tm { // one range term: the sum over i < n of w x vec[i vs] x variable x0 + i xs of space sp
  int sp = 0;
  size_t x0 = 0;
  long long xs = 1;
  u32 n = 1;
  Sy w;
  const T *vec = nullptr;
  long long vs = 1;
};
/* Affine form: a list of range terms plus a constant c. cbad marks a constant that was scaled by a
   device value, which the host cannot evaluate. `Walk::emit` reads the terms only: the operators
   carry c and cbad along, and nothing else in this file reads them. */
// Public sources: affine forms built with overloaded operators: Gurobi Optimization, Gurobi
// Optimizer Reference Manual, version 11.0, C++ API, page "GRBLinExpr",
// https://docs.gurobi.com/projects/optimizer/en/11.0/reference/cpp/linexpr.html; JuMP developers,
// JuMP.jl manual (stable), page "Expressions", section "Affine expressions",
// https://jump.dev/JuMP.jl/stable/manual/expressions/. Here the variables are values of the proof
// and not variables of an optimization model.
struct AF {
  std::vector<Tm> t;
  HE c = ext_base(0);
  bool cbad = false;
  AF() {}
  explicit AF(HE k) : c(k) {}
};
static inline AF operator+(AF a, const AF &b) {
  a.t.insert(a.t.end(), b.t.begin(), b.t.end());
  a.c = hadd(a.c, b.c);
  a.cbad |= b.cbad;
  return a;
}
static inline AF operator*(AF a, const Sy &w) {
  for (auto &x : a.t) x.w = x.w * w;
  if (w.n && !hz0(a.c)) a.cbad = true;
  a.c = hmul(a.c, w.c);
  return a;
}
static inline AF operator-(AF a, const AF &b) {
  return a + b * Sy(hneg(ext_base(1)));
}
// The form of one range term. n must fit 32 bits, which is not checked.
static inline AF rng(int sp, size_t x0, long long xs, size_t n, const T *vec, long long vs = 1, Sy w = Sy()) {
  AF a;
  Tm t;
  t.sp = sp;
  t.x0 = x0;
  t.xs = xs;
  t.n = (u32)n;
  t.vec = vec;
  t.vs = vs;
  t.w = w;
  a.t.push_back(t);
  return a;
}
static inline AF var(int sp, size_t x, Sy w = Sy()) {
  return rng(sp, x, 1, 1, nullptr, 1, w);
}
static inline AF slot(size_t j, Sy w = Sy()) {
  return var(0, j, w);
}
// A range term, or a part of one, as k_scat reads it. ss: the step in lam (Tm::xs).
// Public sources: see Tm (the run), Sy (the factor c, p, np) and Walk::seal (alone).
struct HJob {
  T *lam = nullptr;
  const T *vec = nullptr;
  long long vs = 1;
  long long ss = 1;
  u32 n = 0, claim = 0;
  int neg = 0, np = 0;
  int alone = 0; // set by `Walk::seal`: no other job it compares adds into this job's range of lam
  T c = ext_base(1);
  const T *p[4] = {nullptr, nullptr, nullptr, nullptr};
};
static const u32 CHN = 1024;
/* Returns entry j of a row of extension-field elements stored as EDEG planes of u64 coordinates,
   first coordinate first: coordinate k of entry j is plane0[j + k * planes]. hz::Tree (trees.cuh)
   stores its mask and pad rows this way (`h_plane`, `plane_stride`). */
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// c, a thread's own short array, where unrolling fixes the index of each entry: M. Milakov, "Fast
// Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
__device__ __forceinline__ T at_planes(const u64 *plane0, long long planes, long long j) {
  u64 c[EDEG];
#pragma unroll
  for (int k = 0; k < EDEG; k++) c[k] = plane0[j + k * planes];
  return ext_from(c);
}
// Atomic addition modulo p = 2^64 - 2^32 + 1 (FIELD_P).
// Public sources for the techniques of this routine.
// An atomic operation made with compare-and-swap. The idea is public: a loop that repeats
// atomicCAS() until no other write has come between: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", cited at at_planes, "Legacy Atomic Functions".
__device__ __forceinline__ void atomic_add_mod_p(u64 *p, u64 v) {
  unsigned long long old = *p, as;
  do {
    as = old;
    old = atomicCAS((unsigned long long *)p, as, (unsigned long long)gf_add((u64)as, v));
  } while (old != as);
}
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", cited at
// at_planes.
__device__ __forceinline__ void atomAddE(T *p, T v) {
#pragma unroll
  for (int k = 0; k < EDEG; k++)
    if (coord(v, k)) atomic_add_mod_p((u64 *)p + k, coord(v, k));
}
/* One warp per chunk: CH[w] = (job index, first term) covers at most CHN terms i of its job. For
   each term i it adds k * vec[i*vs] (k if vec is null) to lam[i*ss]; a job without lam is skipped.
   k = kap[claim] * c * *p[0] * .. * *p[np - 1], negated if neg is set. Launch with at least
   nch / 8 blocks of 256 threads. A job with `alone` set (see HJob) adds without atomics. */
// Public source: many problems in one launch, from a list made on the host: NVIDIA, CUTLASS
// documentation, "Grouped Kernel Schedulers", section "Precomputing schedules on the host:
// GroupScheduleMode::kHostPrecompute",
// https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html.
// Public source: long items cut into chunks of bounded size: Zihao Ye, Lequn Chen, Ruihang Lai,
// Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind
// Krishnamurthy, Luis Ceze, "FlashInfer: Efficient and Customizable Attention Engine for LLM
// Inference Serving", 2025, section 3.3.1, Algorithm 1, step 4, https://arxiv.org/abs/2501.01005.
// Public source: a warp per unit of work: Muhammad Osama, Serban D. Porumbescu, John D. Owens, "A
// Programming Model for GPU Load Balancing", PPoPP 2023, section 5.2.2,
// https://arxiv.org/abs/2301.04792.
// Here the items are scattered additions, not CUTLASS's grouped matrix products or FlashInfer's
// attention, and the bound is a constant.
// Public sources for the choice between plain and atomic additions: see Walk::seal.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(256)
    k_scat(const HJob *__restrict__ J, const uint2 *__restrict__ CH, u32 nch, const T *kap) {
  const u32 w = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (w >= nch) return;
  const uint2 c = CH[w];
  const HJob j = J[c.x];
  if (!j.lam) return;
  const u32 i1 = min(j.n, c.y + CHN);
  T k = F::mul(j.c, kap[j.claim]);
  for (int q = 0; q < j.np; q++) k = F::mul(k, *j.p[q]);
  if (j.neg) k = F::sub(F::zero(), k);
  const typename F::P kp = F::prep(k);
  if (j.alone)
    for (u32 i = c.y + lane; i < i1; i += 32) {
      T *at = j.lam + (long long)i * j.ss;
      *at = F::add(*at, j.vec ? F::mulp(kp, j.vec[(long long)i * j.vs]) : k);
    }
  else
    for (u32 i = c.y + lane; i < i1; i += 32)
      atomAddE(j.lam + (long long)i * j.ss, j.vec ? F::mulp(kp, j.vec[(long long)i * j.vs]) : k);
}
// kap[c] = theta^c for c < n: the coefficient of claim c in the batched equation. Claims are
// numbered from 0 in the order in which they were added.
// Public source: A. Menezes, P. van Oorschot, S. Vanstone, Handbook of Applied Cryptography, CRC
// Press, 1996, chapter 14, section 14.6.1, Algorithm 14.76 "Right-to-left binary exponentiation",
// https://cacr.uwaterloo.ca/hac/about/chap14.pdf, run by each thread on its own exponent c.
__global__ void k_batch_powers(const T *theta, u32 n, T *kap) {
  const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  T power = F::one(), square = *theta;
  for (u32 e = c; e; e >>= 1, square = F::mul(square, square))
    if (e & 1) power = F::mul(power, square);
  kap[c] = power;
}
// Draws theta from the transcript Tr (transcript.cuh). Launch: one thread.
__global__ void k_batch_draw(Tr *tr, T *theta) {
  Tr t = *tr;
  tr_point(t, Round::BAT_SIGMA, theta, 1);
  *tr = t;
}
// One masked sumcheck, as psc_degrees() registers it (in_y: its degree_in_y; eps, beta: see Rec).
// k_psc_rows fills `out` for Walk::row(). out[0 .. nw): the value at the challenges of the hiding
// polynomial (the random polynomial added to the summand), as a linear form in its nw mask entries.
// out[nw] = eps P; P is 1 under equality weights, else the product of the challenges *r[j]. If tau,
// the challenge after a product layer's end values, is set, out[nw + 1] = 1 - tau.
constexpr int PSC_ROUNDS = psc::MAX_ROUNDS;
struct PscDev {
  int nr = 0, form = 0, in_y = 0, nw = 0;
  short degree[PSC_ROUNDS] = {};
  const T *r[PSC_ROUNDS] = {}, *eps = nullptr, *beta = nullptr, *tau = nullptr;
  T *out = nullptr;
};
// One thread for each record.
__global__ void k_psc_rows(const PscDev *R, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const PscDev &d = R[i];
  d.out[d.nw] = psc::row_form<F>((psc::Form)d.form, d.nr, [&d](int j) { return *d.r[j]; }, [&d](int j) { return (int)d.degree[j]; },
                                 d.in_y, *d.eps, d.in_y ? *d.beta : F::zero(), d.out);
  if (d.tau) d.out[d.nw + 1] = F::sub(F::one(), *d.tau);
}
// o[i] = in[i] / eps[0] for i < n. Launch: one block.
__global__ void k_psc_over_eps(const T *in, int n, const T *eps, T *o) {
  const T e = F::inv(eps[0]);
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = F::mul(in[i], e);
}
struct EqD {
  const T *z = nullptr;
  int n = 0;
  T *o = nullptr;
};
// For each table d of D: o[x] = product over i < n of (bit n - 1 - i of x ? z[i] : 1 - z[i]), for
// x < 2^n. An entry map[b] = (table, y) denotes the entries 256 y .. 256 y + 255 of that table.
// Launch with one block of 256 threads per entry of map.
// Public sources for the techniques of this routine.
// Warp reduction by shuffles. The idea is public: J. Luitjens, "Faster Parallel Reductions on
// Kepler", NVIDIA Technical Blog, 2014, "Shuffle On Down" and "Shuffle Warp Reduce",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
__global__ void __launch_bounds__(256) k_eqB(const EqD *D, const uint2 *map) {
  __shared__ T high, upper[16], lower[16];
  const uint2 m = map[blockIdx.x];
  const EqD d = D[m.x];
  const int t = threadIdx.x, n = d.n, nl = min(n, 8), n0 = min(nl, 4), n1 = nl - n0;
  auto pick = [&](int i, unsigned bit) {
    const T r = d.z[i];
    return bit ? r : F::sub(F::one(), r);
  };
  if (t < 32) { // lower[b]: the factor of the lowest n0 bits; upper[b]: that of the n1 bits above them
    const int b = t & 15, nb = t < 16 ? n0 : n1, i0 = t < 16 ? n - n0 : n - nl;
    if (b < (1 << nb)) {
      T v = F::one();
      for (int q = 0; q < nb; q++) v = F::mul(v, pick(i0 + q, (b >> (nb - 1 - q)) & 1));
      (t < 16 ? lower : upper)[b] = v;
    }
  } else if (t < 64) {
    T v = F::one();
    for (int i = t - 32; i < n - nl; i += 32) v = F::mul(v, pick(i, (unsigned)(m.y >> (n - 1 - i - 8)) & 1));
    for (int o = 16; o; o >>= 1) v = F::mul(v, shfl_down<F>(v, o));
    if (t == 32) high = v;
  }
  __syncthreads();
  const size_t x = (size_t)m.y * 256 + t;
  if (x >> n) return;
  d.o[x] = F::mul(F::mul(high, upper[(t >> n0) & 15]), lower[t & ((1 << n0) - 1)]);
}
// k_gaB sets o[x] = src[idx[x]] for x < n. map and launch: as for k_eqB.
struct GaD {
  const T *src = nullptr;
  const u32 *idx = nullptr;
  u32 n = 0;
  T *o = nullptr;
};
__global__ void k_gaB(const GaD *D, const uint2 *map) {
  const uint2 m = map[blockIdx.x];
  const GaD d = D[m.x];
  const u32 x = m.y * 256 + threadIdx.x;
  if (x < d.n) d.o[x] = d.src[d.idx[x]];
}
// One entry of the array that k_atB reads. k_atB sets *o to the product over q < nb of z[q] where
// bit nb - 1 - q of idx is set, else 1 - z[q].
struct AtD {
  const T *z = nullptr;
  int nb = 0;
  u64 idx = 0;
  T *o = nullptr;
};
__global__ void k_atB(const AtD *D, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const AtD d = D[i];
  T v = F::one();
  for (int q = 0; q < d.nb; q++) {
    const T r = d.z[q];
    v = F::mul(v, ((d.idx >> (d.nb - 1 - q)) & 1) ? r : F::sub(F::one(), r));
  }
  *d.o = v;
}
// o[i] = (*g)^i for i < n. Launch: one thread.
__global__ void k_pows(const T *g, int n, T *o) {
  T v = F::one();
  for (int i = 0; i < n; i++) {
    o[i] = v;
    v = F::mul(v, *g);
  }
}
// o[x] = er[(x >> kap) & (2^io - 1)] * 2^(x & (2^kap - 1)) for x < n. The power of 2 is not
// reduced, so it must be below p.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_wcol(const T *er, int io, int kap, size_t n, T *o) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < n; x += (size_t)gridDim.x * blockDim.x)
    o[x] = F::mul(er[(x >> kap) & (((size_t)1 << io) - 1)], ext_base(1ull << (x & ((1 << kap) - 1))));
}
// lam: where k_scat adds up the batched equation's coefficients, one for each variable; a space
// without lam gets none. neg: k_scat negates what it adds.
// A virtual row of a committed array X is the vector sum_r cv[r] X[r][.], for coefficients cv.
struct Space { // an array of variables: slots, a committed vector, a virtual row or matrix columns
  // Public sources: Apache TVM documentation, 'InferBound Pass', section 'Phase 4: Consolidate across all consumers',
  // https://tvm.apache.org/docs/arch/inferbound.html; J. Demmel, M. Hoemmen, Y. Hida, J. Riedy, 'Non-Negative Diagonals
  // and High Performance on Low-Profile Matrices from Householder QR', LAPACK Working Note 203, 2008, abstract,
  // https://www.netlib.org/lapack/lawnspdf/lawn203.pdf.
  // TVM: a range set to the union of its consumers' requests. The Working Note: a vector's trailing
  // zeros skipped. Here later passes skip what is outside [lo, hi), where lam is zero.
  size_t lo = (size_t)-1, hi = 0; // the range [lo, hi) of variables that any term reads
  // the terms' ranges, recorded only when lam is set; segs() merges those that overlap or touch
  std::vector<std::pair<size_t, size_t>> rg;
  // Public source: ranges sorted and merged: H. Pages, M. Lawrence, P. Aboyoun, IRanges
  // (Bioconductor package), manual page 'inter-range-methods: Inter range transformations', section
  // Details, function reduce, https://rdrr.io/pkg/IRanges/man/inter-range-methods.html. What
  // differs here: nothing in the merging itself.
  std::vector<std::pair<size_t, size_t>> segs() const {
    auto v = rg;
    std::sort(v.begin(), v.end());
    std::vector<std::pair<size_t, size_t>> o;
    for (auto &r : v) {
      if (!o.empty() && r.first <= o.back().second)
        o.back().second = std::max(o.back().second, r.second);
      else
        o.push_back(r);
    }
    return o;
  }
  T *lam = nullptr;
  // If Kw is not 0, lam has rows of Kw entries, lamPitch elements apart. len: number of variables.
  size_t lamPitch = 0;
  size_t Kw = 0, len = 0;
  int neg = 0;
};
// A masked sumcheck (PSC, defined in blocks/iop/psc.h), as psc() and psc_degrees() return it.
// eps, beta: the addresses of two of its challenges. It runs on (hiding polynomial) + eps *
// (summand), and end value j is sent as (value) + beta * (inner mask j). tau: see PscDev.
struct Rec {
  int id = -1;
  int nr = 0;          // its rounds
  size_t slot0 = 0;    // the number of its first slot and first mask entry
  size_t nw = 0;       // the entries of its hiding polynomial: the numbers slot0 .. slot0 + nw - 1
  size_t shown = 0;    // its end values: the slots, and inner masks, slot0 + nw .. slot0 + nw + shown - 1
  T *out = nullptr;    // PscDev::out
  const T *eps = nullptr, *beta = nullptr, *tau = nullptr;
};
// Order of calls. Set-up: init(); fill sp (lam, lamPitch, Kw, len and neg are the caller's to set)
// and scratch; psc(), lin(), row(), layer(), eq(), gather(), eqAt() as needed; seal(). In each
// proof: zero the spaces' lam, then finish(). A failed check of this file exits with code 2, a
// failed CUDA call under CK (prover/common/common.h) with code 1; no kernel launch is checked.
struct Walk {
  std::vector<Space> sp;
  std::vector<HJob> jobs;
  u32 nclaim = 0;
  std::vector<PscDev> recs;
  // The message array's scratch plane (psc_device.cuh): element j is slot j's. Set it before psc().
  const T *scratch = nullptr;
  std::vector<EqD> eqs;
  std::vector<GaD> gas;
  std::vector<AtD> ats;
  std::vector<std::function<void()>> prep;
  size_t poolN = 0, poolAt = 0;
  T *pool = nullptr;
  HJob *dJ = nullptr;
  uint2 *dCH = nullptr, *dEqM = nullptr, *dGaM = nullptr;
  u32 nch = 0, nEqB = 0, nGaB = 0;
  PscDev *dRec = nullptr;
  EqD *dEq = nullptr;
  GaD *dGa = nullptr;
  AtD *dAt = nullptr;
  T *kap = nullptr;
  T *theta = nullptr; // the batching challenge; kap[c] = theta^c
  void init(size_t poolElems) {
    poolN = poolElems;
    hz::alloc(pool, poolN * sizeof(T), hz::FORK);
    CK(cudaMemset(pool, 0, poolN * sizeof(T)));
  }
  T *take(size_t n) {
    if (poolAt + n > poolN) {
      printf("walk: the pool of %zu elements is exhausted\n", poolN);
      exit(2);
    }
    T *p = pool + poolAt;
    poolAt += n;
    return p;
  }
  const T *cvec(const std::vector<HE> &v) {
    T *p = take(v.size());
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
  // eq(), gather() and eqAt() register a vector that run_prep() fills in each proof (see k_eqB,
  // GaD, AtD) and return its address.
  const T *eq(const T *z, int n) {
    EqD d;
    d.z = z;
    d.n = n;
    d.o = take((size_t)1 << n);
    eqs.push_back(d);
    return d.o;
  }
  const T *gather(const T *src, const std::vector<u32> &idx) {
    u32 *di;
    hz::alloc(di, idx.size() * 4 + 4, hz::SHAPE);
    CK(cudaMemcpy(di, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    GaD d;
    d.src = src;
    d.idx = di;
    d.n = (u32)idx.size();
    d.o = take(idx.size());
    gas.push_back(d);
    return d.o;
  }
  const T *eqAt(const T *z, int nb, u64 idx) {
    AtD d;
    d.z = z;
    d.nb = nb;
    d.idx = idx;
    d.o = take(1);
    ats.push_back(d);
    return d.o;
  }
  // Registers a sumcheck. r: the address of each round's challenge, in order. degrees: of the
  // round polynomials; under EQ2, SUM2 and SUM3 they must be the form's own, 2, 2 and 3, which is
  // not checked. form: how the variables are summed. slot0: its first slot. shown: how many maps'
  // end values are sent. degree_in_y: the summand's degree in those maps; 0 if and only if shown
  // is 0. tau: see PscDev, or null.
  Rec psc_degrees(psc::Form form, const std::vector<int> &degrees, const std::vector<const T *> &r, size_t slot0,
                  int degree_in_y, size_t shown, const T *tau) {
    if (r.size() > (size_t)PSC_ROUNDS || degrees.size() != r.size() || !scratch || (degree_in_y == 0) != (shown == 0)) {
      printf("internal error: walk: a sum-check of %zu rounds, %zu degrees, degree %d in the shown maps, %zu shown%s\n", r.size(),
             degrees.size(), degree_in_y, shown, scratch ? "" : "; the scratch plane is not set");
      exit(2);
    }
    PscDev d;
    d.nr = (int)r.size(), d.form = (int)form, d.in_y = degree_in_y, d.nw = 1 + degree_in_y;
    for (int j = 0; j < d.nr; j++) d.r[j] = r[j], d.degree[j] = (short)degrees[j], d.nw += degrees[j];
    d.eps = scratch + slot0;
    d.beta = degree_in_y ? scratch + slot0 + (d.nw - degree_in_y) : nullptr; // the first slot of the round in the further variable
    d.tau = tau;
    d.out = take((size_t)d.nw + 2);
    recs.push_back(d);
    Rec o;
    o.id = (int)recs.size() - 1, o.nr = d.nr, o.slot0 = slot0, o.nw = (size_t)d.nw, o.shown = shown, o.out = d.out;
    o.eps = d.eps, o.beta = d.beta, o.tau = tau;
    return o;
  }
  // kind 0: equality weights, degree 2; 1: plain sums, degree 2; 2: plain sums, degree 3. Another
  // kind is not rejected.
  Rec psc(int kind, const std::vector<const T *> &r, size_t slot0, int degree_in_y, size_t shown, const T *tau) {
    const psc::Form form = kind == 0 ? psc::Form::EQ2 : kind == 1 ? psc::Form::SUM2 : psc::Form::SUM3;
    return psc_degrees(form, std::vector<int>(r.size(), kind == 2 ? 3 : 2), r, slot0, degree_in_y, shown, tau);
  }
  Rec psc_degrees(const std::vector<int> &degrees, const std::vector<const T *> &r, size_t slot0, int degree_in_y, size_t shown,
                  const T *tau) {
    return psc_degrees(psc::Form::NODES, degrees, r, slot0, degree_in_y, shown, tau);
  }
  // Not checked: that sp has a space t.sp.
  // Public sources: one table of records in GPU memory, which a kernel reads whole in one launch,
  // in the manner of an instruction tensor and on-GPU interpreter: B. Spector, A. Singhal, D. Fu,
  // C. Re, "ThunderMLA: FlashMLA, Faster and Fused-er!", Hazy Research blog, Stanford, 4 March
  // 2025, https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla; B. Spector, J. Juravsky,
  // S. Sul, O. Dugan, D. Lim, D. Fu, S. Arora, C. Re, "Look Ma, No Bubbles! Designing a Low-Latency
  // Megakernel for Llama-1B", Hazy Research blog, Stanford, 27 May 2025,
  // https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles. Here all records have one form.
  void emit(const AF &f, u32 claim) {
    for (auto &t : f.t) {
      Space &S = sp[t.sp];
      // The range that encloses all terms: see Space::lo, hi. Needs xs >= 0, and xs > 0 where Kw is
      // set (a division below); neither is checked.
      S.lo = std::min(S.lo, t.x0);
      S.hi = std::max(S.hi, (size_t)(t.x0 + (size_t)(t.n - 1) * (size_t)t.xs + 1));
      if (S.lam) S.rg.push_back({t.x0, (size_t)(t.x0 + (size_t)(t.n - 1) * (size_t)t.xs + 1)});
      if (t.x0 + (size_t)(t.n - 1) * t.xs >= S.len) {
        printf("internal error: walk: a term reads past the end of space %d (start %zu, count "
               "%u, step %lld, space length %zu)\n",
               t.sp, t.x0, t.n, t.xs, S.len);
        exit(2);
      }
      u32 i = 0;
      while (i < t.n) {
        const size_t x = t.x0 + (size_t)i * t.xs;
        u32 m = t.n - i;
        size_t lm = x;
        // Public source: S. Amarasinghe, W. Lee, B. Greenwald, 'Strength Reduction of Integer Division and Modulo
        // Operations', MIT-LCS-TM-600, November 1999, section 4.3.2, https://dspace.mit.edu/handle/1721.1/149291.
        // Amarasinghe, Lee and Greenwald (MIT-LCS-TM-600, 1999, section 4.3.2) break a loop whose index is divided by a
        // constant into two loops, so that the quotient changes only at the inner loop's boundaries; here the host cuts
        // a run of indices in the same way when it makes the jobs.
        if (S.Kw) {
          const size_t r = x / S.Kw, c = x % S.Kw;
          // Public source: row times pitch plus column: NVIDIA, CUDA Runtime API, module Memory
          // Management, cudaMallocPitch,
          // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html. The pitch
          // here counts elements, not bytes.
          lm = r * S.lamPitch + c;
          const size_t left = (S.Kw - 1 - c) / t.xs + 1;
          if (left < m) m = (u32)left;
        }
        HJob j;
        j.lam = S.lam ? S.lam + lm : nullptr;
        j.vec = t.vec ? t.vec + (long long)i * t.vs : nullptr;
        j.vs = t.vs;
        j.ss = t.xs;
        j.n = m;
        j.claim = claim;
        j.neg = S.neg;
        j.np = t.w.n;
        j.c = t.w.c;
        for (int q = 0; q < 4; q++) j.p[q] = t.w.p[q];
        jobs.push_back(j);
        i += m;
      }
    }
  }
  // Adds the claim that the form f is 0 (up to its constant, which the prover does not form).
  void lin(const AF &f) { emit(f, nclaim++); }
  // The hidden end value j of a sumcheck: (what was sent in its slot) - beta * (inner mask j).
  AF shown(const Rec &R, size_t j) const {
    if (j >= R.shown) {
      printf("internal error: walk: a sum-check shows %zu end values, number %zu is asked for\n", R.shown, j);
      exit(2);
    }
    return var(0, R.slot0 + R.nw + j, Sy(R.beta));
  }
  // Adds the claim that a sumcheck leaves:
  //   <out[0 .. nw), its hiding polynomial's entries> - eps P start + eps kept = public
  // (out, P: see PscDev). start: the form that the sum was claimed to equal. kept: sum_i A_i * (the
  // form of map i at the challenge point) over the maps whose end values are not sent, with the
  // caller's coefficients A_i.
  // Callers must set sp[0].neg, which is not checked: k_scat then negates what it adds in space 0,
  // since a hidden value has minus its mask entry. So the entries themselves get -1 here.
  void row(const Rec &R, const AF &start, const AF &kept = AF()) {
    lin(rng(0, R.slot0, 1, R.nw, R.out, 1, Sy(hneg(ext_base(1)))) - start * Sy(R.out + R.nw) + kept * Sy(R.eps));
  }
  // For a sumcheck whose kept terms are not forms in committed values, because the next sumcheck
  // takes them up: returns their value K as a form. By the equation at row(), up to the constant,
  // K = P start - <out[0 .. nw) / eps, the entries>. Adds no claim: K enters the next start form.
  AF kept_value(const Rec &R, const AF &start) {
    T *o = take(R.nw + 1);
    const T *in = R.out, *eps = R.eps;
    const int n = (int)R.nw + 1;
    prep.push_back([=] { k_psc_over_eps<<<1, 64>>>(in, n, eps, o); });
    return start * Sy(o + R.nw) - rng(0, R.slot0, 1, R.nw, o, 1, Sy(hneg(ext_base(1))));
  }
  // Adds a product layer's claim; returns the next start form (1 - tau) v_0 + tau v_1, where v_0
  // and v_1 are its hidden end values. Needs R.tau, which is not checked.
  AF layer(const Rec &R, const AF &Cstart) {
    row(R, Cstart);
    return shown(R, 0) * Sy(R.out + R.nw + 1) + shown(R, 1) * Sy(R.tau);
  }
  template <class X> static X *up(const std::vector<X> &v) {
    X *d = nullptr;
    hz::alloc(d, v.size() * sizeof(X) + 16, hz::OWN);
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return d;
  }
  // Ends set-up. No lin(), row(), psc(), eq(), gather() or eqAt() may follow it: not checked.
  // vectors_only: the walk has no claim and exists for the vectors that its prep jobs make
  // (gather(), eq(), take()). On such a walk call run_prep(), not finish().
  void seal(bool vectors_only = false) {
    // Public source: atomics only for duplicates: J. Zhang, J. Brown, S. Balay and others, 'The
    // PetscSF Scalable Communication Layer', arXiv 2102.13018, section 5.3,
    // https://arxiv.org/abs/2102.13018.
    // Public source: atomics only for rows that blocks share: M. Steinberger, R. Zayer, H.-P.
    // Seidel, 'Globally Homogeneous, Locally Adaptive Sparse Matrix-Vector Multiplication on the
    // GPU', ICS 2017, section 3.2, DOI 10.1145/3079079.3079086.
    // Here the host decides for each job before the launch.
    // seal() does not compare a job with n > 1 and ss = 0 (all its terms add to one entry): if that
    // entry were in the range of a job marked alone, the plain addition would race with the atomic
    // ones, and an addition could be lost.
    { // a job is alone if its range of lam overlaps no other's; jobs with n > 1, ss = 0 are skipped
      std::vector<std::array<const T *, 2>> stretch(jobs.size());
      std::vector<size_t> order;
      for (size_t j = 0; j < jobs.size(); j++) {
        HJob &J = jobs[j];
        J.alone = 0;
        if (!J.lam || !J.n || (J.n > 1 && J.ss == 0)) continue;
        const T *a = J.lam, *b = J.lam + (long long)(J.n - 1) * J.ss;
        stretch[j] = {std::min(a, b), std::max(a, b) + 1};
        order.push_back(j);
      }
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return stretch[a][0] < stretch[b][0]; });
      for (size_t i = 0; i < order.size();) { // go through the runs of overlapping ranges
        size_t k = i + 1;
        const T *end = stretch[order[i]][1];
        while (k < order.size() && stretch[order[k]][0] < end) end = std::max(end, stretch[order[k++]][1]);
        if (k == i + 1) jobs[order[i]].alone = 1;
        i = k;
      }
    }
    // Public sources: long jobs cut at set-up into chunks of bounded length: FlashInfer (cited at
    // k_scat()), section 3.3.1. Here the plan is made once.
    std::vector<uint2> ch;
    for (size_t j = 0; j < jobs.size(); j++)
      for (u32 i = 0; i < jobs[j].n; i += CHN) ch.push_back(make_uint2((u32)j, i));
    nch = (u32)ch.size();
    dJ = up(jobs);
    dCH = up(ch);
    dRec = up(recs);
    dEq = up(eqs);
    dGa = up(gas);
    dAt = up(ats);
    {
      std::vector<uint2> m;
      for (size_t i = 0; i < eqs.size(); i++)
        for (size_t b = 0; b < (((size_t)1 << eqs[i].n) + 255) / 256; b++) m.push_back(make_uint2((u32)i, (u32)b));
      nEqB = (u32)m.size();
      dEqM = up(m);
    }
    {
      std::vector<uint2> m;
      for (size_t i = 0; i < gas.size(); i++)
        for (u32 b = 0; b < (gas[i].n + 255) / 256; b++) m.push_back(make_uint2((u32)i, b));
      nGaB = (u32)m.size();
      dGaM = up(m);
    }
    hz::alloc(theta, sizeof(T), hz::FORK);
    if ((nclaim == 0) != vectors_only) {
      printf(nclaim ? "internal error: walk: a claim was added to a walk that is kept for its vectors only\n"
                    : "internal error: walk: no claim was added\n");
      exit(2);
    }
    if (nclaim) hz::alloc(kap, (size_t)nclaim * sizeof(T), hz::FORK);
  }
  // Once per proof, after every challenge that any form reads has been drawn. finish() calls it.
  void run_prep() {
    if (!recs.empty()) k_psc_rows<<<(unsigned)((recs.size() + 31) / 32), 32>>>(dRec, (int)recs.size());
    if (nEqB) k_eqB<<<nEqB, 256>>>(dEq, dEqM);
    if (!ats.empty()) k_atB<<<(unsigned)((ats.size() + 63) / 64), 64>>>(dAt, (int)ats.size());
    if (nGaB) k_gaB<<<nGaB, 256>>>(dGa, dGaM);
    for (auto &f : prep) f();
  }
  // Call after the sub-proof's last message. Draws the batching challenge theta, then runs
  // run_prep(), whose prep jobs may read theta, and adds the claims into the spaces' lam, which the
  // caller zeroes first. LS0: must be sp[0].lam, the mask entries' coefficients.
  void finish(Tr *tr, const T *LS0) {
    if (sp.empty() || sp[0].lam != LS0) {
      printf("internal error: walk: finish() is given another vector than space 0 has\n");
      exit(2);
    }
    k_batch_draw<<<1, 1>>>(tr, theta);
    k_batch_powers<<<(nclaim + 127) / 128, 128>>>(theta, nclaim, kap);
    run_prep();
    k_scat<<<(nch + 7) / 8, 256>>>(dJ, dCH, nch, kap);
  }
};
} // namespace hw
