// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The token sumcheck (TSC; "tsc" in names) on the GPU, for the layer sub-proofs and the sub-proof
// of the output head. It reduces a claimed sum over token positions t,
//   Cl = sum_t eq(P, t) sum_m co_m F_m[t] G_m[t],  t < 2^lt,  1 <= lt <= 32 (not checked here),
// to the end values F_m(r), G_m(r): r is the vector of its lt rounds' challenges, top bit first.
// Each F_m and G_m has 2^lt elements of the extension field (type T), stored as [m][t]. eq(P, t) is
// the product over the bits b of t (b = 0 is the top bit) of P[b] if the bit is 1, else 1 - P[b].
// The protocol is the masked sumcheck (PSC; blocks/iop/psc.h): namespace sc_door
// (prover/hiding/sumcheck_door.cuh) forms each message from the values computed here and sends it.
//
// The arrays are folded in place, which destroys them. Tsc::prove() ends with the last round over
// t. The caller runs any further rounds (TscShape::more) and the end, for example with TscEnd,
// because sumchecks differ in which end values they show.
// Include prover/hiding/product_test.cuh first: it defines the stream g_st and includes
// prover/hiding/transcript.cuh, which brings in T, its arithmetic F, the transcript Tr and sc_door.
#pragma once
#include "hiding/place.cuh"
// Written by the kernels of Tsc::prove(): the challenges r[0 .. lt - 1] and the sum Cl.
struct TscOut {
  T r[32], Cl;
};
// What a sumcheck has beyond the lt rounds over t. A map is a factor of the summand, such as F_m.
// A shown map's end value is sent, under its inner mask (c_j at k_tsc_end). The summand's degree in
// the shown maps is also its degree in Y, PSC's further variable, which carries the inner masks.
struct TscShape {
  size_t shown = 0;    // end values that are shown
  int degree_in_y = 0; // the degree of the summand in the maps that these are values of
  // Rounds that the caller runs after those over t, over an index that is summed without weights.
  // They run under equality weights at the coordinate 1/2, which give every value of the index the
  // weight 2^-more: co must contain this factor, so that Cl is 2^-more times the unweighted sum.
  int more = 0;
};
// The two sends, shared by both ways of computing a round (see TSC_NV). tsc_claim() runs before
// round 0: it sends Cl in msg[0] and begins the sumcheck, whose slots start at msg + 1.
// tsc_round() sends round j and returns the challenge r[j]. q[0] is q(0) and q[1] is qinf, the X^2
// coefficient, of
//   q(X) = sum_c eq(P[j+1..], c) sum_m co_m F_m(X, c) G_m(X, c),
// which omits the eq factors of bit j and of the bits already bound.
__device__ __forceinline__ void tsc_claim(Tr &t, T Cl, const T *P, int lt, const TscShape &shape, T *msg) {
  sc_door::claim_alone(sc_door::Site::TOKEN, t, Cl, msg);
  const T half = F::inv(F::small(2));
  (void)sc_door::begin<sc_door::Form::EQ2>(
      sc_door::Site::TOKEN, t, msg + 1, lt + shape.more, [=](int j) { return j < lt ? P[j] : half; }, Cl,
      shape.degree_in_y);
}
__device__ __forceinline__ T tsc_round(Tr &t, int j, const T *q) {
  return sc_door::round<sc_door::Form::EQ2, sc_door::Draw::COIN>(sc_door::Site::TOKEN, j, t, q);
}
// For c < H: S0[c] = sum_m co_m F_m[c] G_m[c],
// Si[c] = sum_m co_m (F_m[c+H] - F_m[c]) (G_m[c+H] - G_m[c]),
// S1[c] = sum_m co_m F_m[c+H] G_m[c+H] if `first` (for Cl), else 0. The arrays still have 2 H
// entries. part[y][0..2][c] receives S0, Si, S1 over the y-th of gridDim.y consecutive ranges of m.
__global__ void k_tsc_sum(const T *Fv, const T *Gv, const T *co, int NT, size_t stride, size_t H, T *part, int first) {
  const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const int per = (NT + gridDim.y - 1) / gridDim.y, m0 = blockIdx.y * per, m1 = min(NT, m0 + per);
  T s0 = F::zero(), si = F::zero(), s1 = F::zero();
  for (int m = m0; m < m1; m++) {
    const T f0 = Fv[m * stride + c], f1 = Fv[m * stride + c + H], g0 = Gv[m * stride + c], g1 = Gv[m * stride + c + H],
            k = co[m];
    s0 = F::add(s0, F::mul(k, F::mul(f0, g0)));
    si = F::add(si, F::mul(k, F::mul(F::sub(f1, f0), F::sub(g1, g0))));
    if (first) s1 = F::add(s1, F::mul(k, F::mul(f1, g1)));
  }
  T *p = part + (size_t)blockIdx.y * 3 * H;
  p[c] = s0;
  p[H + c] = si;
  p[2 * H + c] = s1;
}
// Adds up `part` over y and c, with entry c weighted by eq(P[j+1..], c). If j == 0 it first stores
// and sends Cl (tsc_claim()). It sends round j's message and stores the challenge in o->r[j].
// Launch one block of a multiple of 32 threads, at most 1024.
// The idea is public: a0, ai and a1 share one block reduction by warp shuffles and shared memory
// (the warps' sums are added here by thread 0, there by the first warp): J. Luitjens, "Faster
// Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014, "Block Reduce" and "Interleaving
// Multiple Reductions", https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void k_tsc_msg(const T *part, int ny, size_t H, const T *P, int lt, int j, TscShape shape, Tr *tr, T *msg,
                          TscOut *o) {
  __shared__ T sh[3][32];
  T a0 = F::zero(), ai = F::zero(), a1 = F::zero();
  const int nb = lt - j - 1;
  for (size_t c = threadIdx.x; c < H; c += blockDim.x) {
    T w = F::one();
    for (int i = 0; i < nb; i++) {
      const T r = P[j + 1 + i];
      w = F::mul(w, ((c >> (nb - 1 - i)) & 1) ? r : F::sub(F::one(), r));
    }
    T s0 = F::zero(), si = F::zero(), s1 = F::zero();
    for (int y = 0; y < ny; y++) {
      const T *p = part + (size_t)y * 3 * H;
      s0 = F::add(s0, p[c]);
      si = F::add(si, p[H + c]);
      if (j == 0) s1 = F::add(s1, p[2 * H + c]);
    }
    a0 = F::add(a0, F::mul(w, s0));
    ai = F::add(ai, F::mul(w, si));
    if (j == 0) a1 = F::add(a1, F::mul(w, s1));
  }
  for (int d = 16; d; d >>= 1) {
    a0 = F::add(a0, shfl_down<F>(a0, d));
    ai = F::add(ai, shfl_down<F>(ai, d));
    a1 = F::add(a1, shfl_down<F>(a1, d));
  }
  if ((threadIdx.x & 31) == 0) {
    sh[0][threadIdx.x >> 5] = a0;
    sh[1][threadIdx.x >> 5] = ai;
    sh[2][threadIdx.x >> 5] = a1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T q0 = F::zero(), qi = F::zero(), q1 = F::zero();
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) {
      q0 = F::add(q0, sh[0][w]);
      qi = F::add(qi, sh[1][w]);
      q1 = F::add(q1, sh[2][w]);
    }
    Tr t = *tr;
    if (j == 0) {
      const T Cl = F::add(F::mul(F::sub(F::one(), P[0]), q0), F::mul(P[0], q1));
      o->Cl = Cl;
      tsc_claim(t, Cl, P, lt, shape, msg);
    }
    const T q[2] = {q0, qi};
    o->r[j] = tsc_round(t, j, q);
    *tr = t;
  }
}
// Folds the top half of every F_m and G_m into the bottom half with challenge o->r[j], in place.
// The idea is public: several outputs per thread (thread coarsening): Wen-mei W. Hwu, David Kirk,
// "Advanced Algorithmic Techniques for GPUs, Lecture 4: Thread Coarsening and more on
// Tiling/Blocking", PASI Summer School, Chile, January 5-7, 2011, slide "Outline of Technique",
// https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf.
__global__ void k_tsc_fold(T *Fv, T *Gv, int NT, size_t stride, size_t H, const TscOut *o, int j) {
  const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const T r = o->r[j];
  for (int m = blockIdx.y; m < NT; m += gridDim.y) {
    T *f = Fv + m * stride, *g = Gv + m * stride;
    f[c] = F::add(f[c], F::mul(r, F::sub(f[c + H], f[c])));
    g[c] = F::add(g[c], F::mul(r, F::sub(g[c + H], g[c])));
  }
}
// k_tsc_resident (token_rounds.cuh) computes the last min(lt, TSC_NV) rounds in one launch, and
// k_tsc_sum, k_tsc_msg and k_tsc_fold compute each earlier round. Both ways give the same messages,
// challenges and arrays. k_tsc_resident can compute at most TSC_NV rounds: its eq tables, TscSh::e,
// have 2^TSC_NV elements. It is launched with TSC_TPB = 2^TSC_LTPB threads per block and with at
// most TSC_MAXB blocks.
constexpr int TSC_NV = 10, TSC_LTPB = 8, TSC_TPB = 1 << TSC_LTPB, TSC_MAXB = 128;
// Adds share u of a round's sums to v: S0, Si and, if NQ == 3 (round 0), S1 of k_tsc_sum, over the
// terms m0 <= m < m1, with pair c < H = 2^nb (entries c and c + H of an array) weighted by eq[c].
// There are 2^lnth shares, and no two touch the same entry of Fv or Gv; tsc_sums()
// (token_rounds.cuh) gives each thread of a block one. If `pend` is set, each array still has 4 H
// entries and is first folded in place with the previous round's challenge r, as in k_tsc_fold.
// The idea is public: a compile-time constant as a template parameter: NVIDIA, CUTLASS
// documentation, "Programming Guidelines", "Templates",
// https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/programming_guidelines.html.
// The idea is public: kernel fusion in a sumcheck, the pending fold with the next round's sums:
// "Hardware-friendliness of HyperPlonk, Part2", a note on HackMD under the account Ingonyama,
// https://hackmd.io/@Ingonyama/Hardware-Friendliness-HyperPlonk-Part2.
// The compiler hint is public: #pragma unroll, without an argument and with 1: NVIDIA, "CUDA
// Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// The compiler hint is public: with the loops on q unrolled, a thread's small array s can stay in
// registers: M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog,
// 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int NQ>
HD void tsc_pairs(int u, int lnth, T *Fv, T *Gv, const T *co, size_t stride, int m0, int m1, int nb, bool pend, T r,
                  const T *eq, T (&v)[NQ]) {
  const size_t H = (size_t)1 << nb;
  // The pairs of one term are divided among 2^lth shares, and the terms among 2^(lnth - lth)
  // groups of shares.
  const int lth = nb < lnth ? nb : lnth;
  const size_t TH = (size_t)1 << lth;
  const int ML = 1 << (lnth - lth);
  const F::P rp = F::prep(r);
#pragma unroll 1
  for (size_t c = u & (TH - 1); c < H; c += TH) {
    T s[NQ];
#pragma unroll
    for (int q = 0; q < NQ; q++) s[q] = F::zero();
#pragma unroll 1
    for (int m = m0 + (u >> lth); m < m1; m += ML) {
      T *f = Fv + m * stride + c, *g = Gv + m * stride + c;
      T f0 = f[0], f1 = f[H], g0 = g[0], g1 = g[H];
      if (NQ == 2 && pend) { // Round 0 has no fold before it.
        f0 = F::add(f0, F::mulp(rp, F::sub(f[2 * H], f0)));
        f1 = F::add(f1, F::mulp(rp, F::sub(f[3 * H], f1)));
        g0 = F::add(g0, F::mulp(rp, F::sub(g[2 * H], g0)));
        g1 = F::add(g1, F::mulp(rp, F::sub(g[3 * H], g1)));
        f[0] = f0;
        f[H] = f1;
        g[0] = g0;
        g[H] = g1;
      }
      const F::P k = F::prep(co[m]);
      s[0] = F::add(s[0], F::mulp(k, F::mul(f0, g0)));
      s[1] = F::add(s[1], F::mulp(k, F::mul(F::sub(f1, f0), F::sub(g1, g0))));
      if constexpr (NQ == 3) s[2] = F::add(s[2], F::mulp(k, F::mul(f1, g1)));
    }
    const F::P w = F::prep(eq[c]);
#pragma unroll
    for (int q = 0; q < NQ; q++) v[q] = F::add(v[q], F::mulp(w, s[q]));
  }
}
// Level nb of the eq tables starts at e[2^nb] and has the weights eq(P[lt-nb..], c) of the 2^nb
// pairs c of round lt - nb - 1. The caller sets level 0, e[1], to one. tsc_eq_step() writes entries
// x and 2^(nb - 1) + x of level nb >= 1 from entry x < 2^(nb - 1) of level nb - 1.
HD void tsc_eq_step(T *e, int nb, int x, const T *P, int lt) {
  const T a = e[(1 << (nb - 1)) + x], hi = F::mul(a, P[lt - nb]);
  e[(1 << nb) + (1 << (nb - 1)) + x] = hi;
  e[(1 << nb) + x] = F::sub(a, hi);
}
// Shared memory of k_tsc_resident: eq tables, scratch for a block's sums, P, the latest challenge.
struct TscSh {
  T e[1 << TSC_NV], red[3 * 32], P[32], r;
};
#include "hiding/token_rounds.cuh" // k_tsc_resident, tsc_resident()
// Host-side driver. init() must run first: it allocates the GPU buffers with hz::alloc()
// (prover/hiding/place.cuh); nothing here frees them. prove() queues kernels on stream st for Cl
// and the lt rounds, for NT >= 1 terms (not checked). They update the transcript *tr, write *out
// and fold Fv and Gv in place, which leaves F_m(r) and G_m(r) in entry 0 of each array. They trap
// if PADOFF (prover/hiding/sumcheck_door.cuh) or a P[b] is 0.
// msg has nmsg() message slots of one scalar each: Cl plus the mask entry of its slot (one random
// element, added to one sent scalar); then the sumcheck that starts from Cl: the sum of PSC's
// hiding polynomial (a random polynomial added to the summand), two scalars for each of the
// lt + shape.more rounds, the shape.degree_in_y scalars of the round in Y, the shown end values,
// each under its inner mask. Each slot has its mask entry PADOFF elements after it and a scratch
// cell, which sc_door overwrites, 2 PADOFF elements after it (neither is checked).
struct Tsc {
  // ny: the number of ranges of m in k_tsc_sum, chosen from NTmax. maxb: the largest number of
  // blocks of k_tsc_resident.
  int lt = 0, NTmax = 0, ny = 0, maxb = 0;
  TscShape shape;
  T *part = nullptr;
  TscOut *out = nullptr;
  T *slots = nullptr; // work space of k_tsc_resident: a round's partial sums
  void init(int lt_, int NTmax_, const TscShape &shape_) {
    lt = lt_;
    NTmax = NTmax_;
    shape = shape_;
    ny = NTmax >= 64 ? 32 : NTmax >= 8 ? 4 : 1;
    hz::alloc(part, (size_t)ny * 3 * ((size_t)1 << (lt - 1)) * sizeof(T), hz::SHAPE);
    hz::alloc(out, sizeof(TscOut), hz::SHAPE);
    hz::alloc(slots, tsc_resident_elements(TSC_MAXB) * sizeof(T), hz::SHAPE); // a launch reads only what it wrote
    int dev, sms;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    maxb = std::min(TSC_MAXB, sms);
  }
  size_t nmsg() const {
    return 1 + sc_door::slots(sc_door::Form::EQ2, lt + shape.more, shape.shown, shape.degree_in_y);
  }
  void prove(Tr *tr, const T *P_dev, const T *co, T *Fv, T *Gv, int NT, T *msg, cudaStream_t st = g_st) {
    const size_t stride = (size_t)1 << lt;
    const int j0 = std::max(0, lt - TSC_NV);
    for (int j = 0; j < j0; j++) {
      const size_t H = stride >> (j + 1);
      const unsigned gx = (unsigned)((H + 127) / 128);
      k_tsc_sum<<<dim3(gx, ny), 128, 0, st>>>(Fv, Gv, co, NT, stride, H, part, j == 0);
      k_tsc_msg<<<1, 256, 0, st>>>(part, ny, H, P_dev, lt, j, shape, tr, msg, out);
      k_tsc_fold<<<dim3(gx, std::min(NT, 64)), 128, 0, st>>>(Fv, Gv, NT, stride, H, out, j);
    }
    const int per = (NT + maxb - 1) / maxb, nbk = (NT + per - 1) / per;
    tsc_resident(nbk, Fv, Gv, co, NT, per, stride, P_dev, lt, j0, shape, tr, msg, out, slots, st);
  }
};
// ---- The end of a sumcheck whose summand is sum_m co_m (map a_m) (map b_m) ----
// One product. a, b: the indices of its two maps among the shown end values (TscEnd::value).
//   a >= 0, b >= 0   both are shown;
//   a >= 0, b < 0    the second is not shown: *kept is its true value at the point, and the summand
//                    is affine in that map;
//   a < 0            neither is shown: the term does not depend on Y and is skipped.
// co: the coefficient, or null for 1. A map that occurs in several products is shown once.
struct TscTerm {
  int a, b;
  const T *co, *kept;
};
// Sends the round in Y, then the n end values v_j = *value[j], under their inner masks c_j, as one
// message. Shown map j enters the summand as v_j + c_j Y, so the coefficients of Y and Y^2 are
// sum co (v_a c_b + v_b c_a) + sum co c_a kept, and sum co c_a c_b. Launch one thread. It traps if
// rounds are left, if n or degree_in_y is 0, if a term has a < 0 <= b, or if degree_in_y > 2, for
// which it has no coefficients. It does not check a, b < n: TscEnd::setup() does.
__global__ void k_tsc_end(Tr *tr, const T *const *value, int n, const TscTerm *terms, int nterms) {
  Tr t = *tr;
  if (t.p.in_y > 2) __trap();
  T in_y[2] = {F::zero(), F::zero()};
  for (int m = 0; m < nterms; m++) {
    const TscTerm x = terms[m];
    if (x.a < 0 && x.b >= 0) __trap();
    if (x.a < 0) continue;
    const T co = x.co ? *x.co : F::one(), ca = F::mul(co, sc_door::inner_mask(t, x.a));
    if (x.b < 0) {
      in_y[0] = F::add(in_y[0], F::mul(ca, *x.kept));
      continue;
    }
    const T cb = sc_door::inner_mask(t, x.b);
    in_y[0] = F::add(in_y[0], F::add(F::mul(ca, *value[x.b]), F::mul(F::mul(co, cb), *value[x.a])));
    in_y[1] = F::add(in_y[1], F::mul(ca, cb));
  }
  (void)sc_door::further(sc_door::Site::TOKEN, t, in_y);
  for (int j = 0; j < n; j++) (void)sc_door::end(sc_door::Site::TOKEN, t, *value[j], j);
  sc_door::ends_sent(t, n);
  *tr = t;
}
// The host side of k_tsc_end. setup() runs once, with GPU addresses in value_ and terms_. run()
// comes after the sumcheck's last round. `shape` is the one given to Tsc::init(): sc_door does not
// know how many end values a sumcheck shows, so setup() exits with code 2 unless value_ and terms_
// fit it.
struct TscEnd {
  const T **value = nullptr;
  TscTerm *terms = nullptr;
  int n = 0, nterms = 0;
  void setup(const TscShape &shape, const std::vector<const T *> &value_, const std::vector<TscTerm> &terms_) {
    int degree = 0;
    bool numbers = true;
    for (const TscTerm &x : terms_) {
      degree = std::max(degree, x.a < 0 ? 0 : x.b < 0 ? 1 : 2);
      numbers = numbers && x.a < (int)value_.size() && x.b < (int)value_.size() && !(x.a < 0 && x.b >= 0) &&
                !(x.a >= 0 && x.b < 0 && !x.kept);
    }
    if (value_.size() != shape.shown || degree > shape.degree_in_y || shape.degree_in_y > 2 || !numbers) {
      printf(
          "internal error: the end of a token sum-check does not fit its shape: %zu end values for %zu, terms of degree %d "
          "for %d (at most 2), or a term that names no such end value\n",
          value_.size(), shape.shown, degree, shape.degree_in_y);
      exit(2);
    }
    n = (int)value_.size(), nterms = (int)terms_.size();
    hz::alloc(value, value_.size() * sizeof(const T *), hz::OWN);
    hz::alloc(terms, terms_.size() * sizeof(TscTerm), hz::OWN);
    CK(cudaMemcpy(value, value_.data(), value_.size() * sizeof(const T *), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(terms, terms_.data(), terms_.size() * sizeof(TscTerm), cudaMemcpyHostToDevice));
  }
  void run(Tr *tr, cudaStream_t st = g_st) const { k_tsc_end<<<1, 1, 0, st>>>(tr, value, n, terms, nterms); }
};
