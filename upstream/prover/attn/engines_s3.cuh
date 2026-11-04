// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// SC3 is the attention sub-proof's batched sumcheck of degree 3: it sums W_b(x) A_b(x) B_b(x) over
// b < nb and x < 2^nv, where W_b(x) = wb[b] W0[x >> nlow]. A functor Src (prover/attn/close.cuh)
// reads A_b and B_b as field elements (type T), as integers if Src::INT, or as rows of 2^nlow int16
// if also Src::ROW. Round j binds bit nv - 1 - j of x; h(0), h(-1) and the coefficient of t^3 fix
// its polynomial h(t), given h(0) + h(1). The round kernels compute them as in a sumcheck that
// hides nothing; sc_door (prover/hiding/sumcheck_door.cuh) turns them into messages of PSC, the
// sumcheck with a hiding polynomial (blocks/iop/psc.h), and sends them. Call a_s3_prep(), then
// a_s3_run(). prover/attn/engines.cuh includes this file last, after the helpers that it uses.
// Round j > 0 takes three launches. Public sources for its sums: see prover/attn/engines_sums.cuh.
// First rounds computed from integers: Bagad, Domb, Thaler, ePrint 2024/1046.
#pragma once
#include "attn/engines_sums.cuh"
namespace ga {
// Round 0: partial sums of (h(0), h(-1), h's t^3 coefficient, h(1)) go to part; later rounds omit
// h(1). All round kernels assume blocks of NTH threads, LE = s3_le(), grid s3_grid(): unchecked.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++ Language Extensions",
// "#pragma unroll", https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// the loop on k is unrolled so that v, each thread's four partial sums, stays in registers:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <class Src, int LE>
__global__ void __launch_bounds__(NTH)
    k_s3_r0(Src s, int nv, int nlow, const T *wb, const T *W0, T *part) {
  const int b = blockIdx.y, nvW = nv - nlow;
  const size_t H = (size_t)1 << (nv - 1), nth = H >> LE;
  T v[4] = {F::zero(), F::zero(), F::zero(), F::zero()};
  const F::P dum = F::prep(F::zero());
  for (size_t q = blockIdx.x * (size_t)NTH + threadIdx.x; q < nth; q += (size_t)gridDim.x * NTH) {
    const size_t X0 = ((q >> 5) << (5 + LE)) + (q & 31);
    T s0 = F::zero(), sm = s0, sd = s0, s1 = s0;
    long long i0 = 0, i1 = 0, im = 0, id = 0;
    (void)i0;
    (void)i1;
    (void)im;
    (void)id;
    const int16_t *pa0 = nullptr, *pa1 = nullptr, *pb0 = nullptr, *pb1 = nullptr;
    (void)pa0;
    (void)pa1;
    (void)pb0;
    (void)pb1;
    if constexpr (Src::INT)
      if constexpr (Src::ROW) {
        pa0 = s.Ar(b, X0 >> nlow);
        pa1 = s.Ar(b, (X0 + H) >> nlow);
        pb0 = s.Br(b, X0 >> nlow);
        pb1 = s.Br(b, (X0 + H) >> nlow);
      }
#pragma unroll
    for (int e = 0; e < (1 << LE); e++) {
      const size_t X = X0 + 32 * e;
      if constexpr (Src::INT) {
        long long a0, a1, b0, b1;
        if constexpr (Src::ROW) {
          const size_t ml = ((size_t)1 << nlow) - 1, xl = X & ml, xh = (X + H) & ml;
          a0 = pa0[xl];
          a1 = pa1[xh];
          b0 = pb0[xl];
          b1 = pb1[xh];
        } else {
          a0 = s.Ai(b, X);
          a1 = s.Ai(b, X + H);
          b0 = s.Bi(b, X);
          b1 = s.Bi(b, X + H);
        }
        i0 += a0 * b0;
        i1 += a1 * b1;
        im += (2 * a0 - a1) * (2 * b0 - b1);
        id += (a1 - a0) * (b1 - b0);
      } else {
        const T a0 = s.A(b, X), a1 = s.A(b, X + H), b0 = s.B(b, X), b1 = s.B(b, X + H);
        s3_acc(a0, a1, b0, b1, s0, sm, sd);
        s1 = F::add(s1, F::mul(a1, b1));
      }
    }
    if constexpr (Src::INT) {
      s0 = F::emb(F::fromi(i0));
      s1 = F::emb(F::fromi(i1));
      sm = F::emb(F::fromi(im));
      sd = F::emb(F::fromi(id));
    }
    T w0, w1;
    s3_w(0, nvW, W0, nullptr, X0 >> nlow, dum, false, w0, w1);
    const T dw = F::sub(w1, w0);
    v[0] = F::add(v[0], F::mul(w0, s0));
    v[1] = F::add(v[1], F::mul(F::sub(w0, dw), sm));
    v[2] = F::add(v[2], F::mul(dw, sd));
    v[3] = F::add(v[3], F::mul(w1, s1));
  }
  {
    const F::P x = F::prep(wb[b]);
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] = F::mulp(x, v[k]);
  }
  block_sums<4>(v, 4, part);
}
// Computes h4 (see a_s3_prep()) from round 0's totals. Launch: one thread.
__global__ void k_s3_h4(const T *tot, T *h4) {
  h4[0] = tot[0];
  h4[1] = tot[1];
  h4[2] = tot[2];
  h4[3] = F::add(tot[0], tot[3]);
}
// Begins the sumcheck: sc_door::begin() sends the hiding polynomial's sum in slots[0] and draws
// the element that joins it to the claimed sum h4[3]. 2 is the summand's degree in A_b and B_b. The
// caller must have sent or drawn whatever A_b, B_b, wb, W0 or h4[3] depend on. Launch: one thread.
__global__ void k_s3_begin(const T *h4, int nv, Tr *tr, T *slots) {
  Tr t = *tr;
  (void)sc_door::begin<sc_door::Form::SUM3>(
      sc_door::Site::ATTENTION_S3, t, slots, nv, [](int) { return F::zero(); }, h4[3], 2);
  *tr = t;
}
// Sends a round's three values v and draws the round's challenge *rj. Launch: one thread.
__global__ void k_s3_send(const T *v, Tr *tr, T *rj) {
  Tr t = *tr;
  *rj = sc_door::round<sc_door::Form::SUM3, sc_door::Draw::CHAL>(sc_door::Site::ATTENTION_S3, -1, t, v);
  *tr = t;
}
// Round 1, for any Src: folds A_b and B_b by r[0] and writes them to A and B as [b][2^(nv-1)].
// The blocks with b == 0 also write W1, round 1's table of W (s3_w() in prover/attn/engines.cuh).
// Public sources for the techniques of this routine. The idea is public: kernel fusion in the
// sumcheck prover (one pass folds and sums, saving memory accesses; round 0, the first, only sums):
// "Hardware-friendliness of HyperPlonk, Part2", a note on HackMD under the account Ingonyama,
// "Sumcheck in ICICLE", https://hackmd.io/@Ingonyama/Hardware-Friendliness-HyperPlonk-Part2.
// The loop form and the compiler hints are public: the grid-stride loop, __launch_bounds__ and
// #pragma unroll: the post on grid-stride loops, the Best Practices Guide, the CUDA Programming
// Guide and Milakov's post (cited at k_s3_r0()).
template <class Src, int LE>
__global__ void __launch_bounds__(NTH)
    k_s3_r1(Src s, int nv, int nlow, const T *wb, const T *W0, T *W1, T *__restrict__ A, T *__restrict__ B, T *part,
            const T *r) {
  const T r0 = r[0];
  const F::P rr = F::prep(r0);
  const int b = blockIdx.y, nvW = nv - nlow;
  const size_t H = (size_t)1 << (nv - 2), nth = H >> LE, msk = ((size_t)1 << nlow) - 1;
  T v[3] = {F::zero(), F::zero(), F::zero()};
  T *Ab = A + (size_t)b * 2 * H, *Bb = B + (size_t)b * 2 * H;
  const bool top = 1 < nvW;
  for (size_t q = blockIdx.x * (size_t)NTH + threadIdx.x; q < nth; q += (size_t)gridDim.x * NTH) {
    const size_t X0 = ((q >> 5) << (5 + LE)) + (q & 31);
    T s0 = F::zero(), sm = s0, sd = s0;
#pragma unroll
    for (int e = 0; e < (1 << LE); e++) {
      const size_t X = X0 + 32 * e;
      T a0, a1, b0, b1;
      if constexpr (Src::INT) {
        a0 = foldi(s.Ai(b, X), s.Ai(b, X + 2 * H), r0);
        a1 = foldi(s.Ai(b, X + H), s.Ai(b, X + 3 * H), r0);
        b0 = foldi(s.Bi(b, X), s.Bi(b, X + 2 * H), r0);
        b1 = foldi(s.Bi(b, X + H), s.Bi(b, X + 3 * H), r0);
      } else {
        a0 = fold1(s.A(b, X), s.A(b, X + 2 * H), rr);
        a1 = fold1(s.A(b, X + H), s.A(b, X + 3 * H), rr);
        b0 = fold1(s.B(b, X), s.B(b, X + 2 * H), rr);
        b1 = fold1(s.B(b, X + H), s.B(b, X + 3 * H), rr);
      }
      st16(Ab + X, a0);
      st16(Ab + X + H, a1);
      st16(Bb + X, b0);
      st16(Bb + X + H, b1);
      s3_acc(a0, a1, b0, b1, s0, sm, sd);
    }
    T w0, w1;
    s3_w(1, nvW, W0, W1, X0 >> nlow, rr, b == 0 && (top ? (X0 & msk) == 0 : X0 == 0), w0, w1);
    const T dw = F::sub(w1, w0);
    v[0] = F::add(v[0], F::mul(w0, s0));
    v[1] = F::add(v[1], F::mul(F::sub(w0, dw), sm));
    v[2] = F::add(v[2], F::mul(dw, sd));
  }
  {
    const F::P x = F::prep(wb[b]);
#pragma unroll
    for (int k = 0; k < 3; k++) v[k] = F::mulp(x, v[k]);
  }
  block_sums<3>(v, 3, part);
}
// Round J (1, 2 or 3) for integer sources, computed directly from the integers. After J folds, a
// source y (A or B) has at X the value sum_u eq(r_0 .. r_(J-1); u) y[u 2^(nv-J) + X], where
// eq(r; u) = prod_v (bit J - 1 - v of u ? r_v : 1 - r_v). The eq weights are cut into limbs, and
// weight x integer is added up exactly and reduced once (Limbs, I4 in prover/attn/engines.cuh).
// A sum has 2^J <= 8 terms, each with an int16: far within I4's capacity. The ROW branch requires
// J < nv - nlow. The other branch is never launched: a_s3_run() takes this kernel for ROW sources
// only. WR: also writes the folded tables to A and B as [b][2^(nv-J)], for the later rounds.
// Public sources for the techniques of this routine.
// The idea is public: thread u < 2^J of a block fills eqs[u] in shared memory once, for the whole block to
// read: M. Harris, "Optimizing CUDA", NVIDIA Developer Technology, slides headed "S05: High Performance
// Computing with CUDA", slide "Take Advantage of Shared Memory" (the address runs over two lines):
// https://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/Cours09/CUDA/
// SC07_CUDA_5_Optimization_Harris.pdf.
// Delayed reduction. This is a public idea that is specific to modular arithmetic: J.-G. Dumas,
// P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK
// packages", section 2.1.2, https://arxiv.org/pdf/cs/0601133.
// The loop form and the compiler hints are public: the grid-stride loop, __launch_bounds__ and
// #pragma unroll: the post on grid-stride loops, the Best Practices Guide, the CUDA Programming
// Guide and Milakov's post (cited at k_s3_r0()).
template <class Src, int J, int LE, bool WR>
__global__ void __launch_bounds__(NTH)
    k_s3_ri(Src s, int nv, int nlow, const T *wb, const T *Wp, T *Wn, T *__restrict__ A, T *__restrict__ B, T *part,
            const T *r) {
  __shared__ T rs[J];
  __shared__ Limbs eqs[1 << J];
  if (threadIdx.x < J) rs[threadIdx.x] = r[threadIdx.x];
  __syncthreads();
  if (threadIdx.x < (1 << J)) {
    T e = F::one();
    for (int v = 0; v < J; v++) e = F::mul(e, ((threadIdx.x >> (J - 1 - v)) & 1) ? rs[v] : F::sub(F::one(), rs[v]));
    eqs[threadIdx.x] = limbs_of(e);
  }
  __syncthreads();
  const F::P rr = F::prep(rs[J - 1]);
  const int b = blockIdx.y, nvW = nv - nlow;
  const size_t H = (size_t)1 << (nv - 1 - J), nth = H >> LE, msk = ((size_t)1 << nlow) - 1;
  T v[3] = {F::zero(), F::zero(), F::zero()};
  T *Ab = A + (size_t)b * 2 * H, *Bb = B + (size_t)b * 2 * H;
  const bool top = J < nvW;
  for (size_t q = blockIdx.x * (size_t)NTH + threadIdx.x; q < nth; q += (size_t)gridDim.x * NTH) {
    const size_t X0 = ((q >> 5) << (5 + LE)) + (q & 31);
    T s0 = F::zero(), sm = s0, sd = s0;
    I4 x0[1 << LE], x1[1 << LE], y0[1 << LE], y1[1 << LE];
#pragma unroll
    for (int e = 0; e < (1 << LE); e++) x0[e] = x1[e] = y0[e] = y1[e] = i4zero();
#pragma unroll
    for (int u = 0; u < (1 << J); u++) {
      const Limbs l = eqs[u];
      const size_t o = (size_t)u * 2 * H + X0;
      if constexpr (Src::ROW) {
        const size_t xl = X0 & msk;
        const int16_t *pa0 = s.Ar(b, o >> nlow) + xl, *pa1 = s.Ar(b, (o + H) >> nlow) + xl,
                      *pb0 = s.Br(b, o >> nlow) + xl, *pb1 = s.Br(b, (o + H) >> nlow) + xl;
#pragma unroll
        for (int e = 0; e < (1 << LE); e++) {
          i4mac(x0[e], l, pa0[32 * e]);
          i4mac(x1[e], l, pa1[32 * e]);
          i4mac(y0[e], l, pb0[32 * e]);
          i4mac(y1[e], l, pb1[32 * e]);
        }
      } else {
#pragma unroll
        for (int e = 0; e < (1 << LE); e++) {
          i4mac(x0[e], l, s.Ai(b, o + 32 * e));
          i4mac(x1[e], l, s.Ai(b, o + 32 * e + H));
          i4mac(y0[e], l, s.Bi(b, o + 32 * e));
          i4mac(y1[e], l, s.Bi(b, o + 32 * e + H));
        }
      }
    }
#pragma unroll
    for (int e = 0; e < (1 << LE); e++) {
      const size_t X = X0 + 32 * e;
      const T a0 = i4red(x0[e]), a1 = i4red(x1[e]), b0 = i4red(y0[e]), b1 = i4red(y1[e]);
      if (WR) {
        st16(Ab + X, a0);
        st16(Ab + X + H, a1);
        st16(Bb + X, b0);
        st16(Bb + X + H, b1);
      }
      s3_acc(a0, a1, b0, b1, s0, sm, sd);
    }
    T w0, w1;
    s3_w(J, nvW, Wp, Wn, X0 >> nlow, rr, b == 0 && (top ? (X0 & msk) == 0 : X0 == 0), w0, w1);
    const T dw = F::sub(w1, w0);
    v[0] = F::add(v[0], F::mul(w0, s0));
    v[1] = F::add(v[1], F::mul(F::sub(w0, dw), sm));
    v[2] = F::add(v[2], F::mul(dw, sd));
  }
  {
    const F::P x = F::prep(wb[b]);
#pragma unroll
    for (int k = 0; k < 3; k++) v[k] = F::mulp(x, v[k]);
  }
  block_sums<3>(v, 3, part);
}
// Round j >= 2: folds the tables A and B in place by r[j - 1]. Table b starts at element b 2^lstr.
// Wp and Wn are W's tables of rounds j - 1 and j (s3_w()).
// Public sources for the techniques of this routine.
// The idea is public: kernel fusion in the sumcheck prover: Ingonyama's note (cited at k_s3_r1()).
// The loop form and the compiler hints are public: the grid-stride loop, __launch_bounds__ and
// #pragma unroll: the post on grid-stride loops, the Best Practices Guide, the CUDA Programming
// Guide and Milakov's post (cited at k_s3_r0()).
template <int LE>
__global__ void __launch_bounds__(NTH)
    k_s3_rn(int j, int nv, int nlow, int lstr, const T *wb, const T *Wp, T *Wn, T *__restrict__ A, T *__restrict__ B,
            T *part, const T *r) {
  const F::P rr = F::prep(r[j - 1]);
  const int b = blockIdx.y, nvW = nv - nlow;
  const size_t H = (size_t)1 << (nv - 1 - j), nth = H >> LE, msk = ((size_t)1 << nlow) - 1;
  T v[3] = {F::zero(), F::zero(), F::zero()};
  T *Ab = A + ((size_t)b << lstr), *Bb = B + ((size_t)b << lstr);
  const bool top = j < nvW;
  for (size_t q = blockIdx.x * (size_t)NTH + threadIdx.x; q < nth; q += (size_t)gridDim.x * NTH) {
    const size_t X0 = ((q >> 5) << (5 + LE)) + (q & 31);
    T s0 = F::zero(), sm = s0, sd = s0;
#pragma unroll
    for (int e = 0; e < (1 << LE); e++) {
      const size_t X = X0 + 32 * e;
      const T a0 = fold1(ld16(Ab + X), ld16(Ab + X + 2 * H), rr),
              a1 = fold1(ld16(Ab + X + H), ld16(Ab + X + 3 * H), rr),
              b0 = fold1(ld16(Bb + X), ld16(Bb + X + 2 * H), rr),
              b1 = fold1(ld16(Bb + X + H), ld16(Bb + X + 3 * H), rr);
      st16(Ab + X, a0);
      st16(Ab + X + H, a1);
      st16(Bb + X, b0);
      st16(Bb + X + H, b1);
      s3_acc(a0, a1, b0, b1, s0, sm, sd);
    }
    T w0, w1;
    s3_w(j, nvW, Wp, Wn, X0 >> nlow, rr, b == 0 && (top ? (X0 & msk) == 0 : X0 == 0), w0, w1);
    const T dw = F::sub(w1, w0);
    v[0] = F::add(v[0], F::mul(w0, s0));
    v[1] = F::add(v[1], F::mul(F::sub(w0, dw), sm));
    v[2] = F::add(v[2], F::mul(dw, sd));
  }
  {
    const F::P x = F::prep(wb[b]);
#pragma unroll
    for (int k = 0; k < 3; k++) v[k] = F::mulp(x, v[k]);
  }
  block_sums<3>(v, 3, part);
}
// After the last round, A and B have two entries per b, and Wp has one, or two if nlow == 0. Folds
// each pair by r[nv - 1]: the end values A_b(r) and B_b(r) go to As and Bs, and W_b(r) to Ws.
// The loop form is public: the grid-stride loop (cited at k_s3_r0()).
__global__ void k_s3_end(int nv, int nlow, int lstr, int nb, const T *wb, const T *Wp, const T *A, const T *B,
                         const T *r, T *As, T *Bs, T *Ws) {
  const F::P rr = F::prep(r[nv - 1]);
  const T W = nlow == 0 ? fold1(Wp[0], Wp[1], rr) : Wp[0];
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
    const size_t o = (size_t)b << lstr;
    As[b] = fold1(A[o], A[o + 1], rr);
    Bs[b] = fold1(B[o], B[o + 1], rr);
    Ws[b] = F::mul(wb[b], W);
  }
}
// Round in PSC's further variable Y: in_y gets the coefficients of Y and Y^2 of the summand at the
// challenge point, sum_b W_b (A_b + c_2b Y) (B_b + c_2b+1 Y). c_j, end value j's inner mask, is the
// mask entry of that value's slot (see a_s3_run()). Launch: one block of 256 threads.
// The block's sum is formed by bsum() (prover/attn/engines.cuh); its public source is named there.
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels",
// cited in prover/attn/engines_sums.cuh. The compiler hint is public: __launch_bounds__: the Best
// Practices Guide, "Occupancy" (cited at k_s3_r0()).
__global__ void __launch_bounds__(256)
    k_s3_in_y(int nb, const T *As, const T *Bs, const T *Ws, const Tr *tr, T *in_y) {
  __shared__ T sh[32];
  T y1 = F::zero(), y2 = F::zero();
  for (int b = threadIdx.x; b < nb; b += 256) {
    const T ca = sc_door::inner_mask(*tr, 2 * b), cb = sc_door::inner_mask(*tr, 2 * b + 1);
    y1 = F::add(y1, F::mul(Ws[b], F::add(F::mul(As[b], cb), F::mul(Bs[b], ca))));
    y2 = F::add(y2, F::mul(Ws[b], F::mul(ca, cb)));
  }
  y1 = bsum(y1, sh);
  __syncthreads();
  y2 = bsum(y2, sh);
  if (threadIdx.x == 0) {
    in_y[0] = y1;
    in_y[1] = y2;
  }
}
// Sends the round in Y and draws its challenge, which *tr keeps. Launch: one thread.
__global__ void k_s3_further(const T *in_y, Tr *tr) {
  Tr t = *tr;
  (void)sc_door::further(sc_door::Site::ATTENTION_S3, t, in_y);
  *tr = t;
}
// Stores end value j as it is sent, (end value j) + beta c_j, where beta is the challenge of the
// round in Y: A_b(r) as number 2 b, B_b(r) as 2 b + 1. k_s3_end_send() absorbs them.
// The loop form is public: the grid-stride loop (cited at k_s3_r0()).
__global__ void k_s3_ends(int nb, const T *As, const T *Bs, const Tr *tr) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
    sc_door::end(sc_door::Site::ATTENTION_S3, *tr, As[b], 2 * b);
    sc_door::end(sc_door::Site::ATTENTION_S3, *tr, Bs[b], 2 * b + 1);
  }
}
// Absorbs the n end values as one message; no challenge follows. Launch: one thread.
__global__ void k_s3_end_send(Tr *tr, int n) {
  Tr t = *tr;
  sc_door::ends_sent(t, n);
  *tr = t;
}
// A round's second and third launches: adds the sums that the blocks of grid g wrote, then sends
// the totals and draws the challenge r[round].
static inline void s3_sent(Eng *e, dim3 g, int round, Tr *tr, T *r, cudaStream_t st) {
  sum_parts(e, g, 3, 3, st);
  k_s3_send<<<1, 1, 0, st>>>(totals(e), tr, r + round);
}
// Runs all three launches of round J. le (0, 1 or 2) and g are the round's values of s3_le() and
// s3_grid(). A round kernel's loop step takes 2^le values of X, 32 apart, that share w0 and w1.
template <class Src, int J, bool WR>
static inline void s3_ri(const Src &s, int le, dim3 g, int nv, int nlow, const T *wb, const T *Wp, T *Wn, T *A, T *B,
                         Eng *e, Tr *tr, T *r, cudaStream_t st) {
  if (le == 2)
    k_s3_ri<Src, J, 2, WR><<<g, NTH, 0, st>>>(s, nv, nlow, wb, Wp, Wn, A, B, e->part, r);
  else if (le == 1)
    k_s3_ri<Src, J, 1, WR><<<g, NTH, 0, st>>>(s, nv, nlow, wb, Wp, Wn, A, B, e->part, r);
  else
    k_s3_ri<Src, J, 0, WR><<<g, NTH, 0, st>>>(s, nv, nlow, wb, Wp, Wn, A, B, e->part, r);
  s3_sent(e, g, J, tr, r, st);
}
} // namespace ga
// Computes round 0's sums and the claimed total; does not use the transcript (`Tr`, from
// prover/hiding/transcript.cuh). h4 = (h(0), h(-1), t^3 coefficient of h, claim = h(0) + h(1)),
// where h is round 0's polynomial.
// Unchecked: nv >= 2 and nb <= ga::MAXBLK, as in a_s3_run(); |Ai()|, |Bi()| <= 2^28 (64-bit sums).
template <class Src>
static inline void a_s3_prep(const Src &s, int nv, int nlow, int nb, const T *wb, const T *W0, T *h4, cudaStream_t st,
                             ga::Eng *e) {
  using namespace ga;
  e->base();
  const int le = s3_le(0, nv, nlow);
  const dim3 g = s3_grid(nv - 1, le, nb);
  if (le == 2)
    k_s3_r0<Src, 2><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, e->part);
  else if (le == 1)
    k_s3_r0<Src, 1><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, e->part);
  else
    k_s3_r0<Src, 0><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, e->part);
  sum_parts(e, g, 4, 4, st);
  k_s3_h4<<<1, 1, 0, st>>>(totals(e), h4);
}
// Runs the sumcheck, given h4 of a_s3_prep(). What must be sent or drawn before the call: see
// k_s3_begin(). out: AFront::sc3n(nv, nb) message slots (prover/attn/front.cuh), one per scalar
// sent: the hiding polynomial's sum, 3 for each round, 2 for the round in Y, the 2 nb end values.
// Each slot's mask entry must be PADOFF elements after it (prover/hiding/sumcheck_door.cuh): not
// checked. w: scratch; S3Work (prover/attn/engines.cuh) gives the sizes. r: the nv challenges;
// As[b], Bs[b], Ws[b]: A_b(r), B_b(r), W_b(r), without masks.
template <class Src>
static inline void a_s3_run(const Src &s, int nv, int nlow, int nb, const T *wb, const T *W0, const S3Work &w,
                            const T *h4, Tr *tr, T *out, T *r, T *As, T *Bs, T *Ws, cudaStream_t st, ga::Eng *e) {
  using namespace ga;
  e->base();
  T *Wp = w.W1, *Wn = w.W2;
  int j = 2, lstr = nv - 1;
  bool done = false;
  k_s3_begin<<<1, 1, 0, st>>>(h4, nv, tr, out);
  k_s3_send<<<1, 1, 0, st>>>(h4, tr, r); // round 0 sends h4[0 .. 3)
  if constexpr (Src::INT)
    if (Src::ROW && nv - nlow >= 5 && nlow >= 5) {
      s3_ri<Src, 1, false>(s, s3_le(1, nv, nlow), s3_grid(nv - 2, s3_le(1, nv, nlow), nb), nv, nlow, wb, W0, w.W1, w.A,
                           w.B, e, tr, r, st);
      s3_ri<Src, 2, false>(s, s3_le(2, nv, nlow), s3_grid(nv - 3, s3_le(2, nv, nlow), nb), nv, nlow, wb, w.W1, w.W2,
                           w.A, w.B, e, tr, r, st);
      s3_ri<Src, 3, true>(s, s3_le(3, nv, nlow), s3_grid(nv - 4, s3_le(3, nv, nlow), nb), nv, nlow, wb, w.W2, w.W1, w.A,
                          w.B, e, tr, r, st);
      j = 4;
      lstr = nv - 3;
      Wp = w.W1;
      Wn = w.W2;
      done = true;
    }
  if (!done) {
    const int le = s3_le(1, nv, nlow);
    const dim3 g = s3_grid(nv - 2, le, nb);
    if (le == 2)
      k_s3_r1<Src, 2><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, w.W1, w.A, w.B, e->part, r);
    else if (le == 1)
      k_s3_r1<Src, 1><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, w.W1, w.A, w.B, e->part, r);
    else
      k_s3_r1<Src, 0><<<g, NTH, 0, st>>>(s, nv, nlow, wb, W0, w.W1, w.A, w.B, e->part, r);
    s3_sent(e, g, 1, tr, r, st);
  }
  for (; j < nv; j++) {
    const int le = s3_le(j, nv, nlow);
    const dim3 g = s3_grid(nv - 1 - j, le, nb);
    if (le == 2)
      k_s3_rn<2><<<g, NTH, 0, st>>>(j, nv, nlow, lstr, wb, Wp, Wn, w.A, w.B, e->part, r);
    else if (le == 1)
      k_s3_rn<1><<<g, NTH, 0, st>>>(j, nv, nlow, lstr, wb, Wp, Wn, w.A, w.B, e->part, r);
    else
      k_s3_rn<0><<<g, NTH, 0, st>>>(j, nv, nlow, lstr, wb, Wp, Wn, w.A, w.B, e->part, r);
    s3_sent(e, g, j, tr, r, st);
    std::swap(Wp, Wn);
  }
  k_s3_end<<<(nb + 255) / 256, 256, 0, st>>>(nv, nlow, lstr, nb, wb, Wp, w.A, w.B, r, As, Bs, Ws);
  k_s3_in_y<<<1, 256, 0, st>>>(nb, As, Bs, Ws, tr, totals(e));
  k_s3_further<<<1, 1, 0, st>>>(totals(e), tr);
  k_s3_ends<<<(nb + 255) / 256, 256, 0, st>>>(nb, As, Bs, tr);
  k_s3_end_send<<<1, 1, 0, st>>>(tr, 2 * nb);
}
