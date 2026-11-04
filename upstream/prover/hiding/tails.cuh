// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The "tail" of each layer of a product test: the layer's last, short sumcheck rounds (`PSC` in
// identifiers: the sumcheck with a hiding polynomial). A product test is the grand-product check
// of the lookup argument; each of its layers is a sumcheck over two tables. One launch of k_tails
// runs the tails of a list of layers of one chain. A chain (struct Eng in product_test.cuh, which
// includes this file) is a sequence of layers on one Fiat-Shamir transcript state. A round is wide
// if each of the layer's two tables has more than 2^NL entries (NL: pt_transcript.h). The blocks
// share a wide round's work; block 0 runs the other (narrow) rounds in shared memory. Every block
// reaches tail_barrier() the same number of times: once at the start of each layer that has a wide
// round, and twice in each wide round. Launch k_tails through launch_tail(): it sets the block
// size, the shared memory and the cooperative launch that tail_layer() relies on.
//
// Public sources: NVIDIA, CUDA C++ Programming Guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/), "Cooperative Groups", "Grid
// Synchronization": this_grid().sync() under a cooperative launch, with no more blocks than the GPU
// holds at once; "Warp Shuffle Functions": sums within a warp.
// Public source: a cooperative launch with a priority, by attributes: NVIDIA, CUDA Runtime API,
// enum cudaLaunchAttributeID (cudaLaunchAttributeCooperative, cudaLaunchAttributePriority) and the
// field `cooperative` of cudaLaunchAttributeValue.
// Public source: the effect of the cooperative attribute: NVIDIA, CUDA Driver API, cuLaunchKernelEx
// (CU_LAUNCH_ATTRIBUTE_COOPERATIVE, cuLaunchCooperativeKernel).
// See also, for the grid's barrier and the shuffles: NVIDIA, CUDA Programming Guide,
// "Device-Callable APIs and Intrinsics", "Cooperative Groups API", "Grid Synchronization",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html;
// "C/C++ Language Extensions", "Warp Shuffle Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
#pragma once
#include <cooperative_groups.h>
// The barrier over all blocks of a launch. A launch of one block is not cooperative, and its
// barrier is __syncthreads().
__device__ __forceinline__ void tail_barrier() {
  if (gridDim.x > 1)
    cooperative_groups::this_grid().sync();
  else
    __syncthreads();
}
// Runs rounds j0 + 1 to mu of one layer, sends what the layer shows at its end (kind: TailKind) and updates the transcript state in *st.
// The values computed here are unmasked: the functions of sc_door (sumcheck_door.cuh) mask them,
// store them in the running sumcheck's next message slots and send them. The layer's two end
// values are its tables folded with every challenge. TAIL_PRODUCT (TailKind: product_test.cuh)
// sends both and draws the challenge tau, TAIL_BITS sends the first, and TAIL_LEAVES sends none
// here; each kind stores both in st->endv[id].
// If `begins` is set, the layer's first round is run here and its sumcheck begins here, with
// `out` (not read otherwise) as first slot; with TAIL_LEAVES this traps. If not, k_layer_begin or
// k_leaves_begin (product_test.cuh) must have begun it, or psc::round() (psc_device.cuh) traps.
// src holds the layer's two tables, one after the other, folded with the earlier rounds'
// challenges: 2^n entries each, where n is the number of rounds run here. ch->rho is the point of
// the weights eq(rho, x), the multilinear equality polynomial. ch->r holds the challenges: the
// earlier rounds' are read, the others written. A TAIL_PRODUCT layer writes the next layer's point
// (tau, then r) to chn->rho, unless chn is null, and the true value of that layer's sum to
// st->start[idn], unless idn is negative. A TAIL_LEAVES or TAIL_BITS layer writes only r, from
// chn->rho[1] on.
// bx is the calling block's number among gx blocks. B0 and B1 (2^n and 2^(n - 1) entries) receive
// the wide rounds' folded tables in turn; `part` ((2 + PSC_CHECK_CLAIM) gx entries) receives the
// blocks' partial sums.
// Not checked: these sizes; n <= 22 (E3S in pt_transcript.h); mu < MAXV; id and idn < 128;
// blockDim.x == TPB; with TAIL_LEAVES, that the second table is constant in the last 5 variables.
//
// Public source: Dao and Thaler, "More Optimizations to Sum-Check Proving", ePrint 2024/1210,
// section 1.1: they replace the full table of eq(w, x) by small tables over groups of variables and
// apply one group's factor once to an aggregated inner sum (here, in a wide round: `el` times a
// thread's sums over its tiles).
// Public source: Verschelde and Yu, "GPU acceleration of Newton's method for large systems of
// polynomial equations in double double and quad double arithmetic", arXiv:1402.2626, section 2.1:
// tables of factors that each block computes in shared memory. Here the tables are those of eq.
__device__ void tail_layer(const int bx, const int gx, const T *src, int j0, int mu, int id, int idn, bool begins, int kind, St *st,
                           Chal<F> *ch, Chal<F> *chn, T *B0, T *B1, ShT &S_, T *part, T *out) {
  constexpr int NT = 2; /* the number of tables, and of coefficients sent in a round of degree 2 */
  const int u = threadIdx.x;
  T *e = S_.e, *sh = S_.sh, *tab = S_.tab, *rsh = S_.rsh;
  const int nwide = mu - NL - j0 > 0 ? mu - NL - j0 : 0;
  // The barrier makes what the previous layer wrote (the transcript state, the point rho) visible
  // to this layer.
  if (nwide)
    tail_barrier();
  else if (bx != 0)
    return;
  sc_door::tr_fetch_by_lanes(&S_.tr, &st->tr, u, TPB);
  if (u < mu) S_.rho[u] = ldcg16(ch->rho + u);
  if (u < j0) rsh[u] = ldcg16(ch->r + u);
  __syncthreads();
  // The sumcheck is begun with degree 2 in every variable and in the end values that are sent.
  // That fits TAIL_PRODUCT and TAIL_BITS, but not TAIL_LEAVES.
  if (begins && kind == TAIL_LEAVES) __trap();
  if (begins && u < 32 && bx == 0)
    (void)sc_door::begin_by_warp<sc_door::Form::EQC>(sc_door::Site::LAYER_LATE, &S_.tr, out, mu, [&S_](int q) { return S_.rho[q]; },
                                                     ldcg16(st->start + id), 2, 2);
  if (mu > j0) build3<F>(S_.rho, mu, mu - j0 - 1, e);
  // A round's polynomial is c0 + c1 X + c2 X^2. With table k's entries lo_k and hi_k = lo_k + D_k
  // and their eq weight w (from the small tables e, which build3() in pt_transcript.h fills),
  // summed over all such pairs:
  //   c0 = sum w lo_0 lo_1,   c1 = sum w (lo_0 D_1 + D_0 lo_1),   c2 = sum w D_0 D_1.
  // sc_door takes c1 and c2 (psc::Form::EQC). The + 1 in the summand of TAIL_LEAVES changes only
  // c0. A build for tests with `PSC_CHECK_CLAIM` defined as 1 passes c0 too, and psc::round() traps
  // unless the polynomial fits the running claim.
#if PSC_CHECK_CLAIM
  const T more = kind == TAIL_LEAVES ? F::one() : F::zero();
#endif
  // s: the tables to read. np: if set, they are the previous round's, still to be folded with
  // its challenge. pp: which of B0 and B1 receives the folded tables.
  const T *s = src;
  int pp = 0, j = j0 + 1, np = 0;
  for (; j <= j0 + nwide; j++) {
    // Each table has S entries: nt tiles of 2^LT = TPB pairs (c, c + H), shared among nbk blocks.
    const int ls = mu - j + 1, nb = ls - 1, nbh = nb - LT;
    const size_t S = (size_t)1 << ls, H = S >> 1, nt = (size_t)1 << nbh;
    T *d = pp ? B1 : B0;
    const int nbk = nt < gx ? (int)nt : gx;
    if (bx < nbk) {
      F::P rb = F::prep(np ? rsh[j - 2] : F::zero());
      F::Acc acc[NT + PSC_CHECK_CLAIM];
      for (F::Acc &a : acc) a = F::azero();
      for (size_t t = bx; t < nt; t += gx) {
        const size_t c = (t << LT) + u;
        T lo[NT], hi[NT], D[NT];
        // lo[k] and hi[k] are entries c and c + H of table k in this round. If a fold is pending
        // (np), they are first formed from the previous round's tables with its challenge and
        // stored in d.
        if (np == 0) {
#pragma unroll
          for (int k = 0; k < NT; k++) {
            lo[k] = ldcg16(s + k * S + c);
            hi[k] = ldcg16(s + k * S + c + H);
          }
        } else {
          T a[NT][4];
#pragma unroll
          for (int k = 0; k < NT; k++) {
            const T *p = s + k * 2 * S + c;
            a[k][0] = ldcg16(p);
            a[k][1] = ldcg16(p + S);
            a[k][2] = ldcg16(p + H);
            a[k][3] = ldcg16(p + H + S);
          }
#pragma unroll
          for (int k = 0; k < NT; k++) {
            lo[k] = fold1<F>(rb, a[k][0], a[k][1]);
            hi[k] = fold1<F>(rb, a[k][2], a[k][3]);
          }
#pragma unroll
          for (int k = 0; k < NT; k++) {
            st16(d + k * S + c, lo[k]);
            st16(d + k * S + c + H, hi[k]);
          }
        }
#pragma unroll
        for (int k = 0; k < NT; k++) D[k] = F::sub(hi[k], lo[k]);
        const F::P w = F::prep(wtile<F>(e, t, nbh));
        const F::P wl = F::prep(F::mulp(w, lo[0])), wd = F::prep(F::mulp(w, D[0]));
        F::mac(acc[0], wl, D[1]);
        F::mac(acc[0], wd, lo[1]);
        F::mac(acc[1], wd, D[1]);
#if PSC_CHECK_CLAIM
        F::mac(acc[2], wl, lo[1]);
#endif
      }
      T v[NT + PSC_CHECK_CLAIM];
      const T el = e[(1 << LT) + u];
#pragma unroll
      for (int q = 0; q < NT + PSC_CHECK_CLAIM; q++) v[q] = F::mul(F::ared(acc[q]), el);
      blk_red<F, NT + PSC_CHECK_CLAIM>(v, sh);
      if (u == 0) {
#pragma unroll
        for (int q = 0; q < NT + PSC_CHECK_CLAIM; q++) part[bx * (NT + PSC_CHECK_CLAIM) + q] = v[q];
      }
    }
    tail_barrier(); // all partial sums of the round are written before they are read below
    // Block 0 adds the nbk blocks' partial sums and sends the round.
    if (bx == 0) {
      const int w = u >> 5, l = u & 31;
      if (w < NT + PSC_CHECK_CLAIM) {
        T x = F::zero();
        for (int b = l; b < nbk; b += 32) x = F::add(x, ldcg16(part + b * (NT + PSC_CHECK_CLAIM) + w));
        for (int o = 16; o; o >>= 1) x = F::add(x, shfl_down<F>(x, o));
        if (l == 0) S_.sums[w] = x;
      }
      __syncthreads();
      if (u < SEND2_N) { // threads 0 to SEND2_N - 1 must call `round_by_lanes` together
#if PSC_CHECK_CLAIM
        const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_LATE, j - 1, &S_.tr, S_.sums[0], S_.sums[1],
                                                                F::add(S_.sums[2], more));
#else
        const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_LATE, j - 1, &S_.tr, S_.sums[0], S_.sums[1]);
#endif
        if (u == 0) {
          rsh[j - 1] = r;
          ch->r[j - 1] = r;
        }
      }
    }
    tail_barrier(); // the challenge is written before the other blocks read it below
    if (bx != 0 && u == 0) rsh[j - 1] = ldcg16(ch->r + j - 1);
    __syncthreads();
    if (np) {
      s = d;
      pp ^= 1;
    }
    np = 1;
  }
  if (bx != 0) return;
  {
    const int ls = mu - j + 1;
    const size_t S = (size_t)1 << ls;
    const F::P rb = F::prep(np ? rsh[j - 2] : F::zero());
    for (size_t i0 = u; i0 < NT * S; i0 += 4 * TPB) { /* tab receives the tables of the first narrow round */
      T a[4], b[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const size_t i = i0 + q * TPB, k = i >> ls, x = i & (S - 1);
        if (i < NT * S) {
          const T *p = np ? s + k * 2 * S + x : s + i;
          a[q] = ldcg16(p);
          if (np) b[q] = ldcg16(p + S);
        }
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const size_t i = i0 + q * TPB;
        if (i < NT * S) tab[i] = np ? fold1<F>(rb, a[q], b[q]) : a[q];
      }
    }
    __syncthreads();
    for (; j <= mu; j++) {
      const int nb = mu - j;
      const size_t H = (size_t)1 << nb;
      T v[NT + PSC_CHECK_CLAIM], lo[NT], D[NT];
      for (T &x : v) x = F::zero();
      const bool act = (size_t)u < H;
      if (act) {
#pragma unroll
        for (int k = 0; k < NT; k++) {
          lo[k] = tab[k * S + u];
          D[k] = F::sub(tab[k * S + u + H], lo[k]);
        }
        const F::P w = F::prep(e[H + u]);
        const F::P wl = F::prep(F::mulp(w, lo[0])), wd = F::prep(F::mulp(w, D[0]));
        v[0] = F::add(F::mulp(wl, D[1]), F::mulp(wd, lo[1]));
        v[1] = F::mulp(wd, D[1]);
#if PSC_CHECK_CLAIM
        v[2] = F::mulp(wl, lo[1]);
#endif
      }
      if (H > 32)
        blk_red<F, NT + PSC_CHECK_CLAIM>(v, sh);
      else if (u < 32) {
#pragma unroll
        for (int q = 0; q < NT + PSC_CHECK_CLAIM; q++)
          for (int o = 16; o; o >>= 1) v[q] = F::add(v[q], shfl_down<F>(v[q], o));
      }
#if PSC_CHECK_CLAIM
      v[2] = F::add(v[2], more);
#endif
      if (kind == TAIL_LEAVES && j > mu - 5) { // table 1 is constant in these 5 variables: c2 is 0
#if PSC_CHECK_CLAIM
        v[1] = v[2]; // c0 follows the one coefficient of a round of degree 1
#endif
        if (u == 0) rsh[j - 1] = ch->r[j - 1] = sc_door::round<sc_door::Form::EQC, sc_door::Draw::COIN>(sc_door::Site::LAYER_LATE, j - 1, S_.tr, v, 1);
      } else if (u < SEND2_N) { // threads 0 to SEND2_N - 1 must call `round_by_lanes` together
#if PSC_CHECK_CLAIM
        const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_LATE, j - 1, &S_.tr, v[0], v[1], v[2]);
#else
        const T r = sc_door::round_by_lanes<sc_door::Form::EQC>(sc_door::Site::LAYER_LATE, j - 1, &S_.tr, v[0], v[1]);
#endif
        if (u == 0) {
          rsh[j - 1] = r;
          ch->r[j - 1] = r;
        }
      }
      if (H > 32)
        __syncthreads();
      else
        __syncwarp();
      // rsh[j - 1] is read only under act: for H <= 32 the __syncwarp() above orders its write only
      // for u < 32.
      if (act) {
        const F::P rp = F::prep(rsh[j - 1]);
#pragma unroll
        for (int k = 0; k < NT; k++) tab[k * S + u] = F::add(lo[k], F::mulp(rp, D[k]));
      }
      if (H > 32)
        __syncthreads();
      else
        __syncwarp();
    }
    __syncthreads();
    // tab[0] and tab[S] are the layer's two end values.
    if (kind != TAIL_PRODUCT) {
      if (u == 0) {
        const T value = tab[0];
        st->endv[id][0] = value, st->endv[id][1] = tab[S];
        if (kind == TAIL_BITS) { // t^2 - t at t = value + c Y: the coefficients of Y and Y^2
          const T c = sc_door::inner_mask(S_.tr, 0), vc = F::mul(value, c), in_y[2] = {F::sub(F::add(vc, vc), c), F::mul(c, c)};
          (void)sc_door::further(sc_door::Site::LAYER_LATE, S_.tr, in_y);
          (void)sc_door::ends<sc_door::Draw::NONE>(sc_door::Site::LAYER_LATE, S_.tr, &value, 1);
        }
        sc_door::tr_copy_by_lanes(&st->tr, &S_.tr, 0, 1);
      }
    } else if (u < SEND2_N) { // threads 0 to SEND2_N - 1 must call `layer_ends_by_lanes` together
      const T v[NT] = {tab[0], tab[S]};
      T tau = sc_door::layer_ends_by_lanes(sc_door::Site::LAYER_LATE, &S_.tr, v[0], v[1]);
      if (u == 0) {
        // cheat_at_layer_end() is a test hook that returns false in this repository
        // (prover/hiding/sumcheck_door.cuh), so the assignment below never runs.
        if (mu > 0 && sc_door::cheat_at_layer_end(5, tau, S_.rho[0])) tau = S_.rho[0];
        st->endv[id][0] = v[0];
        st->endv[id][1] = v[1];
        st->claim[id] = F::add(v[0], F::mul(tau, F::sub(v[1], v[0])));
        if (idn >= 0) st->start[idn] = st->claim[id]; // what the next layer's sumcheck claims
        if (chn) chn->rho[0] = tau;
        sc_door::tr_copy_by_lanes(&st->tr, &S_.tr, 0, 1);
      }
    }
    if (chn && u < mu) chn->rho[1 + u] = rsh[u];
    __syncthreads();
  }
}
// Runs the layers of the list dsc->ll, one after another. Not checked: dsc->ll.n <= 24, the length
// of the arrays in struct LL (product_test.cuh).
// The idea is public: a persistent kernel for dependent stages: L. Zhang, M. Wahib, P. Chen,
// J. Meng, X. Wang, T. Endo, S. Matsuoka, "PERKS: a Locality-Optimized Execution Model for
// Iterative Memory-bound GPU Applications", ICS '23, 2023, abstract and section 3,
// https://arxiv.org/pdf/2204.02064.
// The compiler hint is public: __launch_bounds__(n), an upper bound on the block size: NVIDIA,
// CUDA C++ Best Practices Guide, "Occupancy",
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(TPB) k_tails(const ChainD *__restrict__ dsc) {
  extern __shared__ uint4 dyn[];
  ShT &S_ = *reinterpret_cast<ShT *>(dyn);
  const ChainD &c = *dsc;
  for (int i = 0; i < c.ll.n; i++)
    tail_layer((int)blockIdx.x, (int)gridDim.x, c.ll.src[i], c.ll.j0[i], c.ll.mu[i], c.ll.id[i], c.ll.idn[i], c.ll.begins[i] != 0, c.ll.kind[i], c.st,
                     c.chd + c.ll.id[i], c.ll.idn[i] >= 0 ? c.chd + c.ll.idn[i] : nullptr, c.B0, c.B1, S_, c.part,
                     c.ll.out[i]);
}
// desc is in GPU memory. A launch of more than one block is cooperative, because its blocks wait
// for one another at tail_barrier(). It is requested by cudaLaunchAttributeCooperative, so that the
// launch can also carry a priority. Its blocks must all fit on the GPU together. That is not
// checked here: CK() stops the program if the runtime rejects the launch. Sched::launch()
// (product_test.cuh), the caller, requests one block for each SM or, if Sched::isfree is set, at
// most PT_FREE_BLOCKS. The tails of chains that run independently (Sched::run()) may run at the
// same time: for each proving thread at most 32 launches, of at most PT_FREE_BLOCKS blocks each.
// A block waits at the barrier only for blocks of the same launch.
//
// How many blocks a cooperative launch may have: NVIDIA, CUDA Runtime API,
// cudaLaunchCooperativeKernel()
// (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html): the limit is
// cudaDevAttrMultiProcessorCount times what cudaOccupancyMaxActiveBlocksPerMultiprocessor (or
// cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) reports;
// CUDA C++ Programming Guide, "Grid Synchronization" (cited at the top of this file).
// Public source: cooperative kernels side by side, from CUDA 11.2 on: NVIDIA's blog post
// "Enhancing Memory Allocation with New NVIDIA CUDA 11.2 Features", section "Cooperative groups"
// (there: if they fit on the GPU together).
// See also, for the number of blocks: NVIDIA, CUDA Programming Guide, "Device-Callable APIs and
// Intrinsics", "Cooperative Groups API", "Grid Synchronization",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html: the
// same rule (all blocks on the GPU at once).
static inline void launch_tail(const ChainD *desc, int blocks, bool high_priority, cudaStream_t stream) {
  static const bool once = [] {
    CK_ATTRIBUTE(k_tails, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(ShT));
    return true;
  }();
  (void)once;
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(blocks);
  c.blockDim = dim3(TPB);
  c.dynamicSmemBytes = sizeof(ShT);
  c.stream = stream;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (high_priority) {
    at[na].id = cudaLaunchAttributePriority;
    at[na++].val.priority = pt_high();
  }
  if (blocks > 1) {
    at[na].id = cudaLaunchAttributeCooperative;
    at[na++].val.cooperative = 1;
  }
  c.attrs = at;
  c.numAttrs = na;
  CK(cudaLaunchKernelEx(&c, k_tails, desc));
  CK(cudaGetLastError());
}
