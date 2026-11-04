// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Three steps of a layer sub-proof (`fork` in identifiers): the batched inner-product sumcheck SCH
   ("stack"), the weight claims it leaves, and their merge into one pending claim on the committed
   weight matrix W (`Fork::member`), left for the batch's opening. SCH and the merge run on the
   masked sumcheck (PSC; prover/hiding/sumcheck_door.cuh), which shows an end value v as v + beta m
   (m: its inner mask; beta: a challenge). A sub-proof covers a layer group: Lam layers, the LamM
   odd ones mixture-of-experts (MoE) if mo is set. SCH sums As[x] Bs[x]. Both consist of panels
   (`Blk`, "block") of 2^wi rows of 2^wy entries. Row i adds om wt_i <A_i, B_i> (om: `kS_om`). In a
   panel with the selector (`tri`), row e of A is sl[e] xx. Where the token sumcheck TSC(zt) has
   shown a factor (a norm's R, a router's score sc, the expert selector), As has the shown value.
   The gated products' token sums Hh1 and Hh2 are never sent: they are the prover's own vectors, and
   SCH shows their values H1 and X_lamM at its end point. Verifier: verifier/src/vf_layers_check.inc. */
#pragma once
#include "common/index_guard.cuh"
#include "head/claims_lib.cuh"
#include <map>
#include <functional>
namespace gs { // a layer sub-proof's claims on the weights: the three steps described above
using hw::AF;
using hw::h2;
using hw::hadd;
using hw::HE;
using hw::hE;
using hw::hmul;
using hw::hz0;
using hw::Rec;
using hw::rng;
using hw::slot;
using hw::Sy;
using hw::var;
using hw::Walk;
struct TenG {
  int nl = 1;
  long long cnt = 1, J = 1, off = 0;
};
struct Shape {
  int Lam = 1, LamM = 0, mo = 0, d = 0, dq = 0, dkv = 0, dh = 0, H = 0, Gk = 0, f = 0, fe = 0, NE = 0, lkW = 0, L = 0,
      V = 0;
};
struct VD {
  const T *p[4] = {nullptr, nullptr, nullptr, nullptr};
  long long xs[4] = {1, 1, 1, 1};
  T w[4] = {}; // 0 where unused: a launch copies all four to the GPU
  int n = 0;
  T c = ext_base(0);
};
__device__ __forceinline__ T vd_at(const VD &v, size_t y) {
  T a = v.c;
  for (int q = 0; q < v.n; q++) a = F::add(a, F::mul(v.w[q], v.p[q][(long long)y * v.xs[q]]));
  return a;
}
__global__ void kS_vec(VD v, size_t n, T *o) {
  GRID_LOOP(y, n) o[y] = vd_at(v, y);
}
struct SyD {
  T c = ext_base(1);
  const T *p[4] = {nullptr, nullptr, nullptr, nullptr};
  int n = 0;
};
__global__ void kS_sy(const SyD *s, int n, T *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T v = s[i].c;
  for (int q = 0; q < s[i].n; q++) v = F::mul(v, *s[i].p[q]);
  o[i] = v;
}
// The start form's n <= 2 coefficients (Fork::plan()): 1, then a challenge. Launch: one thread.
__global__ void kS_om(Tr *tr, T *om, int n) {
  Tr t = *tr;
  om[0] = F::one();
  if (n > 1) tr_point(t, Round::BAT_SIGMA, om + 1, 1);
  *tr = t;
}
/* One row of a two-factor panel: As[y] = om[0] wt[0] A[y] (without wt[0] if wt is null) and
   Bs[y] = B[y]. As is zero from ny on and Bs from nyB on, up to the padded length 2^wy. A non-null
   sa or sb_ is a scalar that replaces the vector A or B; a non-null lo gives B as the limb pair
   lo[y] + 256 hi[y], whatever sb_ is. */
__global__ void kS_row(VD A, const T *sa, VD B, const T *sb_, const T *lo, const T *hi, const T *om, const T *wt,
                       int ny, int nyB, int wy, T *As, T *Bs) {
  const T c = wt ? F::mul(om[0], wt[0]) : om[0];
  GRID_LOOP(y, (size_t)1 << wy) {
    As[y] = (int)y < ny ? F::mul(c, sa ? sa[0] : vd_at(A, y)) : F::zero();
    Bs[y] = (int)y >= nyB ? F::zero()
            : lo          ? F::add(lo[y], F::mul(hi[y], F::small(256)))
            : sb_         ? sb_[0]
                          : vd_at(B, y);
  }
}
// Bs[i 2^wy + y] *= wt[i], i < ni: for panel "D", whose As is om Hh1 on all f columns, without wt.
__global__ void kS_rows_times(T *Bs, int ni, int wy, const T *wt) {
  GRID_LOOP(x, (size_t)ni << wy) Bs[x] = F::mul(Bs[x], wt[x >> wy]);
}
__global__ void kS_tri(const T *sl, const T *xx, const T *lo, const T *hi, size_t sB, const T *om, const T *wt, int NE,
                       int nk, int wk, int w, T *As, T *Bs) {
  const T c = F::mul(om[0], wt[0]);
  GRID_LOOP(x, (size_t)1 << w) {
    const int e = (int)(x >> wk), kk = (int)(x & (((size_t)1 << wk) - 1));
    if (e >= NE || kk >= nk) {
      As[x] = F::zero();
      Bs[x] = F::zero();
      continue;
    }
    As[x] = F::mul(c, F::mul(sl[e], xx[kk]));
    Bs[x] = F::add(lo[e * sB + kk], F::mul(hi[e * sB + kk], F::small(256)));
  }
}
// One job of kS_at: *o = product over q < nb of (z[q] if bit nb-1-q of idx is 1, else 1 - z[q]).
struct AtG {
  const T *z = nullptr;
  int nb = 0;
  unsigned long long idx = 0;
  T *o = nullptr;
};
__global__ void kS_at(const AtG *D, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const AtG d = D[i];
  T v = F::one();
  for (int q = 0; q < d.nb; q++) {
    const T r = d.z[q];
    v = F::mul(v, ((d.idx >> (d.nb - 1 - q)) & 1) ? r : F::sub(F::one(), r));
  }
  *d.o = v;
}
// A panel's row i: cB = sb ei and cA = cB om wt. A null ei is 1. sb, ei = EI[i]: Fork::allocate().
struct CoD {
  const T *sb = nullptr, *om = nullptr, *wt = nullptr, *ei = nullptr;
};
__global__ void kS_co(const CoD *D, int n, T *cA, T *cB) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const CoD d = D[i];
  const T b = d.ei ? F::mul(*d.sb, *d.ei) : *d.sb;
  cB[i] = b;
  cA[i] = F::mul(b, F::mul(*d.om, *d.wt));
}
/* The end of the batched sumcheck, whose summand at the end point is A* B*, with
     A* = A0 + sum_(i < nu) ku[i] u[i],   B* = B0 + sum_(a < nv) kv[a] v[a].
   u: the values of the never-sent vectors (H1, the X_lamM); v: those of the weight claims; ku, kv:
   public; A0, B0: the rest: forms in the reads (In::U3), up to public constants. Runs the round in
   the further variable Y and shows A0 | u | B0 | v as one message. Launch: one thread. */
__global__ void kS_end(Tr *tr, const T *Astar, const T *Bstar, const T *ku, const T *u, int nu, const T *kv, const T *v,
                       int nv) {
  constexpr sc_door::Site site = sc_door::Site::PAIRS;
  Tr t = *tr;
  T A0 = Astar[0], B0 = Bstar[0], a1 = sc_door::inner_mask(t, 0), b1 = sc_door::inner_mask(t, 1 + nu);
  for (int i = 0; i < nu; i++) {
    A0 = F::sub(A0, F::mul(ku[i], u[i]));
    a1 = F::add(a1, F::mul(ku[i], sc_door::inner_mask(t, 1 + i)));
  }
  for (int a = 0; a < nv; a++) {
    B0 = F::sub(B0, F::mul(kv[a], v[a]));
    b1 = F::add(b1, F::mul(kv[a], sc_door::inner_mask(t, 2 + nu + a)));
  }
  // (A* + a1 Y) (B* + b1 Y): the coefficients of Y and Y^2.
  const T in_y[2] = {F::add(F::mul(Astar[0], b1), F::mul(Bstar[0], a1)), F::mul(a1, b1)};
  (void)sc_door::further(site, t, in_y);
  sc_door::end(site, t, A0, 0);
  for (int i = 0; i < nu; i++) sc_door::end(site, t, u[i], 1 + i);
  sc_door::end(site, t, B0, 1 + nu);
  for (int a = 0; a < nv; a++) sc_door::end(site, t, v[a], 2 + nu + a);
  sc_door::ends_sent(t, 2 + nu + nv);
  *tr = t;
}
/* kap, which kS_bqk reads for panel "QKE", folds the tables Cr (cos) and Sr (sin) of the rotary
   position embedding (RoPE) into lf. If rope is 0, Cr counts as 4096 (fixed-point 1) and Sr as 0.
   kap[2i] = Ct lf[2i] + St lf[2i+1] and kap[2i+1] = Ct lf[2i+1] - St lf[2i], where Ct = sum_t er[t]
   Cr[t][i] and St likewise from Sr. Launch with one block of 256 threads for each i < dh / 2. */
// Public sources for the techniques of this routine.
// The idea is public: a block's sum from its warps' sums: J. Luitjens, "Faster Parallel Reductions
// on Kepler", NVIDIA Technical Blog, 2014, "Block Reduce",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// Here the warps' sums are added in a loop.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void kS_kap(const short *Cr, const short *Sr, int rope, const T *er, int NT, int dh, const T *lf, T *kap) {
  const int i = blockIdx.x;
  __shared__ T sh[2][8];
  T c = F::zero(), s = F::zero();
  for (int t = threadIdx.x; t < NT; t += 256) {
    const int cv = rope ? Cr[(size_t)t * (dh / 2) + i] : 4096, sv = rope ? Sr[(size_t)t * (dh / 2) + i] : 0;
    c = F::add(c, F::mulb(er[t], F::fromi(cv)));
    s = F::add(s, F::mulb(er[t], F::fromi(sv)));
  }
  c = kG_wred(c);
  s = kG_wred(s);
  if ((threadIdx.x & 31) == 0) {
    sh[0][threadIdx.x >> 5] = c;
    sh[1][threadIdx.x >> 5] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T Ct = F::zero(), St = F::zero();
    for (int w = 0; w < 8; w++) {
      Ct = F::add(Ct, sh[0][w]);
      St = F::add(St, sh[1][w]);
    }
    kap[2 * i] = F::add(F::mul(Ct, lf[2 * i]), F::mul(St, lf[2 * i + 1]));
    kap[2 * i + 1] = F::sub(F::mul(Ct, lf[2 * i + 1]), F::mul(St, lf[2 * i]));
  }
}
/* o[kk] = sum_c kap[c] (BQ[r] + 256 BQ[n1 + r]), r = (lam d + kk) dh + c.
 * BQ holds two limbs, the high limb at offset n1. It requires 256 threads per block. */
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(256) kS_bqk(const T *BQ, size_t n1, int lam, int d, int dh, const T *kap, T *o) {
  const int lane = threadIdx.x & 31;
  for (int kk = (int)((blockIdx.x * 256 + threadIdx.x) >> 5); kk < d; kk += (int)gridDim.x * 8) {
    T a = F::zero();
    for (int c = lane; c < dh; c += 32) {
      const size_t r = ((size_t)lam * d + kk) * dh + c;
      a = F::add(a, F::mul(kap[c], F::add(BQ[r], F::mul(BQ[n1 + r], F::small(256)))));
    }
    a = kG_wred(a);
    if (lane == 0) o[kk] = a;
  }
}
__global__ void kS_gat(const T *const *src, size_t n, T *o) {
  GRID_LOOP(i, n) o[i] = *src[i];
}
__global__ void kS_zero(T *o, size_t n) {
  GRID_LOOP(i, n) o[i] = F::zero();
}
__global__ void kS_one(T *o) {
  o[0] = F::one();
}
// o[x] = HV[0] prod_(b < rem) eq(pt[z - 1 - b], bit b of x).
__global__ void kS_low(const T *HV, const T *pt, int z, int rem, T *o) {
  GRID_LOOP(x, (size_t)1 << rem) {
    T r = HV[0];
    for (int b = 0; b < rem; b++) r = F::mul(r, kG_eq1(pt[z - 1 - b], (x >> b) & 1));
    o[x] = r;
  }
}
// kS_cut sets tab[r] = sum_xt eq((pt[0], .., pt[t - 1]), xt) tab2[r 2^t + xt].
// It is linear in tab2, so it may be applied to each limb separately.
__global__ void kS_cut(const T *tab2, size_t n, int t, const T *pt, T *tab) {
  GRID_LOOP(r, n) {
    T a = F::zero();
    for (int xt = 0; xt < (1 << t); xt++) {
      T e = F::one();
      for (int b = 0; b < t; b++) e = F::mul(e, kG_eq1(pt[t - 1 - b], (xt >> b) & 1));
      a = F::add(a, F::mul(e, tab2[(r << t) + xt]));
    }
    tab[r] = a;
  }
}
__global__ void kS_join(const T *lo, const T *hi, size_t n, T *o) {
  GRID_LOOP(i, n) o[i] = F::add(lo[i], F::mul(hi[i], F::small(256)));
}
/* psi[v] += sum_(t: tok_t = v) etE[t];
 * vE = sum_t etE[t] sum_xt eq((pt[0], .., pt[tE - 1]), xt) tabE[tok_t 2^tE + xt].
 * psi: V entries; tabE: V 2^tE. `index_below` flags a token outside 0 .. V - 1 and gives 0. */
__global__ void kS_psi(const int *tok, int V, const T *etE, int NT, T *psi, const T *tabE, int tE, const T *pt, T *vE) {
  if (threadIdx.x || blockIdx.x) return;
  T v = F::zero();
  for (int t = 0; t < NT; t++) {
    const size_t token = index_below(tok[t], V, INDEX_TOKEN);
    psi[token] = F::add(psi[token], etE[t]);
    T a = F::zero();
    for (int xt = 0; xt < (1 << tE); xt++) {
      T e = F::one();
      for (int b = 0; b < tE; b++) e = F::mul(e, kG_eq1(pt[tE - 1 - b], (xt >> b) & 1));
      a = F::add(a, F::mul(e, tabE[(token << tE) + xt]));
    }
    v = F::add(v, F::mul(etE[t], a));
  }
  vE[0] = v;
}
/* phi[q d + k], q < 2^tE, is the coefficient of entry k in vector q of the row passes over Emb (see
 * struct In). It is HV[k >> z] prod_(b < rem) eq(pt[z - 1 - b], bit b of k) if bits rem to z - 1 of
 * k equal q, else 0; rem = z - tE. */
__global__ void kS_phiE(const T *HV, const T *pt, int z, int tE, int d, T *phi) {
  const int rem = z - tE;
  GRID_LOOP(i, (size_t)d << tE) {
    const int q = (int)(i / d), k = (int)(i % d), lo = k & ((1 << z) - 1);
    if ((lo >> rem) != q) {
      phi[i] = F::zero();
      continue;
    }
    T r = HV[k >> z];
    for (int b = 0; b < rem; b++) r = F::mul(r, kG_eq1(pt[z - 1 - b], (lo >> b) & 1));
    phi[i] = r;
  }
}
__global__ void kS_joinE(const T *raw, int V, int tE, T *tab) {
  GRID_LOOP(i, (size_t)V << tE) {
    const size_t v = i >> tE, q = i & (((size_t)1 << tE) - 1);
    const T *r = raw + q * 4 * (size_t)V;
    T a = r[3 * (size_t)V + v];
    for (int l = 2; l >= 0; l--) a = F::add(F::mul(a, F::small(256)), r[(size_t)l * V + v]);
    tab[i] = a;
  }
}
/* State shared by all layer sub-proofs of one Prover: SC is the sumcheck prover, Z the merge's.
 * The sub-proofs share its scratch buffers, so they must run one at a time, on one stream. */
struct Shared {
  Shape s;
  // M[X]: the layout of weight kind X (see ui()); M[12]: kind 9 in a MoE layer; M[13]: embedding.
  CMap M[14];
  CMap MTgr, MTdr, MTe;
  T *phiE2 = nullptr, *beRaw = nullptr, *tabE = nullptr;
  int tE = 0, tG = 0, lamMax = 0, nclMax = 0;
  size_t K = 0;
  CSumcheck SC;
  CEnd Z;
  T *As = nullptr, *Bs = nullptr, *Bk = nullptr, *tab2[2] = {nullptr, nullptr}, *phiL[2] = {nullptr, nullptr},
    *R2j = nullptr, *one = nullptr;
  // Cells in one limb of one layer's weight of kind X; 0 for kind 9 (sized in limb()), 12 and 13.
  size_t U(int X) const {
    const size_t d = s.d;
    switch (X) {
    case 0:
      return d * s.dh * 64;
    case 1:
      return d * s.dkv;
    case 2:
      return (size_t)s.dq * d;
    case 3:
    case 4:
      return d * s.f;
    case 5:
    case 6:
      return d * s.fe;
    case 7:
    case 8:
      return (size_t)s.NE * d * s.fe;
    case 10:
      return (size_t)s.NE * s.fe * d;
    case 11:
      return d * s.NE;
    default:
      return 0;
    }
  }
  // Cells in one limb of kind X for the whole layer group: the limb stride in In::cells' buffer.
  size_t limb(int X) const {
    const size_t nd = s.mo ? s.Lam - s.LamM : s.Lam;
    switch (X) {
    case 0:
    case 1:
    case 2:
      return s.Lam * U(X);
    case 3:
    case 4:
      return nd * U(X);
    case 9:
      return s.mo ? (size_t)s.LamM * (s.f + s.fe) * s.d : (size_t)s.Lam * s.f * s.d;
    default:
      return s.LamM * U(X);
    }
  }
  // Offset at which layer lam's cells start inside a limb of kind X.
  size_t base(int X, int lam) const {
    switch (X) {
    case 0:
    case 1:
    case 2:
      return lam * U(X);
    case 3:
    case 4:
      return (s.mo ? lam / 2 : lam) * U(X);
    case 9:
      return s.mo ? (size_t)(lam / 2) * (s.f + s.fe) * s.d + (size_t)(lam & 1) * s.f * s.d : (size_t)lam * s.f * s.d;
    default:
      return (lam / 2) * U(X);
    }
  }
  // Index of the unit (a tensor's rows in W) of kind X in layer l. Units go by layer and, in a
  // layer, by kind: 6 in a dense layer, 10 in a MoE layer. Then come the embedding's (12) and the
  // head's (13): pass l = s.L. Kind 0: q and k, 1 v, 2 o, 3, 4 dense gating and up, 5, 6 the shared
  // expert's, 7, 8 the routed experts', 9 down (dense or shared), 10 routed down, 11 router.
  int ui(int X, int l) const {
    int n = 0;
    for (int q = 0; q < l; q++) n += (s.mo && (q & 1)) ? 10 : 6;
    if (l >= s.L) return n + (X == 13 ? 1 : 0);
    static const int den[6] = {0, 1, 2, 3, 4, 9}, moe[10] = {0, 1, 2, 5, 6, 7, 8, 9, 10, 11};
    const bool m = s.mo && (l & 1);
    for (int q = 0; q < (m ? 10 : 6); q++)
      if ((m ? moe[q] : den[q]) == X) return n + q;
    printf("internal error: weight-stack step of a layer sub-proof: W has no entry for "
           "weight kind %d in layer %d\n", X, l);
    exit(2);
  }
  // Builds the maps M (MoE ones only if s.mo; 4, 6, 8 use those of 3, 5, 7). As, Bs, Bk: 2^lamMax_
  // elements; Z: nclMax_ claims. Exits with code 2 if lkW (W has 2^lkW columns) does not fit s_.
  void setup(const Shape &s_, int lamMax_, int nclMax_) {
    s = s_;
    lamMax = lamMax_;
    nclMax = nclMax_;
    K = (size_t)1 << s.lkW;
    const size_t d = s.d, f = s.f, fe = s.fe, NE = s.NE;
    auto mk = [&](int at, int X, std::vector<CGrp> g, bool run) {
      CUnit u;
      u.X = X;
      u.nlw = 2;
      u.g = g;
      u.limbStride = limb(X);
      u.lay(s.lkW);
      M[at].setup(u);
      if (run) M[at].setup_run();
    };
    mk(0, 0, {c_grp("hd", 64, 'A', 1), c_grp("cd", s.dh, 'A', 64), c_grp("kk", d, 'L', (size_t)64 * s.dh)}, true);
    mk(1, 1, {c_grp("kk", d, 'L', s.dkv), c_grp("x", s.dkv, 'H', 1)}, true);
    mk(2, 2, {c_grp("kk", s.dq, 'L', d), c_grp("x", d, 'H', 1)}, true);
    mk(3, 3, {c_grp("kk", d, 'L', f), c_grp("x", f, 'H', 1)}, true);
    mk(9, 9, {c_grp("kk", f, 'L', d), c_grp("x", d, 'H', 1)}, true);
    size_t nymax = std::max<size_t>(1, NE); // Z.gam also takes a sum per expert (panels TG, TU, TD)
    if (s.mo) {
      mk(5, 5, {c_grp("kk", d, 'L', fe), c_grp("x", fe, 'H', 1)}, true);
      mk(12, 9, {c_grp("kk", fe, 'L', d), c_grp("x", d, 'H', 1)}, true);
      mk(11, 11, {c_grp("kk", d, 'L', NE), c_grp("x", NE, 'H', 1)}, true);
      mk(7, 7, {c_grp("kk", d, 'L', fe), c_grp("e", NE, 'L', d * fe), c_grp("x", fe, 'H', 1)}, false);
      mk(10, 10, {c_grp("kk", fe, 'L', d), c_grp("e", NE, 'L', fe * d), c_grp("x", d, 'H', 1)}, false);
      tG = M[7].u.g[2].take;
      if (M[10].u.g[2].take) {
        printf("policy: lkW = %d is too large for this mixture-of-experts model: log2(expert width fe) + log2(expert "
               "count NE) must be at least lkW\n",
               s.lkW);
        exit(2);
      }
      CUnit t;
      t.X = 7;
      t.nlw = 2;
      t.g = {c_grp("kk", d, 'L', (size_t)1 << tG), c_grp("e", NE, 'L', d << tG), c_grp("x'", (size_t)1 << tG, 'H', 1)};
      t.lay(s.lkW);
      MTgr.setup(t);
      CUnit t2;
      t2.X = 10;
      t2.nlw = 2;
      t2.g = {c_grp("kk", fe, 'L', 1), c_grp("e", NE, 'L', fe), c_grp("x'", 1, 'H', 1)};
      t2.lay(s.lkW);
      MTdr.setup(t2);
      for (int q = 0; q < 2; q++) {
        tab2[q] = dalC<T>(((size_t)2 * s.LamM * NE * d) << tG);
        phiL[q] = dalC<T>(fe >> tG);
      }
      R2j = dalC<T>(std::max((NE * d) << tG, NE * fe));
    }
    {
      CUnit e;
      e.X = 12;
      e.nlw = 4;
      e.g = {c_grp("v", s.V, 'A', d), c_grp("x", d, 'H', 1)};
      e.lay(s.lkW);
      tE = e.g[1].take;
      if (!e.g[0].take || tE > 3) {
        printf("policy: lkW = %d must lie between ceil(log2 V) and ceil(log2 V) + 3, where V = %d is the vocabulary "
               "size, for "
               "the embedding's claim\n",
               s.lkW, s.V);
        exit(2);
      }
      M[13].setup(e);
      CUnit t;
      t.X = 12;
      t.nlw = 4;
      t.g = {c_grp("v", s.V, 'A', (size_t)1 << tE), c_grp("x'", (size_t)1 << tE, 'H', 1)};
      t.lay(s.lkW);
      MTe.setup(t);
      phiE2 = dalC<T>(d << tE);
      beRaw = dalC<T>(((size_t)4 * s.V) << tE);
      tabE = dalC<T>((size_t)s.V << tE);
    }
    for (auto &m : M)
      if (m.K) nymax = std::max(nymax, m.u.ny);
    SC.setup();
    Z.setup(s.lkW, nclMax, nymax);
    As = dalC<T>((size_t)1 << lamMax);
    Bs = dalC<T>((size_t)1 << lamMax);
    Bk = dalC<T>((size_t)1 << lamMax);
    one = dalC<T>(1);
    kS_one<<<1, 1>>>(one);
  }
};
// The caller's inputs, fixed at set-up. Device pointers must stay valid while the sub-proof uses
// them. g: layer group. NC: number of lookup calls. cseg[c]: c's lookup segments (named key lists).
struct In {
  int g = 0, NC = 0, ls = 0, lt = 0;
  std::vector<std::string> cname;
  std::vector<std::vector<std::string>> cseg;
  std::vector<int> TS; // lookup calls with terms in the token sumcheck TSC(zt); ascending
  std::vector<int> lJ;
  // Lookup call c's end point is (zl_c | zj[c] | zt): layer, lJ[c] column and lt token coordinates.
  // ec[c]: the eq table of zj[c]. el[c]: e_c eq(zl_c, .), one element per layer; e_c is c's
  // coefficient in the lookup forest's join. rt[c]: TSC(zt)'s end point; null if c is not in TS.
  std::vector<const T *> zj, el, ec, rt;
  // pos(s, x): column of index x of segment s. off3(U, pt): offset in U3 of matrix U's table at pt.
  std::function<long long(const std::string &, long long)> pos;
  std::function<size_t(const char *, const std::string &)> off3;
  std::function<TenG(const char *)> ten; // a tensor's layout by name: m a xr sel sc lr row xs o
  // kw(lam, w): layer lam's shift named w, one of kg ku kv ko kd kq kr kgR kuR. On a MoE layer kg
  // and ku are the shared expert's shifts, kgR and kuR the routed experts', and kd serves both.
  std::function<int(int, const char *)> kw;
  // Hh: Hh1, then Hh2, the gated products' token sums. U3: "the reads", committed after the merge.
  const T *Hh = nullptr, *U3 = nullptr;
  T *MSG = nullptr;
  // A message slot is an index into MSG. slSCH, slVE, slMSC: first slots of the batched sumcheck,
  // the embedding's value and the merge. slEndT: the first slot of what TSC(zt) shows at its end: R
  // of the NN Lam instances (NN must be 2), sc of the LamM MoE layers, their selectors, NE each.
  size_t slSCH = 0, slVE = 0, slMSC = 0, slEndT = 0;
  int NN = 2;
  T *rr = nullptr; // the caller's 64 elements: `run_stack` writes SCH's challenges, in round order
  // The merge's inner mask: this sub-proof's mask entry in the request's index-mask message.
  const T *c = nullptr;
  // kept(s, i): GPU address of the true value, at rt, of the factor that TSC(zt) does not show in
  // lookup segment s's term (N1, N2: i is the layer; XR: i is lamM, the index among MoE layers).
  std::function<const T *(const std::string &, int)> kept;
  const T *tab[12] = {}; // kind X's limbs summed over "x" with its factor (kind 0: "hd", factor AH)
  size_t ntab[12] = {};  // the low limb is the first ntab[X] / 2 entries, the high limb the rest
  T *tabw[12] = {};      // X = 7, 8: buffers that `post_rows` fills from SH->tab2 with kS_cut
  const T *AH = nullptr, *lfc = nullptr; // AH[64]: kind 0's factor per q or k head (H + Gk), then 0
  const short *ropeC = nullptr, *ropeS = nullptr;
  std::vector<int> rp;                  // per layer: whether RoPE applies
  std::function<const i8 *(int)> cells; // a kind's cells for the group, valid until the next call
  // Only if g == 0: between preE() and run_merge() the caller must run row passes over the
  // embedding Emb's committed cells, 4 V rows of d in order [limb][v][k]. They write each row's
  // inner product with phiE(q) to beRaw(q), q < 2^SH->tE, limbs not joined. tok: the 2^lt tokens.
  const int *tok = nullptr;
  const T *ztE = nullptr, *pk = nullptr; // points over the 2^lt token positions and a row's d cells
};
struct VForm {
  struct Tm {
    int sp;
    size_t x0;
    long long xs;
    HE w;
  };
  std::vector<Tm> t;
  HE c = hE(0);
  // A scalar row: `at` is the GPU address of its value, and af its form in the reads. `shown`:
  // TSC(zt) has shown the row's values; they are public, and af is empty.
  const T *at = nullptr;
  bool shown = false;
  AF af;
};
struct Blk {
  std::string nm;
  int tri = 0, ws = 0, claim = 0, ni = 1, ny = 1, wy = 0, wi = 0, w = 0, lamM = 0, call = 0, v0 = -1, k0 = 0, kind = -1;
  int virt = -1; // which never-sent vector is A (panel "D": 0) or xx ("TD": 1 + lamM); -1: none
  size_t off = 0;
  std::vector<Sy> wt;
  std::vector<VForm> A, B;
  const T *sl = nullptr; // tri: the selector's NE shown values
  VForm xx;
  std::vector<const T *> lo, hi;
  size_t sB = 0;
  std::vector<int> lam; // layer of row i
};
// Main calls, in order: setup(), walk10(), walk11(); in a proof run_stack(), run_merge(), member().
struct Fork { // one layer sub-proof's state for SCH, its weight claims and their merge
  Shared *SH = nullptr;
  In in;
  Shape s;
  std::vector<Blk> bl;
  std::vector<int> AC;
  int NFC = 0, NTRI = 0, NCL = 0, NV = 0, lam_s = 0, nk = 0;
  int nvirt() const { return 1 + (s.mo ? s.LamM : 0); }
  size_t nshown() const { return 2 + (size_t)nvirt() + NCL; }
  // NCL counts the panels' weight claims. NV adds the embedding's claim, number iE(), if in.g == 0.
  int iE() const { return NCL; }
  size_t total = 0;
  Walk VW; // only for the vectors that the forms read; it records no claims
  std::map<int, int> clVK, clEN;
  T *om = nullptr, *wtv = nullptr, *cA = nullptr, *cB = nullptr, *cT = nullptr, *sbv = nullptr, *rr = nullptr,
    *rs = nullptr, *v = nullptr, *sig = nullptr, *uv = nullptr, *ku_kv = nullptr, *hrow = nullptr,
    *kap = nullptr, *bqk = nullptr, *zer = nullptr, *Tf = nullptr, *fvp = nullptr, *erq = nullptr, *psi = nullptr,
    *etE = nullptr;
  SyD *dSy = nullptr, *dSyEnd = nullptr;
  CoD *dCo = nullptr;
  AtG *dAt = nullptr;
  int nAt = 0;
  int nAtStack = 0; // dAt's first nAtStack jobs read rr, the point that `run_stack` draws
  std::vector<AtG> ats;
  std::map<int, T *> EY, sEY;
  std::map<std::pair<int, int>, T *> EI;
  std::vector<T *> xxv;
  std::vector<CClaim> cl;
  std::vector<std::pair<int, int>> clBI; // per claim: its (panel, row i)
  std::map<std::string, T *> HVx;
  std::vector<T *> HK, PK, HE_, PE_;
  static bool isAcc(const std::string &n) {
    return n == "G" || n == "U" || n == "V" || n == "O" || n == "D" || n == "LR";
  }
  static int tOf(int k) { return k < 16 ? 16 - k : 0; }
  int callOf(const std::string &sg) const {
    for (int c = 0; c < in.NC; c++)
      for (auto &n : in.cseg[c])
        if (n == sg) return c;
    printf("internal error: weight-stack step of a layer sub-proof: segment %s is not in "
           "any call of this layer sub-proof\n",
           sg.c_str());
    exit(2);
  }
  // Builds the list of panels without their rows. The largest get the lowest offsets, so that off
  // is a multiple of 2^w. Needs only in_.NC, cseg, TS, g and SH_.s; allocates no device memory.
  void plan(Shared &SH_, const In &in_) {
    SH = &SH_;
    in = in_;
    s = SH_.s;
    bl.clear();
    AC.clear();
    clVK.clear();
    clEN.clear();
    const int Lam = s.Lam, LM = s.LamM;
    for (int c = 0; c < in.NC; c++) {
      bool a = false;
      for (auto &n : in.cseg[c]) a = a || isAcc(n);
      if (a) AC.push_back(c);
    }
    // c_start, SCH's claimed sum (the start form), has NFC summands: 0 is the lookup forest's join
    // (AC's panels); 1, absent if TS is empty, TSC(zt)'s end form (TS's). Blk::claim: the summand.
    for (int c : AC) clVK[c] = 0;
    for (int c : in.TS) clEN[c] = 1;
    NFC = 1 + (in.TS.empty() ? 0 : 1);
    auto add = [&](const std::string &nm, int claim, int call, int ni, int ny, int tri, int ws, int lamM) {
      Blk b;
      b.nm = nm;
      b.claim = claim;
      b.call = call;
      b.ni = ni;
      b.ny = ny;
      b.tri = tri;
      b.ws = ws;
      b.lamM = lamM;
      b.wy = c_clg(ny);
      b.wi = c_clg(ni);
      b.w = b.wy + b.wi;
      bl.push_back(b);
    };
    for (int c : AC)
      for (auto &n : in.cseg[c]) {
        if (n == "G" || n == "U" || n == "V") add(n, clVK[c], c, Lam, s.d, 0, 1, 0);
        if (n == "O") add(n, clVK[c], c, Lam, s.dq, 0, 1, 0);
        if (n == "D") add(n, clVK[c], c, Lam, s.f, 0, 1, 0);
        if (n == "LR") add(n, clVK[c], c, LM, s.d, 0, 1, 0);
      }
    for (int c : in.TS)
      for (auto &n : in.cseg[c]) {
        if (n == "N1" || n == "N2") add(n + "E", clEN[c], c, Lam, 1, 0, 0, 0);
        if (n == "QK") add("QKE", clEN[c], c, Lam, s.d, 0, 1, 0);
        if (!s.mo) continue;
        if (n == "XR") add("XRE", clEN[c], c, LM, 1, 0, 0, 0);
        if (n == "DD" || n == "MR") add(n + "E", clEN[c], c, LM, s.NE, 0, 0, 0);
        if (n == "G" || n == "U" || n == "D")
          for (int lm = 0; lm < LM; lm++) add("T" + n, clEN[c], c, s.NE, n == "D" ? s.fe : s.d, 1, 1, lm);
      }
    std::vector<int> o(bl.size());
    for (size_t i = 0; i < o.size(); i++) o[i] = (int)i;
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return bl[a].w > bl[b].w; });
    total = 0;
    for (int i : o) {
      bl[i].off = total;
      total += (size_t)1 << bl[i].w;
    }
    lam_s = c_clg(total);
    NCL = 0;
    NTRI = 0;
    nk = 0;
    clBI.clear();
    for (size_t q = 0; q < bl.size(); q++) {
      Blk &b = bl[q];
      b.k0 = nk;
      nk += b.tri ? 1 : b.ni;
      NTRI += b.tri;
      if (!b.ws) continue;
      b.v0 = NCL;
      for (int i = 0; i < (b.tri ? 1 : b.ni); i++) clBI.push_back({(int)q, i});
      NCL += b.tri ? 1 : b.ni;
    }
    NV = iE() + (in.g == 0 ? 1 : 0);
  }
  std::vector<std::vector<int>> nyr;
  T *wDD = nullptr;
  // Builds the panels' rows: vectors over MSG, Hh and U3 (Tm::sp = 0, 2, 3), pairs of limb tables
  // (lo, hi), or scalars. vd() does not check VD's limit of four terms per row.
  void setup_forms() {
    const int Lam = s.Lam, d = s.d, f = s.f, fe = s.fe, NE = s.NE, LS = 1 << in.ls;
    HE p256[8];
    p256[0] = hE(1);
    for (int i = 1; i < 8; i++) p256[i] = hmul(p256[i - 1], hE(256));
    VW.init((size_t)1 << 17);
    // The slots of what TSC(zt) has shown.
    const size_t slR = in.slEndT, slSc = slR + (size_t)in.NN * Lam, slSel = slSc + (size_t)s.LamM;
    std::map<std::string, const T *> EWc, EWHc;
    std::map<int, const T *> ehc;
    auto segn = [&](const std::string &sg) {
      return sg == "XR" ? d : sg == "DD" ? NE : sg == "MR" ? 1 : d;
    };
    auto EW = [&](const std::string &sg) {
      auto it = EWc.find(sg);
      if (it != EWc.end()) return it->second;
      std::vector<u32> ix(segn(sg));
      for (size_t x = 0; x < ix.size(); x++) ix[x] = (u32)in.pos(sg, (long long)x);
      return EWc[sg] = VW.gather(in.ec[callOf(sg)], ix);
    };
    auto EWH = [&](const std::string &sg) {
      auto it = EWHc.find(sg);
      if (it != EWHc.end()) return it->second;
      const int c = callOf(sg);
      if (!ehc.count(c)) ehc[c] = VW.eq(in.zj[c], in.lJ[c] - in.ls);
      std::vector<u32> ix(d / LS);
      for (size_t x = 0; x < ix.size(); x++) ix[x] = (u32)(in.pos(sg, (long long)x * LS) / LS);
      return EWHc[sg] = VW.gather(ehc[c], ix);
    };
    auto sh = [&](const char *U, const std::string &pt, const TenG &X, int i) {
      VForm v;
      for (int l = 0; l < X.nl; l++)
        v.t.push_back({3, in.off3(U, pt) + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J), 1, p256[l]});
      return v;
    };
    auto hSx = [&](const std::string &sg, const char *U, const std::string &pt, const TenG &X, int i) {
      AF a;
      const T *vv = EWH(sg);
      for (int l = 0; l < X.nl; l++)
        a = a + rng(3, in.off3(U, pt) + (size_t)(X.off + ((long long)l * X.cnt + i) * X.J) / LS, 1, (size_t)X.J / LS,
                    vv, 1, Sy(p256[l]));
      return a;
    };
    TenG XBT;
    XBT.nl = in.ten("xs").nl; // the number of limbs of a residual-stream value
    XBT.cnt = 1;
    XBT.J = d;
    XBT.off = 0;
    auto hSst = [&](const std::string &sg, int pos, const std::string &pt) {
      if (pos == 0) return hSx(sg, "UX", pt, XBT, 0);
      if (pos == 2 * Lam) return hSx(sg, "UY", pt, XBT, 0);
      return hSx(sg, "UF", pt, in.ten("xs"), pos - 1);
    };
    // shown(): a scalar that TSC(zt) has shown, in slot sl. kept(): one not shown; a: VForm::af.
    auto shown = [&](size_t sl) {
      VForm v;
      v.at = in.MSG + sl;
      v.shown = true;
      return v;
    };
    auto kept = [&](const std::string &sg, int i, const AF &a) {
      VForm v;
      v.at = in.kept(sg, i);
      v.af = a;
      return v;
    };
    auto raw = [&](Blk &b, int kind, size_t rest) {
      b.lo.push_back(in.tab[kind] + rest);
      b.hi.push_back(in.tab[kind] + in.ntab[kind] / 2 + rest);
    };
    kap = dalC<T>((size_t)Lam * s.dh);
    bqk = dalC<T>((size_t)Lam * d);
    zer = dalC<T>(std::max(d, 1));
    kS_zero<<<grid_blocks(d), 256>>>(zer, d);
    nyr.assign(bl.size(), {});
    for (size_t q = 0; q < bl.size(); q++) {
      Blk &b = bl[q];
      const int c = b.call;
      const std::string &n = b.nm, z = "z", r = "r", Yp = "Y"; // names of points, for `in.off3`
      auto EL = [&](int lam) {
        return Sy(in.el[c] + lam);
      };
      // wt_i (Blk::wt[i]) of a row whose shift k is in.kw(lam, word):
      // e_c eq(zl_c, lam) 2^tOf(k).
      auto WT = [&](int lam, const char *word) {
        return EL(lam) * Sy(h2(tOf(in.kw(lam, word))));
      };
      for (int i = 0; i < (b.tri ? 1 : b.ni); i++) {
        const int lam = b.tri                                                                   ? 2 * b.lamM + 1
                        : b.ni == Lam && !(n == "LR" || n == "XRE" || n == "DDE" || n == "MRE") ? i
                                                                                                : 2 * i + 1;
        const bool M = s.mo && (lam & 1);
        b.lam.push_back(lam);
        nyr[q].push_back(b.ny);
        if (n == "G" || n == "U") {
          const bool gt = n == "G";
          b.A.push_back(sh("UD", z, in.ten("m"), lam));
          b.wt.push_back(WT(lam, gt ? "kg" : "ku"));
          raw(b, M ? (gt ? 5 : 6) : (gt ? 3 : 4), (size_t)(s.mo ? lam / 2 : lam) * d);
        } else if (n == "V") {
          b.A.push_back(sh("UD", z, in.ten("a"), lam));
          b.wt.push_back(WT(lam, "kv"));
          raw(b, 1, (size_t)lam * d);
        } else if (n == "O") {
          b.A.push_back(sh("Uq", z, in.ten("o"), lam));
          b.wt.push_back(WT(lam, "ko"));
          raw(b, 2, (size_t)lam * s.dq);
        } else if (n == "D") {
          VForm v;
          v.t.push_back({2, (size_t)lam * f, 1, hE(1)});
          b.A.push_back(v);
          b.virt = 0;
          nyr[q][i] = M ? fe : f; // length of the B side; the A side has all f columns
          b.wt.push_back(WT(lam, "kd"));
          raw(b, 9, s.mo ? (size_t)(lam / 2) * (f + fe) + (size_t)(lam & 1) * f : (size_t)lam * f);
        } else if (n == "LR") {
          b.A.push_back(sh("UD", z, in.ten("m"), lam));
          b.wt.push_back(WT(lam, "kr"));
          raw(b, 11, (size_t)i * d);
        } else if (n == "N1E" || n == "N2E") {
          const int one = n == "N1E" ? 0 : 1;
          b.A.push_back(shown(slR + (size_t)(2 * lam + one))); // R
          b.B.push_back(kept(one ? "N2" : "N1", lam, hSst(one ? "N2" : "N1", 2 * lam + one, Yp)));
          b.wt.push_back(EL(lam));
        } else if (n == "QKE") {
          b.A.push_back(sh("UD", r, in.ten("a"), lam));
          b.wt.push_back(WT(lam, "kq"));
          b.lo.push_back(bqk + (size_t)lam * d);
          b.hi.push_back(zer);
        } else if (n == "XRE") {
          const TenG Xm = in.ten("m");
          AF bb;
          for (int l = 0; l < Xm.nl; l++)
            bb = bb + rng(3, in.off3("UD", r) + (size_t)(Xm.off + ((long long)l * Xm.cnt + lam) * Xm.J), 1, d, EW("XR"),
                          1, Sy(p256[l]));
          b.A.push_back(shown(slSc + (size_t)i));
          b.B.push_back(kept("XR", i, bb));
          b.wt.push_back(EL(lam));
        } else if (n == "DDE" || n == "MRE") {
          {
            VForm v; // the selector's shown values, a vector of public numbers
            v.t.push_back({0, slSel + (size_t)i * NE, 1, hE(1)});
            v.shown = true;
            b.A.push_back(v);
          }
          b.B.push_back(sh("UR", r, in.ten("lr"), i));
          if (n == "DDE") {
            if (!wDD) {
              wDD = VW.take(1);
              const T *e = EW("DD");
              T *o = wDD;
              const int ne = NE;
              VW.prep.push_back([=] { kG_sum1<<<1, 1024>>>(e, ne, o); });
            }
            b.wt.push_back(EL(lam) * Sy(wDD));
          } else
            b.wt.push_back(EL(lam) * Sy(EW("MR")));
        } else if (b.tri) {
          const bool D_ = n == "TD";
          b.sl = in.MSG + slSel + (size_t)b.lamM * NE;
          if (D_) {
            b.xx.t.push_back({2, (size_t)Lam * f + (size_t)b.lamM * fe, 1, hE(1)});
            b.virt = 1 + b.lamM;
          } else
            b.xx = sh("UD", r, in.ten("xr"), b.lamM);
          b.sB = D_ ? fe : d;
          b.kind = D_ ? 10 : n == "TG" ? 7 : 8;
          raw(b, b.kind, (size_t)b.lamM * NE * b.sB);
          b.wt.push_back(WT(lam, D_ ? "kd" : n == "TG" ? "kgR" : "kuR"));
        } else {
          printf("internal error: weight-stack step of a layer sub-proof: unknown block name %s "
                 "in the sub-proof's block list\n",
                 n.c_str());
          exit(2);
        }
      }
    }
    VW.seal(true);
  }
  // Allocates the buffers that setup_forms(), setup_claims() and plan_record() do not. Uploads dSy
  // (the factors of each wt) and dCo (kS_co's inputs): one entry per row, but one per panel with
  // the selector. EY[wy] is the eq table of rr's last wy coordinates and EI[{wy, wi}] that of the
  // wi coordinates before them; run_stack() fills both. A panel's sb is eq(p, off >> w), where p is
  // rr's first lam_s - w coordinates; a job appended to ats computes it. xxv: the `tri` panels' xx.
  void allocate() {
    om = dalC<T>(NFC + 1);
    wtv = dalC<T>(nk);
    cA = dalC<T>(nk);
    cB = dalC<T>(nk);
    cT = dalC<T>(NTRI + 1);
    sbv = dalC<T>(bl.size());
    rr = in.rr;
    rs = dalC<T>(64);
    v = dalC<T>(NV + 1);
    sig = dalC<T>(NTRI + 1);
    uv = dalC<T>(nvirt());
    ku_kv = dalC<T>(nvirt() + NCL);
    hrow = dalC<T>(s.Lam);
    Tf = dalC<T>(SH->K, Place::Result);
    fvp = dalC<T>(1);
    erq = dalC<T>((size_t)1 << in.lt);
    {
      std::vector<SyD> sy(nk);
      std::vector<CoD> co(nk);
      for (auto &b : bl) {
        if (!EY.count(b.wy)) EY[b.wy] = dalC<T>((size_t)1 << b.wy);
        if (!EI.count({b.wy, b.wi})) EI[{b.wy, b.wi}] = dalC<T>((size_t)1 << b.wi);
        const size_t q = &b - &bl[0];
        AtG a;
        a.z = rr;
        a.nb = lam_s - b.w;
        a.idx = b.off >> b.w;
        a.o = sbv + q;
        ats.push_back(a);
        for (int i = 0; i < (b.tri ? 1 : b.ni); i++) {
          SyD &y = sy[b.k0 + i];
          y.c = b.wt[i].c;
          y.n = b.wt[i].n;
          for (int p = 0; p < 4; p++) y.p[p] = b.wt[i].p[p];
          CoD &c = co[b.k0 + i];
          c.sb = sbv + q;
          c.om = om + b.claim;
          c.wt = wtv + b.k0 + i;
          c.ei = b.tri ? nullptr : EI[{b.wy, b.wi}] + i;
        }
        if (b.tri) xxv.push_back(dalC<T>(b.ny));
      }
      dSy = Walk::up(sy);
      dCo = Walk::up(co);
      // kS_end's coefficients: ku of the never-sent vectors' values, then kv of the weight values.
      std::vector<SyD> e(nvirt() + NCL);
      auto put = [&](SyD &y, std::initializer_list<const T *> p) {
        for (const T *x : p) y.p[y.n++] = x;
      };
      int jt = 0;
      for (auto &b : bl) {
        const size_t q = &b - &bl[0];
        if (b.tri) {
          if (b.virt >= 0) put(e[b.virt], {cT + jt, sig + jt});
          put(e[nvirt() + b.v0], {cB + b.k0});
          jt++;
          continue;
        }
        if (b.virt >= 0) put(e[b.virt], {sbv + q, om + b.claim});
        if (!b.ws) continue;
        for (int i = 0; i < b.ni; i++) {
          put(e[nvirt() + b.v0 + i], {cB + b.k0 + i});
          if (b.virt >= 0) put(e[nvirt() + b.v0 + i], {wtv + b.k0 + i});
        }
      }
      dSyEnd = Walk::up(e);
    }
  }
  // X: weight kind; `at`: its map in Shared::M. Not checked here: fe and NE are powers of 2; in.pos
  // must map each aligned run of 2^z indices (`facX`), in order, to an aligned run of columns.
  void setup_claims() {
    const int Lam = s.Lam, d = s.d, f = s.f, fe = s.fe, NE = s.NE;
    cl.assign(NV, CClaim());
    HK.assign(NV, nullptr);
    PK.assign(NV, nullptr);
    HE_.assign(NV, nullptr);
    PE_.assign(NV, nullptr);
    auto facX = [&](CFacD &fd, const std::string &sg, long long x0, size_t R) {
      const int c = callOf(sg), z = c_tz(R), o = (int)(R >> z);
      T *HV = dalC<T>(o);
      for (int hi = 0; hi < o; hi++) {
        AtG a;
        a.z = in.zj[c];
        a.nb = in.lJ[c] - z;
        a.idx = (unsigned long long)(in.pos(sg, x0 + ((long long)hi << z)) >> z);
        a.o = HV + hi;
        ats.push_back(a);
      }
      fd.HV = HV;
      fd.pt = in.zj[c] + (in.lJ[c] - z);
    };
    for (int a = 0; a < NCL; a++) {
      const Blk &b = bl[clBI[a].first];
      const int i = clBI[a].second, lam = b.lam[b.tri ? 0 : i], l = in.g * Lam + lam;
      const bool M = s.mo && (lam & 1);
      const std::string &n = b.nm;
      CClaim &c = cl[a];
      int X = -1, at = -1;
      std::string sg;
      long long x0 = 0;
      size_t R = 0;
      if (n == "G" || n == "U") {
        X = n == "G" ? (M ? 5 : 3) : (M ? 6 : 4);
        at = M ? 5 : 3;
        sg = n;
        R = M ? fe : f;
      } else if (n == "V") {
        X = at = 1;
        sg = n;
        R = s.dkv;
      } else if (n == "O") {
        X = at = 2;
        sg = n;
        R = d;
      } else if (n == "D") {
        X = 9;
        at = M ? 12 : 9;
        sg = n;
        R = d;
      } else if (n == "LR") {
        X = at = 11;
        sg = n;
        R = NE;
      } else if (n == "QKE") {
        X = at = 0;
      } else if (n == "TG" || n == "TU") {
        X = n == "TG" ? 7 : 8;
        at = 7;
        sg = n == "TG" ? "G" : "U";
        x0 = fe;
        R = fe;
      } else if (n == "TD") {
        X = at = 10;
        sg = "D";
        R = d;
      }
      c.map = &SH->M[at];
      c.ui = SH->ui(X, l);
      c.d = c.map->desc();
      const CUnit &u = c.map->u;
      const int ikk = X == 0 ? 2 : 0;
      const CGrp &k = u.g[ikk];
      HK[a] = dalC<T>(k.R >> k.z);
      PK[a] = dalC<T>(k.z + 1);
      c.d.f[ikk].HV = HK[a];
      c.d.f[ikk].pt = PK[a];
      if (X == 0) {
        c.d.f[0].AV = in.AH;
        c.d.f[1].AV = kap + (size_t)lam * s.dh;
        c.mode = c.map->run ? 2 : 0;
        continue;
      }
      facX(c.d.f[u.g.size() - 1], sg, x0, R);
      if (b.tri) {
        const CGrp &e = u.g[1];
        HE_[a] = dalC<T>(1);
        PE_[a] = dalC<T>(e.z + 1);
        c.d.f[1].HV = HE_[a];
        c.d.f[1].pt = PE_[a];
        c.mode = 1;
        c.tab = SH->R2j;
        c.mapT = X == 10 ? &SH->MTdr : &SH->MTgr;
        c.dT = c.mapT->desc();
        c.dT.f[0] = c.d.f[0];
        c.dT.f[1] = c.d.f[1];
        c.dT.f[2].HV = SH->one;
        c.dT.f[2].pt = c.d.f[2].pt;
      } else
        c.mode = c.map->run ? 2 : 0;
    }
    if (NV > iE()) {
      CClaim &c = cl[iE()];
      c.map = &SH->M[13];
      c.ui = SH->ui(12, s.L);
      c.d = c.map->desc();
      psi = dalC<T>(s.V);
      etE = dalC<T>((size_t)1 << in.lt);
      c.d.f[0].AV = psi;
      const int z = c_tz(d), o = d >> z, lPd = c_clg(d);
      T *HV = dalC<T>(o);
      for (int hi = 0; hi < o; hi++) {
        AtG a;
        a.z = in.pk;
        a.nb = lPd - z;
        a.idx = hi;
        a.o = HV + hi;
        ats.push_back(a);
      }
      c.d.f[1].HV = HV;
      c.d.f[1].pt = in.pk + (lPd - z);
      c.mode = 1;
      c.tab = SH->tabE;
      c.mapT = &SH->MTe;
      c.dT = c.mapT->desc();
      c.dT.f[0].AV = psi;
      c.dT.f[1].HV = SH->one;
      c.dT.f[1].pt = in.pk;
    }
    // Moves the nAtStack jobs that read rr to the front, order kept: at_before_run() skips them.
    nAtStack =
        (int)(std::stable_partition(ats.begin(), ats.end(), [&](const AtG &a) { return a.z == rr; }) - ats.begin());
    nAt = (int)ats.size();
    dAt = Walk::up(ats);
    rho_ = dalC<T>(NV);
    rm_ = dalC<T>(64);
    ka_ = dalC<T>(NV);
    beta_ = dalC<T>(1);
  }
  /* Lists the device addresses of `member`'s elements. `run_merge` ends by gathering them into
   * recD: a later sub-proof may overwrite buffers such as in.zj before the host reads them. */
  void plan_record() {
    {
      std::vector<const T *> src;
      src.push_back(fvp);
      src.push_back(beta_);
      for (int j = 0; j < s.lkW; j++) src.push_back(rm_ + j);
      for (int a = 0; a < NV; a++) {
        src.push_back(ka_ + a);
        const CClaimD &dd = cl[a].d;
        for (int i = 0; i < dd.ng; i++) {
          const CFacD &fc = dd.f[i];
          if (fc.cls == 2) {
            if (!fc.take)
              for (unsigned x = 0; x < fc.R; x++) src.push_back(fc.AV + x);
          } else {
            for (unsigned x = 0; x < (fc.R >> fc.z); x++) src.push_back(fc.HV + x);
            const int rem = fc.z - fc.take;
            const T *p0 = fc.cls == 0 ? fc.pt : fc.pt + fc.take;
            for (int x = 0; x < rem; x++) src.push_back(p0 + x);
          }
        }
        recN.push_back(src.size());
      }
      nrec = src.size();
      dSrc = Walk::up(src);
      recD = dalC<T>(nrec, Place::Result);
    }
  }
  void setup(Shared &SH_, const In &in_) {
    plan(SH_, in_);
    if (lam_s > SH->lamMax || NV > SH->nclMax) {
      printf("internal error: weight-stack step of a layer sub-proof: shared buffers too "
             "small for this sub-proof (it needs 2^%d entries and %d claims; the set-up "
             "allowed 2^%d and %d)\n",
             lam_s, NV, SH->lamMax, SH->nclMax);
      exit(2);
    }
    setup_forms();
    allocate();
    setup_claims();
    plan_record();
  }
  VD vd(const VForm &f) const {
    VD o;
    o.n = (int)f.t.size();
    for (int q = 0; q < o.n; q++) {
      o.p[q] = (f.t[q].sp == 0 ? in.MSG : f.t[q].sp == 2 ? in.Hh : in.U3) + f.t[q].x0;
      o.xs[q] = f.t[q].xs;
      o.w[q] = f.t[q].w;
    }
    o.c = f.c;
    return o;
  }
  // Runs the kS_at jobs after the first nAtStack. Their points (in.zj, in.pk) must already be set.
  void at_before_run() {
    const int n = nAt - nAtStack;
    if (n > 0) kS_at<<<(n + 63) / 64, 64>>>(dAt + nAtStack, n);
  }
  T *phiLow(int kind) { return SH->phiL[kind - 7]; }
  T *tab2(int kind) { return SH->tab2[kind - 7]; }
  /* Call after the points in.zj are drawn. Fills phiLow for kind 7 and kind 8 (panels "TG", "TU").
   * The caller's row passes read phiLow to fill tab2, from which `post_rows` fills in.tabw. */
  void pre_rows() {
    if (!s.mo) return;
    at_before_run();
    for (int a = 0; a < NCL; a++) {
      const Blk &b = bl[clBI[a].first];
      if (!b.tri || b.kind == 10 || b.lamM) continue;
      const CFacD &x = cl[a].d.f[2];
      kS_low<<<grid_blocks((size_t)s.fe >> SH->tG), 256>>>(x.HV, x.pt, x.z, x.z - SH->tG, phiLow(b.kind));
    }
  }
  // phiE(q) is valid only after preE(), which fills it if NV > iE(). Both buffers: see struct In.
  T *phiE(int q) { return SH->phiE2 + (size_t)q * s.d; }
  T *beRaw(int q) { return SH->beRaw + (size_t)q * 4 * s.V; }
  void preE() {
    if (NV == iE()) return;
    at_before_run();
    const CFacD &x = cl[iE()].d.f[1];
    kS_phiE<<<grid_blocks((size_t)s.d << SH->tE), 256>>>(x.HV, x.pt, x.z, SH->tE, s.d, SH->phiE2);
  }
  void post_rows() {
    if (!s.mo) return;
    for (int a = 0; a < NCL; a++) {
      const Blk &b = bl[clBI[a].first];
      if (!b.tri || b.kind == 10 || b.lamM) continue;
      const size_t n = in.ntab[b.kind];
      kS_cut<<<grid_blocks(n), 256>>>(tab2(b.kind), n, SH->tG, cl[a].d.f[2].pt, in.tabw[b.kind]);
    }
  }
  // Runs SCH on the transcript tr (`Tr`: prover/hiding/transcript.cuh): draws om, builds As and Bs,
  // runs the rounds, shows the end values A0 | uv | B0 | v (kS_end). Then uv[0] is H1, uv[1 + lamM]
  // is X_lamM, and rr + (lam_s - w) is the point of the lowest w index bits, top bit first.
  void run_stack(Tr *tr) {
    Shared &H = *SH;
    const int Lam = s.Lam, d = s.d;
    T *As = H.As, *Bs = H.Bs;
    const size_t N = (size_t)1 << lam_s;
    kS_om<<<1, 1>>>(tr, om, NFC);
    VW.run_prep();
    kS_sy<<<(nk + 63) / 64, 64>>>(dSy, nk, wtv);
    {
      const int cq = callOf("QK");
      k_eq<<<gsl_blocks_of((size_t)1 << in.lt), 256>>>(in.rt[cq], in.lt, erq);
      for (int lam = 0; lam < Lam; lam++) {
        kS_kap<<<s.dh / 2, 256>>>(in.ropeC, in.ropeS, in.rp[lam], erq, 1 << in.lt, s.dh, in.lfc,
                                  kap + (size_t)lam * s.dh);
        kS_bqk<<<(unsigned)((d + 7) / 8), 256>>>(in.tab[0], in.ntab[0] / 2, lam, d, s.dh, kap + (size_t)lam * s.dh,
                                                 bqk + (size_t)lam * d);
      }
    }
    if (total < N) kG_zero<<<grid_blocks(N - total), 256>>>(As + total, Bs + total, N - total);
    int jt = 0;
    for (size_t q = 0; q < bl.size(); q++) {
      Blk &b = bl[q];
      if (b.tri) {
        kS_vec<<<grid_blocks(b.ny), 256>>>(vd(b.xx), b.ny, xxv[jt]);
        kS_tri<<<grid_blocks((size_t)1 << b.w), 256>>>(b.sl, xxv[jt], b.lo[0], b.hi[0], b.sB, om + b.claim, wtv + b.k0,
                                                b.ni, b.ny, b.wy, b.w, As + b.off, Bs + b.off);
        jt++;
        continue;
      }
      for (int i = 0; i < (1 << b.wi); i++) {
        T *a = As + b.off + ((size_t)i << b.wy), *bb = Bs + b.off + ((size_t)i << b.wy);
        if (i >= b.ni) {
          kG_zero<<<grid_blocks((size_t)1 << b.wy), 256>>>(a, bb, (size_t)1 << b.wy);
          continue;
        }
        const VForm &A = b.A[i];
        const bool sc = A.at != nullptr;
        kS_row<<<grid_blocks((size_t)1 << b.wy), 256>>>(
            sc ? VD() : vd(A), A.at, b.ws || sc ? VD() : vd(b.B[i]), !b.ws && sc ? b.B[i].at : nullptr,
            b.ws ? b.lo[i] : nullptr, b.ws ? b.hi[i] : nullptr, om + b.claim, b.virt >= 0 ? nullptr : wtv + b.k0 + i,
            b.virt >= 0 ? b.ny : nyr[q][i], nyr[q][i], b.wy, a, bb);
      }
    }
    CK(cudaMemcpyAsync(H.Bk, Bs, N * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    // Bk keeps B for the weight claims' values. Panel "D" then takes its wt_i in Bs, not in As.
    for (auto &b : bl)
      if (!b.tri && b.virt >= 0) kS_rows_times<<<grid_blocks((size_t)b.ni << b.wy), 256>>>(Bs + b.off, b.ni, b.wy, wtv + b.k0);
    H.SC.run(1, lam_s, 0, As, Bs, nullptr, tr, in.MSG + in.slSCH, rr, 2);
    kG_rev<<<1, 64>>>(rr, lam_s, rs);
    for (auto &e : EY) k_eq<<<gsl_blocks_of((size_t)1 << e.first), 256>>>(rr + (lam_s - e.first), e.first, e.second);
    for (auto &e : EI)
      k_eq<<<gsl_blocks_of((size_t)1 << e.first.second), 256>>>(rr + (lam_s - e.first.first - e.first.second), e.first.second,
                                                         e.second);
    kS_at<<<(nAt + 63) / 64, 64>>>(dAt, nAt);
    kS_co<<<(nk + 63) / 64, 64>>>(dCo, nk, cA, cB);
    jt = 0;
    for (auto &b : bl) {
      if (b.tri) {
        kG_dot<<<1, 256>>>(EI[{b.wy, b.wi}], b.sl, 0, b.ni, sig + jt);
        if (b.virt >= 0) kG_dot<<<1, 256>>>(EY[b.wy], xxv[jt], 0, b.ny, uv + b.virt);
        CK(cudaMemcpyAsync(cT + jt, cA + b.k0, sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        jt++;
      } else if (b.virt >= 0) { // H1: Hh1's extension at the point of the panel's bits
        kG_dot<<<b.ni, 256>>>(EY[b.wy], in.Hh, (size_t)s.f, b.ny, hrow);
        kG_dot<<<1, 256>>>(EI[{b.wy, b.wi}], hrow, 0, b.ni, uv + b.virt);
      }
    }
    for (auto &b : bl) {
      if (!b.ws) continue;
      if (!b.tri)
        kG_dot<<<b.ni, 256>>>(EY[b.wy], H.Bk + b.off, (size_t)1 << b.wy, b.ny, v + b.v0);
      else {
        kG_dot<<<b.ni, 256>>>(EY[b.wy], H.Bk + b.off, (size_t)1 << b.wy, b.ny, H.Z.gam);
        kG_dot<<<1, 256>>>(EI[{b.wy, b.wi}], H.Z.gam, 0, b.ni, v + b.v0);
      }
    }
    kS_sy<<<(nvirt() + NCL + 63) / 64, 64>>>(dSyEnd, nvirt() + NCL, ku_kv);
    kS_end<<<1, 1>>>(tr, As, Bs, ku_kv, uv, nvirt(), ku_kv + nvirt(), v, NCL);
    CK(cudaGetLastError()); // the slots are sent: a failed launch would leave their old content
  }
  // In layer group 0, first sends the embedding's value in a separate message. Then runs the merge:
  // CEnd::send_merge() (claims_lib.cuh). fvp keeps its last slot, the merged value under its mask.
  void run_merge(Tr *tr) {
    Shared &H = *SH;
    if (NV > iE()) {
      k_eq<<<gsl_blocks_of((size_t)1 << in.lt), 256>>>(in.ztE, in.lt, etE);
      kS_zero<<<grid_blocks(s.V), 256>>>(psi, s.V);
      kS_joinE<<<grid_blocks((size_t)s.V << H.tE), 256>>>(H.beRaw, s.V, H.tE, H.tabE);
      kS_psi<<<1, 1>>>(in.tok, s.V, etE, 1 << in.lt, psi, H.tabE, H.tE, cl[iE()].d.f[1].pt, v + iE());
      kG_sendv<<<1, 1>>>(tr, in.MSG + in.slVE, v + iE(), 1);
    }
    // The factors at the batched sumcheck's point; rs[j] is the coordinate of index bit j. The
    // first wy coordinates give (HK, PK) on the index group "kk"; in a `tri` panel the next wi give
    // (HE_, PE_) on "e". Then the tables: mode 1 claim by claim, as the routed experts' claims
    // share R2j; modes 0 and 2 by weight kind, as the next in.cells() may overwrite a kind's cells.
    for (int a = 0; a < NCL; a++) {
      const Blk &b = bl[clBI[a].first];
      const CUnit &u = cl[a].map->u;
      const CGrp &k = u.g[u.X == 0 ? 2 : 0];
      kG_facE<<<1, 64>>>(rs, b.wy, k.z, (int)(k.R >> k.z), HK[a], PK[a]);
      if (b.tri) kG_facE<<<1, 64>>>(rs + b.wy, b.wi, u.g[1].z, 1, HE_[a], PE_[a]);
    }
    H.Z.cl = cl;
    H.Z.NCL = NV;
    H.Z.TH_ = Tf;
    for (int a = 0; a < NV; a++) {
      if (cl[a].mode != 1) continue;
      if (a < NCL) {
        const Blk &b = bl[clBI[a].first];
        if (b.kind == 10)
          kS_join<<<grid_blocks((size_t)b.ni * b.sB), 256>>>(b.lo[0], b.hi[0], (size_t)b.ni * b.sB, H.R2j);
        else {
          const size_t n = ((size_t)b.ni * b.sB) << H.tG, n1 = (in.ntab[b.kind] / 2) << H.tG;
          kS_join<<<grid_blocks(n), 256>>>(tab2(b.kind) + b.lamM * n, tab2(b.kind) + n1 + b.lamM * n, n, H.R2j);
        }
      }
      H.Z.tables(a, a + 1);
    }
    {
      static const int order[9] = {0, 1, 2, 3, 4, 5, 6, 9, 11};
      for (int X : order) {
        const i8 *base = nullptr;
        for (int a = 0; a < NCL; a++) {
          if (cl[a].mode == 1) continue;
          const Blk &b = bl[clBI[a].first];
          const int lam = b.lam[clBI[a].second];
          const bool M = s.mo && (lam & 1);
          const std::string &n = b.nm;
          const int Xa = n == "G"    ? (M ? 5 : 3)
                         : n == "U"  ? (M ? 6 : 4)
                         : n == "V"  ? 1
                         : n == "O"  ? 2
                         : n == "D"  ? 9
                         : n == "LR" ? 11
                                     : 0;
          if (Xa != X) continue;
          if (!base) base = in.cells(X);
          H.Z.cl[a].cells = base + H.base(X, lam);
          H.Z.tables(a, a + 1);
        }
      }
    }

    H.Z.send_merge(H.SC, tr, in.MSG + in.slMSC, in.c);
    CK(cudaGetLastError()); // the slots are sent: a failed launch would leave their old content
    CK(cudaMemcpyAsync(fvp, in.MSG + in.slMSC + H.Z.merge_slots() - 1, sizeof(T), cudaMemcpyDeviceToDevice,
                       cudaStreamPerThread));
    CK(cudaMemcpyAsync(rho_, H.Z.rho, NV * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    CK(cudaMemcpyAsync(rm_, H.Z.rm, s.lkW * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    CK(cudaMemcpyAsync(ka_, H.Z.ka, NV * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    CK(cudaMemcpyAsync(beta_, H.Z.beta, sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    kS_gat<<<grid_blocks(nrec), 64>>>(dSrc, nrec, recD);
  }
  T *rho_ = nullptr, *rm_ = nullptr, *ka_ = nullptr, *beta_ = nullptr, *recD = nullptr;
  const T **dSrc = nullptr;
  size_t nrec = 0;
  std::vector<size_t> recN;
  // After run_merge(), returns the inputs of the pending claim on the committed weights: the
  // merge's shown value fvp, mask factor beta and point rm (lkW elements), NV, then per claim ui,
  // ka and row factors. NV, ui: one u32. An element: EDEG coordinates, two u32 each, low first.
  std::vector<u32> member() const {
    std::vector<T> h(nrec);
    CK(cudaMemcpy(h.data(), recD, sizeof(T) * nrec, cudaMemcpyDeviceToHost));
    std::vector<u32> w;
    auto put = [&](const T &x) {
      for (int k = 0; k < EDEG; k++) {
        w.push_back((u32)coord(x, k));
        w.push_back((u32)(coord(x, k) >> 32));
      }
    };
    size_t at = 0;
    for (; at < 2 + (size_t)s.lkW; at++) put(h[at]);
    w.push_back((u32)NV);
    for (int a = 0; a < NV; a++) {
      w.push_back((u32)cl[a].ui);
      for (; at < recN[a]; at++) put(h[at]);
    }
    return w;
  }
  // Returns w times the form of a panel's row f, without its constant: f.af for a scalar, else the
  // sum of f's terms over n entries weighted by e. A row of shown values is public and has no form.
  AF vecAF(const VForm &f, size_t n, const T *e, Sy w) const {
    AF a;
    if (f.shown) return a;
    if (f.at) return f.af * w;
    for (auto &t : f.t) a = a + rng(t.sp, t.x0, t.xs, n, e, 1, Sy(t.w) * w);
    return a;
  }
  // Adds SCH's three claims to W (`Walk`, prover/hiding/walk.cuh: the sub-proof's linear claims):
  // the one that SCH leaves, and that A0 and B0 equal their forms (see kS_end). R: SCH as the
  // caller registered it by W.psc(1, the addresses in.rr + j, in.slSCH, 2, nshown(), nullptr); only
  // its two counts are checked. gClaim must contain c_start's NFC summands, in plan()'s order.
  void walk10(Walk &W, const Rec &R, const std::vector<AF> &gClaim) {
    if ((int)gClaim.size() != NFC) {
      printf("internal error: weight-stack step of a layer sub-proof: got %zu summands of the start form c_start from the "
             "calling code, expected %d\n", gClaim.size(), NFC);
      exit(2);
    }
    if (R.shown != nshown() || R.nr != lam_s) {
      printf("internal error: weight-stack step of a layer sub-proof: the calling code has registered the sum-check with %zu "
             "shown values and %d rounds, expected %zu and %d\n",
             R.shown, R.nr, nshown(), lam_s);
      exit(2);
    }
    AF cs0 = gClaim[0]; // om[0] is 1; not multiplied in, as a `Sy` takes at most four GPU factors
    for (int i = 1; i < NFC; i++) cs0 = cs0 + gClaim[i] * Sy(om + i);
    W.row(R, cs0);
    AF Alin, Blin;
    int jt = 0;
    for (size_t q = 0; q < bl.size(); q++) {
      const Blk &b = bl[q];
      const T *ey = EY.at(b.wy);
      if (b.tri) {
        if (b.virt < 0) Alin = Alin + vecAF(b.xx, b.ny, ey, Sy(cT + jt) * Sy(sig + jt));
        jt++;
        continue;
      }
      if (b.virt >= 0) continue;
      for (int i = 0; i < b.ni; i++) {
        Alin = Alin + vecAF(b.A[i], nyr[q][i], ey, Sy(cA + b.k0 + i));
        if (!b.ws) Blin = Blin + vecAF(b.B[i], nyr[q][i], ey, Sy(cB + b.k0 + i));
      }
    }
    W.lin(W.shown(R, 0) - Alin);
    W.lin(W.shown(R, 1 + (size_t)nvirt()) - Blin);
    RS = R;
  }
  Rec RS;
  // Call after walk10(). Adds the merge's claim to W. It starts from the NV values weighted by
  // rho_: the batched sumcheck's NCL and, in layer group 0, the embedding's. The merge's one shown
  // value is in no form: the claim on it leaves the sub-proof (CEnd::send_merge()).
  void walk11(Walk &W) {
    std::vector<const T *> r(s.lkW);
    for (int j = 0; j < s.lkW; j++) r[j] = rm_ + j;
    const Rec R = W.psc(1, r, in.slMSC, 1, 1, nullptr);
    AF start;
    for (int a = 0; a < NCL; a++) start = start + W.shown(RS, 2 + (size_t)nvirt() + a) * Sy(rho_ + a);
    if (NV > iE()) start = start + slot(in.slVE, Sy(rho_ + iE()));
    W.row(R, start);
  }
};
} // namespace gs
