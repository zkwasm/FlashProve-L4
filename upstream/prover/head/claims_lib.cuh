// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The prover's side of a sub-proof's claims on W, the committed weight matrix, and of the merge,
   which reduces them to one claim that is proved later. A sub-proof (`fork` in identifiers) is the
   argument for one part of the forward pass.

   A unit is a tensor's rows in W. A cell is one signed byte of W, and a weight is split into limbs
   of one cell each. Claim a states the value of sum_c eps_a[c] * T_a[c], where
   T_a[c] = sum_y gam_a[y] * cell[y][c], over its unit's columns c and rows y. The merge is the
   masked sumcheck (PSC) of sum_a rho_a * sum_c eps_a[c] * T_a[c] (rho_a: kG_rho).

   A slot is one element of the sub-proof's message array and takes one sent scalar. Slot i's mask
   entry is entry i of the sub-proof's committed mask row. The kernels compute unmasked values;
   namespace sc_door (prover/hiding/sumcheck_door.cuh) masks, stores and sends them as PSC
   prescribes (blocks/iop/psc.h). One exception: kG_merge_end masks its value itself.
   Only the set-up functions allocate memory. This header is not self-contained: it needs, among
   others, prover/hiding/transcript.cuh (T, an element of the extension field E; F, E's arithmetic;
   Tr, the Fiat-Shamir transcript) and prover/prover/forward.cuh (GRID_LOOP, a grid-stride loop; `grid_blocks`,
   its number of blocks). */
#pragma once
#include "common/element_from_hash.h"
#include "hiding/rotation.h"
#include <vector>
#include <string>
#include <algorithm>
// GPU memory for n elements, plus 256 bytes. `Place` (prover/prover/fork_memory.h) states how long
// a buffer is valid and who shares it.
template <class X> static X *dalC(size_t n, Place place = Place::Own) {
  X *p_ = nullptr;
  CK(fork_alloc((void **)&p_, n * sizeof(X) + 256, place));
  return p_;
}
// ceil(log2 R)
static inline int c_clg(size_t R) {
  int b = 0;
  while (((size_t)1 << b) < R) b++;
  return b;
}
// trailing zero bits of R; R must not be 0
static inline int c_tz(size_t R) {
  int z = 0;
  while (!((R >> z) & 1)) z++;
  return z;
}
/* One index group of a unit: an index with R values. `CUnit::lay` moves `take` of its bits into the
   column number, from bit cb0 up, and leaves `left` values for the row number. For classes 'L' and
   'H', 2^z is the largest power of two that divides R: 'L' moves bits 0 .. take - 1, 'H' moves bits
   z - take .. z - 1. 'A' has z = 0 and moves all ceil(log2 R) bits if `lay` has that many left,
   otherwise none. In a table's layout (CClaim::mapT), stride counts elements of tab, not cells. */
struct CGrp {
  const char *nm = "";
  size_t R = 1;
  char cls = 'L';
  size_t stride = 0; // cells of the prover's storage from one index value to the next
  int z = 0, take = 0, cb0 = 0;
  size_t left = 1;
  // The index value with column part col and row part lf.
  size_t index(size_t col, size_t lf) const {
    if (cls == 'A') return take ? col : lf;
    if (cls == 'L') return (lf << take) | col;
    const int rem = z - take;
    return ((lf >> rem) << z) | (col << rem) | (lf & (((size_t)1 << rem) - 1));
  }
};
// A unit's layout in W: its index groups, and how their bits are split between W's rows and
// columns.
struct CUnit {
  // X: the kind of tensor; l: not used; nlw: limbs per weight; ncb: column bits in use; lk: log2 of
  // W's number of columns.
  int X = 0, l = 0, nlw = 2, ncb = 0, lk = 0;
  // At most four groups, which nothing checks.
  std::vector<CGrp> g;
  // ny: the unit's number of rows divided by nlw. limbStride: cells from one limb of a weight to
  // the next.
  size_t ny = 1, limbStride = 0;
  // Gives each group in turn its column bits, out of lk_ bits; computes ny and ncb.
  void lay(int lk_) {
    lk = lk_;
    int left = lk, cb = 0;
    ny = 1;
    for (auto &q : g) {
      q.z = q.cls == 'A' ? 0 : c_tz(q.R);
      q.take = 0;
      q.cb0 = cb;
      if (q.cls == 'A') {
        const int b = c_clg(q.R);
        if (b <= left) q.take = b;
        if (q.R == 1) q.take = 0;
        q.left = q.take || q.R == 1 ? 1 : q.R;
      } else {
        q.take = std::min(q.z, left);
        q.left = q.R >> q.take;
      }
      left -= q.take;
      cb += q.take;
      ny *= q.left;
    }
    ncb = cb;
  }
  // Writes the mixed-radix digits of row y (without the limb) to lf. The last group is the most
  // significant.
  void digits(size_t y, size_t *lf) const {
    for (size_t i = 0; i < g.size(); i++) {
      lf[i] = y % g[i].left;
      y /= g[i].left;
    }
  }
  // Offsets into CClaim::cells or tab (see kG_Tcells). colOff() returns false if the unit has no
  // cell in column c.
  size_t rowOff(size_t y) const {
    size_t lf[4], o = 0;
    digits(y, lf);
    for (size_t i = 0; i < g.size(); i++) o += g[i].stride * g[i].index(0, lf[i]);
    return o;
  }
  bool colOff(size_t c, size_t &o) const {
    o = 0;
    if (c >> ncb) return false;
    for (auto &q : g) {
      const size_t col = (c >> q.cb0) & (((size_t)1 << q.take) - 1);
      if (q.cls == 'A' && q.take && col >= q.R) return false;
      o += q.stride * q.index(col, 0);
    }
    return true;
  }
};
static inline CGrp c_grp(const char *nm, size_t R, char cls, size_t stride) {
  CGrp q;
  q.nm = nm;
  q.R = R;
  q.cls = cls;
  q.stride = stride;
  return q;
}
// A warp's sum by shuffles, valid in lane 0. The idea is public: a reduction tree of shuffles:
// J. Luitjens, "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014, "Shuffle Warp
// Reduce", https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// All 32 lanes of a warp must call it together: the shuffles' mask is ~0u.
__device__ __forceinline__ T kG_wred(T v) {
  for (int o = 16; o; o >>= 1) v = F::add(v, shfl_down<F>(v, o));
  return v;
}
/* One sumcheck round. It requires 256 threads per block. Added over b < gridDim.x, part[2*b] gives
   the sum over q < H of x[q]*y[q], and part[2*b+1] that of (x[q+H]-x[q])*(y[q+H]-y[q]), both summed
   over the na array pairs (x, y) = (X + a*st, Y + a*st). Pair a is multiplied by co[a] if co is
   non-null. */
// A block's sum. The idea is public: a two-level reduction, by warps: J. Luitjens, "Faster Parallel
// Reductions on Kepler", "Block Reduce" (cited at kG_wred()). There shuffles in the first warp
// combine the warps' partial sums; here one thread adds them.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(256)
    kG_sum(int na, size_t H, size_t st, const T *X, const T *Y, const T *co, T *part) {
  __shared__ T sh[2][8];
  T g0 = F::zero(), gi = F::zero();
  for (int a = 0; a < na; a++) {
    const T *x = X + a * st, *y = Y + a * st;
    T s0 = F::zero(), si = F::zero();
    for (size_t q = blockIdx.x * (size_t)256 + threadIdx.x; q < H; q += (size_t)gridDim.x * 256) {
      const T x0 = x[q], x1 = x[q + H], y0 = y[q], y1 = y[q + H];
      s0 = F::add(s0, F::mul(x0, y0));
      si = F::add(si, F::mul(F::sub(x1, x0), F::sub(y1, y0)));
    }
    if (co) {
      s0 = F::mul(co[a], s0);
      si = F::mul(co[a], si);
    }
    g0 = F::add(g0, s0);
    gi = F::add(gi, si);
  }
  g0 = kG_wred(g0);
  gi = kG_wred(gi);
  if ((threadIdx.x & 31) == 0) {
    sh[0][threadIdx.x >> 5] = g0;
    sh[1][threadIdx.x >> 5] = gi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T a = F::zero(), b = F::zero();
    for (int w = 0; w < 8; w++) {
      a = F::add(a, sh[0][w]);
      b = F::add(b, sh[1][w]);
    }
    part[2 * blockIdx.x] = a;
    part[2 * blockIdx.x + 1] = b;
  }
}
// Begins a sumcheck of `rounds` rounds of degree 2: sc_door::begin() sends the sum of the hiding
// polynomial U, a random polynomial, and draws the challenge eps; the rounds run on U + eps *
// (summand). An end value f (a map's value at the rounds' challenges) that is shown is sent as
// f + beta c: c is its inner mask, and beta the challenge of a last round, in a further variable Y.
// degree_in_y: the degree of the summand in the maps whose end values will be shown, 0 if none will
// be. The claimed sum 0 is a placeholder: a round of Form::SUM2 sends the two coefficients that its
// check does not fix (kG_sum's two sums), so nothing that is sent depends on the claim. Launch: one
// thread.
__global__ void kG_begin(Tr *tr, T *slots, int rounds, int degree_in_y) {
  Tr t = *tr;
  (void)sc_door::begin<sc_door::Form::SUM2>(
      sc_door::Site::PAIRS, t, slots, rounds, [](int) { return F::zero(); }, F::zero(), degree_in_y);
  *tr = t;
}
// Adds kG_sum's nb pairs of partial sums, sends the two sums as the message of round `round` and
// writes the round's challenge to r[0]. It requires one block of 256 threads.
// A block's sum. The idea is public: J. Luitjens, "Faster Parallel Reductions on Kepler", "Block
// Reduce" (cited at kG_wred()); for what differs, see kG_sum().
// The compiler hint is public: __launch_bounds__: NVIDIA's Best Practices Guide (cited at
// kG_sum()).
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void __launch_bounds__(256) kG_msg(int nb, const T *part, Tr *tr, int round, T *r) {
  __shared__ T sh[2][8];
  T a = F::zero(), b = F::zero();
  for (int i = threadIdx.x; i < nb; i += 256) {
    a = F::add(a, part[2 * i]);
    b = F::add(b, part[2 * i + 1]);
  }
  a = kG_wred(a);
  b = kG_wred(b);
  if ((threadIdx.x & 31) == 0) {
    sh[0][threadIdx.x >> 5] = a;
    sh[1][threadIdx.x >> 5] = b;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    T q0 = F::zero(), qi = F::zero();
    for (int w = 0; w < 8; w++) {
      q0 = F::add(q0, sh[0][w]);
      qi = F::add(qi, sh[1][w]);
    }
    const T g[2] = {q0, qi};
    Tr t = *tr;
    r[0] = sc_door::round<sc_door::Form::SUM2, sc_door::Draw::COIN>(sc_door::Site::PAIRS, round, t, g);
    *tr = t;
  }
}
// Folds each of the na array pairs in place at the challenge r[0].
// The loop form is public: the grid-stride loop (cited at kG_sum()).
__global__ void kG_fold(int na, size_t H, size_t st, T *X, T *Y, const T *r) {
  const T rr = r[0];
  for (int a = blockIdx.y; a < na; a += gridDim.y) {
    T *x = X + a * st, *y = Y + a * st;
    for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x; q < H; q += (size_t)gridDim.x * blockDim.x) {
      const T x0 = x[q], y0 = y[q];
      x[q] = F::add(x0, F::mul(rr, F::sub(x[q + H], x0)));
      y[q] = F::add(y0, F::mul(rr, F::sub(y[q + H], y0)));
    }
  }
}
// The sumcheck of sum_a co[a] * sum_q X_a[q] * Y_a[q], over many array pairs at once. Used by the
// merge (CEnd::send_merge()) and, with one pair, by the layer sub-proofs' sumcheck that reads the
// weights (prover/head/stack.cuh).
struct CSumcheck {
  T *part = nullptr;
  // most blocks in a launch of kG_sum
  int nblk = 528;
  void setup() { part = dalC<T>(2 * (size_t)nblk); }
  static size_t slots(int n, size_t shown, int degree_in_y) {
    return sc_door::slots(sc_door::Form::SUM2, n, shown, degree_in_y);
  }
  /* X and Y each hold na arrays of 2^n elements at stride st, which are folded in place. co holds
   * one coefficient per array pair; a null co means all 1. msg is the sumcheck's first slot;
   * `slots` gives their number. rr[i] receives the challenge of round i, which binds index bit
   * n - 1 - i. On return X_a[0] and Y_a[0] hold each array's multilinear extension at the point rr.
   * If degree_in_y is not 0, the caller must still run the sumcheck's end: the round in Y, then
   * the end values. */
  void run(int na, int n, size_t st, T *X, T *Y, const T *co, Tr *tr, T *msg, T *rr, int degree_in_y) {
    kG_begin<<<1, 1>>>(tr, msg, n, degree_in_y);
    for (int i = 0; i < n; i++) {
      const size_t H = (size_t)1 << (n - 1 - i);
      const int nb = (int)std::min<size_t>(nblk, (H + 255) / 256);
      kG_sum<<<nb, 256>>>(na, H, st, X, Y, co, part);
      kG_msg<<<1, 256>>>(nb, part, tr, i, rr + i);
      kG_fold<<<(dim3((unsigned)std::min<size_t>(1024, (H + 255) / 256), (unsigned)std::min(na, 16))), 256>>>(
          na, H, st, X, Y, rr + i);
    }
  }
};
// A claim's factor for one index group. Index value x has the coefficient AV[x] in class 'A', and
// otherwise HV[x >> z] times the product over the bits v < z of eq(pt[z - 1 - v], bit v of x) (eq:
// kG_eq1).
struct CFacD {
  const T *HV = nullptr, *pt = nullptr, *AV = nullptr;
  int cls = 0 /* class: 0 = L, 1 = H, 2 = A */, z = 0, take = 0, cb0 = 0;
  unsigned R = 1, left = 1;
};
// One claim's factors, one for each of its ng groups, with CUnit's ncb and lk.
struct CClaimD {
  CFacD f[4];
  int ng = 0, ncb = 0, lk = 0;
};
// eq(z, bit): z if the bit is set, otherwise 1 - z.
__device__ __forceinline__ T kG_eq1(T z, int bit) {
  return bit ? z : F::sub(F::one(), z);
}
// ep[c] = eps_a[c], the claim's coefficient for column c < 2^lk: the product of the groups' factors
// (CFacD) for the index bits in c, or 0 if the unit has no cell in column c. Each block first
// builds small tables in shared memory: the column bits of each 'L' or 'H' group are cut into runs
// of at most six, and a run's table has the products of kG_eq1 over its bits. It requires blocks of
// 256 threads and at most 33 column bits in all, so that four groups have at most eight runs.
// Nothing checks that here, but lk bounds the column bits, and the 2^lk field elements of ep, 24
// bytes each, do not fit in GPU memory at lk = 33. The kernel reads the claim's factors and the
// layout, and no cell.
// The kernel is our own.
// Public source: a few threads fill shared tables: Harris's slides "Optimizing CUDA", as cited at
// k_s3_ri (prover/attn/engines_s3.cuh).
// Public source: the split of eq(z, x) into eq(z1, x1) eq(z2, x2): Q. Dao, J. Thaler, "More
// Optimizations to Sum-Check Proving", IACR ePrint 2024/1210, section 1.1, equation (4),
// https://eprint.iacr.org/2024/1210.pdf; prover/hiding/gram_products.cuh uses the same split
// (eqlo).
// The loop form: the grid-stride loop (cited at kG_sum()).
__global__ void __launch_bounds__(256) kG_eps(CClaimD d, T *ep) {
  constexpr int RUN = 6, RUNS = 8; // a run's bits; the most runs: see above
  __shared__ T table[RUNS][1 << RUN];
  __shared__ int run_from[RUNS], run_mask[RUNS]; // a run's first bit in c, and the mask of its bits
  __shared__ T point[4][32];                     // the groups' points, fetched in parallel
#pragma unroll
  for (int i = 0; i < 4; i++)
    if (i < d.ng && d.f[i].take && d.f[i].cls != 2 && (int)threadIdx.x >> 5 == i && (int)threadIdx.x % 32 < d.f[i].z)
      point[i][threadIdx.x % 32] = d.f[i].pt[threadIdx.x % 32];
  __syncthreads();
  int runs = 0;
#pragma unroll // so that d.f is indexed by constants and needs no stack frame
  for (int i = 0; i < 4; i++) {
    if (i >= d.ng || !d.f[i].take || d.f[i].cls == 2) continue;
    for (int b0 = 0; b0 < d.f[i].take; b0 += RUN, runs++) {
      const int bits = min(RUN, d.f[i].take - b0);
      if (threadIdx.x == 0) run_from[runs] = d.f[i].cb0 + b0, run_mask[runs] = (1 << bits) - 1;
      // 64 threads fill each run's table, four runs at a time: hence the 256 threads.
      if ((int)(threadIdx.x >> RUN) == (runs & 3) && (int)(threadIdx.x & 63) < 1 << bits) {
        const int e = threadIdx.x & 63;
        T r = F::one();
        for (int b = 0; b < bits; b++) {
          const int lob = d.f[i].cls == 0 ? b0 + b : d.f[i].z - d.f[i].take + b0 + b;
          const int at = d.f[i].z - 1 - lob;
          r = F::mul(r, kG_eq1(at < 32 ? point[i][at] : d.f[i].pt[at], (e >> b) & 1));
        }
        table[runs][e] = r;
      }
    }
  }
  __syncthreads();
  GRID_LOOP(c, (size_t)1 << d.lk) {
    T r = runs ? table[0][(c >> run_from[0]) & run_mask[0]] : F::one();
    for (int k = 1; k < runs; k++) r = F::mul(r, table[k][(c >> run_from[k]) & run_mask[k]]);
#pragma unroll
    for (int i = 0; i < 4; i++)
      if (i < d.ng && d.f[i].take && d.f[i].cls == 2) {
        const size_t col = (c >> d.f[i].cb0) & (((size_t)1 << d.f[i].take) - 1);
        r = F::mul(r, col >= d.f[i].R ? F::zero() : d.f[i].AV[col]);
      }
    ep[c] = (c >> d.ncb) ? F::zero() : r;
  }
}
// gam[y] is the claim's coefficient for row y < ny: gam_a[y] without the limb's factor 256^l.
__global__ void kG_gam(CClaimD d, size_t ny, T *gam) {
  GRID_LOOP(y0, ny) {
    size_t y = y0;
    T r = F::one();
    for (int i = 0; i < d.ng; i++) {
      const CFacD &f = d.f[i];
      const unsigned lf = (unsigned)(y % f.left);
      y /= f.left;
      if (f.cls == 2) {
        if (!f.take && f.R > 1)
          r = F::mul(r, f.AV[lf]);
        else if (!f.take && f.R == 1)
          r = F::mul(r, f.AV[0]);
      } else {
        const int rem = f.z - f.take;
        r = F::mul(r, f.HV[lf >> rem]);
        for (int b = 0; b < rem; b++) {
          const int lob = f.cls == 0 ? f.take + b : b;
          r = F::mul(r, kG_eq1(f.pt[f.z - 1 - lob], (lf >> b) & 1));
        }
      }
    }
    gam[y0] = r;
  }
}
/* T_a from the cells (CClaim's mode 0), as partial sums:
     sum over s < gridDim.y of part[s*K + c]
       = sum over y < ny of gam[y] * sum_l 256^l * cells[rowOff[y] + colOff[c] + l*limbStride]
   plus an offset that kG_Tfin removes: each limb pair gets +32896 (128 + 128*256), so that the
   integer factor is never negative. Up to 4,096 products are accumulated before one reduction
   (F::smac, F::sred in prover/hiding/pt_field.h, which allow (terms) x (greatest integer
   factor) <= 2^32; here it is 2^12 x 2^16). colOff[c] = ~0 marks a column without cells. NL, the
   number of limbs, is 2 or 4. The launch needs a thread for each column c < K. */
template <int NL>
__global__ void kG_Tcells(const i8 *cells, size_t limbStride, const unsigned long long *rowOff, const T *gam, int ny,
                          const unsigned long long *colOff, size_t K, T *part) {
  const size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (c >= K) return;
  const int per = (ny + gridDim.y - 1) / gridDim.y, y0 = blockIdx.y * per, y1 = min(ny, y0 + per);
  const unsigned long long co = colOff[c];
  T *o = part + (size_t)blockIdx.y * K + c;
  if (co == ~0ull) {
    *o = F::zero();
    return;
  }
  T acc = F::zero();
  for (int yb = y0; yb < y1; yb += 4096) {
    F::SAcc a = F::szero(), b = F::szero();
    const int ye = min(y1, yb + 4096);
    for (int y = yb; y < ye; y++) {
      const size_t at = rowOff[y] + co;
      const F::SP e = F::sprep(gam[y]);
      F::smac(a, e, (u32)((int)cells[at] + 256 * (int)cells[at + limbStride] + 32896));
      if (NL == 4)
        F::smac(b, e, (u32)((int)cells[at + 2 * limbStride] + 256 * (int)cells[at + 3 * limbStride] + 32896));
    }
    acc = F::add(acc, F::sred(a));
    if (NL == 4) acc = F::add(acc, F::mul(F::sred(b), F::small(65536)));
  }
  *o = acc;
}
// Adds kG_Tcells' ns partial sums per column and subtracts the limb offset times gsum[0], the sum
// of gam.
__global__ void kG_Tfin(const T *part, int ns, size_t K, const unsigned long long *colOff, const T *gsum, int nl,
                        T *out) {
  const T off = F::mul(gsum[0], F::small(nl == 4 ? 32896ull * 65537ull : 32896ull));
  GRID_LOOP(c, K) {
    if (colOff[c] == ~0ull) {
      out[c] = F::zero();
      continue;
    }
    T a = F::zero();
    for (int s = 0; s < ns; s++) a = F::add(a, part[(size_t)s * K + c]);
    out[c] = F::sub(a, off);
  }
}
// gw[lf], for lf < o * 2^rem: kG_gam's product for a class-'H' group at row digit lf. A null HV
// means all 1.
__global__ void kG_gw(const T *HV, const T *pt, int z, int rem, int o, T *gw) {
  GRID_LOOP(i, (size_t)o << rem) {
    T r = HV ? HV[i >> rem] : F::one();
    for (int b = 0; b < rem; b++) r = F::mul(r, kG_eq1(pt[z - 1 - b], (i >> b) & 1));
    gw[i] = r;
  }
}
// gk[lf], for lf < left: the same product for a class-'L' group. Here HV must not be null.
__global__ void kG_gk(const T *HV, const T *pt, int z, int take, int left, T *gk) {
  const int rem = z - take;
  GRID_LOOP(i, left) {
    T r = HV[i >> rem];
    for (int b = 0; b < rem; b++) r = F::mul(r, kG_eq1(pt[z - 1 - (take + b)], (i >> b) & 1));
    gk[i] = r;
  }
}
/* T_a from a table (mode 1): out[c] = sum over y < ny of gam[y] * tab[rowOff[y] + colOff[c]]. tab
   has field elements, with the limbs already combined. */
__global__ void kG_Ttab(const T *tab, const unsigned long long *rowOff, const T *gam, int ny,
                        const unsigned long long *colOff, size_t K, T *out) {
  GRID_LOOP(c, K) {
    const unsigned long long co = colOff[c];
    if (co == ~0ull) {
      out[c] = F::zero();
      continue;
    }
    T a = F::zero();
    for (int y = 0; y < ny; y++) a = F::add(a, F::mul(gam[y], tab[rowOff[y] + co]));
    out[c] = a;
  }
}
// o[0] = sum of v[0..n). It requires one block of 1024 threads.
// A block's sum. The idea is public: J. Luitjens, "Faster Parallel Reductions on Kepler", "Block
// Reduce" (cited at kG_wred()); for what differs, see kG_sum().
// The compiler hint is public: __launch_bounds__: NVIDIA's Best Practices Guide (cited at
// kG_sum()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at kG_msg()).
__global__ void __launch_bounds__(1024) kG_sum1(const T *v, size_t n, T *o) {
  __shared__ T sh[32];
  T a = F::zero();
  for (size_t i = threadIdx.x; i < n; i += 1024) a = F::add(a, v[i]);
  a = kG_wred(a);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    T s = F::zero();
    for (int w = 0; w < 32; w++) s = F::add(s, sh[w]);
    o[0] = s;
  }
}
// One unit layout's offset tables in GPU memory and mode 2's data (see CClaim). Claims share it by
// pointer (CClaim::map, mapT).
struct CMap {
  CUnit u;
  unsigned long long *rowOff = nullptr, *colOff = nullptr;
  size_t K = 0;
  int run = 0, ikk = 0, o = 1, z = 0, rem = 0;
  u32 *colTab = nullptr;
  /* Sets run = 1 and builds colTab if the kernels of mode 2 can read the unit (o, z and rem:
   * prover/head/claims_segments.cuh; g[ikk] is the class-'L' group). The conditions: 2 limbs;
   * z >= 3; the groups are, in this order, {L, H} with H's stride 1 and L's stride equal to H's R,
   * or {A, A, L} where both A groups have column bits and L's stride is the product of their R, a
   * power of 2; rem is at most 8 and is not 1; an entry of T_a sums at most 2^16 bytes, which span
   * less than 2^32 bytes; L's stride and limbStride are multiples of 8 (of 16 if rem >= 4). Not
   * checked: each index pair of the A groups must have an offset below the product of their R. */
  void setup_run() {
    const size_t ng = u.g.size();
    std::vector<u32> ct;
    if (ng == 2 && u.g[0].cls == 'L' && u.g[1].cls == 'H' && u.g[1].stride == 1 && u.g[0].stride == u.g[1].R) {
      const CGrp &x = u.g[1];
      ikk = 0;
      o = (int)(x.R >> x.z);
      z = x.z;
      rem = x.z - x.take;
      ct.resize((size_t)1 << x.take);
      for (size_t i = 0; i < ct.size(); i++) ct[i] = (u32)(i << x.cb0);
    } else if (ng == 3 && u.g[0].cls == 'A' && u.g[1].cls == 'A' && u.g[2].cls == 'L' && u.g[0].take && u.g[1].take &&
               u.g[2].stride == u.g[0].R * u.g[1].R && !((u.g[0].R * u.g[1].R) & (u.g[0].R * u.g[1].R - 1))) {
      ikk = 2;
      o = 1;
      z = c_clg(u.g[0].R * u.g[1].R);
      rem = 0;
      ct.assign((size_t)1 << z, ~0u);
      for (size_t a = 0; a < u.g[0].R; a++)
        for (size_t b = 0; b < u.g[1].R; b++)
          ct[a * u.g[0].stride + b * u.g[1].stride] = (u32)((a << u.g[0].cb0) | (b << u.g[1].cb0));
    } else
      return;
    if (z < 3 || rem > 8 || u.nlw != 2 || (u.g[ikk].stride & 7) || (u.limbStride & 7)) return;
    const CGrp &k = u.g[ikk];
    const size_t bytes = (2 * k.left * o) << rem;
    const size_t span =
        u.limbStride + ((k.left - 1) << k.take) * k.stride + ((size_t)(o - 1) << z) + ((size_t)1 << rem);
    if (rem == 1 || bytes > 65536 || span >> 32) return;
    if (rem >= 4 && ((k.stride | u.limbStride) & 15)) return;
    run = 1;
    colTab = dalC<u32>(ct.size());
    CK(cudaMemcpy(colTab, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice));
  }
  void setup(const CUnit &u_) {
    u = u_;
    K = (size_t)1 << u.lk;
    std::vector<unsigned long long> r(u.ny), c(K);
    for (size_t y = 0; y < u.ny; y++) r[y] = u.rowOff(y);
    for (size_t x = 0; x < K; x++) {
      size_t o;
      c[x] = u.colOff(x, o) ? o : ~0ull;
    }
    rowOff = dalC<unsigned long long>(u.ny);
    colOff = dalC<unsigned long long>(K);
    CK(cudaMemcpy(rowOff, r.data(), u.ny * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(colOff, c.data(), K * 8, cudaMemcpyHostToDevice));
  }
  // A CClaimD with this unit's layout; the caller sets HV, pt and AV.
  CClaimD desc() const {
    CClaimD d;
    d.ng = (int)u.g.size();
    d.ncb = u.ncb;
    d.lk = u.lk;
    for (int i = 0; i < d.ng; i++) {
      const CGrp &q = u.g[i];
      d.f[i].cls = q.cls == 'L' ? 0 : q.cls == 'H' ? 1 : 2;
      d.f[i].z = q.z;
      d.f[i].take = q.take;
      d.f[i].cb0 = q.cb0;
      d.f[i].R = (unsigned)q.R;
      d.f[i].left = (unsigned)q.left;
    }
    return d;
  }
};
/* Builds one group's factor (HV, pt: see CFacD) from a point p with one coordinate per index bit:
   index value x < o * 2^z gets the coefficient product over v < nb of eq(p[v], bit v of x). */
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at kG_msg()).
__global__ void kG_facE(const T *p, int nb, int z, int o, T *HV, T *pt) {
  const int t = threadIdx.x;
  for (int hi = t; hi < o; hi += blockDim.x) {
    T r = F::one();
    for (int v = z; v < nb; v++) r = F::mul(r, kG_eq1(p[v], (hi >> (v - z)) & 1));
    HV[hi] = r;
  }
  for (int i = t; i < z; i += blockDim.x) pt[i] = p[z - 1 - i];
}
__global__ void kG_zero(T *a, T *b, size_t n) {
  GRID_LOOP(i, n) {
    a[i] = F::zero();
    b[i] = F::zero();
  }
}
// Orders CSumcheck::run()'s challenges rr by index bit. One block of at least n threads.
__global__ void kG_rev(const T *rr, int n, T *byBit) {
  const int v = threadIdx.x;
  if (v < n) byBit[v] = rr[n - 1 - v];
}
// o[r] = sum over y < n of e[y] M[r s + y]. It requires one block of 256 threads for each row r.
// A block's sum. The idea is public: J. Luitjens, "Faster Parallel Reductions on Kepler", "Block
// Reduce" (cited at kG_wred()); for what differs, see kG_sum().
// The compiler hint is public: __launch_bounds__: NVIDIA's Best Practices Guide (cited at
// kG_sum()).
// The loop form is public: the block-stride loop: CUDA Programming Guide (cited at kG_msg()).
__global__ void __launch_bounds__(256) kG_dot(const T *e, const T *M, size_t s, int n, T *o) {
  __shared__ T sh[8];
  const T *m = M + blockIdx.x * s;
  T a = F::zero();
  for (int y = threadIdx.x; y < n; y += 256) a = F::add(a, F::mul(e[y], m[y]));
  a = kG_wred(a);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    T v = F::zero();
    for (int w = 0; w < 8; w++) v = F::add(v, sh[w]);
    o[blockIdx.x] = v;
  }
}
// Sends the n values v, each masked by its slot's mask entry, as one message in the slots
// msg[0 .. n), outside a sumcheck. Launch: one thread.
__global__ void kG_sendv(Tr *tr, T *msg, const T *v, int n) {
  Tr t = *tr;
  sc_door::claims(sc_door::Site::WEIGHT_CLAIMS, t, v, n, msg);
  *tr = t;
}
// The merge's coefficients: one challenge x is drawn, and rho[a] = x^a for a < na. Launch: one
// thread.
__global__ void kG_rho(Tr *tr, T *rho, int na) {
  Tr t = *tr;
  T x;
  tr_point(t, Round::BAT_SIGMA, &x, 1);
  T p = F::one();
  for (int a = 0; a < na; a++, p = F::mul(p, x)) rho[a] = p;
  *tr = t;
}
// The end of the merge's sumcheck (see CEnd::send_merge()): the round in Y, in which the summand
// is fv + c Y and whose challenge goes to beta[0], then the end value v = fv + beta c. c[0] is not
// the mask entry of v's slot: sc_door::in_the_clear() adds nothing to v, and that mask entry stays
// as it was drawn, unused. Launch: one thread.
__global__ void kG_merge_end(Tr *tr, const T *fv, const T *c, T *beta) {
  constexpr sc_door::Site site = sc_door::Site::WEIGHT_CLAIMS;
  Tr t = *tr;
  // c[0] is not tested for 0, on purpose. The reason is an argument, not a proof: the hiding
  // argument (blocks/iop/batt_shown.h, "Hiding") uses nothing of c but that it is uniform. 0 is a
  // value like any other. Apart from v, which c masks, this kernel sends c only in the coefficient
  // of the round in Y, under its slot's mask entry; "Hiding" names the other scalars that c enters.
  // A program that rejects or flags c = 0 draws c from E without 0, which is not the distribution
  // that the argument assumes.
  // Two stronger tests catch a c[0] that was never filled or has lost its value. c[0] is a copy of
  // an entry of the request's index-mask message (masks for its claims on W). (1) The host later
  // proves the linear claim "fv + beta c = v" with the message's c: any difference makes that claim
  // false, and the honest prover's check of its own sumcheck (blocks/iop/psc.h) stops the process,
  // with no file for any request (no test of this repository plants such a fault in cp_prove;
  // blocks/iop/tests/psc_stops_test.cpp triggers that check of psc.h at toy size). (2) Just before
  // hd::close() (prover/hiding/door.cuh) the caller compares c[0] with the entry by value
  // (ch::host.index_mask_entry_holds), which also catches an earlier sub-proof's mask.
  // An entry that is empty or wrong at its source, the host's table of that message, is looked for
  // by a third self-check (own-test-index-mask): it compares the whole table, right after it is
  // sent, and each entry again when it is taken, with a second implementation of the draw, and ends
  // the process on a difference. What it does not see: the places of SEND's pad, which it skips,
  // and a fault that the draw and its second implementation share (the same key and context): there
  // all copies would agree, the proof would verify, and fv would be sent unmasked.
  // (SEND, blocks/iop/send.h, sends an oracle message; its pad is the random element that masks the
  // answer to the message's out-of-domain sample.) send_prove() there checks the pad's places
  // itself: each is empty before the pad is written (own-test-pad-place) and, after the message is
  // sent, holds the pad as it was drawn (own-test-pad-kept). own-test-pad-kept also compares each
  // pad, as the message's table holds it after the sending, with its draw computed again from the
  // draw's number. In cp_prove and cp_commit the second implementation computes it, so this check
  // too misses a fault that the draw and its second implementation share. Where a generator does
  // not number its draws, the comparison is left out.
  // A __trap() here would also leave the CUDA context unusable, so that the wipe of GPU memory at
  // the program's end would fail.
  const T in_y[1] = {c[0]};
  beta[0] = sc_door::further(site, t, in_y);
  sc_door::in_the_clear(site, t, F::add(fv[0], F::mul(beta[0], c[0])), const_cast<T *>(sc_door::end_slots(t)));
  tr_empty(t, Round::PSC_VALUES); // the end values' round, in which the verifier's message is empty
  *tr = t;
}
// ka and fv from the folded arrays: X[a * st] = eps_a(rm), Y[a * st] = T_a(rm). Launch: one thread.
__global__ void kG_fin(int na, size_t st, const T *X, const T *Y, const T *rho, T *ka, T *fv) {
  T f = F::zero();
  for (int a = 0; a < na; a++) {
    ka[a] = F::mul(rho[a], X[a * st]); // rho_a eps_a(rm), where rm is the merge's point
    f = F::add(f, F::mul(ka[a], Y[a * st]));
  }
  fv[0] = f; // the merge's end value: the sum over a of ka[a] * T_a(rm)
}
__global__ void kG_TH(int na, size_t K, const T *Tor, const T *ka, T *o) {
  GRID_LOOP(c, K) {
    T a = F::zero();
    for (int q = 0; q < na; q++) a = F::add(a, F::mul(ka[q], Tor[q * K + c]));
    o[c] = a; // the sum over the claims q of ka[q] T_q[c]
  }
}
// One claim on W: its unit's layout (map), its factors (d), and what T_a is computed from.
struct CClaim {
  const CMap *map = nullptr;
  CClaimD d;
  /* ui: the unit's number among W's units. mode selects how T_a is computed: 0 from the cells, 1
   * from a table (tab, mapT, dT), 2 from the cells read in runs. Mode 2 needs map->run == 1, which
   * nothing checks. It uses mode 0's kernels unless the address `cells` is a multiple of 8 (of 16
   * if rem >= 4). */
  int ui = 0, mode = 0;
  const i8 *cells = nullptr;
  const T *tab = nullptr;
  const CMap *mapT = nullptr;
  /* dT holds, for mode 1, the claim's factors in the table's layout (mapT). There the 'H' group
   * keeps only its column bits (its range is 2^take), so its HV is {1}. */
  CClaimD dT;
};
#include "head/claims_segments.cuh"
/* The NCL weight claims of one sub-proof (cl) and the merge's buffers in GPU memory. ep, Tor and Tw
 * have K = 2^lk elements for each claim: eps_a, T_a, and the copy of T_a that the merge folds. rho,
 * rm, beta: the merge's coefficients, its point, and its challenge in Y. ka, fv, TH_: see kG_fin
 * and kG_TH. The rest is scratch; part: ns partial sums per column. */
struct CEnd {
  int lk = 0, NCL = 0, ns = 4;
  size_t K = 0, nymax = 1;
  std::vector<CClaim> cl;
  T *ep = nullptr, *Tor = nullptr, *Tw = nullptr, *rho = nullptr, *ka = nullptr, *fv = nullptr, *beta = nullptr, *rm = nullptr,
    *TH_ = nullptr, *gam = nullptr, *gs = nullptr, *part = nullptr, *gk = nullptr, *gw = nullptr;
  u32 *segOff = nullptr, *gb = nullptr;
  // Allocates for NCL_ claims whose layouts (map, mapT) have lk == lk_ and ny <= nymax_, which
  // nothing checks. th is TH_'s Place: Place::Result if TH_ is read after the sub-proof's run has
  // finished.
  void setup(int lk_, int NCL_, size_t nymax_, Place th = Place::Own) {
    lk = lk_;
    NCL = NCL_;
    K = (size_t)1 << lk;
    nymax = nymax_;
    cl.resize(NCL);
    ep = dalC<T>(NCL * K);
    Tor = dalC<T>(NCL * K);
    Tw = dalC<T>(NCL * K);
    rho = dalC<T>(NCL);
    ka = dalC<T>(NCL);
    fv = dalC<T>(1);
    beta = dalC<T>(1);
    rm = dalC<T>(64);
    TH_ = dalC<T>(K, th);
    gam = dalC<T>(nymax);
    gs = dalC<T>(1);
    part = dalC<T>((size_t)ns * K);
    gk = dalC<T>(nymax);
    gw = dalC<T>(nymax);
    // An entry of T_a sums 2 ny bytes; nksMax counts them in steps of 32, rounded up. Mode 2's
    // tables take at most one word (segOff) and 2 EDEG words (gb) for each byte (EDEG: E's degree).
    const size_t nksMax = nymax / 16 + 2;
    segOff = dalC<u32>(32 * nksMax);
    gb = dalC<u32>(64 * EDEG * nksMax);
  }
  // Builds eps_a and T_a for the claims a in [a0, a1), or [a0, NCL) when a1 < 0.
  void tables(int a0 = 0, int a1 = -1) {
    for (int a = a0; a < (a1 < 0 ? NCL : a1); a++) {
      const CClaim &c = cl[a];
      const CUnit &u = c.map->u;
      // No more than 264 blocks (twice the 132 multiprocessors of an H100): every block rebuilds
      // the small tables.
      kG_eps<<<std::min(grid_blocks(K), 264), 256>>>(c.d, ep + a * K);
      if (c.mode == 1) {
        const CUnit &t = c.mapT->u;
        kG_gam<<<grid_blocks(t.ny), 256>>>(c.dT, t.ny, gam);
        kG_Ttab<<<grid_blocks(K), 256>>>(c.tab, c.mapT->rowOff, gam, (int)t.ny, c.mapT->colOff, K, Tor + a * K);
      } else if (c.mode == 2 && !((size_t)c.cells & (c.map->rem >= 4 ? 15 : 7))) {
        const CMap &m = *c.map;
        const CGrp &k = u.g[m.ikk];
        const CFacD &fk = c.d.f[m.ikk];
        kG_gk<<<grid_blocks(k.left), 64>>>(fk.HV, fk.pt, k.z, k.take, (int)k.left, gk);
        if (m.ikk == 0)
          kG_gw<<<grid_blocks((size_t)m.o << m.rem), 64>>>(c.d.f[1].HV, c.d.f[1].pt, m.z, m.rem, m.o, gw);
        else
          kG_gw<<<1, 1>>>(nullptr, nullptr, 0, 0, 1, gw);
        CK(cudaMemsetAsync(Tor + a * K, 0, K * sizeof(T), cudaStreamPerThread));
        const size_t nrows = (size_t)1 << (k.take + m.z - m.rem);
        kG_T_segments(c.cells, u, k, m, nrows, ns, gk, gw, segOff, gb, part, K, Tor + a * K, cudaStreamPerThread);
      } else {
        kG_gam<<<grid_blocks(u.ny), 256>>>(c.d, u.ny, gam);
        kG_sum1<<<1, 1024>>>(gam, u.ny, gs);
        const int s = u.ny >= 64 ? ns : 1;
        if (u.nlw == 4)
          kG_Tcells<4><<<dim3((unsigned)((K + 127) / 128), s), 128>>>(c.cells, u.limbStride, c.map->rowOff, gam,
                                                                      (int)u.ny, c.map->colOff, K, part);
        else
          kG_Tcells<2><<<dim3((unsigned)((K + 127) / 128), s), 128>>>(c.cells, u.limbStride, c.map->rowOff, gam,
                                                                      (int)u.ny, c.map->colOff, K, part);
        kG_Tfin<<<grid_blocks(K), 256>>>(part, s, K, c.map->colOff, gs, u.nlw, Tor + a * K);
      }
    }
  }
  // The last of the merge's slots takes v.
  size_t merge_slots() const { return CSumcheck::slots(lk, 1, 1); }
  // The merge, then ka, fv and TH_. ep and Tor must be filled, as by tables(); ep is folded in
  // place. msgM is the merge's first slot. The claims' values have been sent before: as end values
  // of the sumcheck that reads the weights, or by kG_sendv. The merge's whole summand counts as
  // one map, whose end value fv is shown as v = fv + beta c. The sub-proof's own equation (proved
  // by hd::close()) never names fv: "fv + beta c = v" leaves the sub-proof as a pending claim, a
  // linear claim on W that is proved later. So the inner mask c is no entry of the sub-proof's own
  // mask row: c[0], in GPU memory, is an entry of the request's index-mask message, which the
  // caller has fetched.
  void send_merge(CSumcheck &SC, Tr *tr, T *msgM, const T *c) {
    kG_rho<<<1, 1>>>(tr, rho, NCL);
    CK(cudaMemcpyAsync(Tw, Tor, NCL * K * sizeof(T), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    SC.run(NCL, lk, K, ep, Tw, rho, tr, msgM, rm, 1);
    kG_fin<<<1, 1>>>(NCL, K, ep, Tw, rho, ka, fv);
    kG_merge_end<<<1, 1>>>(tr, fv, c, beta);
    kG_TH<<<grid_blocks(K), 256>>>(NCL, K, Tor, ka, TH_);
  }
};
