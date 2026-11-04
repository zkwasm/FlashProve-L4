// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The prover's forward pass through the output head, `HeadForward::forward`: the final norm (output
   aF), the logits, and the committed cells and lookup keys of the head's sub-proof (the proof's
   argument for this part of the model). A cell is one committed small integer; a lookup key is a
   value that the sub-proof looks up in a public table.
   Include prover/prover/forward.cuh, and what that file needs, first: it defines GRID_LOOP, prover_gemm, Pos2
   and E_LR and includes prover/prover/layout.h (Ten, Seg, Call, Piece, ResidualInt). The launches and their
   grids do not depend on the request's range (r, e) or on its tokens. Outside `rows_out` and after
   its thread's first cuBLAS call, `forward` neither allocates nor synchronizes. A value found
   outside its range sets a bit of the error word `err` and the computation goes on: the caller must
   read the word. A token outside the vocabulary or a negative histogram index sets the request's
   failure bit `INDEX` instead (`index_below`). The lie:: hooks (prover/common/lie.cuh) falsify
   values only in a malicious-prover test build that is not part of this repository.
   lam_t[v] = sum over k of aF[k][t] Wh[v][k] is the logit of vocabulary entry v at token position
   t < T (Wh: the head's weights). The request's range is r <= t < e; its positions below T - 1 are
   the counted positions. The checked token w_t is the request's token t + 1 or, at the last
   position, the next token y (k_head_y). For every t in the range and every v, `forward` writes the
   margin D[t][v] = lam_t[w_t] + lift_t - lam_t[v] - (1 if v < w_t) - Pub[t][v] as three digits, and
   the table value e[t][v] = EX[the middle digit] (EX: head_ex_table; Pub: head_public_term).
   k_head_budget sets lift_t from the honest cost (`price` in identifiers): the least gap_t + 2^sh j,
   j >= 1, that as lift_t makes the position's table values sum to at most 2^F (k_head_price; gap_t is
   the logit gap, k_head_chosen). sh, F, sT, C, theta and cap are lines of the parameter file
   (`policy.txt`): see HeadShape. Each of three sweeps (gaps, costs, margins) computes the logits by
   slices of the vocabulary. docs/model.md, "8 Embedding, head and output", defines aF and lam_t. */
/* Public sources for the kernels of this file. NVIDIA, CUDA C++ Programming Guide: "Atomic
   Functions" (atomicMax() on 64-bit words, atomicAdd() on 32-bit and on 64-bit words, atomicOr());
   "Synchronization Functions" (__syncthreads()); "Compute Capability 7.x", "Independent Thread
   Scheduling" (its code sample's sum by halving: k_head_price); "Warp Shuffle Functions"
   (__shfl_down_sync(): k_head_chosen). NVIDIA Technical Blog, "CUDA Pro Tip: Write Flexible Kernels
   with Grid-Stride Loops" (GRID_LOOP). The products use `prover_gemm`, which calls cuBLAS's `cublasGemmEx`
   or, for T < 256 or a d that is not a multiple of 16, its own kernel. Logits recomputed, not
   stored: Wijmans et al., arXiv:2411.09009; Chen et al., arXiv:1604.06174.
   See also, for the kernels' CUDA functions: NVIDIA, CUDA Programming Guide, "C/C++ Language
   Extensions",
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html:
   "Atomic Functions", "Thread Block Synchronization Functions", "Warp Shuffle Functions".
   The core of how the head treats the softmax of the logits (a committed shift, here lam_t[w_t] +
   lift_t, of which c and b are committed, lift_t = c + 2^37 b; looked-up digits; a table of the
   exponential; a lowest digit that is only range-checked; a condition on the sum) is that of
   zkAttn, which is for the softmax of attention: Sun, Li, Zhang, "zkLLM: Zero Knowledge Proofs for
   Large Language Models", arXiv:2404.16109, sections 5.1, 5.1.1 and 5.2 with Protocol 2. The head
   differs: only the middle digit has a table of the exponential (in zkAttn the number of such
   digits is a parameter), every rounding goes against the prover, the sum is bounded only from
   above, and the shift itself is used: at a counted position lift_t is then at least 2^sT times
   w_t's negative log-probability under the softmax of lam_t / 2^sT, and the acceptance policy (the
   rule that every checked token must meet) bounds c and b. */
#pragma once
#include "common/lie.cuh"
#include "common/index_guard.cuh"
#include "common/made_up_alone.h"
#include <mutex>
// The head's shape and parameters. T = 2^lt token positions; d: the residual stream's width; V: the
// vocabulary's size. la: see JV in make_head_layout. A lookup call has at least 2^nmin keys. The final
// norm adds epsF under the root, scales the reciprocal by 2^MF (norm_ledger) and shifts its output
// right by KF bits (k_norm_out). docs/model.md, "5 The norm", defines the norm and its clamp.
struct HeadShape {
  int lt = 0, d = 0, V = 0, la = 0, nmin = 10, MF = 0, KF = 0;
  long long epsF = 1;
  int xl = 4; // limbs of a value of the residual stream (prover/prover/layout.h)
  const char *lists = nullptr; // the parameter file's callH lines as "name:seg,seg;name:seg"
  u32 C = 0, theta = 0, cap = 0; // the parameter file's lines of these names: see k_head_budget
  int sh = 8;                    // the line sh: see margin_digits
  int s = 15;                    // the line s: the model's logit is an integer logit / 2^s; the prover never reads it
  int sT = 4, F = 24;            // the lines sT and F: the table is head_ex_table(sT - sh, F)
};
// After the final norm's digits, a position's row of digits (`ledger` in identifiers) has HEAD_BUDGET_DIGITS
// more, for k_head_budget's six numbers: c (5), b (1), u (5), k (5), z (2), zs (3), low digit first.
enum { HEAD_BUDGET_DIGITS = 21 };
/* The public table EX, 2^16 entries: a decaying exponential, EX[u] >= 2^F exp(-u / 2^q), as every
   rounding is upward. G_0 = 2^(F + 17), G_(u+1) = ceil(G_u N_q / 2^32), EX[u] = ceil(G_u / 2^17),
   N_q = ceil(2^32 exp(-2^-q)). q is 4 .. 10, or -4 for the step table (at F = 24: 2^24, 2, 1, 1,
   ...). EX[0] = 2^F. EX never increases and is 1 from some index below 2^15 on. */
static inline std::vector<u32> head_ex_table(int q, int F) {
  static const u64 N[7] = {4034748383ull, 4162825045ull, 4228380000ull, 4261543596ull,
                           4278222806ull, 4286586875ull, 4290775040ull};
  if (q != -4 && (q < 4 || q > 10)) {
    printf("policy: sT - sh must be -4 or from 4 to 10\n");
    exit(2);
  }
  const u64 step = q == -4 ? 484 : N[q - 4];
  std::vector<u32> EX(65536);
  unsigned __int128 G = (unsigned __int128)1 << (F + 17);
  for (auto &x : EX) {
    x = (u32)((G + 131071) >> 17);
    G = (G * step + 0xFFFFFFFFull) >> 32;
  }
  return EX;
}
// lift_t at a counted position whose honest cost is above C (b = 1 in k_head_budget). Logit gaps stay
// below it if |Wh| <= 2^12, which this file does not check, and if the parameter file meets the
// condition `no-wrap-H-three-digits` (prover/common/parameter_file.h).
constexpr long long HEAD_BIG = 1ll << 37;
// The bit of the error word for a request that exceeds the parameter file's bounds theta or cap.
// The word's lower bits are the range flags of prover/prover/forward.cuh, E_LR among them.
enum { E_BOUNDS = 64 };
// The tensors of cells. In the matrix hf: aw and ra (the final norm's unclamped output and its
// remainder's high digit), d1 and d2 (a margin's middle and low digit), e (the table value minus
// 1). In hd: aF. In hr: the rows of digits (`row`). Cells are int8, [feature][t]; a wider value
// takes two or three limbs at radix 256.
enum TenH { Y_aw, Y_ra, Y_d1, Y_d2, Y_e, Y_aF, Y_row, NTENH };
// Where each tensor is in its matrix and each lookup segment in its lookup call. In a lookup call
// every key of its segments (named lists of keys) is to occur in a public table. Keys: u16 at
// column * T + t. G[m]: the number of features of matrix m (0 hf, 1 hd, 2 hr). Pd: d rounded up to
// a power of two.
struct HeadLayout {
  HeadShape s;
  int T = 0;
  long long JV = 0, Pd = 0;
  std::vector<u32> EX; // head_ex_table(s.sT - s.sh, s.F)
  int UT = 0;          // the first index at which EX holds 1
  Ten t[NTENH];
  long long G[3] = {0, 0, 0};
  std::vector<Seg> seg;
  std::vector<Call> call;
  int segid(const char *n) const {
    for (size_t i = 0; i < seg.size(); i++)
      if (seg[i].name == n) return (int)i;
    return -1;
  }
  // The column of entry x of segment sg inside its call.
  long long pos(int sg, long long x) const {
    const Call &c = call[seg[sg].call];
    for (int p : seg[sg].pc)
      if (x >= c.pc[p].s && x < c.pc[p].s + c.pc[p].size) return c.pc[p].off + x - c.pc[p].s;
    printf("internal error: head sub-proof: an entry beyond a segment's length was "
           "requested\n");
    exit(2);
  }
};
// Builds the layout: places the tensors in their matrices and parses the lookup calls from s.lists.
// Not checked here: that s.lists has each of the nine segments exactly once (`lay_out_matrices` in
// prover/head/hidden_fork.cuh checks it).
static inline HeadLayout make_head_layout(const HeadShape &s) {
  HeadLayout L;
  L.s = s;
  L.T = 1 << s.lt;
  L.Pd = 1ll << p2c(s.d);
  const long long ls2 = 1ll << (s.la - s.lt);
  L.JV = (s.V + ls2 - 1) / ls2 * ls2; // V rounded up to a multiple of 2^(la - lt): the size of d1, d2, e and HD0 to HD2
  const long long d = s.d, ledger = norm_row_len(s.xl) + HEAD_BUDGET_DIGITS, JVa = 1ll << (p2c(L.JV) - 1);
  L.EX = head_ex_table(s.sT - s.sh, s.F);
  while (L.EX[L.UT] != 1) L.UT++;
  Ten tt[NTENH] = {{"aw", 2, I_ALL, d, 0, 1},    {"ra", 2, I_ALL, d, 0, 1},
                   {"d1", 2, I_ALL, L.JV, 0, 1}, {"d2", 1, I_ALL, L.JV, 0, 1},
                   {"e", 3, I_ALL, L.JV, 0, 1},  {"aF", 2, I_ALL, d, 1, 1},
                   {"row", 1, I_ALL, ledger, 2, 1}};
  for (int i = 0; i < NTENH; i++) {
    Ten &X = L.t[i];
    X = tt[i];
    X.on = true;
    X.cnt = 1;
    X.off = L.G[X.mat];
    L.G[X.mat] += X.nl * X.J;
  }
  struct SD {
    const char *n;
    Kind k;
    long long len;
  };
  // HE0 (vocabulary entries below JVa) and HE1 (from JVa on) look up a middle digit u with EX[u].
  const SD all[] = {{"NF", K_R16, d},      {"NFH", K_R16, d},   {"HD0", K_R16, L.JV}, {"HD1", K_R16, L.JV},
                    {"ROW", K_R8, ledger}, {"HD2", K_R8, L.JV}, {"CNF", K_FC, d},     {"HE0", K_FE, JVa},
                    {"HE1", K_FE, L.JV - JVa}};
  if (!s.lists) {
    printf("internal error: head sub-proof: the layout was requested without the call "
           "lists\n");
    exit(2);
  }
  std::string t = s.lists;
  const int fl = std::max(0, s.nmin - s.lt);
  size_t p = 0;
  while (p < t.size()) {
    size_t e = t.find(';', p);
    if (e == std::string::npos) e = t.size();
    std::string one = t.substr(p, e - p);
    size_t c = one.find(':');
    Call C;
    C.name = one.substr(0, c);
    C.kind = C.name.rfind("R16", 0) == 0  ? K_R16
             : C.name.rfind("R8", 0) == 0 ? K_R8
             : C.name.rfind("FE", 0) == 0 ? K_FE
                                          : K_FC;
    size_t q = c + 1;
    while (q < one.size()) {
      size_t r = one.find(',', q);
      if (r == std::string::npos) r = one.size();
      std::string nm = one.substr(q, r - q);
      q = r + 1;
      const SD *sd = nullptr;
      for (auto &a : all)
        if (nm == a.n) sd = &a;
      if (!sd || sd->k != C.kind) {
        printf("policy: callH segment %s is unknown or in a call of the wrong kind: calls named R16... take NF, NFH, "
               "HD0, HD1; R8... take ROW, HD2; FE... take HE0, HE1; others take CNF\n",
               nm.c_str());
        exit(2);
      }
      Seg S;
      S.name = sd->n;
      S.kind = sd->k;
      S.n = sd->len;
      S.call = (int)L.call.size();
      int id = (int)L.seg.size();
      L.seg.push_back(S);
      C.seg.push_back(id);
      /* A segment is cut into one power-of-two piece for each one-bit of its length. The call
         places all pieces largest first and keeps the order of pieces of equal size. */
      long long st = 0;
      for (int b = 40; b >= 0; b--)
        if ((sd->len >> b) & 1) {
          Piece P;
          P.seg = id;
          P.s = st;
          P.size = 1ll << b;
          C.pc.push_back(P);
          st += 1ll << b;
        }
    }
    std::stable_sort(C.pc.begin(), C.pc.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    long long off = 0;
    for (size_t i = 0; i < C.pc.size(); i++) {
      C.pc[i].off = off;
      off += C.pc[i].size;
      L.seg[C.pc[i].seg].pc.push_back((int)i);
    }
    C.total = off;
    C.lJ = std::max(p2c(off), fl);
    L.call.push_back(C);
    p = e + 1;
  }
  return L;
}
/* Writes n rows of the head's weight matrix, from vocabulary entry v0 on, as two signed byte limbs
   w0 + 256 w1, each [n][d]. Wh is [V][d], or null if the weights are generated from the seed
   (stream 14 of synth_weight). Not checked: a weight above 32,639 does not fit the two limbs. */
// Given Wh, where size and alignment allow, a thread takes four weights: 8 bytes read, two words
// written, no division. Public source: "CUDA Pro Tip: Increase Performance with Vectorized Memory
// Access" (NVIDIA's developer blog), cited in prover/head/claims_segments.cuh.
__global__ void k_head_limbs(const int16_t *Wh, u64 seed, int d, int V, int v0, int n, i8 *w0, i8 *w1) {
  const size_t all = (size_t)n * d;
  const int16_t *from = Wh + (Wh ? (size_t)v0 * d : 0);
  if (Wh && all % 4 == 0 && (uintptr_t)from % 8 == 0 && (uintptr_t)w0 % 4 == 0 && (uintptr_t)w1 % 4 == 0) {
    GRID_LOOP(x, all / 4) {
      const uint2 four = ((const uint2 *)from)[x];
      u32 low = 0, high = 0;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int wv = (int16_t)((q < 2 ? four.x : four.y) >> 16 * (q & 1)), b0 = ((wv + 128) & 255) - 128;
        low |= (u32)(b0 & 255) << 8 * q, high |= (u32)(((wv - b0) >> 8) & 255) << 8 * q;
      }
      ((u32 *)w0)[x] = low, ((u32 *)w1)[x] = high;
    }
    return;
  }
  GRID_LOOP(i, all) {
    const int v = v0 + (int)(i / d), k = (int)(i % d);
    const int wv = Wh ? (int)Wh[(size_t)v * d + k] : synth_weight(seed, 14, (u64)k * V + v, 13);
    const int b0 = ((wv + 128) & 255) - 128;
    w0[i] = (i8)b0;
    w1[i] = (i8)((wv - b0) >> 8);
  }
}
// The logit at index i = j T + t, from a slice's three limb products, each [j][t].
__device__ __forceinline__ i64 head_logit(const int *C0, const int *C1, const int *C2, size_t i) {
  return (i64)C0[i] + ((i64)C1[i] << 8) + ((i64)C2[i] << 16);
}
/* Updates each position's running maximum with a slice's logits (n entries from v0 on). best[t]
   packs (logit + 2^42) << 20 | (2^20 - 1 - v). The packing needs V <= 2^20 and -2^42 <= logit <
   3 * 2^42 (not checked here). Its maximum then belongs to the smallest v with the largest logit:
   the argmax. Requires a launch with at least T threads in x and (n + 63) / 64 blocks in y. */
// Public source: a value and an identifier packed into one 64-bit word: described for the
// lightest edge of a set in Fallin et al., "A High-Performance MST Implementation for GPUs", SC
// '23, section 3.2.
// Public source: signed or reversed keys compared as unsigned ones: described for sort keys in
// NVIDIA's CUB documentation, cub::DeviceRadixSort, "Transformations".
// Public source: the index stored reversed in the low bits: described for the lanes of a warp in a
// post of the NVIDIA Developer Forums thread "Reduction question, searching for the lane index of
// the thread with the highest value".
// Here the maximum runs over the vocabulary and the threads' results meet in an atomic one, only
// the index field is reversed, and the signed logit is made non-negative by adding a constant.
__global__ void k_head_best(const int *C0, const int *C1, const int *C2, int T, int v0, int n, unsigned long long *best) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x), j1 = min(n, ((int)blockIdx.y + 1) * 64);
  if (t >= T) return;
  unsigned long long m = 0;
  for (int j = (int)blockIdx.y * 64; j < j1; j++) {
    const unsigned long long p = ((unsigned long long)(head_logit(C0, C1, C2, (size_t)j * T + t) + (1ll << 42)) << 20) |
                                 (unsigned long long)(0xFFFFF - (v0 + j));
    if (p > m) m = p;
  }
  atomicMax(best + t, m);
}
// y = the last position's argmax if the range includes it (e = T), otherwise 0. rows holds r and e.
// Public sources for the packed word: see k_head_best.
__global__ void k_head_y(const unsigned long long *best, const u32 *rows, int T, int *y) {
  *y = lie::at(lie::TOKEN, 0, T - 1, rows[1] == (u32)T ? 0xFFFFF - (int)(best[T - 1] & 0xFFFFF) : 0); // a test hook (prover/common/lie.cuh): the value itself
}
/* win[t] = w_t. chosen[t] = lam_t[w_t]. gap[t] is the logit gap, the maximum over v of lam_t[v] -
   lam_t[w_t] + (1 if v < w_t), which is 0 exactly if w_t is the argmax. It is computed as the
   largest logit minus lam_t[w_t], plus 1 if the argmax has a lower index than w_t: among the
   entries with the largest logit the first maximizes the formula, and no other entry gives more. a0
   and a1 are aF's limbs, [t][k]. Requires a launch of at least 32 T threads, in blocks of a
   multiple of 32 threads. */
// Public sources for the packed word: see k_head_best.
// The idea is public: a warp reduction by shuffles: J. Luitjens, "Faster Parallel Reductions on
// Kepler", NVIDIA Technical Blog, 2014, "Shuffle Warp Reduce",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
__global__ void k_head_chosen(const i8 *a0, const i8 *a1, int T, int d, int V, const int16_t *Wh, u64 seed,
                            const int *tok, const int *y, const unsigned long long *best, int *win, i64 *chosen,
                            i64 *gap) {
  const int t = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  const long long named = t + 1 < T ? tok[t + 1] : *y;
  // A token outside 0 .. V - 1 counts as 0 and sets the failure bit `INDEX` (`index_below`). The
  // test hook lie::pass() is false here; if true, the bit is skipped and the verifier decides.
  const int w = (named < 0 || named >= V) && lie::pass(lie::HEAD_WINNER) ? 0 : (int)index_below(named, V, INDEX_WINNER);
  i64 a = 0;
  for (int k = lane; k < d; k += 32) {
    const size_t c = (size_t)t * d + k;
    const int av = (int)a0[c] + 256 * (int)a1[c];
    const int wv = Wh ? (int)Wh[(size_t)w * d + k] : synth_weight(seed, 14, (u64)k * V + w, 13);
    a += (i64)av * wv;
  }
  for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
  if (lane) return;
  a = lie::at(lie::LOGIT, 0, t, a, 1); // row 0, second = 1: lam_t[w_t], as gap and margins read it
  win[t] = w;
  chosen[t] = a;
  gap[t] = (i64)(best[t] >> 20) - (1ll << 42) - a + (0xFFFFF - (int)(best[t] & 0xFFFFF) < w ? 1 : 0);
}
/* Adds a slice's entries to each position's histogram. Let base_t = lam_t[w_t] + gap_t and i =
   floor((base_t + 2^sh - lam_t[v] - (1 if v < w_t)) / 2^sh), which is at least 1. hist[t][i] counts
   the entries v with that i, only for i < UT. From UT on the table is 1, so k_head_price needs only
   the number of such entries. */
// Public source: only non-missing entries collected into buckets: described for split finding in
// gradient-boosted trees in Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System", 2016,
// https://arxiv.org/abs/1603.02754, section 3.4, Algorithm 3;
// Public source: missing entries' statistics by subtraction: described for the same algorithm on a
// GPU in Mitchell and Frank, "Accelerating the XGBoost algorithm using GPU computing", PeerJ
// Computer Science 3:e127, 2017, "Missing values".
// Public source: a histogram that ignores out-of-range elements: PyTorch's torch.histc, its
// documented behavior.
// Here the class left out is every entry at or beyond the cut-off, each entry is still computed and
// tested, and counts are subtracted from V, not gradient sums.
// Public sources for a search on such a histogram: see k_head_price.
__global__ void k_head_hist(const int *C0, const int *C1, const int *C2, int T, int v0, int n, const i64 *chosen,
                          const i64 *gap, const int *win, int sh, int UT, u32 *hist) {
  GRID_LOOP(i, (size_t)n * T) {
    const int v = v0 + (int)(i / T), t = (int)(i % T);
    const i64 at = (chosen[t] + gap[t] + (1ll << sh) - head_logit(C0, C1, C2, i) - (v < win[t] ? 1 : 0)) >> sh;
    if (at < UT) atomicAdd(hist + (size_t)t * UT + index_below(at, UT, INDEX_HISTOGRAM), 1u);
  }
}
/* The honest cost of each position. sum(j) is the sum over all V entries of EX[i + j], with i as in
   k_head_hist (an index from UT on reads 1). Bisection finds j', the least j >= 0 with sum(j) <= 2^F:
   the sum never increases with j, and at j = 2^15 it is V. price[t] = gap_t + 2^sh (1 + j') and
   spare[t] = 2^F - sum(j'). With lift_t = price[t] an entry's middle digit is i + j', or an index
   at which the table also reads 1. Requires a launch of T blocks of 256 threads. */
// Public source: a threshold search on a histogram built beforehand: described for split finding
// in LightGBM's documentation, Features, "Optimization in Speed and Memory Usage".
// Public source: every candidate threshold evaluated on such a histogram: described for the
// clipping threshold of int8 calibration in the TPU-MLIR developer manual, Sophgo, SDK
// v23.09.01-lts, 7.3.1, "KLD Algorithm".
// Here the search is a bisection for a shift of the table index by whole bins, each position has
// its own histogram, and the histogram loses nothing, because the table depends only on the binned
// value.
// The idea is public: a tree reduction with sequential addressing: M. Harris, "Optimizing Parallel
// Reduction in CUDA", NVIDIA Developer Technology, slides "Parallel Reduction: Sequential
// Addressing" and "Reduction #3: Sequential Addressing",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void k_head_price(const u32 *hist, const u32 *EX, int UT, int V, int F, int sh, const i64 *gap, i64 *price,
                           i64 *spare) {
  __shared__ unsigned long long part[256];
  const int t = (int)blockIdx.x;
  const u32 *row = hist + (size_t)t * UT;
  auto all = [&](unsigned long long mine) { // returns the total of the partial sums 'mine'
    part[threadIdx.x] = mine;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
      if ((int)threadIdx.x < o) part[threadIdx.x] += part[threadIdx.x + o];
      __syncthreads();
    }
    const unsigned long long sum = part[0];
    __syncthreads();
    return sum;
  };
  unsigned long long counted = 0;
  for (int b = (int)threadIdx.x; b < UT; b += 256) counted += row[b];
  // Public sources for counting the left-out entries by subtraction: see k_head_hist.
  const unsigned long long tail = (unsigned long long)V - all(counted);
  auto sum = [&](int j) {
    unsigned long long a = 0;
    for (int b = (int)threadIdx.x; b < UT; b += 256) a += (unsigned long long)row[b] * EX[min(b + j, UT)];
    return tail + all(a);
  };
  int low = 0, high = 32768; // needs sum(high) = V <= 2^F; the least such j is in low .. high
  while (low < high) {
    const int middle = (low + high) / 2;
    if (sum(middle) <= 1ull << F)
      high = middle;
    else
      low = middle + 1;
  }
  const unsigned long long at_it = sum(low);
  if (threadIdx.x) return;
  price[t] = lie::at(lie::PRICE, 0, t, gap[t] + ((i64)(1 + low) << sh));
  spare[t] = (i64)((1ull << F) - at_it);
}
/* Sets every position's six committed numbers by the honest prover's rule. C clips a position's
   cost, theta bounds the mean of the clipped costs, and cap the number of clipped positions. At a
   counted position: if price <= C then c = price and b = 0, otherwise c = 0 and b = 1; k = C - c;
   zs = spare[t], or 2^F - V if b = 1, where every table value is 1. With R counted positions, the
   slack theta R - sum of (c + C b) goes into u at the first counted positions, at most 2^40 - 1 per
   position, and the slack cap - sum of b is z at the first counted position. A negative slack sets
   E_BOUNDS in the error word and counts as 0. At every other position all six are 0. lift[t] = c +
   b HEAD_BIG. Each digit goes to the row of digits, from feature NR on, as the digit minus 128, and
   unchanged to the keys of the segment ROW. Requires a launch of one block. */
// Public source for the rule of k_head_budget under the step table (a position's gap to the top score
// is clipped, averaged and compared with a bound): it follows Token-DiFR's score at temperature 0,
// Karvonen et al., "DiFR: Inference Verification Despite Nondeterminism", arXiv:2511.20621 (the
// margin: section 4.2, eq. (1); the clip: 4.2; the mean: 5.2; the threshold: 4.1). It differs in
// four ways: a token tied with the argmax has gap 1, not 0; each cost is the gap plus the constant
// 2^sh; cap bounds the number of clipped positions; theta is a line of the parameter file, not a
// calibrated threshold. PUBLISHED-WORKS.md, "A published method", cites the paper in full.
__global__ void k_head_budget(const i64 *price, const i64 *spare, const u32 *rows, int T, int V, int F, u32 C, u32 theta,
                            u32 cap, int NR, i8 *row, u16 *keyRow, const int *posRow, i64 *lift, u32 *err) {
  __shared__ unsigned long long spent, clipped;
  const int r = (int)rows[0], end = min((int)rows[1], T - 1);
  if (!threadIdx.x) spent = clipped = 0;
  __syncthreads();
  unsigned long long s = 0, n = 0;
  for (int t = r + (int)threadIdx.x; t < end; t += (int)blockDim.x) {
    s += price[t] <= (i64)C ? (unsigned long long)price[t] : C;
    n += price[t] > (i64)C;
  }
  atomicAdd(&spent, s);
  atomicAdd(&clipped, n);
  __syncthreads();
  const i64 R = max(end - r, 0), most = (1ll << 40) - 1;
  i64 left = (i64)theta * R - (i64)spent, rows_left = (R ? (i64)cap : 0) - (i64)clipped;
  if ((left < 0 || rows_left < 0) && !threadIdx.x && !lie::pass(lie::HEAD_BOUNDS)) atomicOr(err, (u32)E_BOUNDS);
  left = max(left, (i64)0), rows_left = max(rows_left, (i64)0);
  for (int t = (int)threadIdx.x; t < T; t += (int)blockDim.x) {
    const bool counted = t >= r && t < end, over = counted && price[t] > (i64)C;
    const i64 c = lie::at(lie::COST, 0, t, counted && !over ? price[t] : 0); // k and the margins' lift follow it
    auto digits = [&](int at, int m, i64 v) {
      for (int i = 0; i < m; i++) {
        const int dg = (int)((v >> (8 * i)) & 255);
        row[(size_t)(NR + at + i) * T + t] = (i8)(dg - 128);
        keyRow[(size_t)posRow[NR + at + i] * T + t] = (u16)dg;
      }
    };
    digits(0, 5, c);
    digits(5, 1, lie::at(lie::COST, 0, t, (i64)(over ? 1 : 0), 1));
    digits(6, 5, lie::at(lie::COST, 0, t, counted ? min(max(left - (i64)(t - r) * most, (i64)0), most) : 0, 2));
    digits(11, 5, lie::at(lie::COST, 0, t, counted ? (i64)C - c : 0, 3));
    digits(16, 2, lie::at(lie::COST, 0, t, t == r && counted ? rows_left : 0, 4));
    digits(18, 3, lie::at(lie::COST, 0, t, !counted ? 0 : over ? (1ll << F) - V : spare[t], 5));
    lift[t] = c + (over ? HEAD_BIG : 0);
  }
}
/* The public term Pub[t][v] that k_head_margins subtracts from a margin: always 0. This function is
   the only place where such a term enters the prover's margins. */
__device__ __forceinline__ i64 head_public_term(int, int) {
  return 0;
}
/* Splits a margin D >= 0 into three digits, D = K2 + 2^sh K1 + 2^(sh + 15) K0. K2 = D mod 2^sh is
   the low digit. Let m = floor(D / 2^sh). If m < 2^15, K1 = m and K0 = 0; otherwise K1 = 2^15 + (m
   mod 2^15) and K0 = floor(m / 2^15) - 1, so the top digit overlaps the middle one by one bit. K0
   stays below 2^16 as long as D < (2^16 + 1) 2^(sh + 15). */
__device__ __forceinline__ void margin_digits(i64 D, int sh, int &K0, int &K1, int &K2) {
  const i64 m = D >> sh;
  K2 = (int)(D & ((1ll << sh) - 1));
  K1 = m < 32768 ? (int)m : 32768 + (int)(m & 32767);
  K0 = m < 32768 ? 0 : (int)(m >> 15) - 1;
}
/* Writes the margins of n vocabulary entries from v0 on, at all T positions, from the slice's three
   products. In the request's range D is the margin defined at the top of this file, split by
   margin_digits. K0 is HD0's key and has no cell. K1 is HD1's key, and the cells d1 hold K1 - 32896 (two
   limbs). K2 is HD2's key, and the cell d2 holds K2 - 128. A D outside [0, (2^16 + 1) 2^(sh + 15))
   does not fit the digits and counts as 0; in the range it also sets E_LR in the error word.
   Outside the range the three keys and these three cells are 0. The three cells e hold the byte
   digits of EX[K1] - 1, each minus 128; outside the range K1 counts as 0 here too, so each is 127
   (at F = 24). */
// Source for showing that no candidate exceeds a value, by non-negative differences: Liu, Xie, Zhang, "zkCNN: Zero
// Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy", Cryptology ePrint Archive 2021/673,
// section 4.3, condition (1) (this condition and its number were not checked against the paper).
// Source for the same: Chen, Waiwitlikhit, Stoica, Kang, "ZKML: An Optimizing System for ML Inference in
// Zero-Knowledge Proofs", EuroSys 2024, section 5.1.
// In zkCNN a second condition shows that the value is attained or is 0; here there is no second condition.
// Source for greedy decoding shown in this way over the whole vocabulary: Wang, Lou, Li, Yu, Hu, "zkAgent: Verifiable
// LLM Agent Execution via One-Shot Transcript Proofs", Cryptology ePrint Archive 2026/199, section 3.3, "Greedy
// decoding", of its version of February 2026 (titled "zkAgent: Verifiable Agent Execution via One-Shot Complete LLM
// Inference Proof"). There the conclusion is that y is an argmax; here the difference also has the tie-break term (1
// if v < w_t) and lift_t.
__global__ void k_head_margins(const int *C0, const int *C1, const int *C2, int T, int v0, int n, const i64 *chosen,
                             const i64 *lift, const int *win, const u32 *rows, int sh, const u32 *EX, i8 *d1,
                             size_t st1, i8 *d2, i8 *e3, u16 *k0, u16 *k1, u16 *k2, Pos2 p0, Pos2 p1, Pos2 p2,
                             u32 *err) {
  const int r = (int)rows[0], e = (int)rows[1];
  const i64 reach = 65537ll << (sh + 15);
  GRID_LOOP(i, (size_t)n * T) {
    const int v = v0 + (int)(i / T), t = (int)(i % T);
    const bool stated = t >= r && t < e;
    i64 D = lie::at(lie::MARGIN, v, t, chosen[t] + lift[t] - head_logit(C0, C1, C2, i) - (v < win[t] ? 1 : 0) - head_public_term(t, v));
    if (D < 0 || D >= reach) {
      if (stated && !lie::pass(lie::HEAD_MARGIN)) atomicOr(err, (u32)E_LR);
      D = 0;
    }
    int K0, K1, K2;
    margin_digits(D, sh, K0, K1, K2);
    K0 = lie::at(lie::DIGIT, v, t, K0, 1), K1 = lie::at(lie::DIGIT, v, t, K1, 2), K2 = lie::at(lie::DIGIT, v, t, K2, 3); // second: 1 top, 2 middle, 3 low (0 is a digit of the final norm's row of digits)
    const size_t c = (size_t)v * T + t;
    write_limbs(stated ? K1 - 32896 : 0, 2, d1 + c, st1);
    d2[c] = (i8)(stated ? K2 - 128 : 0);
    k0[(size_t)p0.at(v) * T + t] = (u16)(stated ? K0 : 0);
    k1[(size_t)p1.at(v) * T + t] = (u16)(stated ? K1 : 0);
    k2[(size_t)p2.at(v) * T + t] = (u16)(stated ? K2 : 0);
    const u32 less_one = lie::at(lie::EXPONENTIAL, v, t, EX[stated ? lie::inside(K1, 65536) : 0]) - 1; // `lie::inside`: a test hook that returns K1; EX has 2^16 entries
    for (int l = 0; l < 3; l++) e3[c + l * st1] = (i8)((int)((less_one >> (8 * l)) & 255) - 128);
  }
}
// The forward pass and its buffers, which the sub-proof (prover/head/hidden_fork.cuh) owns and
// allocates. whsrc: Wh, or null for weights generated from `seed`. hf, hd, hr: the matrices of
// cells. key[c], keyN[c]: lookup call c's keys and their number. Rv, sigv: each position's
// reciprocal and root in the final norm; Wg: that norm's scratch (norm_ledger). y: the next token.
struct HeadForward {
  HeadLayout L;
  u64 seed = 0;
  int cm = 2047; // aF is clamped to [-cm, cm]; cm = 2^(cw - 1) - 1 for the cell width cw = 12
  const int16_t *whsrc = nullptr;
  i8 *hf = nullptr, *hd = nullptr, *hr = nullptr;
  u16 *key[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // the pass writes no keys for HE0 and HE1
  std::vector<size_t> keyN;
  u64 *Rv = nullptr, *sigv = nullptr;
  u32 *err = nullptr;
  // err: the error word. norm() begins by zeroing it, unless raise_flags_in() has pointed err at
  // the caller's word.
  bool zeroes_its_flags = true;
  void raise_flags_in(u32 *word) { err = word, zeroes_its_flags = false; }
  unsigned long long *Wg = nullptr;
  int *y = nullptr, *posRow = nullptr; // posRow: the columns of ROW's norm_row_len(xl) + HEAD_BUDGET_DIGITS entries in its call
  const u32 *EX = nullptr;    // L.EX on the device, 2^16 words
  cublasHandle_t h = nullptr; // cuBLAS's handle for the products: see forward()
  char *scratch = nullptr;    // scratch_bytes() bytes, aligned to 256; no pass reads what an earlier pass left in it
  // The number of vocabulary entries in a slice. A slice's limbs and products take 2 d + 12 T bytes
  // per entry.
  int slice() const { return std::min(L.s.V, 8192); }
  // The parts of the scratch buffer, in order. Each begins at a multiple of 256 bytes.
  enum Part {
    P_BLAS, P_A0, P_A1, P_W0, P_W1, P_C0, P_C1, P_C2, P_BEST, P_CHOSEN, P_GAP, P_LIFT, P_PRICE, P_SPARE, P_WIN, P_HIST,
    NPARTS
  };
  size_t part_bytes(int p) const {
    const size_t T = (size_t)L.T, d = (size_t)L.s.d, n = (size_t)slice();
    switch (p) {
      case P_BLAS: return (size_t)32 << 20; // cuBLAS's work space (source: prover/common/blas.h)
      case P_A0:
      case P_A1: return T * d;
      case P_W0:
      case P_W1: return n * d;
      case P_C0:
      case P_C1:
      case P_C2: return n * T * sizeof(int);
      case P_BEST:
      case P_CHOSEN:
      case P_GAP:
      case P_LIFT:
      case P_PRICE:
      case P_SPARE: return T * 8;
      case P_HIST: return T * (size_t)L.UT * sizeof(u32);
      default: return T * sizeof(int);
    }
  }
  size_t part_at(int p) const {
    size_t at = 0;
    for (int q = 0; q < p; q++) at += (part_bytes(q) + 255) / 256 * 256;
    return at;
  }
  size_t scratch_bytes() const { return part_at(NPARTS); }
  template <class X> X *part(int p) const { return (X *)(scratch + part_at(p)); }
  // Gathers the pieces (at most eight) of segment sg within its call into a Pos2.
  Pos2 pos2(const char *sg) const {
    int id = L.segid(sg);
    const Seg &S = L.seg[id];
    const Call &c = L.call[S.call];
    if (S.pc.size() > 8) {
      printf("policy: a segment length has more than eight one-bits, so it would be cut into more than eight pieces\n");
      exit(2);
    }
    Pos2 p{};
    p.n = (int)S.pc.size();
    for (int i = 0; i < p.n; i++) {
      p.s[i] = (int)c.pc[S.pc[i]].s;
      p.off[i] = c.pc[S.pc[i]].off;
    }
    return p;
  }
  // kp: the key array of segment sg's call. cell: the first cell of limb l of tensor X.
  u16 *kp(const char *sg) const { return key[L.seg[L.segid(sg)].call]; }
  i8 *cell(TenH X, int l) const {
    const Ten &t = L.t[X];
    i8 *m = t.mat == 0 ? hf : t.mat == 1 ? hd : hr;
    return m + (size_t)(t.off + (long long)l * t.J) * L.T;
  }
  // The final norm at all T positions. xL is the last layer's output, [k][t], on the device.
  void norm(const ResidualInt *xL, cudaStream_t s) {
    const int T = L.T, d = L.s.d;
    if (zeroes_its_flags) CK(cudaMemsetAsync(err, 0, 4, s));
    norm_ledger(L.s.xl, xL, T, d, Wg, L.s.MF, L.s.epsF, cell(Y_row, 0), kp("ROW"), posRow, 0, Rv, sigv, err, 1, s);
    k_norm_out<<<grid_blocks((size_t)T * d), 256, 0, s>>>(xL, Rv, T, d, L.s.KF, cell(Y_aw, 0), cell(Y_ra, 0), cell(Y_aF, 0),
                                                    (size_t)d * T, (size_t)d * T, kp("NF"), kp("NFH"), kp("CNF"),
                                                    pos2("NF"), pos2("NFH"), pos2("CNF"), 0, 0, 0, 1, cm);
  }
  // A slice's limbs and their four products. The 32-bit sums are exact for d < 2^16: each has at
  // most 2 d terms of at most 2^14.
  void products(int v0, int n, cudaStream_t s) const {
    const int T = L.T, d = L.s.d;
    i8 *a0 = part<i8>(P_A0), *a1 = part<i8>(P_A1), *w0 = part<i8>(P_W0), *w1 = part<i8>(P_W1);
    k_head_limbs<<<grid_blocks(((size_t)n * d + 3) / 4), 256, 0, s>>>(whsrc, seed, d, L.s.V, v0, n, w0, w1);
    prover_gemm(h, a0, w0, part<int>(P_C0), T, n, d, 0, s);
    prover_gemm(h, a1, w0, part<int>(P_C1), T, n, d, 0, s);
    prover_gemm(h, a0, w1, part<int>(P_C1), T, n, d, 1, s);
    prover_gemm(h, a1, w1, part<int>(P_C2), T, n, d, 0, s);
    // An empty test hook. It is given only C0, in each of the three sweeps that compute the slice.
    lie::in_memory<int>(0, lie::HEAD, lie::LOGIT, s, [&](long long v, long long t, long long second) -> int * {
      return second != 0 || v < v0 || v >= v0 + n || t < 0 || t >= T ? nullptr : part<int>(P_C0) + (v - v0) * T + t;
    });
  }
  /* Test-only output. If the environment variable CHAIN_ROWS_OUT=<file> is set, `forward` appends
     one record of what it found at every position. The record is derived from the weights, so it is
     written only in a run on synthetic weights under a test seed (`made_up_under_a_tests_seed`,
     prover/common/made_up_alone.h). A production build (SERVING_BUILD) never reads the variable
     (TESTENV, prover/common/setting.h); in every other build the prover program cp_prove rejects
     any other run that has it set (prover/prover/isk.cuh, "test-made-up-alone"). Nothing in the
     record enters a proof or a response. A record is little-endian: the 32-bit words 0x32535752 and
     T; the T tokens, 32 bits each; each position's argmax, 32 bits; each position's logit gap, 64
     bits; each position's honest cost (k_head_price), 64 bits. A tool that is not part of this
     repository writes the same format. */
  void rows_out(const int *tok, cudaStream_t s) const {
    const char *path = made_up_under_a_tests_seed() ? TESTENV("CHAIN_ROWS_OUT") : nullptr;
    if (!path) return;
    const size_t T = (size_t)L.T;
    std::vector<unsigned long long> best(T);
    std::vector<u32> record(2 + 6 * T);
    record[0] = 0x32535752u, record[1] = (u32)T;
    CK(cudaMemcpyAsync(&record[2], tok, T * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(best.data(), part<unsigned long long>(P_BEST), T * 8, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(&record[2 + 2 * T], part<i64>(P_GAP), T * 8, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(&record[2 + 4 * T], part<i64>(P_PRICE), T * 8, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    for (size_t t = 0; t < T; t++) record[2 + T + t] = 0xFFFFFu - (u32)(best[t] & 0xFFFFF);
    static std::mutex one_at_a_time; // serializes the appends to the file
    std::lock_guard<std::mutex> held(one_at_a_time);
    FILE *f = fopen(path, "ab");
    if (!f || fwrite(record.data(), 4, record.size(), f) != record.size() || fclose(f)) {
      printf("CHAIN_ROWS_OUT: %s cannot be written\n", path);
      exit(1);
    }
  }
  // tok holds the request's T tokens, each below V. rows holds r and e. Both are on the device. No
  // cell or key is written for the padding entries V <= v < JV.
  void forward(const ResidualInt *xL, const int *tok, const u32 *rows, cudaStream_t s = cudaStreamPerThread) {
    const int T = L.T, d = L.s.d, V = L.s.V, step = slice();
    const int *C0 = part<int>(P_C0), *C1 = part<int>(P_C1), *C2 = part<int>(P_C2);
    unsigned long long *best = part<unsigned long long>(P_BEST);
    i64 *chosen = part<i64>(P_CHOSEN), *gap = part<i64>(P_GAP), *lift = part<i64>(P_LIFT);
    i64 *price = part<i64>(P_PRICE), *spare = part<i64>(P_SPARE);
    u32 *hist = part<u32>(P_HIST);
    const int F = L.s.F;
    int *win = part<int>(P_WIN);
    h = blas::blas(s, part<char>(P_BLAS));
    lie::here(0, lie::HEAD, s); // an empty test hook: the head is layer 0 of the part lie::HEAD
    // The final norm's kernels are a layer's, so their test hooks have a layer norm's value names
    // (lie::SIGMA, ..). The hook below, empty too, is given xL, not its committed cells.
    lie::in_memory<ResidualInt>(0, lie::HEAD, lie::STREAM, s, [&](long long k, long long t, long long second) -> ResidualInt * {
      return second != 0 || k < 0 || k >= d || t < 0 || t >= T ? nullptr : const_cast<ResidualInt *>(xL) + k * T + t;
    });
    norm(xL, s);
    transpose_i8(cell(Y_aF, 0), part<i8>(P_A0), d, T, s);
    transpose_i8(cell(Y_aF, 1), part<i8>(P_A1), d, T, s);
    CK(cudaMemsetAsync(best, 0, (size_t)T * 8, s));
    for (int v0 = 0; v0 < V; v0 += step) {
      const int n = std::min(step, V - v0);
      products(v0, n, s);
      k_head_best<<<dim3((unsigned)((T + 255) / 256), (unsigned)((n + 63) / 64)), 256, 0, s>>>(C0, C1, C2, T, v0, n,
                                                                                             best);
    }
    k_head_y<<<1, 1, 0, s>>>(best, rows, T, y);
    k_head_chosen<<<(int)(((size_t)T * 32 + 255) / 256), 256, 0, s>>>(part<i8>(P_A0), part<i8>(P_A1), T, d, V, whsrc,
                                                                    seed, tok, y, best, win, chosen, gap);
    CK(cudaMemsetAsync(hist, 0, part_bytes(P_HIST), s));
    for (int v0 = 0; v0 < V; v0 += step) {
      const int n = std::min(step, V - v0);
      products(v0, n, s);
      k_head_hist<<<grid_blocks((size_t)n * T), 256, 0, s>>>(C0, C1, C2, T, v0, n, chosen, gap, win, L.s.sh, L.UT, hist);
    }
    k_head_price<<<T, 256, 0, s>>>(hist, EX, L.UT, V, F, L.s.sh, gap, price, spare);
    k_head_budget<<<1, 256, 0, s>>>(price, spare, rows, T, V, F, L.s.C, L.s.theta, L.s.cap, norm_row_len(L.s.xl), cell(Y_row, 0),
                                  kp("ROW"), posRow, lift, err);
    for (int v0 = 0; v0 < V; v0 += step) {
      const int n = std::min(step, V - v0);
      products(v0, n, s);
      k_head_margins<<<grid_blocks((size_t)n * T), 256, 0, s>>>(C0, C1, C2, T, v0, n, chosen, lift, win, rows, L.s.sh, EX,
                                                       cell(Y_d1, 0), (size_t)L.JV * T, cell(Y_d2, 0),
                                                       cell(Y_e, 0), kp("HD0"),
                                                       kp("HD1"), kp("HD2"), pos2("HD0"), pos2("HD1"), pos2("HD2"),
                                                       err);
    }
    // A test hook, empty in this repository, given the cells of hf after the forward pass. A test
    // build outside this repository alters one cell here; nothing is computed again from it.
    lie::in_memory<i8>(0, lie::HEAD, lie::CELL, s, [&](long long feature, long long t, long long second) -> i8 * {
      return second != 0 || feature < 0 || feature >= 4ll * d + 6 * L.JV || t < 0 || t >= T ? nullptr : hf + feature * T + t;
    });
    rows_out(tok, s);
  }
};
