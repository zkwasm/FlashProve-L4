// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The error-correcting code C_F, which encodes the rows of a weight table (all the weights, as a
// matrix of long rows), for the host and the GPU. This file accepts lengths n = 2^ln up to 2^34
// (2^32 by the `table` line of prover/examples/target_full_f2/policy.txt). The encoding transform
// is a fast Fourier transform, in layers of butterflies. Its table of twiddle factors would take
// 8 n bytes at such a length (row_tree::tables() in prover/common/hash_tree.cuh stops at 2^24), so
// nothing here reads one. A position's evaluation point is computed from at most 32 stored roots
// of unity (`Domain`), and rows are encoded on runs (aligned blocks of 2^m positions), each of
// which needs a table of only 2^m entries.
// The code is plain Reed-Solomon on cosets modulo p = 2^64 - 2^32 + 1.
// An element of the field F is a u64, and every element that a function here writes as a u64 is
// canonical (below p). The file has no state. No function allocates GPU memory or waits for the
// GPU; kernels run on the caller's stream. A failed check or launch ends the process.
// prover/hash_commitment/hf_code_ref.h computes the same values by the defining sums, and
// prover/hash_commitment/tests/hf_code_test.cu compares the two at lengths from 2^1 to 2^34 (2^32
// is not among them), at the long ones only at some positions and runs.
//
// The idea is public: J. W. Cooley, J. W. Tukey, "An Algorithm for the Machine Calculation of
// Complex Fourier Series", Mathematics of Computation 19 (1965): a transform of length r s as
// transforms of length r on sums over the other index.
// The layers' own sources are cited in prover/common/code_layers.cuh.
#pragma once
#include "common/field_host.h"    // HD, u32, u64
#include "common/hash_tree_ref.h" // the field's arithmetic without CUDA
#include "field/field_names.h"
#ifdef __CUDACC__
#include "common/common.h"
#include "common/code_layers.cuh"
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace hf {
// A plain C++ compiler can compile this file, so that a test can run the host functions on a
// machine without CUDA. The file then omits the kernels and the functions that launch them, and
// takes the field's three operations from namespace row_commit_ref (prover/common/hash_tree_ref.h), because
// prover/common/common.h needs CUDA's headers. Either way, the three operations compute
// modulo p: gf_add(), gf_sub(), gf_mul() and row_commit_ref.
#ifdef __CUDACC__
HD u64 code_add(u64 a, u64 b) {
  return gf_add(a, b);
}
HD u64 code_sub(u64 a, u64 b) {
  return gf_sub(a, b);
}
HD u64 code_mul(u64 a, u64 b) {
  return gf_mul(a, b);
}
#else
HD u64 code_add(u64 a, u64 b) {
  return row_commit_ref::add(a, b);
}
HD u64 code_sub(u64 a, u64 b) {
  return row_commit_ref::sub(a, b);
}
HD u64 code_mul(u64 a, u64 b) {
  return row_commit_ref::mul(a, b);
}
#endif
// The largest ln that domain() accepts. A parameter file (`policy.txt`) may ask for at most 32
// (blocks/frame/parameters.h); HfTableParams (prover/hash_commitment/hf_table_params.h) accepts up to HF_LN_MOST = 33.
constexpr int MOST_LN = 34;
// short_runs_encode() accepts runs of at most 2^MOST_SHORT positions: a run of that size fills the
// shared array of k_short_runs (32 KiB).
constexpr int MOST_SHORT = 12;

// Called when an internal check fails (in this file: of an argument or a constant): prints `what`
// and exits with code 4. After a failed launch, `CK` (prover/common/common.h) exits with code 1.
static inline void code_refuse(const char *what) {
  printf("hf_code: internal error: %s\n", what);
  exit(4);
}
// The smallest b with 2^b >= K: factor_product(f, j) with j < K reads only f[0 .. b - 1].
HD int bits_for(u64 K) {
  int b = 0;
  while (b < 64 && ((u64)1 << b) < K) b++;
  return b;
}

// ---------------------------------------------------------------------------------------------------------------------
// (1) The domain, points and factors.
// ---------------------------------------------------------------------------------------------------------------------

// The code symbol of a row of KD coefficients c_0 .. c_(KD - 1) at position I is
//     Enc_F(c)[I] = the sum over i < KD of c_i bas_i(z_I),   z_I = point(D, I).
// The basis value bas_i(z) is the product of the factors f_a(z), defined below, over the set bits
// a of i. Part (3) rests on two properties of the factors: f_a(z_I) does not depend on the bits of
// I below bit a, and bit a only flips its sign. Which indices i are data and which are masks
// (random coefficients encoded with the data) is decided by blk::Code (blocks/code/code.h), not
// here.
//
// The multiplicative group modulo p has order 2^32 (2^32 - 1), and 7 generates it. Let
// s = min(ln, 32), and let c = I >> s number the coset. Position I has the point
//     z_I = 7^(1 + c) times the product of root[i] over the set bits i < s of I,
// where root[i] has order 2^(i + 1).
// The points form one coset of the subgroup of order 2^s up to length 2^32, and 2^(ln - 32) cosets
// beyond it, because no subgroup of two-power order is larger. They are the points of
// blocks/code/cosets.h: prover/hash_commitment/tests/hf_code_test.cu compares them at some
// positions. f_a(z) = z^(2^a), so bas_i(z) = z^i. The two properties hold for a < s. For a >= s,
// f_a depends only on c and no bit flips its sign, so a run's height and a transform's layers end
// at s. run_high() also limits KD to 2^s.

// The domain of the code of length 2^ln: root[i] = w^(2^(31 - i)), i < s, where
// w = 7^(5 (2^32 - 1)) has order 2^32. Kernels take a `Domain` by value (264 bytes).
struct Domain {
  int ln, s;
  u64 root[32];
};
static inline Domain domain(int ln) {
  if (ln < 1 || ln > MOST_LN) code_refuse("domain: the code's length must be from 2^1 to 2^34");
  u64 w = 1, b = 7;
  for (u64 e = 5 * 0xFFFFFFFFull; e; e >>= 1, b = code_mul(b, b))
    if (e & 1) w = code_mul(w, b);
  Domain D{};
  D.ln = ln, D.s = ln < 32 ? ln : 32;
  for (int i = 31; i >= 0; i--, w = code_mul(w, w)) // w has the order 2^(i + 1)
    if (i < D.s) D.root[i] = w;
  if (D.root[0] != FIELD_P - 1 || w != 1) code_refuse("domain: the root of unity does not have the expected order");
  return D;
}
// A position from two sampled 32-bit words: (o0 + 2^32 o1) mod 2^ln. o1 matters only if ln > 32.
// Only prover/hash_commitment/tests/hf_code_test.cu calls it.
HD u64 position_of(u32 o0, u32 o1, int ln) {
  return (o0 | ((u64)o1 << 32)) & (((u64)1 << ln) - 1);
}
// z_I. I must be below 2^ln, which is not checked: the first loop runs I >> s times.
HD u64 point(const Domain &D, u64 I) {
  u64 z = 7;
  for (u64 c = I >> D.s; c; c--) z = code_mul(z, 7); // there are at most four cosets
  for (int i = 0; i < D.s; i++)
    if ((I >> i) & 1) z = code_mul(z, D.root[i]);
  return z;
}
// f[a] = f_a(z_I) for a < count.
HD void factors(const Domain &D, u64 I, int count, u64 *f) {
  u64 x = point(D, I);
  for (int a = 0; a < count; a++) {
    f[a] = x;
    x = code_mul(x, x);
  }
}
// The product of f[a] over the set bits a of j. If f points at factor f_b, this is the part of a
// basis value that comes from bits b, b + 1, .. of its index.
HD u64 factor_product(const u64 *f, u64 j) {
  u64 v = 1;
  for (int a = 0; (j >> a) != 0; a++)
    if ((j >> a) & 1) v = code_mul(v, f[a]);
  return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// (2) The factors of many positions.
// ---------------------------------------------------------------------------------------------------------------------

// fac[q nf + a] = f_a(z_(pos[q])) for q < count and a < nf <= ln. pos[q] is taken modulo 2^ln.
static inline std::vector<u64> position_factors_host(const Domain &D, const u64 *pos, int count, int nf) {
  if (nf < 1 || nf > D.ln) code_refuse("position_factors_host: the number of factors must be from 1 to ln");
  std::vector<u64> fac((size_t)std::max(count, 0) * nf);
  for (int q = 0; q < count; q++) factors(D, pos[q] & (((u64)1 << D.ln) - 1), nf, fac.data() + (size_t)q * nf);
  return fac;
}

// ---------------------------------------------------------------------------------------------------------------------
// (3) Runs.
// ---------------------------------------------------------------------------------------------------------------------

// Run h of height m is the set of 2^m positions h 2^m + a, a < 2^m. The factors f_m, f_(m + 1), ..
// are constant on it; prover/hash_commitment/tests/hf_code_test.cu checks this only at length 2^8,
// on the factors that prover/hash_commitment/hf_code_ref.h computes. So for a row of coefficients
// c_0 .. c_(KD - 1)
//     Enc_F(c)[h 2^m + a] = the sum over i' < 2^m of bas_(i')(z_(h 2^m + a)) V[i'],
//     V[i'] = the sum over x with x 2^m + i' < KD of high[x] c[x 2^m + i'],
//     high[x] = the product of f_(m + k)(z_(h 2^m)) over the set bits k of x.
// V is the row's fold, which the caller computes, and the high[x] are the fold weights. The first
// line is the last m layers, m - 1 .. 0, of the transform of length n on these positions. Of the
// table of length n, layer L < m reads entries 2^(ln - 1 - L) + (h 2^(m - 1 - L) + t),
// t < 2^(m - 1 - L). The run's table has them at 2^(m - 1 - L) + t, which is the layout of a table
// for length 2^m. So the layers of prover/common/code_layers.cuh work on a run unchanged, with m in
// place of ln. run_encode() serves the committer (the program cp_commit), which encodes a table in
// long runs; from height code_groups::LEAST_HEIGHT on, the committer calls
// code_groups::run_encode() instead (prover/hash_commitment/hf_code_groups.cuh).
// short_runs_encode() serves an opening of the table at sampled positions, which encodes only the
// short runs around them.
//
// Folds and code symbols are hidden values: the owner of a buffer that contains them must erase it.
// Points, factors, basis values, tables and fold weights depend only on public or sampled values.

// Entry t < 2^m of the table of run h of height m. Entry 0 is 0. Entry 2^(m - 1 - L) + tt is f_L at
// position h 2^m + tt 2^(L + 1), whose bit L is 0. The arguments are not checked.
HD u64 run_table_entry(const Domain &D, int m, u64 h, u64 t) {
  if (!t) return 0;
  int d = 0;
  while ((t >> (d + 1)) != 0) d++;
  const int ln = D.ln, L = m - 1 - d;
  const u64 up = (h << d) | (t - ((u64)1 << d)); // the position's bits L + 1 .. ln - 1
  (void)ln;
  u64 x = point(D, up << (L + 1));
  for (int a = 0; a < L; a++) x = code_mul(x, x);
  return x;
}
// The largest height of a run, which is also the largest number of layers of a transform: one
// coset has 2^s positions.
HD int most_height(const Domain &D) {
  return D.s;
}
#ifdef __CUDACC__
// tw[x 2^m + t] = entry t of the table of run h[x] (h0 + x if h is null). Launch: ceil(2^m / 256)
// by (the number of runs) blocks of 256 threads. `Entry` is u64, for the entries themselves, or
// code_layers::Prepared, for the form in which a butterfly takes them.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
template <class Entry>
__global__ void __launch_bounds__(256)
    k_run_tables(const Domain D, int m, const u64 *__restrict__ h, u64 h0, Entry *__restrict__ tw) {
  const u64 t = (u64)blockIdx.x * 256 + threadIdx.x;
  if (t >> m) return;
  code_layers::put_factor(tw, ((size_t)blockIdx.y << m) + t, run_table_entry(D, m, h ? h[blockIdx.y] : h0 + blockIdx.y, t));
}
// The tables of the runs h[x], x < count, in the layout of k_run_tables. h is in GPU memory. Its
// run numbers must be below 2^(ln - m), which is not checked.
static inline void run_tables(const Domain &D, int m, const u64 *h, int count, u64 *tw, cudaStream_t st) {
  if (count <= 0) return;
  if (m < 0 || m > most_height(D)) code_refuse("run_tables: a run's height must be from 0 to ln, and within one coset");
  if (!h) code_refuse("run_tables: the list of run numbers is missing");
  for (int x = 0; x < count; x += 32768) { // the grid's second extent ends at 65,535
    k_run_tables<u64><<<dim3((unsigned)((((u64)1 << m) + 255) / 256), (unsigned)std::min(32768, count - x)), 256, 0, st>>>(
        D, m, h + x, 0, tw + ((size_t)x << m));
    CK(cudaGetLastError());
  }
}
// The table of the single run h: 2^m entries of either type (see k_run_tables).
template <class Entry> static inline void run_table(const Domain &D, int m, u64 h, Entry *tw, cudaStream_t st) {
  if (m < 0 || m > most_height(D)) code_refuse("run_table: a run's height must be from 0 to ln, and within one coset");
  if (m < D.ln ? (h >> (D.ln - m)) != 0 : h != 0) code_refuse("run_table: the run's number must be below 2^(ln - m)");
  k_run_tables<Entry><<<dim3((unsigned)((((u64)1 << m) + 255) / 256), 1), 256, 0, st>>>(D, m, nullptr, h, tw);
  CK(cudaGetLastError());
}
#endif // __CUDACC__
// The fold weights high[x], x < count, of run h of height m. Host code. A row of KD coefficients
// needs count = ceil(KD / 2^m).
static inline std::vector<u64> run_high(const Domain &D, int m, u64 h, u64 count) {
  if (m < 0 || m > most_height(D)) code_refuse("run_high: a run's height must be from 0 to ln, and within one coset");
  if (m < D.ln ? (h >> (D.ln - m)) != 0 : h != 0) code_refuse("run_high: the run's number must be below 2^(ln - m)");
  if (m + bits_for(count) > most_height(D))
    code_refuse("run_high: count, the number of weights, must be at most 2^(ln - m), and the coefficients must fit one coset");
  u64 f[MOST_LN];
  factors(D, h << m, D.ln, f);
  std::vector<u64> high(count);
  for (u64 x = 0; x < count; x++) high[x] = factor_product(f + m, x);
  return high;
}
#ifdef __CUDACC__
// Applies the last m layers, in place, to `rows` rows on one run h, whose table is tw. On entry
// cw[r 2^m + i'] = V[i'] of row r, as any 64-bit word of its residue class. On return
// cw[r 2^m + a] = Enc_F(row r)[h 2^m + a], canonical.
template <class Entry> static inline void run_encode(u64 *cw, const Entry *tw, int m, int rows, cudaStream_t st) {
  if (rows <= 0) return;
  if (m < 1 || m > 30) code_refuse("run_encode: a run's height must be from 1 to 30");
  const int last = std::min(m, code_layers::TILE);
  if ((((u64)rows << m) >> last) > 0x7FFFFFFFull) code_refuse("run_encode: too many rows for one launch");
  code_layers::upper_layers(cw, tw, m, m, last, rows, st);
  code_layers::last_layers<u64>(cw, tw, m, last, rows, nullptr, 0, 0, st);
}
// One block encodes one row of one short run, with the m layers in shared memory.
// in[(x rows + r) 2^m + i'] is V[i'] of row r for the x-th run, as any word of its residue class.
// The row's code symbol at position a of that run goes to out[(x 2^m + a) rows + r] if by_column,
// otherwise to out[(x rows + r) 2^m + a]. Launch: rows by (the number of runs) blocks of at most
// 1,024 threads.
// The compiler hint: __launch_bounds__, cited at k_run_tables.
__global__ void __launch_bounds__(1024)
    k_short_runs(const u64 *__restrict__ in, const u64 *__restrict__ tw, int m, u32 rows, bool by_column,
                 u64 *__restrict__ out) {
  __shared__ u64 s[1 << MOST_SHORT];
  const u32 A = 1u << m, r = blockIdx.x;
  const size_t x = blockIdx.y;
  const u64 *w = tw + (x << m);
  for (u32 i = threadIdx.x; i < A; i += blockDim.x) s[i] = code_layers::canonical(in[((x * rows + r) << m) + i]);
  __syncthreads();
  for (int L = m - 1; L >= 0; L--) {
    for (u32 b = threadIdx.x; b < A / 2; b += blockDim.x) {
      const u32 up = b >> L, at = (up << (L + 1)) | (b & ((1u << L) - 1));
      const u64 a0 = s[at], t = gf_mul(s[at + (1u << L)], w[(1u << (m - 1 - L)) + up]);
      s[at] = gf_add(a0, t), s[at + (1u << L)] = gf_sub(a0, t);
    }
    __syncthreads();
  }
  for (u32 a = threadIdx.x; a < A; a += blockDim.x)
    out[by_column ? ((x << m) + a) * rows + r : ((x * rows + r) << m) + a] = s[a];
}
// Applies the last m layers to `count` short runs of `rows` rows each. tw contains the tables of
// these runs, as run_tables() writes them. `in` and `out` are as in k_short_runs. Without by_column
// they may be the same buffer: a block reads its 2^m words, passes a barrier and only then writes
// those same words, which no other block touches. Strictly, such a call breaks the promise of the
// kernel's __restrict__ on both pointers. An opening calls it so (hf_side_symbols() in
// prover/hash_commitment/hf_open_side.cuh), and prover/hash_commitment/tests/hf_code_test.cu checks
// the result of such a call.
// With by_column the two must not overlap, and only in == out is rejected. The rows' symbols at one
// position are then adjacent, as in a column of the weight table. Only the test passes by_column.
static inline void short_runs_encode(const u64 *in, const u64 *tw, int m, int rows, int count, bool by_column,
                                     u64 *out, cudaStream_t st) {
  if (rows <= 0 || count <= 0) return;
  if (m < 0 || m > MOST_SHORT) code_refuse("short_runs_encode: a short run's height must be from 0 to 12");
  if (by_column && in == out) code_refuse("short_runs_encode: with by_column, in and out must be two buffers");
  const unsigned threads = (unsigned)std::min(1024, std::max(32, (1 << m) / 2));
  for (int x = 0; x < count; x += 32768) {
    k_short_runs<<<dim3((unsigned)rows, (unsigned)std::min(32768, count - x)), threads, 0, st>>>(
        in + (((size_t)x * rows) << m), tw + ((size_t)x << m), m, (u32)rows, by_column,
        out + (((size_t)x * rows) << m));
    CK(cudaGetLastError());
  }
}
#endif // __CUDACC__
} // namespace hf

// ---------------------------------------------------------------------------------------------------------------------
// Entry point that takes the parameters.
// ---------------------------------------------------------------------------------------------------------------------
// Host code. f[a] = f_a(z_I) for a < P.lc; I must be below 2^P.ln, which is not checked. `Params`
// is any type with the fields ln and lc (the number of bits of a coefficient index), such as
// hf::HfParams (prover/hash_commitment/hf_params.h).
// Only prover/hash_commitment/tests/hf_code_test.cu calls hf_f_values().
template <class Params> inline void hf_f_values(const Params &P, uint64_t I, uint64_t *f) {
  hf::factors(hf::domain(P.ln), I, P.lc, (u64 *)f);
}
