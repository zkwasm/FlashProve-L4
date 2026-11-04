// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests prover/hash_commitment/hf_code.cuh (namespace hf), the code that encodes the rows of the
// weight table (a matrix of all the weights), against hf_code_ref.h beside it (namespace hfr), a
// slow reference that computes each value from its definition.
// Part 1 needs no GPU: cpu_checks() verifies facts that hf_code.cuh relies on, without calling it;
// host_checks() compares its host functions with the reference (position_of(): with constants).
// Part 2, gpu_checks(), does the same for its GPU functions, with tables of u64, never of
// code_layers::Prepared; at lengths 2^9, 2^33 and 2^34 (the longest) on selected runs or positions
// only. Known answers from another program (code_test_values.py beside this file) exist for
// hf_f_values() only: nine, in host_checks().
//
// Terms. The code has n = 2^ln positions. Position I has a point z_I and factors f_a(z_I), which
// are z_I^(2^a) in the Goldilocks field (p = 2^64 - 2^32 + 1). For a row of coefficients c, the
// code symbol Enc_F(c)[I] is the sum over i of c[i] times the product of f_a(z_I) over the set bits
// a of i. Run h of height m is the 2^m positions from h 2^m on. Its fold is 2^m sums of c, weighted
// by products of the high factors (f_a with a >= m). The encoding transform's m layers (layers()
// below), with the run's table of factors, turn the fold into the run's code symbols.
#include "hash_commitment/hf_code.cuh"
#include "hash_commitment/hf_code_ref.h"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <set>
#include <vector>
using std::vector;

static int failed = 0, checks = 0;
static void check(bool ok, const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vprintf(format, arguments);
  va_end(arguments);
  printf(": %s\n", ok ? "ok" : "FAILS");
  checks++, failed += !ok;
}
static size_t differing(const vector<u64> &x, const vector<u64> &y) {
  size_t count = x.size() == y.size() ? 0 : 1;
  for (size_t i = 0; i < x.size() && i < y.size(); i++) count += x[i] != y[i];
  return count;
}
// Compares got with want. Negative control: flipping one bit of got must change the number of
// differing words by one. An empty got fails.
static void same(vector<u64> got, const vector<u64> &want, const char *format, ...) {
  char name[400];
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(name, sizeof name, format, arguments);
  va_end(arguments);
  const size_t differ = differing(got, want);
  bool seen = !got.empty();
  if (seen) {
    got[got.size() / 2] ^= 1;
    const size_t now = differing(got, want);
    seen = now == differ + 1 || (differ && now == differ - 1);
  }
  check(!differ && seen, "%s: %zu of %zu words differ; negative control: a changed word is %s", name, differ,
        want.size(), seen ? "counted" : "not counted");
}
// Fixed seed: the same values every time.
static u64 g_state = 20261010;
static u64 word() { // SplitMix64
  u64 z = (g_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// row_commit_ref (prover/common/hash_tree_ref.h) has the field's arithmetic on the CPU.
static u64 element() {
  return word() % row_commit_ref::P;
}
static vector<u64> elements(size_t count) {
  vector<u64> v(count);
  for (u64 &x : v) x = element();
  return v;
}
// The transform's layers m - 1 .. 0 on 2^m words with the table tw, on the CPU (on the GPU:
// prover/common/code_layers.cuh). Layer L pairs positions 2^L apart:
// (a, b) := (a + w b, a - w b) with w = tw[2^(m - 1 - L) + (position >> (L + 1))].
static vector<u64> layers(vector<u64> v, const vector<u64> &tw, int m) {
  for (int L = m - 1; L >= 0; L--)
    for (u64 at = 0; at < v.size(); at++)
      if (!((at >> L) & 1)) {
        const u64 a = v[at], t = row_commit_ref::mul(v[at + ((u64)1 << L)], tw[((size_t)1 << (m - 1 - L)) + (at >> (L + 1))]);
        v[at] = row_commit_ref::add(a, t), v[at + ((u64)1 << L)] = row_commit_ref::sub(a, t);
      }
  return v;
}

static void cpu_checks() {
  for (int ln : {1, 2, 5, 9}) { // The points all differ; positions 2 k and 2 k + 1 have z and -z.
    std::set<u64> seen;
    bool pairs = true;
    for (u64 I = 0; I < ((u64)1 << ln); I++) {
      const u64 z = hfr::factors(ln, I, 1)[0];
      seen.insert(z), pairs = pairs && row_commit_ref::add(z, hfr::factors(ln, I ^ 1, 1)[0]) == 0;
    }
    check(pairs && seen.size() == ((size_t)1 << ln) && !seen.count(0), "ln = %d: the points differ, none is 0, neighbours are z and -z", ln);
  }
  { // A position has the high factors of run h exactly if it is in run h (ln = 8 only; m < 8).
    const int ln = 8;
    for (int m = 0; m <= ln; m++) {
      size_t wrong = 0;
      for (u64 h = 0; h < ((u64)1 << (ln - m)); h++) {
        const vector<u64> f0 = hfr::factors(ln, h << m, ln);
        for (u64 I = 0; I < ((u64)1 << ln); I++) {
          const vector<u64> f = hfr::factors(ln, I, ln);
          bool equal = true;
          for (int a = m; a < ln; a++) equal = equal && f[a] == f0[a];
          if (m < ln && equal != ((I >> m) == h)) wrong++;
        }
      }
      check(!wrong, "ln = %d, m = %d: f_m, .. are equal exactly on the run: %zu positions otherwise", ln, m, wrong);
    }
  }
  for (int ln : {3, 7, 10}) { // The run 0 of height ln is the whole domain.
    row_commit_ref::Code code;
    code.init(ln);
    same(hfr::run_table(ln, ln, 0), vector<u64>(code.tw.begin(), code.tw.end()),
         "ln = %d: the table of the run 0 of height ln is the transform's table", ln);
  }
  { // Fold, then short sum or the layers: Enc_F on every run, for odd numbers of coefficients.
    const int ln = 9;
    for (u64 dim : {(u64)1, (u64)3, (u64)77, (u64)255, (u64)257}) {
      const vector<u64> c = elements(dim);
      vector<u64> want((size_t)1 << ln);
      for (u64 I = 0; I < want.size(); I++) want[I] = hfr::enc_at(ln, c.data(), dim, I);
      for (int m : {0, 1, 4, 7, 9}) {
        vector<u64> by_sum, by_layers;
        for (u64 h = 0; h < ((u64)1 << (ln - m)); h++) {
          const vector<u64> V = hfr::run_fold(ln, m, h, c.data(), dim), s = hfr::run_finish(ln, m, h, V),
                            t = layers(V, hfr::run_table(ln, m, h), m);
          by_sum.insert(by_sum.end(), s.begin(), s.end()), by_layers.insert(by_layers.end(), t.begin(), t.end());
        }
        same(by_sum, want, "ln = %d, %llu coefficients, m = %d: fold and short sum give Enc_F", ln,
             (unsigned long long)dim, m);
        same(by_layers, want, "ln = %d, %llu coefficients, m = %d: fold, then the layers with the runs' tables: Enc_F",
             ln, (unsigned long long)dim, m);
      }
    }
  }
}

// A synthetic element of the cubic extension field E (hfr::E): (1000 family + 3 v + 1, .. + 2,
// .. + 3), squared six times. made(1, v) is made(v) of code_test_values.py.
static hfr::E made(u64 family, u64 v) {
  hfr::E x = {1000 * family + 3 * v + 1, 1000 * family + 3 * v + 2, 1000 * family + 3 * v + 3};
  for (int i = 0; i < 6; i++) x = hfr::e_mul(x, x);
  return x;
}
// The sum over i < prefix of the product over v of (one[v] if bit v of i is set, else zero[v]);
// v = 0 is the most significant of zero.size() bits. One product for each set bit of the prefix:
// the i that agree with it above that bit, have 0 there and are free below.
// Requires prefix < 2^zero.size() (not checked).
static hfr::E prefix_sum(const vector<hfr::E> &zero, const vector<hfr::E> &one, u64 prefix) {
  const int n = (int)zero.size();
  hfr::E out = {0, 0, 0}, head = {1, 0, 0};
  for (int v = 0; v < n; v++)
    if ((prefix >> (n - 1 - v)) & 1) {
      hfr::E term = hfr::e_mul(head, zero[v]);
      for (int w = v + 1; w < n; w++) term = hfr::e_mul(term, hfr::e_add(zero[w], one[w]));
      out = hfr::e_add(out, term), head = hfr::e_mul(head, one[v]);
    } else
      head = hfr::e_mul(head, zero[v]);
  return out;
}
// Edge positions, then random ones. With ln > 32 the second coset starts at position 2^32.
static vector<u64> some_positions(int ln, int count) {
  const u64 n = (u64)1 << ln;
  vector<u64> pos = {0, 1, n - 1, n / 2, n / 2 - 1};
  if (ln > 32) pos.insert(pos.end(), {((u64)1 << 32) - 1, (u64)1 << 32, ((u64)1 << 32) + 1});
  while ((int)pos.size() < count) pos.push_back(word() & (n - 1));
  return pos;
}

// The reference's factors come from blk::gl64_factors() (blocks/code/cosets.h).
// hfr::run_high() is given a random position of the run: every position must give the same weights.
static void host_checks() {
  for (int ln : {1, 2, 7, 12, 24, 33, 34}) {
    const hf::Domain D = hf::domain(ln);
    const vector<u64> pos = some_positions(ln, 40);
    vector<u64> want;
    for (u64 I : pos) {
      const vector<u64> f = hfr::factors(ln, I, ln);
      want.insert(want.end(), f.begin(), f.end());
    }
    same(hf::position_factors_host(D, pos.data(), (int)pos.size(), ln), want, "ln = %d: position_factors_host()", ln);
    for (int m : {0, 1, 5, 7})
      if (m <= ln) {
        const u64 h = word() & (((u64)1 << (ln - m)) - 1), count = (u64)1 << std::min(ln - m, 6);
        vector<u64> table((size_t)1 << m);
        for (u64 t = 0; t < table.size(); t++) table[t] = hf::run_table_entry(D, m, h, t);
        same(table, hfr::run_table(ln, m, h), "ln = %d, m = %d: run_table_entry()", ln, m);
        same(hf::run_high(D, m, h, count), hfr::run_high(ln, m, (h << m) + (word() & (((u64)1 << m) - 1)), count),
             "ln = %d, m = %d: run_high()", ln, m);
      }
  }
  // Tall runs, up to height 30, the largest that run_encode() accepts: sampled entries of the
  // table, and all weights of the fold.
  for (int m : {20, 27, 30}) {
    const int ln = 33;
    const hf::Domain D = hf::domain(ln);
    const u64 h = word() & (((u64)1 << (ln - m)) - 1);
    vector<u64> got, want;
    for (int L = 0; L < m; L++)
      for (int i = 0; i < 3; i++) {
        const u64 most = ((u64)1 << (m - 1 - L)) - 1, tt = i == 0 ? 0 : i == 1 ? most : word() & most;
        got.push_back(hf::run_table_entry(D, m, h, ((u64)1 << (m - 1 - L)) + tt));
        want.push_back(hfr::factors(ln, (h << m) + (tt << (L + 1)), L + 1)[L]);
      }
    same(got, want, "ln = 33, m = %d: run_table_entry() at three entries of every layer", m);
    const u64 all = (u64)1 << (hf::most_height(D) - m); // at most 2^32 coefficients
    same(hf::run_high(D, m, h, all), hfr::run_high(ln, m, (h << m) + (word() & (((u64)1 << m) - 1)), all),
         "ln = 33, m = %d: run_high(), all weights", m);
  }
  // A position from two words: at ln = 33 bit 0 of the second word counts; at ln <= 32 it does not.
  check(hf::position_of(0xFFFFFFFFu, 0xFFFFFFFFu, 33) == 0x1FFFFFFFFull && hf::position_of(5, 2, 33) == 5 &&
            hf::position_of(5, 3, 33) == 0x100000005ull && hf::position_of(0x80000005u, 1, 31) == 5 &&
            hf::position_of(0x80000005u, 1, 32) == 0x80000005ull,
        "position_of(): (o0 + 2^32 o1) mod 2^ln at ln = 31, 32, 33");
  // Known answers: Enc_F's sum at position I for one row of KD coefficients in E, computed from
  // hf_f_values()'s factors (`lc` is their number), against the value that code_test_values.py
  // prints. Coefficient i is the product over v < 29 of made(1, v) if bit 28 - v of i is set, else
  // of 1 - made(1, v). Positions: 0, 1, 2^32 - 1 .. 2^32 + 1, 2^33 - 1 at ln = 33; 2^31 - 1 ..
  // 2^31 + 1 at ln = 32. That program's docstring says how far its values are independent of
  // hf_code.cuh: least at the three positions from 2^32 on.
  const u64 KD = 446693375; // 3 x 2^27 + 42 x 2^20 - 1 coefficients
  struct Lengths {
    int ln, lc;
  };
  struct Known {
    int ln;
    u64 I;
    u32 w[6];
  };
  const Known known[] = {{33, 0, {0xc45107c1, 0xb56c4103, 0x5c58c9f3, 0x3d6c398f, 0xd81ff377, 0xf51a58f4}},
                         {33, 1, {0x822966dc, 0xc96e2c17, 0xcaea8cb2, 0x7c5fdc27, 0x19e6d221, 0x8d651ebb}},
                         {33, 4294967295ull, {0x46b6c481, 0x01cfb014, 0xb4acd559, 0x110854f9, 0x47495942, 0xf561a014}},
                         {33, 4294967296ull, {0x4d673f79, 0x11898a33, 0x6c8f703a, 0x3b5bd782, 0xf8cf5002, 0x832dee37}},
                         {33, 4294967297ull, {0xa635a508, 0x56b29623, 0x285b0894, 0xdbc02397, 0xe70f684a, 0xd05ecf65}},
                         {33, 8589934591ull, {0x739079ef, 0x295b89d4, 0xa7aa0c4a, 0xe77bb87d, 0x7a2d2535, 0x67cccb64}},
                         {32, 2147483647ull, {0x5bce0adb, 0xd46cf0eb, 0x496bc073, 0x5ed98bc4, 0xe907e942, 0x4de87f37}},
                         {32, 2147483648ull, {0xc0baf746, 0x8ec99e6e, 0x93b4d412, 0x175c6a09, 0xeac1c456, 0xc918f231}},
                         {32, 2147483649ull, {0x760bf5f7, 0xe8c42fef, 0x383985ab, 0x2856003c, 0x01d01a8f, 0xe2d2e1a5}}};
  for (const Known &k : known) {
    const Lengths S = {k.ln, 29};
    vector<u64> f(S.lc);
    hf_f_values(S, k.I, f.data());
    vector<hfr::E> zero, one;
    for (int v = 0; v < S.lc; v++) {
      const hfr::E pt = made(1, v);
      zero.push_back(hfr::e_sub(hfr::E{1, 0, 0}, pt)), one.push_back(hfr::e_scale(pt, f[S.lc - 1 - v]));
    }
    const hfr::E got = prefix_sum(zero, one, KD);
    same(vector<u64>(got.begin(), got.end()),
         vector<u64>{k.w[0] | ((u64)k.w[1] << 32), k.w[2] | ((u64)k.w[3] << 32), k.w[4] | ((u64)k.w[5] << 32)},
         "hf_f_values() at the length 2^%d, I = %llu: the known answer of code_test_values.py", k.ln, (unsigned long long)k.I);
  }
}

#ifdef __CUDACC__
#define ST cudaStreamPerThread
template <class T> struct Gpu {
  T *p = nullptr;
  size_t count = 0;
  explicit Gpu(size_t count_) : count(count_) { CK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T))); }
  explicit Gpu(const vector<T> &v) : Gpu(v.size()) {
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Gpu() { cudaFree(p); }
  vector<T> host() const {
    vector<T> v(count);
    CK(cudaStreamSynchronize(ST));
    CK(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
static void gpu_checks() {
  // Tables of runs: every run at two small lengths, then four runs at 2^33 and 2^34, one of which
  // starts at position 2^32.
  for (int ln : {6, 11}) {
    const hf::Domain D = hf::domain(ln);
    for (int m = 0; m <= ln; m++) {
      const int count = 1 << (ln - m);
      vector<u64> h(count), want;
      for (int x = 0; x < count; x++) {
        h[x] = (u64)(count - 1 - x);
        const vector<u64> t = hfr::run_table(ln, m, h[x]);
        want.insert(want.end(), t.begin(), t.end());
      }
      Gpu<u64> hh(h), tw((size_t)count << m);
      hf::run_tables(D, m, hh.p, count, tw.p, ST);
      same(tw.host(), want, "ln = %d, m = %d: run_tables(), all runs", ln, m);
    }
  }
  for (int ln : {33, 34})
    for (int m : {1, 7, 11}) {
      const hf::Domain D = hf::domain(ln);
      vector<u64> h = {0, ((u64)1 << (ln - m)) - 1, (u64)1 << (32 - m), word() & (((u64)1 << (ln - m)) - 1)}, want;
      for (u64 x : h) {
        const vector<u64> t = hfr::run_table(ln, m, x);
        want.insert(want.end(), t.begin(), t.end());
      }
      Gpu<u64> hh(h), tw(h.size() << m), one((size_t)1 << m);
      hf::run_tables(D, m, hh.p, (int)h.size(), tw.p, ST);
      same(tw.host(), want, "ln = %d, m = %d: run_tables(), four runs", ln, m);
      hf::run_table(D, m, h[3], one.p, ST);
      same(one.host(), hfr::run_table(ln, m, h[3]), "ln = %d, m = %d: run_table()", ln, m);
    }
  { // m = 21 is the smallest height at which the upper layers of `run_encode` take two launches.
    // The table is compared with hf::run_table_entry(), not the reference.
    const int ln = 33, m = 21;
    const hf::Domain D = hf::domain(ln);
    const u64 A = (u64)1 << m, h = word() & (((u64)1 << (ln - m)) - 1), dim = 2 * A + 12345;
    const vector<u64> coef = elements(dim);
    Gpu<u64> V(hfr::run_fold(ln, m, h, coef.data(), dim)), tw((size_t)A);
    hf::run_table(D, m, h, tw.p, ST);
    const vector<u64> table = tw.host();
    vector<u64> got, want;
    for (int i = 0; i < 4000; i++) {
      const u64 t = i < 2 ? (u64)i : i == 2 ? A - 1 : word() & (A - 1);
      got.push_back(table[t]), want.push_back(hf::run_table_entry(D, m, h, t));
    }
    same(got, want, "ln = 33, m = 21: run_table() at 4,000 entries");
    hf::run_encode(V.p, tw.p, m, 1, ST);
    const vector<u64> symbols = V.host();
    got.clear(), want.clear();
    for (int i = 0; i < 24; i++) {
      const u64 a = i < 2 ? (u64)i : i == 2 ? A - 1 : word() & (A - 1);
      got.push_back(symbols[a]), want.push_back(hfr::enc_at(ln, coef.data(), dim, (h << m) + a));
    }
    same(got, want, "ln = 33, m = 21, 2^22 + 12,345 coefficients: run_encode() gives Enc_F at 24 positions");
  }
  { // More than 32,768 runs: each function's loop makes a second launch. Also runs of height 0,
    // and input words that are not canonical (not below p).
    const int ln = 33, count = 40000;
    const hf::Domain D = hf::domain(ln);
    vector<u64> pos(count), want_tables, in, want_runs;
    for (u64 &I : pos) I = word() & (((u64)1 << ln) - 1);
    vector<u64> h(count);
    // V2[0] is small, so that p can be added to it within 64 bits.
    const u64 V2[2] = {element() % 1000, element()};
    for (int q = 0; q < count; q++) {
      h[q] = pos[q] >> 1;
      const vector<u64> t = hfr::run_table(ln, 1, h[q]);
      want_tables.insert(want_tables.end(), t.begin(), t.end());
      in.push_back(V2[0] + row_commit_ref::P), in.push_back(V2[1]); // the first word is not canonical
      for (u64 a = 0; a < 2; a++) want_runs.push_back(hfr::enc_at(ln, V2, 2, (h[q] << 1) + a));
    }
    Gpu<u64> p(pos), hh(h), tw((size_t)count * 2), i(in), o(in.size());
    hf::run_tables(D, 1, hh.p, count, tw.p, ST);
    same(tw.host(), want_tables, "ln = 33, m = 1: run_tables(), 40,000 runs");
    hf::short_runs_encode(i.p, tw.p, 1, 1, count, true, o.p, ST);
    same(o.host(), want_runs, "ln = 33, m = 1: short_runs_encode(), 40,000 runs, words that are not canonical");
    Gpu<u64> t0((size_t)count), o0((size_t)count);
    hf::run_tables(D, 0, p.p, count, t0.p, ST);
    hf::short_runs_encode(p.p, t0.p, 0, 1, count, false, o0.p, ST);
    vector<u64> as_they_are = pos;
    same(o0.host(), as_they_are, "ln = 33, m = 0: a run of one position: the symbol is the one sum");
  }
  // run_high() and run_encode(); the folds are the reference's.
  struct Shape {
    int ln, m, rows;
    u64 dim;
  };
  for (const Shape &s : vector<Shape>{{9, 1, 3, 77},    {9, 4, 3, 257},    {9, 9, 2, 255},   {13, 12, 2, 1001},
                                      {14, 13, 1, 4097}, {15, 14, 1, 16387}, {33, 7, 5, 1025}, {33, 13, 1, 8195},
                                      {33, 12, 2, 777},  {34, 7, 2, 513}}) {
    const hf::Domain D = hf::domain(s.ln);
    const u64 A = (u64)1 << s.m, runs = (u64)1 << (s.ln - s.m), nx = (s.dim + A - 1) >> s.m;
    const size_t pitch = s.dim + 3;
    const vector<u64> coef = elements(pitch * s.rows);
    vector<u64> hs = {0, runs - 1, word() & (runs - 1)};
    for (u64 h : hs) {
      const vector<u64> high = hf::run_high(D, s.m, h, nx);
      same(high, hfr::run_high(s.ln, s.m, (h << s.m) + (word() & (A - 1)), nx), "ln = %d, m = %d, run %llu: run_high()",
           s.ln, s.m, (unsigned long long)h);
      vector<u64> fold, want;
      for (int r = 0; r < s.rows; r++) {
        const vector<u64> V = hfr::run_fold(s.ln, s.m, h, coef.data() + r * pitch, s.dim);
        fold.insert(fold.end(), V.begin(), V.end());
        for (u64 a = 0; a < A; a++) want.push_back(hfr::enc_at(s.ln, coef.data() + r * pitch, s.dim, (h << s.m) + a));
      }
      Gpu<u64> V(fold), tw((size_t)A);
      hf::run_table(D, s.m, h, tw.p, ST);
      hf::run_encode(V.p, tw.p, s.m, s.rows, ST);
      same(V.host(), want, "ln = %d, m = %d, run %llu, %llu coefficients: run_encode() gives Enc_F", s.ln, s.m,
           (unsigned long long)h, (unsigned long long)s.dim);
    }
    if (s.m > hf::MOST_SHORT) continue;
    // The same runs together, once in each order of the output.
    vector<u64> in, by_row, by_column(hs.size() * s.rows << s.m);
    for (size_t x = 0; x < hs.size(); x++)
      for (int r = 0; r < s.rows; r++) {
        const vector<u64> V = hfr::run_fold(s.ln, s.m, hs[x], coef.data() + r * pitch, s.dim);
        in.insert(in.end(), V.begin(), V.end());
        for (u64 a = 0; a < A; a++) {
          const u64 v = hfr::enc_at(s.ln, coef.data() + r * pitch, s.dim, (hs[x] << s.m) + a);
          by_row.push_back(v), by_column[((x << s.m) + a) * s.rows + r] = v;
        }
      }
    Gpu<u64> hh(hs), tws(hs.size() << s.m), i(in), o(in.size());
    hf::run_tables(D, s.m, hh.p, (int)hs.size(), tws.p, ST);
    hf::short_runs_encode(i.p, tws.p, s.m, s.rows, (int)hs.size(), true, o.p, ST);
    same(o.host(), by_column, "ln = %d, m = %d: short_runs_encode(), by column", s.ln, s.m);
    hf::short_runs_encode(i.p, tws.p, s.m, s.rows, (int)hs.size(), false, i.p, ST);
    same(i.host(), by_row, "ln = %d, m = %d: short_runs_encode(), by row, in place", s.ln, s.m);
  }
}

#endif

// Build from prover/, into $OUT/hc/test_build/tests/ (OUT defaults to prover/build):
//   both parts, by nvcc: ./build.sh hc_unit_test tests/hf_code_test
//   part 1 only, by g++: ./build.sh hc_host_test tests/hf_code_test (makes hf_code_test_host)
// build.sh adds -DBLK_FIELD=2 (the field's number) and -I../blocks: the includes need both.
// A passing run ends with the line "0 of N checks failed" and exit code 0.
// Calls that hf_code.cuh rejects (code_refuse(), exit code 4) are not tried.
int main() {
  cpu_checks();
  host_checks();
#ifdef __CUDACC__
  gpu_checks();
#else
  printf("part 2 needs a build by nvcc and a GPU: not run\n");
#endif
  printf("%d of %d checks failed\n", failed, checks);
  return failed ? 1 : 0;
}
