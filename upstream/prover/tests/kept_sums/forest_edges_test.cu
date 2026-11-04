// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests, on a GPU, the field arithmetic of the prover's kernels for the lookup argument (which is
// to show that every key occurs in a public table), on random inputs and at the extremes of each
// function's input range. The functions are called directly: no proof is made or checked. Under
// test, in prover/hiding/: gram_products.cuh (namespace `grm`), forest_keys.cuh (`fk`),
// forest_fused_folds.cuh and forest_fused_second.cuh (`ff`), pt_field.h (struct `FK2`),
// pt_transcript.h (`U4`). Expected values are computed in plain integers of up to 128 bits with the
// compiler's remainder, never with the prover's field functions. The program exits with code 1 if a
// printed line ends in FAILS or CK() sees a CUDA error, else with 0. Build and run (the field is
// p = 2^64 - 2^32 + 1; the source path is relative to prover/):
//   prover/tests/unit.sh both tests/kept_sums/forest_edges_test.cu <directory>
#include "hiding/product_test.cuh"

#include "hiding/forest_keys.cuh"

#include "hiding/forest_fused.cuh"

#include "hiding/forest_fused_folds.cuh"

#include "hiding/forest_fused_first.cuh"

#include "hiding/forest_fused_second.cuh"
#include <string>
#include <vector>
typedef unsigned __int128 V3U;
typedef __int128 V3I;
static const u64 V3P = FIELD_P;
// Prints and counts one result line. The test in SUITE.md that runs this program expects main()'s
// last line with its count, so a new call needs its pattern changed.
static int v3failures = 0, v3lines = 0;
static void v3line(bool ok, const char *what, size_t cases, size_t bad) {
  v3lines++, v3failures += !ok;
  printf("%s: %zu cases, %zu wrong: %s\n", what, cases, bad, ok ? "ok" : "FAILS");
}
// Reference arithmetic modulo p. hpow2(s) is 2^s; hsigned(v) is the residue of a signed v.
static u64 v3mul(u64 a, u64 b) { return (u64)((V3U)a * b % V3P); }
static u64 v3add(u64 a, u64 b) { return (u64)(((V3U)a + b) % V3P); }
static u64 v3red(u64 a) { return a % V3P; }
static u64 v3neg(u64 a) { return (V3P - a % V3P) % V3P; }
static u64 hpow2(int s) { u64 r = 1; for (int i = 0; i < s; i++) r = v3add(r, r); return r; }
static u64 hsigned(long long v) { return v < 0 ? v3neg((u64)0 - (u64)v) : v3red((u64)v); }
// Reference element of the cubic extension field, x^3 = FIELD_W; h3mul() is the schoolbook product.
// `T` (prover/hiding/transcript.cuh) is the prover's element; coord(x, j) reads its coordinate j.
struct V3H3 { u64 c[3]; };
static V3H3 h3mul(const V3H3 &a, const V3H3 &b) {
  const u64 w = FIELD_W;
  V3H3 r;
  r.c[0] = v3add(v3mul(a.c[0], b.c[0]), v3mul(w, v3add(v3mul(a.c[1], b.c[2]), v3mul(a.c[2], b.c[1]))));
  r.c[1] = v3add(v3add(v3mul(a.c[0], b.c[1]), v3mul(a.c[1], b.c[0])), v3mul(w, v3mul(a.c[2], b.c[2])));
  r.c[2] = v3add(v3add(v3mul(a.c[0], b.c[2]), v3mul(a.c[2], b.c[0])), v3mul(a.c[1], b.c[1]));
  return r;
}
static V3H3 h3(const T &x) { return V3H3{{coord(x, 0), coord(x, 1), coord(x, 2)}}; }
static bool v3same(const T &x, const V3H3 &y) { return coord(x, 0) == y.c[0] && coord(x, 1) == y.c[1] && coord(x, 2) == y.c[2]; }
// GPU memory: v3up() uploads, v3room() allocates, v3home() downloads and frees.
template <class X> X *v3up(const std::vector<X> &v) {
  X *p;
  CK(cudaMalloc(&p, v.size() * sizeof(X)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
  return p;
}
template <class X> X *v3room(size_t n) {
  X *p;
  CK(cudaMalloc(&p, n * sizeof(X)));
  CK(cudaMemset(p, 0x5A, n * sizeof(X)));
  return p;
}
template <class X> std::vector<X> v3home(X *p, size_t n) {
  std::vector<X> v(n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost));
  CK(cudaFree(p));
  return v;
}
// The largest loose word. A loose word represents its residue modulo p but need not be below p; a
// canonical word is below p.
static const u64 LOOSE_MAX = ~0ull;
// Random values (printed: drawn) are mix64() of a counter: each run of a build has the same inputs.
static u64 v3counter = 1;
static u64 v3draw() { return mix64(v3counter++); }
// Returns a word in 0 .. most, often one of `edges`. `MOST`: eight signed bytes at their largest.
static u64 word_upto(u64 most) {
  const u64 MOST = 0x7F7F7F7F7F7F7F7Full;
  const u64 edges[] = {0, 1, 2, most, most - 1, V3P - 1, V3P, V3P + 1, V3P - 2, MOST, MOST + 1, MOST - 1, 0xFFFFFFFFull, 0xFFFFFFFEull, 0x100000000ull,
                       0xFFFFFFFF00000000ull, 0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull, 0x8080808080808080ull, 0x7F7F7F7E7F7F7F81ull, V3P + 0xFFFFFFFEull, V3P + 0x7FFFFFFFull};
  const u64 z = v3draw();
  if ((z & 3) == 0) {
    const u64 e = edges[(z >> 8) % (sizeof edges / sizeof edges[0])];
    if (e <= most) return e;
  }
  if ((z & 3) == 1 && most == ~0ull) return V3P + (v3draw() % (0 - V3P)); // a word from p on
  u64 x = v3draw();
  if (most != ~0ull) x %= most + 1;
  return x;
}
static u64 v3canon() { return word_upto(V3P - 1); }
static T v3element() { return T{v3canon(), v3canon(), v3canon()}; }

__global__ void k_digits_of(const u64 *x, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = grm::signed_digits(x[i]);
}
// grm::gram_times_2_to(x, s) and grm::signed_turn(v, s) return x 2^s and v 2^s modulo p, for s below
// grm::TURNS; 2^TURNS = 1 modulo p.
__global__ void k_turns(const u64 *x, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int s = 0; s < grm::TURNS; s++) o[(size_t)i * grm::TURNS + s] = grm::gram_times_2_to(x[i], s);
}
__global__ void k_signed_turns(const long long *v, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int s = 0; s < grm::TURNS; s++) o[(size_t)i * grm::TURNS + s] = grm::signed_turn(v[i], s);
}
// o[8 i + q]: q < 4: e_1 .. e_4 (elementary symmetric polynomials) of four values; 4, 5: e_1, e_2
// of the first two; 6, 7: the product of the last two, loose and canonical.
__global__ void k_coefficients(const u64 *t, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u64 t4[4] = {t[4 * i], t[4 * i + 1], t[4 * i + 2], t[4 * i + 3]}, t2[2] = {t[4 * i], t[4 * i + 1]};
  u64 c4[4], c2[2];
  fk::Coefficients<4, false>::of(t4, c4);
  fk::Coefficients<2, false>::of(t2, c2);
  for (int k = 0; k < 4; k++) o[8 * i + k] = c4[k];
  o[8 * i + 4] = c2[0], o[8 * i + 5] = c2[1];
  o[8 * i + 6] = fk::mul_loose(t[4 * i + 2], t[4 * i + 3]), o[8 * i + 7] = fk::canonical(fk::loose(o[8 * i + 6]));
}
__global__ void k_mul_loose(const u64 *x, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[2 * i] = fk::mul_loose(x[2 * i], x[2 * i + 1]), o[2 * i + 1] = fk::canonical(x[2 * i]);
}
// A `P` for fk::add_dots<P>(), like fk::LooseRows: coefficient i, 1 <= i <= D, is below 2^bits(i).
template <int D_, int B0, int B1> struct V3Rows { // coefficient i has B0 bits for odd i, B1 for even i
  static constexpr int D = D_;
  static __host__ __device__ constexpr int bits(int i) { return (i & 1) ? B0 : B1; }
};
template <class P> __global__ void k_dots(const u64 *cf, const fk::Powers<P::D> *pw, const u64 *x0, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u64 c[4][P::D];
  for (int k = 0; k < 4; k++)
    for (int d = 0; d < P::D; d++) c[k][d] = cf[((size_t)i * 4 + k) * P::D + d];
  for (int j = 0; j < 3; j++) {
    u64 x[4];
    for (int k = 0; k < 4; k++) x[k] = x0[((size_t)i * 3 + j) * 4 + k];
    fk::add_dots<P>(c, fk::FromPowers<P::D>{pw + i}, j, x);
    for (int k = 0; k < 4; k++) o[((size_t)i * 3 + j) * 4 + k] = x[k];
  }
}
// Tests fk::add_dots<P>(): x[k] += the sum over d of cf[k][d] times coordinate j of W_(d+1). `pw`
// has W_1 .. W_D in two limbs, W = l + 2^LIMB h. add_dots() has one path for coefficients of up to
// 28 bits, by P::bits(), and one for wider ones; main()'s widths are on both sides of 28.
template <class P> void play_dots(const char *name, int n) {
  constexpr int D = P::D, LIMB = 32;
  std::vector<u64> cf((size_t)n * 4 * D), x0((size_t)n * 12);
  std::vector<fk::Powers<D>> pw(n);
  for (int i = 0; i < n; i++) {
    const bool all_most = i < 4; // everything at its greatest
    for (int k = 0; k < 4; k++)
      for (int d = 0; d < D; d++) {
        const int b = P::bits(d + 1);
        const u64 most = b >= fk::LOOSE_BITS ? LOOSE_MAX : (1ull << b) - 1;
        cf[((size_t)i * 4 + k) * D + d] = all_most ? most : word_upto(most);
      }
    for (int d = 0; d < D; d++)
      for (int j = 0; j < 3; j++) {
        // i < 2: both limbs at their largest (no canonical W); i < 4: p - 1; else random, canonical
        const u64 w = v3canon();
        pw[i].l[d][j] = i < 2 ? (u32)((1ull << LIMB) - 1) : i < 4 ? (u32)((V3P - 1) & ((1ull << LIMB) - 1)) : (u32)(w & ((1ull << LIMB) - 1));
        pw[i].h[d][j] = i < 2 ? ~0u : i < 4 ? (u32)((V3P - 1) >> LIMB) : (u32)(w >> LIMB);
      }
    for (int j = 0; j < 3; j++) pw[i].w0[j] = 0;
    for (int q = 0; q < 12; q++) x0[(size_t)i * 12 + q] = all_most ? LOOSE_MAX : word_upto(LOOSE_MAX);
  }
  u64 *gc = v3up(cf), *gx = v3up(x0), *go = v3room<u64>((size_t)n * 12);
  fk::Powers<D> *gp = v3up(pw);
  k_dots<P><<<(n + 127) / 128, 128>>>(gc, gp, gx, n, go);
  CK(cudaGetLastError());
  const std::vector<u64> got = v3home(go, (size_t)n * 12);
  size_t bad = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 4; k++) {
        u64 want = v3red(x0[((size_t)i * 3 + j) * 4 + k]);
        for (int d = 0; d < D; d++) {
          const V3U w = (V3U)pw[i].l[d][j] + ((V3U)pw[i].h[d][j] << LIMB);
          want = v3add(want, v3mul(v3red(cf[((size_t)i * 4 + k) * D + d]), (u64)(w % V3P)));
        }
        const u64 g = got[((size_t)i * 3 + j) * 4 + k];
        if (v3red(g) != want || g > LOOSE_MAX) bad++;
      }
  v3line(!bad, name, (size_t)n * 12, bad);
  CK(cudaFree(gc)); CK(cudaFree(gx)); CK(cudaFree(gp));
}
// Writes six results for each i, in the order of the six GPU lines that main() prints for them.
// mac() .. ared() is a sum of products that is reduced once; smac() .. sred() is a sum of elements
// times 32-bit keys. In both smac() cases the keys add up to just below 2^32, so that the 64-bit
// sum over a limb near 2^32 is nearly full. U4::get(): the product of a - v over four 16-bit v.
__global__ void k_fk2(const T *a, const T *b, int n, T *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[6 * i] = FK2::mul(a[i], b[i]);
  const FK2::P p = FK2::prep(a[i]);
  o[6 * i + 1] = FK2::mulp(p, b[i]);
  FK2::Acc s = FK2::azero();
  for (int k = 0; k < 1000; k++) FK2::mac(s, p, b[i]);
  o[6 * i + 2] = FK2::ared(s);
  FK2::SAcc z = FK2::szero();
  FK2::smac(z, FK2::sprep(a[i]), 0xFFFFFFFFu);
  o[6 * i + 3] = FK2::sred(z);
  z = FK2::szero();
  for (int k = 0; k < 256; k++) FK2::smac(z, FK2::sprep(k & 1 ? a[i] : b[i]), 0xFFFFFFu);
  o[6 * i + 4] = FK2::sred(z);
  U4<FE> u;
  u.prep(a[i]);
  const u32 v0 = (u32)coord(b[i], 0) & 65535, v1 = (u32)(coord(b[i], 0) >> 16) & 65535, v2 = (i & 1) ? 65535 : (u32)coord(b[i], 1) & 65535, v3 = (i & 2) ? 65535 : (u32)coord(b[i], 2) & 65535;
  o[6 * i + 5] = u.get((i & 4) ? 65535 : v0, (i & 4) ? 65535 : v1, v2, v3);
}
__global__ void k_from_sums(const int *s, const u64 *start, int n, u64 *o) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = ff::from_sums(s + 8 * i, start[i]);
}
__global__ void k_constant_words(const T *K, int n, u32 *o) { // [i][shift 0 / 32][j][g]
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int sh = 0; sh < 2; sh++)
    for (int j = 0; j < 3; j++)
      for (int g = 0; g < 8; g++) o[(((size_t)i * 2 + sh) * 3 + j) * 8 + g] = ff::constant_word(K[i], j, 32 * sh, g);
}
// Tests ff::second_range() and ff::join_second() against direct sums, on the table words and keys
// that `kind` selects (see below). 2^dT keys, left keys in the first half and right keys in the
// second, make 2^dT / 128 indices. A chunk is 2^lc indices; its results are multiplied by one `hi`.
// A piece is k indices; each 32-bit sum runs over one piece, and 16,384 is the largest k that
// second_range() accepts (ff::second::PIECE_MOST). Left entry a of an index (`u`) is W_0 + the sum
// over e < 8 of left key 8 a + e times W_(e+1), with W_0 .. W_8 from the table. A chunk has 520
// results: p < 8: the sum over its indices of left entry p; p = 8 + 64 jp + 8 a + ap: the sum of
// left entry a times right key 8 ap + jp.
static void play_second(const char *name, int dT, int lc, int k, int kind) {
  const size_t N = (size_t)1 << dT, SC = N / 128, chunk = (size_t)1 << lc;
  const int nch = (int)(SC >> lc), pieces = (int)(chunk / k);
  std::vector<uint16_t> keys(N);
  std::vector<u32> table(54 * chunk);
  std::vector<T> hi(nch);
  for (auto &x : keys) x = kind == 0 || kind == 3 ? 65535 : (uint16_t)v3draw();
  // kind 0: every key 65,535 and every bit of every coordinate set (p or more: not canonical);
  // 1: coordinates p - 1, random keys; 2: random canonical coordinates, keys and `hi` (`hi` is 1
  // otherwise); 3: the carry. The kernel computes a left entry's coordinate as an integer, removes
  // its bits from 2^64 on, of value v, and adds v (2^64 mod p) to the low 64 bits. With
  // W_0 = carry_first, W_1 = .. = W_8 = carry_others and every key 65,535 this addition passes 2^64
  // again, so the result's ninth byte is 1. On random inputs v is about
  // 8 * 2^15 * 2^63 / 2^64 = 2^17 and v (2^32 - 1) about 2^49, so about one coordinate in 2^15
  // carries (worked out, not measured). The block below repeats the kernel's steps in 128 bits to
  // confirm the carry.
  const u64 carry_first = 0x200000000ull, carry_others = V3P - 1;
  if (kind == 3) {
    const V3U lower = (V3U)(u32)carry_first + (V3U)8 * 65535 * (u32)carry_others, upper = (V3U)(carry_first >> 32) + (V3U)8 * 65535 * (carry_others >> 32),
              b = upper + (lower >> 32), low = (lower & 0xFFFFFFFFull) | ((b & 0xFFFFFFFFull) << 32), sum = low + (V3U)ff::second::TWO_64 * (b >> 32);
    const bool taken = (sum >> 64) == 1 && carry_first < V3P && carry_others < V3P &&
                       (u64)(sum % V3P) == v3add(carry_first, v3mul(8 * 65535, carry_others));
    v3line(taken, "second_range, the carry: by the kernel's steps in 128 bits the number left has 65 bits and is the entry modulo p", 1, !taken);
  }
  for (size_t pl = 0; pl < 27; pl++)
    for (size_t c = 0; c < chunk; c++) {
      const u64 w = kind == 3 ? (pl < 3 ? carry_first : carry_others)
                    : kind == 0 ? ~0ull : kind == 1 ? V3P - 1 : v3canon();
      table[2 * pl * chunk + c] = (u32)w, table[(2 * pl + 1) * chunk + c] = (u32)(w >> 32);
    }
  for (auto &x : hi) x = kind == 2 ? v3element() : T{1, 0, 0};
  uint16_t *gk = v3up(keys);
  u32 *gt = v3up(table);
  T *gh = v3up(hi), *out = v3room<T>((size_t)nch * 520);
  int *C = v3room<int>(SC / k * ff::second::ROWS_A * ff::second::ROWS_B);
  fk::Tree t;
  t.keys = gk;
  ff::second_range(t, dT, (int)chunk, gt, 0, SC, k, C, 0);
  ff::join_second(C, pieces, nch, gh, out, 0);
  const std::vector<T> got = v3home(out, (size_t)nch * 520);
  size_t bad = 0;
  for (int ch = 0; ch < nch; ch++) {
    std::vector<V3H3> want(520, V3H3{{0, 0, 0}});
    for (size_t cc = 0; cc < chunk; cc++) {
      const size_t c = ((size_t)ch << lc) + cc;
      for (int a = 0; a < 8; a++) {
        V3H3 u;
        for (int j = 0; j < 3; j++) {
          auto W = [&](int i) { return v3red(table[(size_t)(2 * (3 * i + j)) * chunk + cc] | (u64)table[(size_t)(2 * (3 * i + j) + 1) * chunk + cc] << 32); };
          u64 x = W(0);
          for (int e = 0; e < 8; e++) x = v3add(x, v3mul(keys[64 * c + 8 * a + e], W(e + 1)));
          u.c[j] = x;
        }
        for (int j = 0; j < 3; j++) want[a].c[j] = v3add(want[a].c[j], u.c[j]);
        for (int ap = 0; ap < 8; ap++)
          for (int jp = 0; jp < 8; jp++)
            for (int j = 0; j < 3; j++) {
              u64 &w = want[8 + 64 * jp + 8 * a + ap].c[j];
              w = v3add(w, v3mul(u.c[j], keys[N / 2 + 64 * c + 8 * ap + jp]));
            }
      }
    }
    for (int p = 0; p < 520; p++)
      if (!v3same(got[(size_t)ch * 520 + p], h3mul(want[p], h3(hi[ch])))) bad++;
  }
  v3line(!bad, name, (size_t)nch * 520, bad);
  CK(cudaFree(gk)); CK(cudaFree(gt)); CK(cudaFree(gh)); CK(cudaFree(C));
}
int main() {
  dev_info();
  printf("BLK_FIELD %d, p = %llx, TURNS %d, LOOSE_BITS %d\n", BLK_FIELD, (unsigned long long)V3P, grm::TURNS, fk::LOOSE_BITS);
  const int n = 1 << 16;
  { // signed_digits on words from its whole input range
    const u64 most = ~0ull;
    std::vector<u64> x(n);
    for (auto &w : x) w = word_upto(most);
    u64 *gx = v3up(x), *go = v3room<u64>(n);
    k_digits_of<<<n / 128, 128>>>(gx, n, go);
    const std::vector<u64> got = v3home(go, n);
    size_t bad = 0, from_p = 0, above = 0;
    for (int i = 0; i < n; i++) {
      V3I v = 0;
      for (int m = 7; m >= 0; m--) v = v * 256 + (signed char)(got[i] >> (8 * m));
      const bool exact = v == (V3I)x[i] || v == (V3I)x[i] - (V3I)V3P;
      if (!exact) bad++;
      from_p += x[i] >= V3P, above += x[i] > 0x7F7F7F7F7F7F7F7Full;
    }
    printf("  (%zu words from p on, %zu above 0x7F7F..)\n", from_p, above);
    v3line(!bad, "signed_digits: the digits stand for x or x - p", n, bad);
    CK(cudaFree(gx));
  }
  { // gram_times_2_to: any word
    const int m = 4096;
    std::vector<u64> x(m);
    for (auto &w : x) w = word_upto(~0ull);
    u64 *gx = v3up(x), *go = v3room<u64>((size_t)m * grm::TURNS);
    k_turns<<<m / 128, 128>>>(gx, m, go);
    const std::vector<u64> got = v3home(go, (size_t)m * grm::TURNS);
    size_t bad = 0;
    for (int s = 0; s < grm::TURNS; s++) {
      const u64 f = hpow2(s);
      for (int i = 0; i < m; i++) bad += got[(size_t)i * grm::TURNS + s] != v3mul(v3red(x[i]), f);
    }
    v3line(!bad, "gram_times_2_to(x, s), every s below TURNS, canonical result", (size_t)m * grm::TURNS, bad);
    CK(cudaFree(gx));
  }
  { // signed_turn
    const int m = 4096;
    std::vector<long long> v(m);
    const long long top = (1ll << 61) - 1;
    for (int i = 0; i < m; i++) {
      const long long e[] = {0, 1, -1, top, -top, top - 1, 1 - top, 1ll << 60, -(1ll << 60), 0xFFFFFFFFll, -0xFFFFFFFFll, 1ll << 32, -(1ll << 32)};
      v[i] = i < 13 ? e[i] : (long long)(v3draw() >> (3 + v3draw() % 50)) * ((v3draw() & 1) ? 1 : -1);
    }
    long long *gv = v3up(v);
    u64 *go = v3room<u64>((size_t)m * grm::TURNS);
    k_signed_turns<<<m / 128, 128>>>(gv, m, go);
    const std::vector<u64> got = v3home(go, (size_t)m * grm::TURNS);
    size_t bad = 0;
    for (int s = 0; s < grm::TURNS; s++) {
      const u64 f = hpow2(s);
      for (int i = 0; i < m; i++) bad += v3red(got[(size_t)i * grm::TURNS + s]) != v3mul(hsigned(v[i]), f) || got[(size_t)i * grm::TURNS + s] >= V3P;
    }
    v3line(!bad, "signed_turn(v, s), |v| < 2^61, every s below TURNS", (size_t)m * grm::TURNS, bad);
    CK(cudaFree(gv));
  }
  { // Coefficients of table values below 2^61
    // fk::table_of() (prover/hiding/forest_plan.h) rejects larger ones.
    std::vector<u64> t(4 * n);
    for (int i = 0; i < 4 * n; i++) t[i] = i < 16 ? (1ull << 61) - 1 : word_upto((1ull << 61) - 1);
    u64 *gt = v3up(t), *go = v3room<u64>(8 * (size_t)n);
    k_coefficients<<<n / 128, 128>>>(gt, n, go);
    const std::vector<u64> got = v3home(go, 8 * (size_t)n);
    size_t bad = 0;
    for (int i = 0; i < n; i++) {
      const u64 a = v3red(t[4 * i]), b = v3red(t[4 * i + 1]), c = v3red(t[4 * i + 2]), d = v3red(t[4 * i + 3]);
      const u64 e1 = v3add(v3add(a, b), v3add(c, d)), ab = v3mul(a, b), cd = v3mul(c, d), e2 = v3add(v3add(ab, cd), v3mul(v3add(a, b), v3add(c, d))),
                e3 = v3add(v3mul(ab, v3add(c, d)), v3mul(cd, v3add(a, b))), e4 = v3mul(ab, cd);
      const u64 *g = &got[8 * (size_t)i];
      bool ok = v3red(g[0]) == e1 && v3red(g[1]) == e2 && v3red(g[2]) == e3 && v3red(g[3]) == e4 && g[4] == t[4 * i] + t[4 * i + 1] && v3red(g[5]) == ab && v3red(g[6]) == cd && g[7] == cd;
      for (int k = 0; k < 4; k++) ok = ok && g[k] <= LOOSE_MAX;
      ok = ok && g[0] == t[4 * i] + t[4 * i + 1] + t[4 * i + 2] + t[4 * i + 3];
      bad += !ok;
    }
    v3line(!bad, "Coefficients<4, false>, <2, false>, mul_loose, canonical", n, bad);
    CK(cudaFree(gt));
  }
  { // mul_loose and canonical on loose words
    std::vector<u64> x(2 * n);
    for (auto &w : x) w = word_upto(LOOSE_MAX);
    u64 *gx = v3up(x), *go = v3room<u64>(2 * (size_t)n);
    k_mul_loose<<<n / 128, 128>>>(gx, n, go);
    const std::vector<u64> got = v3home(go, 2 * (size_t)n);
    size_t bad = 0;
    for (int i = 0; i < n; i++) bad += v3red(got[2 * i]) != v3mul(v3red(x[2 * i]), v3red(x[2 * i + 1])) || got[2 * i] > LOOSE_MAX || got[2 * i + 1] != v3red(x[2 * i]);
    v3line(!bad, "mul_loose, canonical on any loose words", n, bad);
    CK(cudaFree(gx));
  }
  play_dots<fk::LooseRows<1>>("add_dots, LooseRows<1>", n);
  play_dots<fk::LooseRows<2>>("add_dots, LooseRows<2>", n);
  play_dots<fk::LooseRows<4>>("add_dots, LooseRows<4>", n);
  play_dots<fk::LooseRows<8>>("add_dots, LooseRows<8>", n);
  play_dots<V3Rows<8, 28, 28>>("add_dots, eight coefficients of 28 bits", n);
  play_dots<V3Rows<8, 16, 16>>("add_dots, eight coefficients of 16 bits", n);
  play_dots<V3Rows<4, 28, fk::LOOSE_BITS>>("add_dots, 28 bits and loose by turns", n);
  play_dots<V3Rows<2, 17, 32>>("add_dots, 17 and 32 bits", n);
  play_dots<V3Rows<8, 28, 64>>("add_dots, eight: 28 bits and 64 by turns", n);
  play_dots<V3Rows<8, 29, 29>>("add_dots, eight coefficients of 29 bits", n);
  play_dots<V3Rows<8, 48, 61>>("add_dots, eight: 48 and 61 bits", n);
  { // FK2 (GPU and host) and U4 (GPU) against the schoolbook product with the compiler's remainder
    std::vector<T> a(n), b(n);
    for (int i = 0; i < n; i++) a[i] = i < 8 ? T{V3P - 1, V3P - 1, V3P - 1} : v3element(), b[i] = i < 4 ? T{V3P - 1, V3P - 1, V3P - 1} : v3element();
    T *ga = v3up(a), *gb = v3up(b), *go = v3room<T>(6 * (size_t)n);
    k_fk2<<<n / 128, 128>>>(ga, gb, n, go);
    const std::vector<T> got = v3home(go, 6 * (size_t)n);
    size_t bad[6] = {0, 0, 0, 0, 0, 0}, host_bad = 0;
    for (int i = 0; i < n; i++) {
      const V3H3 x = h3(a[i]), y = h3(b[i]), xy = h3mul(x, y);
      V3H3 w;
      bad[0] += !v3same(got[6 * (size_t)i], xy), bad[1] += !v3same(got[6 * (size_t)i + 1], xy);
      for (int j = 0; j < 3; j++) w.c[j] = v3mul(xy.c[j], 1000);
      bad[2] += !v3same(got[6 * (size_t)i + 2], w);
      for (int j = 0; j < 3; j++) w.c[j] = v3mul(x.c[j], 0xFFFFFFFFull);
      bad[3] += !v3same(got[6 * (size_t)i + 3], w);
      const V3H3 one_term = w;
      for (int j = 0; j < 3; j++) w.c[j] = v3mul(v3add(x.c[j], y.c[j]), v3mul(128, 0xFFFFFF));
      bad[4] += !v3same(got[6 * (size_t)i + 4], w);
      const u32 v0 = (u32)coord(b[i], 0) & 65535, v1 = (u32)(coord(b[i], 0) >> 16) & 65535, v2 = (i & 1) ? 65535 : (u32)coord(b[i], 1) & 65535, v3 = (i & 2) ? 65535 : (u32)coord(b[i], 2) & 65535;
      const u32 v[4] = {(i & 4) ? 65535 : v0, (i & 4) ? 65535 : v1, v2, v3};
      V3H3 pr{{1, 0, 0}};
      for (int k = 0; k < 4; k++) pr = h3mul(pr, V3H3{{v3add(x.c[0], v3neg(v[k])), x.c[1], x.c[2]}});
      bad[5] += !v3same(got[6 * (size_t)i + 5], pr);
      // the host bodies
      FK2::Acc s = FK2::azero();
      const FK2::P p = FK2::prep(a[i]);
      FK2::mac(s, p, b[i]), FK2::mac(s, p, b[i]);
      V3H3 twice;
      for (int j = 0; j < 3; j++) twice.c[j] = v3add(xy.c[j], xy.c[j]);
      FK2::SAcc z = FK2::szero();
      FK2::smac(z, FK2::sprep(a[i]), 0xFFFFFFFFu);
      host_bad += !v3same(FK2::mul(a[i], b[i]), xy) || !v3same(FK2::mulp(p, b[i]), xy) || !v3same(FK2::ared(s), twice) || !v3same(FK2::sred(z), one_term) || !v3same(FK::mul(a[i], b[i]), xy);
    }
    v3line(!bad[0], "FK2::mul (GPU)", n, bad[0]);
    v3line(!bad[1], "FK2::prep, mulp (GPU)", n, bad[1]);
    v3line(!bad[2], "FK2::mac x 1000, ared (GPU)", n, bad[2]);
    v3line(!bad[3], "FK2::smac, one term of 2^32 - 1 (GPU)", n, bad[3]);
    v3line(!bad[4], "FK2::smac, 256 terms of 2^24 - 1 (GPU)", n, bad[4]);
    v3line(!bad[5], "U4<FE>::get (GPU)", n, bad[5]);
    v3line(!host_bad, "FK2::mul, mulp, mac, ared, smac, sred and FK::mul (host bodies)", n, host_bad);
    CK(cudaFree(ga)); CK(cudaFree(gb));
  }
  { // from_sums
    std::vector<int> s(8 * n);
    std::vector<u64> start(n);
    const int top = (1 << 24) - 1;
    for (int i = 0; i < n; i++) {
      for (int m = 0; m < 8; m++) s[8 * i + m] = i == 0 ? top : i == 1 ? -top : (int)(v3draw() % (2 * (u64)top + 1)) - top;
      start[i] = i < 2 ? V3P - 1 : v3canon();
    }
    int *gs = v3up(s);
    u64 *gx = v3up(start), *go = v3room<u64>(n);
    k_from_sums<<<n / 128, 128>>>(gs, gx, n, go);
    const std::vector<u64> got = v3home(go, n);
    size_t bad = 0;
    for (int i = 0; i < n; i++) {
      u64 want = start[i];
      for (int m = 0; m < 8; m++) want = v3add(want, v3mul(hsigned(s[8 * i + m]), hpow2(8 * m)));
      bad += got[i] != want;
    }
    v3line(!bad, "ff::from_sums, |s| < 2^24", n, bad);
    CK(cudaFree(gs)); CK(cudaFree(gx));
  }
  { // constant_word
    const int m = 4096;
    std::vector<T> K(m);
    for (auto &x : K) x = v3element();
    T *gK = v3up(K);
    u32 *go = v3room<u32>((size_t)m * 48);
    k_constant_words<<<m / 128, 128>>>(gK, m, go);
    const std::vector<u32> got = v3home(go, (size_t)m * 48);
    size_t bad = 0;
    for (int i = 0; i < m; i++)
      for (int sh = 0; sh < 2; sh++)
        for (int j = 0; j < 3; j++)
          for (int b = 0; b < 4; b++) {
            u64 v = 0;
            for (int g = 7; g >= 0; g--) v = v3add(v3mul(v, 256), hsigned((signed char)(got[(((size_t)i * 2 + sh) * 3 + j) * 8 + g] >> (8 * b))));
            bad += v != v3mul(coord(K[i], j), hpow2(32 * sh + 8 * b));
          }
    v3line(!bad, "ff::constant_word, shifts 0 and 32", (size_t)m * 24, bad);
    CK(cudaFree(gK));
  }
  play_second("second_range, everything at its greatest, a piece of 2^14", 21, 14, 16384, 0);
  play_second("second_range, coordinates p - 1, drawn keys, a piece of 2^14", 21, 14, 16384, 1);
  play_second("second_range, drawn, two chunks of 2^12, pieces of 2^10 (four a chunk)", 20, 12, 1024, 2);
  play_second("second_range, drawn, chunks of 2^9, pieces of 2^9", 17, 9, 512, 2);
  play_second("second_range, the carry in every entry, a piece of 2^14", 21, 14, 16384, 3);
  play_second("second_range, the carry in every entry, chunks of 2^9, pieces of 2^9", 17, 9, 512, 3);
  if (v3failures) printf("%d of %d lines fail\n", v3failures, v3lines); else printf("all %d lines ok\n", v3lines);
  return v3failures ? 1 : 0;
}
