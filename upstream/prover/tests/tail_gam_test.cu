// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests, on a GPU, the kernels of prover/hiding/tail_gam.cuh, which serve the approximate range
// check (`tail` in identifiers); kh_gam() only for TAU = 128 and TAU = 160 random combinations.
// p is the field's prime, 2^64 - 2^32 + 1: `FIELD_P` names it; gf_add() adds modulo it.
// A plain sum is what kh_gam() must give for column c: the sum over u of v[u] Cf[c * TAU + u]
// modulo p (v: the combinations' weights; Cf: their coefficients), computed on the host in 128-bit
// integers.
#include <cstdio>
#include <vector>

#include "hiding/pt_field.h"
#include "hiding/tail_gam.cuh"

static u64 state = 88172645463325252ull;
static u64 word() {  // xorshift64: G. Marsaglia, "Xorshift RNGs", Journal of Statistical Software 8 (14), 2003
  state ^= state << 13, state ^= state >> 7, state ^= state << 17;
  return state;
}
static int failures = 0;

// Negative control: like kh_gam(), but the PER words of a lane are added in 64 bits and reduced
// once, by (s[e] & p) + (s[e] >> 61) and a conditional subtraction. That is right for
// p = 2^61 - 1 and wrong for any other prime, so it is wrong here. main() fails if the control
// never differs from the plain sums. old_rule() sets no failure bit. TAU and the block size must
// be multiples of 32 (not checked).
// The fold modulo a Mersenne prime is a public idea that is specific to arithmetic modulo Mersenne
// primes: M. Thorup, "High Speed Hashing for Integers and Strings", notes dated May 14, 2019,
// section 2.2.1, equation (8), https://arxiv.org/pdf/1504.06804.
template <int TAU>
__global__ void old_rule(const signed char *Cf, size_t k, const Ext *v, Ext *g) {
  constexpr int PER = TAU / 32;
  const int lane = threadIdx.x & 31;
  const size_t w0 = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5, nw = ((size_t)gridDim.x * blockDim.x) >> 5;
  for (size_t c = w0; c < k; c += nw) {
    u64 s[3] = {0, 0, 0};
    for (int i = 0; i < PER; i++) {
      const int x = Cf[c * TAU + lane + 32 * i];
      for (int e = 0; e < 3; e++) s[e] += x > 0 ? coord(v[lane + 32 * i], e) : x < 0 ? FIELD_P - coord(v[lane + 32 * i], e) : 0;
    }
    for (int e = 0; e < 3; e++) {
      u64 x = (s[e] & FIELD_P) + (s[e] >> 61);
      if (x >= FIELD_P) x -= FIELD_P;
      for (int o = 16; o; o >>= 1) x = gf_add(x, __shfl_xor_sync(~0u, x, o));
      s[e] = x;
    }
    if (lane == 0) g[c] = ext_from(s);
  }
}
static size_t old_rule_wrong = 0;

// Compares kh_gam() and old_rule() with the plain sums. `kind` sets every coordinate of every
// v[u]: (0) pseudo-random below p; (1) p - 1, at which a sum reduced only at its end overflows
// 64 bits, as p is near 2^64; (2) 1. Column 0 is all 1, column 1 all -1; the others are
// pseudo-random in {-1, 0, 1}. run() also checks kh_gam()'s failure bit, ZERO_ELEMENT
// (prover/common/slot_flags.h).
template <int TAU>
static void run(int kind) {
  const size_t k = 777;
  std::vector<Ext> v(TAU), want(k), got(k);
  for (Ext &x : v) {
    u64 c[3];
    for (u64 &y : c) y = kind == 0 ? word() % FIELD_P : kind == 1 ? FIELD_P - 1 : 1;
    x = ext_from(c);
  }
  std::vector<signed char> Cf(k * TAU);
  for (size_t c = 0; c < k; c++)
    for (int u = 0; u < TAU; u++) Cf[c * TAU + u] = c == 0 ? 1 : c == 1 ? -1 : (signed char)(word() % 4 == 0 ? 1 : word() % 3 == 0 ? -1 : 0);
  for (size_t c = 0; c < k; c++) {
    u64 s[3];
    for (int e = 0; e < 3; e++) {
      unsigned __int128 sum = 0;
      for (int u = 0; u < TAU; u++) sum += Cf[c * TAU + u] > 0 ? coord(v[u], e) : Cf[c * TAU + u] < 0 ? FIELD_P - coord(v[u], e) : 0;
      s[e] = (u64)(sum % FIELD_P);
    }
    want[c] = ext_from(s);
  }
  signed char *dC;
  Ext *dv, *dg;
  CK(cudaMalloc(&dC, Cf.size())); CK(cudaMalloc(&dv, TAU * sizeof(Ext))); CK(cudaMalloc(&dg, k * sizeof(Ext)));
  CK(cudaMemcpy(dC, Cf.data(), Cf.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, v.data(), TAU * sizeof(Ext), cudaMemcpyHostToDevice));
  const unsigned none[4] = {0, 0, 0, 0};
  CK(cudaMemcpyToSymbol(g_live_flags_device, none, sizeof none));
  kh_gam<TAU><<<16, 256>>>(dC, k, dv, dg);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), dg, k * sizeof(Ext), cudaMemcpyDeviceToHost));
  {  // the flag is raised if and only if a plain sum is zero (with equal weights a column with as many 1 as -1 has such a sum)
    size_t zeros = 0;
    for (size_t c = 0; c < k; c++) zeros += !(coord(want[c], 0) | coord(want[c], 1) | coord(want[c], 2));
    unsigned flags[4];
    CK(cudaMemcpyFromSymbol(flags, g_live_flags_device, sizeof flags));
    if (flags[0] != (zeros ? (unsigned)ZERO_ELEMENT : 0u))
      failures++, printf("FAILED: %d combinations, weights of kind %d: %zu plain sums are zero, the flags are %x\n", TAU, kind, zeros, flags[0]);
    CK(cudaMemcpyToSymbol(g_live_flags_device, none, sizeof none));
  }
  size_t wrong = 0;
  for (size_t c = 0; c < k; c++)
    for (int e = 0; e < 3; e++) wrong += coord(got[c], e) != coord(want[c], e);
  if (wrong) failures++, printf("FAILED: %d combinations, weights of kind %d: %zu coordinates differ from the plain sums\n", TAU, kind, wrong);
  old_rule<TAU><<<16, 256>>>(dC, k, dv, dg);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), dg, k * sizeof(Ext), cudaMemcpyDeviceToHost));
  for (size_t c = 0; c < k; c++)
    for (int e = 0; e < 3; e++) old_rule_wrong += coord(got[c], e) != coord(want[c], e);
  CK(cudaFree(dC)); CK(cudaFree(dv)); CK(cudaFree(dg));
}

// Compares the tables of kf_powers() and kf_tie_powers() with repeated multiplication, and large
// exponents with repeated squaring and with base^(a + b) = base^a base^b.
static void powers() {
  u64 c[3] = {word() % FIELD_P, word() % FIELD_P, word() % FIELD_P};
  const Ext base = ext_from(c);
  const unsigned n = 5000;
  const int nb = 13;
  std::vector<unsigned long long> e = {0, 1, 2, 127, 128, 1ull << 31, (1ull << 31) + 12345, (1ull << 40) + 7, (1ull << 31) + 12345 + (1ull << 40) + 7};
  Ext *db, *dt, *dr, *dE, *dp;
  unsigned long long *de;
  CK(cudaMalloc(&db, sizeof(Ext))); CK(cudaMalloc(&dt, n * sizeof(Ext))); CK(cudaMalloc(&dr, nb * sizeof(Ext))); CK(cudaMalloc(&dE, n * sizeof(Ext)));
  CK(cudaMalloc(&dp, e.size() * sizeof(Ext))); CK(cudaMalloc(&de, e.size() * 8));
  CK(cudaMemcpy(db, &base, sizeof(Ext), cudaMemcpyHostToDevice)); CK(cudaMemcpy(de, e.data(), e.size() * 8, cudaMemcpyHostToDevice));
  kf_powers<<<8, 256>>>(db, nullptr, n, dt);
  kf_powers<<<1, 256>>>(db, de, e.size(), dp);
  kf_tie_powers<<<8, 256>>>(db, nb, n, dr, dE);
  CK(cudaDeviceSynchronize());
  std::vector<Ext> t(n), E(n), r(nb), p(e.size());
  CK(cudaMemcpy(t.data(), dt, n * sizeof(Ext), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(E.data(), dE, n * sizeof(Ext), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(r.data(), dr, nb * sizeof(Ext), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(p.data(), dp, e.size() * sizeof(Ext), cudaMemcpyDeviceToHost));
  auto same = [](const Ext &a, const Ext &b) { return coord(a, 0) == coord(b, 0) && coord(a, 1) == coord(b, 1) && coord(a, 2) == coord(b, 2); };
  size_t wrong = 0;
  Ext x = ext_base(1);
  for (unsigned i = 0; i < n; i++, x = k3_mul(x, base)) {
    wrong += !same(t[i], x) + !same(E[i], x);
    Ext by_bits = ext_base(1);  // base^i from r: r[j] goes with bit nb - 1 - j of i
    for (int j = 0; j < nb; j++)
      if ((i >> (nb - 1 - j)) & 1) by_bits = k3_mul(by_bits, r[j]);
    wrong += !same(by_bits, x);
  }
  Ext square = base;  // base^(2^31), base^(2^40) by squarings alone
  Ext at31 = base, at40 = base;
  for (int s = 1; s <= 40; s++) {
    square = k3_mul(square, square);
    if (s == 31) at31 = square;
    if (s == 40) at40 = square;
  }
  wrong += !same(p[0], ext_base(1)) + !same(p[1], base) + !same(p[2], t[2]) + !same(p[3], t[127]) + !same(p[4], t[128]) + !same(p[5], at31);
  wrong += !same(p[6], k3_mul(at31, tail_power(base, 12345))) + !same(tail_power(base, 12345), k3_mul(k3_mul(t[4999], t[4999]), t[2347])) + !same(p[7], k3_mul(at40, t[7])) + !same(p[8], k3_mul(p[6], p[7]));
  if (wrong) failures++, printf("FAILED: %zu powers differ\n", wrong);
}

// tail_power() multiplies by k3_mul(); prover/hiding/walk.cuh multiplies by `F::mul`, which is
// FK2::mul() of prover/hiding/pt_field.h. Both must give the same element, on the GPU and on the
// host.
__global__ void both_products(const Ext *a, const Ext *b, size_t n, Ext *by_k3, Ext *by_walk) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) by_k3[i] = k3_mul(a[i], b[i]), by_walk[i] = FK2::mul(a[i], b[i]);
}
static void products() {
  const size_t n = 20000;
  std::vector<Ext> a(n), b(n), x(n), y(n);
  for (size_t i = 0; i < n; i++) {
    u64 c[6];
    for (u64 &w : c) w = i < 64 ? ((i >> (&w - c)) & 1 ? FIELD_P - 1 : 1) : word() % FIELD_P;  // first all patterns of p - 1 and 1, then drawn words
    a[i] = ext_from(c), b[i] = ext_from(c + 3);
  }
  Ext *da, *db, *dx, *dy;
  CK(cudaMalloc(&da, n * sizeof(Ext))); CK(cudaMalloc(&db, n * sizeof(Ext))); CK(cudaMalloc(&dx, n * sizeof(Ext))); CK(cudaMalloc(&dy, n * sizeof(Ext)));
  CK(cudaMemcpy(da, a.data(), n * sizeof(Ext), cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), n * sizeof(Ext), cudaMemcpyHostToDevice));
  both_products<<<32, 256>>>(da, db, n, dx, dy);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(x.data(), dx, n * sizeof(Ext), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), dy, n * sizeof(Ext), cudaMemcpyDeviceToHost));
  size_t wrong = 0;
  for (size_t i = 0; i < n; i++) {
    const Ext h = k3_mul(a[i], b[i]), w = FK2::mul(a[i], b[i]);
    for (int e = 0; e < 3; e++) wrong += coord(x[i], e) != coord(y[i], e) || coord(x[i], e) != coord(h, e) || coord(h, e) != coord(w, e);
  }
  if (wrong) failures++, printf("FAILED: k3_mul and FK2::mul differ in %zu coordinates\n", wrong);
}

int main() {
  for (int kind = 0; kind < 3; kind++) run<128>(kind), run<160>(kind);
  powers();
  products();
  if (old_rule_wrong == 0) failures++, printf("FAILED: the control: the old rule differs in no coordinate; it should be false in this field\n");
  unsigned flags[4];
  CK(cudaMemcpyFromSymbol(flags, g_live_flags_device, sizeof flags));
  {  // a column of zeros
    std::vector<signed char> Cf(128, 0);
    std::vector<Ext> v(128);
    u64 one[3] = {1, 0, 0};
    for (Ext &x : v) x = ext_from(one);
    signed char *dC;
    Ext *dv, *dg;
    CK(cudaMalloc(&dC, 128)); CK(cudaMalloc(&dv, 128 * sizeof(Ext))); CK(cudaMalloc(&dg, sizeof(Ext)));
    CK(cudaMemcpy(dC, Cf.data(), 128, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, v.data(), 128 * sizeof(Ext), cudaMemcpyHostToDevice));
    kh_gam<128><<<1, 256>>>(dC, 1, dv, dg);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpyFromSymbol(flags, g_live_flags_device, sizeof flags));
    if (!(flags[0] & ZERO_ELEMENT)) failures++, printf("FAILED: a zero sum did not raise ZERO_ELEMENT\n");
  }
  printf("tail_gam_test: %s (field %d; the old rule, planted as a control, differs in %zu coordinates)\n", failures ? "FAILED" : "ok", BLK_FIELD, old_rule_wrong);
  return failures ? 1 : 0;
}
