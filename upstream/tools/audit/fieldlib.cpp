// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A second implementation of the field and its code, for the audit with the secret key
// (tools/audit/README.md); tools/audit/field.py loads it. It calls no function of the library,
// blocks/.
// f = 2 ("field 2") is this repository's one field, p = 2^64 - 2^32 + 1. Its points and symbols
// follow docs/format/FORMAT.md, rule R12.11 (tested by tools/audit/test_field.py).
// f = 1 ("field 1") is P1 = 2^61 - 1 with a circle code, which the library cannot be built for
// (blocks/base.h rejects every field but 2). Field 1's factors() is a retyping, statement for
// statement, of library functions that are not in this
// repository, with the generator's two constants copied from them. No document specifies it, and
// no known answer checks it.
//   g++ -O2 -fno-ipa-modref -shared -fPIC -o libfield.so fieldlib.cpp
#include <cstddef>
#include <cstdint>
#include <vector>
typedef uint64_t u64;
typedef unsigned __int128 u128;
static const u64 P1 = ((u64)1 << 61) - 1, P2 = 0xFFFFFFFF00000001ull;
static inline u64 P(int f) { return f == 1 ? P1 : P2; }
static inline u64 mul(int f, u64 a, u64 b) { return (u64)((u128)a * b % P(f)); }
static inline u64 add(int f, u64 a, u64 b) { return (u64)(((u128)a + b) % P(f)); }
static inline u64 sub(int f, u64 a, u64 b) { return (u64)(((u128)a + P(f) - b) % P(f)); }
static u64 power(int f, u64 a, u64 e) {
  u64 r = 1;
  for (; e; e >>= 1, a = mul(f, a, a))
    if (e & 1) r = mul(f, r, a);
  return r;
}
// The factors f_0 .. f_(count-1) at position x of a code of length 2^a: basis function j is the
// product of the f_i over the bits i of j that are 1.
// Not checked: a <= 32 at field 2 (R12.11).
static void factors(int f, unsigned a, u64 x, unsigned count, u64 *out) {
  if (f == 2) {  // R12.11: z_x = 7 prod_{set bit i of x} w^(2^(31 - i)); basis_j = z_x^j
    const u64 w = power(2, 7, 5 * (((u64)1 << 32) - 1));
    u64 z = 7;
    for (unsigned i = 0; i < a; i++)
      if ((x >> i) & 1) z = mul(2, z, power(2, w, (u64)1 << (31 - i)));
    for (unsigned i = 0; i < count; i++) out[i] = z, z = mul(2, z, z);
    return;
  }
  // Field 1 (n = 2^a; s_i: bit i of x; G: the generator, (gx, gy) at the start):
  // g = G^(2^(60 - a)) of order 2 n;
  // e = 1 + sum_{i >= 1} s_i 2^(a - i + 1), replaced by 2 n - e if s_0 = 1;
  // f_0 = zy, f_1 = zx, f_(i+1) = 2 f_i^2 - 1 for i >= 1
  // The circle code and this basis are those of U. Haboeck, D. Levit, S. Papini, "Circle STARKs",
  // IACR ePrint 2024/278 (Definition 3; sections 3.2 and 4).
  u64 gx = 0x1696969696969695ull, gy = 0x1A5A5A5A5A5A5A5Aull;
  auto times = [](u64 ax, u64 ay, u64 bx, u64 by, u64 &cx, u64 &cy) {
    const u64 nx = sub(1, mul(1, ax, bx), mul(1, ay, by)), ny = add(1, mul(1, ax, by), mul(1, ay, bx));
    cx = nx, cy = ny;
  };
  for (unsigned i = a; i < 60; i++) times(gx, gy, gx, gy, gx, gy);
  u64 e = 1;
  for (unsigned i = 1; i < a; i++)
    if ((x >> i) & 1) e += (u64)1 << (a - i + 1);
  if (x & 1) e = ((u64)2 << a) - e;
  u64 zx = 1, zy = 0;
  for (; e; e >>= 1, times(gx, gy, gx, gy, gx, gy))
    if (e & 1) times(zx, zy, gx, gy, zx, zy);
  if (count > 0) out[0] = zy;
  if (count > 1) out[1] = zx;
  for (unsigned i = 2; i < count; i++) out[i] = sub(1, mul(1, 2, mul(1, out[i - 1], out[i - 1])), 1);
}
static unsigned log_up(u64 n) {
  unsigned k = 0;
  while (((u64)1 << k) < n) k++;
  return k;
}
// sum_{j < n} c[j * stride] * basis_j
static u64 combine(int f, const u64 *c, size_t n, size_t stride, const u64 *fac, std::vector<u64> &work) {
  if (!n) return 0;
  const unsigned k = log_up(n);
  work.assign((size_t)1 << k, 0);
  for (size_t j = 0; j < n; j++) work[j] = c[j * stride];
  size_t len = (size_t)1 << k;
  for (unsigned i = 0; i < k; i++, len >>= 1)
    for (size_t j = 0; j < len / 2; j++) work[j] = add(f, work[2 * j], mul(f, fac[i], work[2 * j + 1]));
  return work[0];
}
static u64 basis(int f, const u64 *fac, u64 j) {
  u64 r = 1;
  for (unsigned i = 0; j; j >>= 1, i++)
    if (j & 1) r = mul(f, r, fac[i]);
  return r;
}
extern "C" {
// The symbols at nx positions xs of one row under the code (a, ell, r) of field f: ell message
// entries, a random tail of r.
// Entries and symbols have deg coordinates, interleaved: coordinate d of entry j is at
// [j * deg + d]; out likewise.
// msg or tail may be null: that part then counts as zeros.
// Field 2 (R12.11): the tail has the lowest r coefficients, the message the next ell; 40 factors
// are computed, enough for ell, r < 2^40 (not checked).
// Field 1: the message starts at coefficient 0, the tail at ell rounded up to a multiple of the
// least power of two >= r.
void row_symbols(int f, unsigned a, u64 ell, u64 r, unsigned deg, const u64 *msg, const u64 *tail, const u64 *xs, size_t nx, u64 *out) {
  std::vector<u64> work;
  u64 message_at, tail_at;
  if (f == 1) {
    const u64 step = (u64)1 << log_up(r);
    message_at = 0, tail_at = (ell + step - 1) / step * step;
  } else
    message_at = r, tail_at = 0;
  u64 fac[64];
  for (size_t i = 0; i < nx; i++) {
    factors(f, a, xs[i], 63 < a + 1 ? 63 : (f == 1 ? a : 40), fac);
    const u64 ms = basis(f, fac, message_at), ts = basis(f, fac, tail_at);
    for (unsigned d = 0; d < deg; d++) {
      u64 s = 0;
      if (msg) s = add(f, s, mul(f, ms, combine(f, msg + d, ell, deg, fac, work)));
      if (tail && r) s = add(f, s, mul(f, ts, combine(f, tail + d, r, deg, fac, work)));
      out[i * deg + d] = s;
    }
  }
}
u64 f_mul(int f, u64 a, u64 b) { return mul(f, a, b); }
}
