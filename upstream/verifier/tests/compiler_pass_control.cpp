// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Reproduces a miscompilation by the ipa-modref pass of g++ 13.3.0 at -O2 -fno-inline, which is why
// verifier/build.sh passes -fno-ipa-modref. verifier/tests/compiler_pass_control.sh builds and runs
// this program and explains the bug. The code is reduced from a CPU prover that is not in
// this repository and keeps its function names. It adds two triples; the factor argc stops the
// compiler from computing the sum at build time. There is no undefined behaviour, so a run without
// arguments must print "(1011, 2022, 3033)" and exit with code 0; anything else, with exit code 1,
// is a compiler bug.
#include <cstdint>
#include <cstdio>
typedef uint64_t u64;
static const u64 P = (1ull << 61) - 1;
static u64 fadd(u64 a, u64 b) { u64 s = a + b; return s >= P ? s - P : s; }
struct E {
  u64 k0, k1, k2;
  E() : k0(0), k1(0), k2(0) {}
  E(u64 a_, u64 b_, u64 c_) : k0(a_), k1(b_), k2(c_) {}
};
static u64 ecoord(E x, int i) { return i == 0 ? x.k0 : i == 1 ? x.k1 : x.k2; }
static E efrom(const u64 *k) { return E(k[0], k[1], k[2]); }
static E eadd(E x, E y) { return E(fadd(x.k0, y.k0), fadd(x.k1, y.k1), fadd(x.k2, y.k2)); }
static inline E operator+(E a, E b) { return eadd(a, b); }
namespace mb { struct E { u64 c[3]; }; }
static mb::E toMb(const E &e) { return mb::E{{ecoord(e, 0), ecoord(e, 1), ecoord(e, 2)}}; }
static E fromMb(const mb::E &e) { u64 k[3] = {e.c[0], e.c[1], e.c[2]}; return efrom(k); }
static E mbPadFv(const E &fv, const E &pad) { return fv + pad; }
int main(int argc, char **) {
  const mb::E fv = {{11, 22, 33}}, pad = {{1000ull * argc, 2000, 3000}};
  const mb::E s = toMb(mbPadFv(fromMb(fv), fromMb(pad)));
  printf("(%llu, %llu, %llu)\n", (unsigned long long)s.c[0], (unsigned long long)s.c[1], (unsigned long long)s.c[2]);
  return s.c[0] == 1011 && s.c[1] == 2022 && s.c[2] == 3033 ? 0 : 1;
}
