// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

#pragma once
// Host-side code compiled into the programs `cp_commit` and `cp_prove`: arithmetic in the field F_p
// and its cubic extension E, the tree hash TH() and a file writer. Neither program calls wr().
// All state is in static globals: one copy per translation unit, for one thread at a time.
// Before including this header inside a namespace, include the headers below outside it.
#include "common/hash.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
using std::vector;
typedef unsigned __int128 u128;
// The field's prime, p = 2^64 - 2^32 + 1 (Goldilocks). fsub() expects arguments below P and does
// not check that; fadd() and fmul() reduce any arguments.
static const u64 P = FIELD_P;  // field/field_names.h, which common/setting.h has included
static const u64 XW = FIELD_W; // x^3 = XW in E
static u64 mix64(u64 z) { // one step of SplitMix64
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// Field operations by division: slow, and they share no code with the GPU's.
static u64 fadd(u64 a, u64 b) {
  return (u64)(((u128)a + b) % P);
}
static u64 fsub(u64 a, u64 b) {
  return a >= b ? a - b : (u64)((u128)a + P - b);
}
static u64 fmul(u64 a, u64 b) {
  return (u64)((u128)a * b % P);
}
// An element of the extension field: EDEG coordinates over F_p (prover/common/setting.h) and no
// padding, so an array of E has the same bytes as the GPU's array of `Ext`
// (prover/common/common.h). This arithmetic shares no code with the GPU's, and the two must agree.
struct E { // k0 + k1 x + k2 x^2, x^3 = XW; one or two arguments do not compile
  u64 k0, k1, k2;
  E() : k0(0), k1(0), k2(0) {}
  E(u64 c0, u64 c1, u64 c2) : k0(c0), k1(c1), k2(c2) {}
  E(u64) = delete;
  E(u64, u64) = delete;
};
static E ebase(u64 v) {
  return E(v, 0, 0);
}
static E efrom(const u64 *k) {
  return E(k[0], k[1], k[2]);
}
static E eadd(E x, E y) {
  return E(fadd(x.k0, y.k0), fadd(x.k1, y.k1), fadd(x.k2, y.k2));
}
static E esub(E x, E y) {
  return E(fsub(x.k0, y.k0), fsub(x.k1, y.k1), fsub(x.k2, y.k2));
}
static E emul(E x, E y) {
  return E(fadd(fmul(x.k0, y.k0), fmul(fadd(fmul(x.k1, y.k2), fmul(x.k2, y.k1)), XW)),
           fadd(fadd(fmul(x.k0, y.k1), fmul(x.k1, y.k0)), fmul(fmul(x.k2, y.k2), XW)),
           fadd(fadd(fmul(x.k0, y.k2), fmul(x.k1, y.k1)), fmul(x.k2, y.k0)));
}
static E esc(E x, u64 c) {
  return E(fmul(x.k0, c), fmul(x.k1, c), fmul(x.k2, c));
}
static_assert(sizeof(E) == 8 * EDEG, "a field element must have no padding bytes");
// The zero and the one of the extension field.
static const E E0 = ebase(0), E1 = ebase(1);
// The tree hash, as TH() in verifier/src/vf_common.h: chunks of THPW = 16 DW words are hashed under
// the number TH_LIST, then chunks of the list of their digests under TH_UPPER, until one digest is
// left. Every list is hashed at least once. A list of fewer than DW words ends the program.
static vector<u32> TH(vector<u32> w) {
  if (w.size() < DW) {
    fprintf(stderr,
            "internal error: the tree hash of %zu words was requested; it is defined for %d words or more; the program "
            "ends\n",
            w.size(), DW);
    exit(2);
  }
  for (u32 num = TH_LIST;; num = TH_UPPER) {
    const size_t nc = (w.size() + THPW - 1) / THPW;
    vector<u32> o(nc * DW);
    for (size_t c = 0; c < nc; c++) earlier_hash_h_digest(&o[c * DW], num, &w[c * THPW], std::min<size_t>(THPW, w.size() - c * THPW));
    w.swap(o);
    if (nc == 1) return w;
  }
}
struct Call {
  int no, n, iota, k, kap = 4;
  int fn = 0; // 0 range, 1 general function table, 2 byte-limb function table
  int t0 = 0; // index of the row with the public keys (padding keys)
  vector<u64> key;
  vector<u32> row;
  vector<u64> tau;
  vector<u32> pub;
  vector<u8> bits;
  E alpha, vk, vb[2];
  vector<E> zk, zb[2];
  vector<E> rv0, rt0, X0, Y0;
  size_t slotR = 0;
};
// The line `D` of the parameter file (`policy.txt`); HfShape::from_policy() sets it. The weight
// matrix has a multiple of D rows (layoutW()).
static int D;
// Writes a new file and calls fsync() on it, not on its directory. A path that already exists
// is rejected, so wr() never overwrites an input or an earlier result. On any failure wr() removes
// the file, if it has created it, and exits with code 2.
static void wr(const std::string &p, const void *d, size_t n) {
  const int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  bool good = fd >= 0;
  // A test build writes a header line that begins "NO PROOF" here, before the file's first byte.
  // No file of this repository defines this hook.
  // prover/weights_commitment/common.cuh includes this header inside namespace rf; the hook's
  // namespace is at global scope.
#ifdef TEST_HOOKS
  test_hooks::write_this_builds_file_head(fd, p, good); // may abandon the file
#endif
  for (const char *at = (const char *)d; good && n;) {
    const ssize_t w = write(fd, at, n);
    if (w <= 0)
      good = false;
    else
      at += w, n -= (size_t)w;
  }
  if (good && fsync(fd)) good = false;
  if (fd >= 0 && close(fd)) good = false;
  if (!good) {
    perror(p.c_str());
    if (fd >= 0) unlink(p.c_str());
    exit(2);
  }
}
// The smallest l with 2^l >= x.
static int l2(size_t x) {
  int l = 0;
  while (((size_t)1 << l) < x) l++;
  return l;
}
