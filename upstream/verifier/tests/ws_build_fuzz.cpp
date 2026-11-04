// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Fuzz test of WsLay::build() (verifier/src/vf_weights_layout.h), which computes the layout of the
// committed weight matrix W from a model's shape (`WsShape`, whose fields are explained there) and
// lk, the number of W's column bits. The verifier program cp_verify takes both from the parameter
// file, except WsShape::eh, and build() must reject bad ones itself, so nothing filters them here:
// each number is plausible, random or an edge value (such as 0, 2^k and its neighbours, 2^64 - 1).
// Most of these shapes cannot come from a parameter file (cp_verify always sets WsShape::eh to 1,
// for example).
// With `refuseThrows` set, a rejection is thrown (`Refusal`, verifier/src/vf_common.h) instead of
// ending the program, and is counted by its name. Every unit (`WsUnit`, a run of W's rows) of a
// built layout must fit within the layout's m0 rows and lk column bits. That is the program's only
// check: it is meant for gcc's address and undefined-behaviour sanitizers.
// Whether a layout is correct is not tested here: verifier/tests/weights_twin_test.cpp compares
// six small shapes' layouts with those of layoutW() (prover/common/weight_layout.h).
// Usage: ws_build_fuzz [<shapes> [<seed>]]. Exit code 0: no unit out of bounds; 3: one was; after a
// sanitizer's report, the sanitizer's own. SUITE.md's test runs 300000 shapes with seed 1.
// Build, from verifier/src: g++ -O1 -g -fno-ipa-modref -D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined -fno-sanitize-recover=all -std=c++17 -pthread -I . -I ../../blocks
//   -I ../../prover -DBLK_FIELD=2 -DBLK_HASH=1 -DDIGEST_WORDS=8 ../tests/ws_build_fuzz.cpp
// verifier/src/vf_weights_pending_claim.h includes verifier/src/vf_weights_layout.h.
// -fno-ipa-modref: as in verifier/build.sh, which gives the reason.
#include "vf_smallness_matrices.h"
#include "vf_attention_shape.h"
#include "vf_subproof_engine.h"
#include "vf_smallness_test.h"
#include "vf_norm_digit_rows.h"
#include "vf_weights_pending_claim.h"
#include <map>
#include <random>
int main(int argc, char **argv) {
  const u64 runs = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1000000;
  std::mt19937_64 rnd(argc > 2 ? strtoull(argv[2], nullptr, 10) : 1);
  vector<u64> edge = {0, 1, 2, 3, 5, 7, 12, 31, 32, 33, 48, 63, 64, 65, 96, 100, 127, 128, 129, 512, 1000, 2047, 2048, 2049, 5120, 8192, 16384, 202048};
  for (int k = 2; k < 64; k++)
    for (long d = -1; d <= 1; d++) edge.push_back((1ull << k) + (u64)d);
  edge.push_back(~0ull);
  // lowBias: per cent chance of a plausible size, a power of two up to 2^13. Such a size passes
  // build()'s range tests (1 to 2^24), so that many shapes get past them to the later tests.
  auto pick = [&](int lowBias) -> u64 {
    const u64 r = rnd();
    if (r % 100 < (u64)lowBias) return 1ull << (rnd() % 14); // a plausible size
    if (r % 3 == 0) return rnd() >> (rnd() % 64);
    return edge[rnd() % edge.size()];
  };
  std::map<string, u64> names;
  u64 built = 0, units = 0, most = 0;
  refuseThrows = true;
  for (u64 i = 0; i < runs; i++) {
    WsShape s;
    const int bias = (int)(rnd() % 101);
    s.D = rnd() % 4 ? (rnd() & 1 ? 2048 : 32) : pick(0);
    s.L = rnd() % 8 ? 1 + rnd() % (rnd() % 16 ? 9 : 4100) : pick(0);
    s.d = pick(bias), s.dh = pick(bias), s.dkv = pick(bias), s.dq = pick(bias), s.f = pick(bias), s.fe = pick(bias), s.NE = pick(bias), s.V = pick(bias);
    s.eh = rnd() % 16 ? (u32)(rnd() & 1) : (u32)pick(0);
    s.MOE = rnd() % 16 ? (u32)(rnd() & 1) : (u32)pick(0);
    const u64 n = rnd() % 32 ? std::min<u64>(s.L, 5000) : rnd() % 5000;
    s.moe.assign(n, 0);
    for (u64 l = 0; l < n; l++) s.moe[l] = rnd() % 64 ? (u8)(s.MOE && (l & 1)) : (u8)rnd();
    const u32 lk = rnd() % 4 ? 1 + (u32)(rnd() % 20) : (u32)pick(0);
    try {
      WsLay lay;
      lay.build(s, lk);
      built++, units += lay.U.size(), most = std::max(most, lay.m0);
      for (auto &u : lay.U) { // `off`: first row; `ncb`: number of column bits
        if (u.off + u.rows > lay.m0 || u.ncb > lay.lk) { printf("OUT OF ITS OWN BOUNDS at run %llu\n", (unsigned long long)i); return 3; }
      }
    } catch (const Refusal &r) {
      names[r.name]++;
    }
  }
  printf("ws_build_fuzz: %llu shapes: %llu built (%llu units; the most rows %llu = 2^%.2f)\n", (unsigned long long)runs, (unsigned long long)built, (unsigned long long)units, (unsigned long long)most, log2((double)most));
  for (auto &x : names) printf("  refused %s: %llu\n", x.first.c_str(), (unsigned long long)x.second);
  return 0;
}
