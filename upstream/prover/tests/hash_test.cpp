// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Compares prover/common/hash.h with known values. Run it from prover/.
// (1) The vector file, verifier/tests/hash_vectors.txt: lines "H <num> <L> <the state, SW words of
//     8 hex digits>" on the test list of L words: word k is ((k + 1) * 0x89ABCDEF) mod 2^32,
//     k = 0 .. L - 1. earlier_hash_h_state() must give the state, and earlier_hash_h_digest() its first DW words, the digest.
//     No domain number <num> is 0, so a function that leaves num and L out of the compression's
//     counter is detected. verifier/tests/known_answers/format_hash.py prints this file, and its
//     compression function has the same outside source as f2b::F() in prover/common/blake2b_f.h:
//     a fault that both took from that source is not detected by (1).
// (2) With num = 0 and l = 0 in every call, earlier_hash_h_state_with_length() is BLAKE2b-512 on the list's bytes. The test
//     checks seven digests computed by Python's hashlib.blake2b on the same lists and, by one call on the block "abc",
//     the digest that RFC 7693 prints in its Appendix A.
// (3) earlier_hash_bare_c() is earlier_hash_c() with the length 0 and without the last call's flag; each of earlier_hash_c()'s seven arguments matters.
// It does not cover the tree hash (prover/tests/tree_hash_test.cu reads the vector file's lines
// "TH"), any Merkle tree (tree_f()), the GPU branches of prover/common/blake2b_f.h, a
// kernel's own implementation of a chain, or a counter of 2^32 bytes or more.
#include "common/hash.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static std::vector<u32> list(size_t L) {
  std::vector<u32> w(L);
  for (size_t k = 0; k < L; k++) w[k] = (u32)(k + 1) * 0x89ABCDEFu;
  return w;
}
static std::string words(const u32 *w, int n) {
  std::string o;
  char c[9];
  for (int i = 0; i < n; i++) snprintf(c, sizeof c, "%08x", w[i]), o += c;
  return o;
}
static std::string bytes(const u32 *w, int n) {
  std::string o;
  char c[3];
  for (int i = 0; i < 4 * n; i++) snprintf(c, sizeof c, "%02x", (w[i / 4] >> (8 * (i % 4))) & 255), o += c;
  return o;
}
static const struct {
  size_t L;
  const char *digest;
} PLAIN[] = {
    {0, "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419"
        "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"},
    {1, "c74f45aec3b548f46f0d4e7c3a4d426513b81b177287dc3c7e196bb156efd3d9"
        "831a0e1e08c1d2998988901e5b9c499daa8eeaf8eeddf223bac39c0cf2afbb75"},
    {31, "81f155cefec9cfce063c8a7297b37ab7521aab0c40caab1c7052b970e2b6ce96"
         "5344843c6f392cb3366fabe6555491002e7d41fea67c4c826a0a2bbfb4bd7c5b"},
    {32, "7e02942cea254c380bb3835a78de0656698bbc6931a3ce04b8770771ed11b63f"
         "5b49c3feddc2d5c88e9c1715fe59dc4fbcf55ed8b16060cc96c23ddb5feb3093"},
    {33, "3734e1e8975ba81aecf30c6cb711a3a7719b85154eaad3a0b665fc51b17ddc2f"
         "575b47e27a9397b427578b6e6fb9b17c6381551f1b53583d98db7dac545fe46a"},
    {64, "f32d951f9a6a510286dba60e6212987c71921ce01569969e500f46490adc875a"
         "e9fb32f4c1502f2c6311f57fd51681d21e07345c0eeed1c50a2681e9ebb87c42"},
    {300, "2d61f0e983abe2b9c5ee66f52b84779355417195ed49af3e635cf1bb788a87cd"
          "5da8ed447cd50bca484a84c16c5a30b75c1119960ad56b72260c0e8e7395c202"},
};
// RFC 7693, Appendix A: BLAKE2b-512("abc").
static const char *const ABC = "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
                               "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923";
int main() {
  int bad = 0, lines = 0;
  const char *const vectors = "../verifier/tests/hash_vectors.txt";
  FILE *f = fopen(vectors, "r");
  if (!f) {
    printf("hash_test: %s cannot be read: run it from prover/\n", vectors);
    return 2;
  }
  char text[600], kind[8], hex[400];
  unsigned num;
  size_t L;
  while (fgets(text, sizeof text, f)) {
    if (sscanf(text, "%7s %u %zu %399s", kind, &num, &L, hex) != 4 || strcmp(kind, "H")) continue;
    const std::vector<u32> w = list(L);
    u32 s[SW], d[DW];
    earlier_hash_h_state(s, num, w.data(), L);
    earlier_hash_h_digest(d, num, w.data(), L);
    lines++;
    if (!num || words(s, SW) != hex || memcmp(s, d, 4 * DW)) bad++, printf("H %u %zu: a different answer\n", num, L);
  }
  fclose(f);
  if (lines != 44) bad++, printf("%d lines begin with H, not 44\n", lines);
  for (const auto &p : PLAIN) {
    const std::vector<u32> w = list(p.L);
    u32 s[SW];
    earlier_hash_h_state_with_length(s, 0, 0, w.data(), p.L, 1);
    if (bytes(s, SW) != p.digest) bad++, printf("BLAKE2b-512 on %zu words: a different answer\n", p.L);
  }
  {
    u32 s[SW], m[BLKW] = {0x00636261u}; // the three bytes, then zeros
    earlier_hash_initial_state(s);
    earlier_hash_c(s, m, 3, 0, 0, 0, true);
    if (bytes(s, SW) != ABC) bad++, printf("BLAKE2b-512 on \"abc\": a different answer\n");
  }
  {
    const std::vector<u32> w = list(SW + BLKW);
    auto answer = [&](u32 c0, u32 c1, u32 l, u32 num_, bool last, int bent) {
      u32 cv[SW], m[BLKW];
      memcpy(cv, w.data(), 4 * SW), memcpy(m, w.data() + SW, 4 * BLKW);
      if (bent == 1) cv[SW - 1] ^= 1;
      if (bent == 2) m[BLKW - 1] ^= 1;
      earlier_hash_c(cv, m, c0, c1, l, num_, last);
      return words(cv, SW);
    };
    u32 cv[SW];
    memcpy(cv, w.data(), 4 * SW);
    earlier_hash_bare_c(cv, w.data() + SW, 5, 6, 7);
    const std::string base = answer(5, 6, 0, 7, false, 0);
    if (words(cv, SW) != base) bad++, printf("a bare call is not earlier_hash_c with the length 0 and no flag\n");
    const std::string others[] = {answer(4, 6, 0, 7, false, 0), answer(5, 7, 0, 7, false, 0),
                                  answer(5, 6, 1, 7, false, 0), answer(5, 6, 0, 8, false, 0),
                                  answer(5, 6, 0, 7, true, 0),  answer(5, 6, 0, 7, false, 1),
                                  answer(5, 6, 0, 7, false, 2)};
    for (const std::string &o : others)
      if (o == base) bad++, printf("an argument of earlier_hash_c has no effect on the result\n");
  }
  printf("hash_test: %d vectors, %zu plain digests and the RFC's, seven arguments: %s\n", lines,
         sizeof PLAIN / sizeof *PLAIN, bad ? "FAILS" : "passes");
  return bad != 0;
}
