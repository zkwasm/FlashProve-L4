// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Holds BLAKE2b's compression function against values that it did not produce: RFC 7693, Appendix A (BLAKE2b-512 of "abc"), and 13 digests that
// Python's hashlib.blake2b gave, at input lengths around the borders of a block, at five output lengths, with and without a key. BLAKE2b is written
// here on Blake2bF::F: a parameter block, then a walk over the blocks of the whole string, following RFC 7693, section 3.3.
// What these values do not reach: salt and personalization (mode_test.cpp, against a library's values), the counter's upper half (older_test.cpp).
#include "hash/hash.h"
#include <cstring>
#include <string>
#include <vector>
using namespace blk;

static std::string hex(const u8 *b, size_t n) {
  std::string s;
  char t[3];
  for (size_t i = 0; i < n; i++) snprintf(t, sizeof t, "%02x", b[i]), s += t;
  return s;
}

// ---- BLAKE2b on F ----
static std::vector<u8> blake2b(size_t outlen, const std::vector<u8> &key, const std::vector<u8> &in) {
  const u8 block[8] = {(u8)outlen, (u8)key.size(), 1, 1}; // the parameter block's first word: digest length, key length, fanout, depth; all else is 0
  u64 h[8] = {blake2b_IV0, blake2b_IV1, blake2b_IV2, blake2b_IV3, blake2b_IV4, blake2b_IV5, blake2b_IV6, blake2b_IV7};
  for (int k = 0; k < 8; k++) h[0] ^= (u64)block[k] << 8 * k;
  std::vector<u8> all;
  if (!key.empty()) all = key, all.resize(128, 0);
  all.insert(all.end(), in.begin(), in.end());
  const size_t blocks = all.empty() ? 1 : (all.size() + 127) / 128;
  for (size_t k = 0; k < blocks; k++) {
    u64 m[16] = {0};
    const size_t have = all.size() - 128 * k < 128 ? all.size() - 128 * k : 128;
    for (size_t i = 0; i < have; i++) m[i / 8] |= (u64)all[128 * k + i] << 8 * (i % 8);
    Blake2bF::F(h, m, 128 * k + have, 0, k + 1 == blocks);
  }
  std::vector<u8> out(outlen);
  for (size_t i = 0; i < outlen; i++) out[i] = (u8)(h[i / 8] >> 8 * (i % 8));
  return out;
}
// len bytes of the test list, starting at word number skip: word k is ((k + 1) * 0x89ABCDEF) mod 2^32, lowest byte first.
static std::vector<u8> list_bytes(size_t len, u32 skip = 0) {
  std::vector<u8> out(len);
  for (size_t i = 0; i < len; i++) out[i] = (u8)((skip + (u32)(i / 4) + 1) * 0x89ABCDEFu >> 8 * (i % 4));
  return out;
}
// hashlib.blake2b(list_bytes(in), digest_size=out, key=list_bytes(key, 300)).hexdigest()
static const struct {
  size_t in, out, key;
  const char *digest;
} CASES[] = {
    {0, 64, 0,
     "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419"
     "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"},
    {1, 64, 0,
     "0c850ed290a87f91004c8fd3c640c0820ece1d2d94600295b6f84e98d1b59918"
     "c901fdaa0e696ef654dfbd37eb5d9f0cf6e9d21cd84b74efd395fe80f68f1f13"},
    {127, 64, 0,
     "9658e063dd0aadb0471a6d2eb5df13e43ce73204a4e6a7182e0a1db5bbfed420"
     "452266567c8a2c062adf7be9be2cb30783b67acb4ab9c527b9e364e6778255ca"},
    {128, 64, 0,
     "7e02942cea254c380bb3835a78de0656698bbc6931a3ce04b8770771ed11b63f"
     "5b49c3feddc2d5c88e9c1715fe59dc4fbcf55ed8b16060cc96c23ddb5feb3093"},
    {129, 64, 0,
     "b99b841c944b2c588415d2d997640ccbf7b55a4a2a8f9162e22ab19f3394546c"
     "39110335e0be0380d0a33c6e3cf7840cdd44d1f052554dde5d85f4c32a7cb340"},
    {256, 64, 0,
     "f32d951f9a6a510286dba60e6212987c71921ce01569969e500f46490adc875a"
     "e9fb32f4c1502f2c6311f57fd51681d21e07345c0eeed1c50a2681e9ebb87c42"},
    {257, 32, 0,
     "be8cb865836ba519ffe40c50e27386bdfdbf412d429995607af7c881d8b2dd48"},
    {1000, 20, 0,
     "9aae6dc2a23430e9b63a6d4be3f9718b11d7a963"},
    {0, 64, 64,
     "1e937a4638b074af51b8f3f0d1a8a851ef8dcd3ece9825e1fc1f12cff81186f9"
     "20e386237eb7fe093be8d22b31dab1f244a43148adb610811fbbd765b240bbc0"},
    {3, 32, 1,
     "648431c428d15da12530e22c2a43812e4bd85f0c003411895f3111a2eb935592"},
    {128, 48, 32,
     "747e6694b18feaa9f00d8b764fc1226d2757935010126bb251f870ddaec1cf41"
     "546232c310801fc44d7d1f90acb07187"},
    {255, 1, 63,
     "2e"},
    {1024, 64, 64,
     "2637cee7503e529317cae95d31c092314325495eabbfaa9833cdb0fd8a663a92"
     "3162d44e0ced0e82adb2f9fefd2750177f37e076c116e6bc96c221a1e6ce6079"},
};

int main() {
  int bad = 0, checks = 0;
  auto same = [&](const char *what, const std::string &got, const std::string &want) {
    checks++;
    if (got != want) bad++, printf("\n%s: got %s, expected %s", what, got.c_str(), want.c_str());
  };
  const std::vector<u8> abc = {'a', 'b', 'c'};
  same("BLAKE2b-512 of abc", hex(blake2b(64, {}, abc).data(), 64),
       "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
       "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
  for (const auto &c : CASES) {
    const std::string what = "hashlib's digest of " + std::to_string(c.out) + " bytes on " + std::to_string(c.in) + " bytes under a key of " + std::to_string(c.key);
    same(what.c_str(), hex(blake2b(c.out, list_bytes(c.key, 300), list_bytes(c.in)).data(), c.out), c.digest);
  }
  printf("%s%d known values of BLAKE2b: %s\n", bad ? "\n" : "", checks, bad ? "NOT all met" : "all met");
  return bad != 0;
}
