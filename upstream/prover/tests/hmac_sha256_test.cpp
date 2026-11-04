// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Tests HmacSha256 (prover/common/hmac_sha256.h), and Sha256::digest() and Sha256::hex()
// (prover/weights_commitment/sha256.h), against known values. CPU only. Build and run from prover/:
//   tests/unit.sh both tests/hmac_sha256_test.cpp <directory>
// Exit code: 0 if no check failed, else 1. prover/tests/unit.sh enables the x86 SHA instructions,
// so Sha256's portable path is not run.
// Public source: RFC 4231, "Identifiers and Test Vectors for HMAC-SHA-224, HMAC-SHA-256,
// HMAC-SHA-384, and HMAC-SHA-512", test cases 1 to 4. Their keys are shorter than HmacSha256's
// 32-byte key. HMAC-SHA-256 pads a key with zeros to 64 bytes, so padding to 32 changes no value.
// The other HMAC values are from Python:
//   python3 -c "import hmac, hashlib; print(hmac.new(key, message, hashlib.sha256).hexdigest())"
#include "common/hmac_sha256.h"
#include <cstdio>
#include <string>
#include <vector>

static int checks = 0, differ = 0;
static void check(bool holds, const std::string &what) {
  checks++;
  if (!holds) differ++, printf("DIFFERS: %s\n", what.c_str());
}
static std::string hex(const uint8_t *bytes, size_t n) {
  static const char *d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; i++) s.push_back(d[bytes[i] >> 4]), s.push_back(d[bytes[i] & 15]);
  return s;
}
using Bytes = std::vector<uint8_t>;
// The HMAC as hex digits, with the message fed in pieces of `piece` bytes. `key`: at most 32 bytes.
static std::string hmac(Bytes key, const Bytes &message, size_t piece) {
  key.resize(32, 0);
  uint8_t out[32];
  HmacSha256 mac(key.data());
  for (size_t at = 0; at < message.size(); at += piece) mac.update(message.data() + at, std::min(piece, message.size() - at));
  mac.finish(out);
  return hex(out, 32);
}
static Bytes made(size_t n, unsigned (*byte)(unsigned)) {
  Bytes b(n);
  for (size_t i = 0; i < n; i++) b[i] = (uint8_t)byte((unsigned)i);
  return b;
}
static Bytes text(const char *s) { return Bytes(s, s + strlen(s)); }

int main() {
  auto count = [](unsigned i) { return i; };
  const Bytes k1 = made(32, count), k2(32, 0xaa), k3(32, 0);
  const struct {
    const char *name;
    Bytes key, message;
    const char *value;
  } cases[] = {
      {"RFC 4231, test case 1", Bytes(20, 0x0b), text("Hi There"), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"},
      {"RFC 4231, test case 2", text("Jefe"), text("what do ya want for nothing?"), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"},
      {"RFC 4231, test case 3", Bytes(20, 0xaa), Bytes(50, 0xdd), "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"},
      {"RFC 4231, test case 4", made(25, [](unsigned i) { return i + 1; }), Bytes(50, 0xcd), "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"},
      {"the key 00 .. 1f, no message", k1, {}, "d38b42096d80f45f826b44a9d5607de72496a415d3f4a1a8c88e3bb9da8dc1cb"},
      {"the key 00 .. 1f, abc", k1, text("abc"), "f0133729c4163dede81e21cd47839256da58171238c8a0d874397c73b14e1e47"},
      {"32 bytes aa, 63 bytes", k2, made(63, count), "f658d66c2476f752e14b8526425ba3aacdc826b4552e281de610b3585bb73b8a"},
      {"32 bytes aa, 64 bytes", k2, made(64, count), "6763d2af2a4335939ec3b7a8ca46b0ceac0416b253aaeda81f06d467a7e9b546"},
      {"32 bytes aa, 65 bytes", k2, made(65, count), "bd2cb5e5d51d98b7e6f79cf664fffe6465b9dba59244029716ae005181cfd43e"},
      {"the zero key, 1000 bytes", k3, made(1000, [](unsigned i) { return 7 * i; }), "3a545fcdca736b77518dcc120c179ea109623d46d840556bdd953ddda159578c"},
      {"the key 00 .. 1f, 100000 bytes", k1, made(100000, [](unsigned i) { return i * i; }), "db78fa2cd84b351d13ef1f4051d6d533febb90d90e19640c0e93425015a45122"},
  };
  for (const auto &c : cases)
    for (const size_t piece : {(size_t)1 << 20, (size_t)1, (size_t)7, (size_t)64})
      check(hmac(c.key, c.message, piece) == c.value, std::string(c.name) + ", fed in pieces of " + std::to_string(piece));
  {
    Sha256 one, other;
    uint8_t raw[32];
    one.update("abc", 3), other.update("abc", 3);
    one.digest(raw);
    check(hex(raw, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" && other.hex() == hex(raw, 32),
          "SHA-256 of abc (FIPS 180-4's example): the 32 bytes, and the same as hex digits");
  }
  printf("hmac_sha256_test: %d checks, %d failed\n", checks, differ);
  return differ ? 1 : 0;
}
