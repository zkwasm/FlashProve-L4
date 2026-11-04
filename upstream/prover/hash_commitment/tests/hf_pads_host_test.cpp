// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CPU test of the pads of a table (prover/hash_commitment/hf_pads.cuh). It needs
// neither a GPU nor CUDA.
//   (1) The three implementations of the rule give the known answers of hf_pads_known.h, some of
//       whose blocks need a retry: hf_pads.cuh, the CPU reference version (hf_mask_ref.h) and
//       prover/second_writing/mask_generator.h.
//   (2) The three agree on all pads of toy tables with 1, 2 and 4 stored answers.
//   (3) [own-test-pads] passes on the generated pads, and fails on: all pads 0; one pad 0; one pad
//       truncated to 8 or to 24 bits; one row's pads in all rows; one answer's pads for two
//       answers; the first masks in place of the pads; another key's pads; a missing second copy
//       of the key.
// Build, from prover/: ./build.sh hc_host_test tests/hf_pads_host_test
// Last line: "hf_pads_host_test: PASS, <n> checks", exit status 0, or FAIL, status 1.
#include "hash_commitment/hf_pads.cuh"
#include "second_writing/mask_generator.h"
#include <cstdio>
#include <cstring>
#include <functional>

static int checks = 0, bad = 0;
static void expect(bool good, const char *what) {
  checks++;
  if (!good) bad++, printf("FAILS: %s\n", what);
}
static void bytes_of(const char *hex, uint8_t out[32]) {
  for (int i = 0; i < 32; i++) {
    unsigned x = 0;
    sscanf(hex + 2 * i, "%2x", &x);
    out[i] = (uint8_t)x;
  }
}
struct Keys { // one key in the forms that the three implementations take
  uint8_t bytes[32];
  hf_key32 fast{};
  coins::Key second{};
  explicit Keys(const uint8_t b[32]) {
    memcpy(bytes, b, 32);
    memcpy(fast.w, b, 32);
    for (int i = 0; i < 8; i++) second[i] = b[4 * i] | b[4 * i + 1] << 8 | b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
  }
};
// A table's parameters, as far as the pads' functions read them.
static hf::HfTableParams table(size_t rows, int answers) {
  hf::HfTableParams P;
  P.R = rows, P.answers = answers, P.i0 = 64;
  return P;
}

int main() {
  const uint64_t full = hf_mask_ref::FULL;
  int again = 0;
  for (const HfKnownPads &row : HF_KNOWN_PADS) { // (1)
    uint8_t b[32];
    bytes_of(row.key, b);
    const Keys k(b);
    uint64_t m[8];
    hf::pad_block(k.fast, row.r, row.g, HF_MASK_LIMIT, m);
    const std::array<uint64_t, 8> second = coins::pads(k.second, row.r, row.g);
    for (int i = 0; i < 8; i++) {
      uint32_t tries = 0;
      expect(m[i] == row.pads[i], "hf_pads.cuh does not give a known answer");
      expect(hf_mask_ref::pad(k.bytes, row.r, 8 * (uint64_t)row.g + i, full, &tries) == row.pads[i], "hf_mask_ref.h does not give a known answer");
      expect(second[i] == row.pads[i], "mask_generator.h does not give a known answer");
      expect((tries != 0) == (row.second_try != 0), "hf_mask_ref.h takes another try than the known answer's");
    }
    again += row.second_try;
  }
  expect(again >= 4, "fewer than four known answers have a second try");
  uint8_t b[32], other[32];
  for (int i = 0; i < 32; i++) b[i] = (uint8_t)(7 * i + 3), other[i] = (uint8_t)(5 * i + 1);
  const Keys k(b), k2(other);
  for (int answers : {1, 2, 4}) {
    const hf::HfTableParams P = table(24, answers);
    const size_t count = (size_t)P.pads();
    hf::TablePads pads;
    hf::draw_table_pads(P, &k.fast, pads);
    expect(pads.host.size() == P.R * count, "another number of pads than three for each row and answer");
    for (size_t r = 0; r < P.R; r++) // (2)
      for (size_t j = 0; j < count; j++) {
        expect(pads.host[r * count + j] == hf_mask_ref::pad(k.bytes, (uint32_t)r, j, full), "hf_pads.cuh and hf_mask_ref.h differ");
        expect(pads.host[r * count + j] == coins::pads(k.second, (coins::Word)r, (coins::Word)(j / 8))[j % 8], "hf_pads.cuh and mask_generator.h differ");
      }
    // (3)
    expect(hf::pads_own_test(P, &k.fast, k.bytes, pads) == hf::OWN_GOOD, "[own-test-pads] fails on the generated pads");
    const std::vector<uint64_t> drawn = pads.host;
    auto fails = [&](const char *what, const std::function<void(std::vector<uint64_t> &)> &change) {
      pads.host = drawn;
      change(pads.host);
      expect(pads.host != drawn, "an alteration of the pads changes no value");
      expect(hf::pads_own_test(P, &k.fast, k.bytes, pads) == hf::OWN_PADS, what);
    };
    fails("all pads 0 pass", [](std::vector<uint64_t> &p) { std::fill(p.begin(), p.end(), 0); });
    fails("one pad 0 passes", [&](std::vector<uint64_t> &p) { p[5 * count + 1] = 0; });
    fails("a pad cut to 8 bits passes", [&](std::vector<uint64_t> &p) { p[7 * count] &= 0xFF; });
    fails("a pad cut to 24 bits passes", [&](std::vector<uint64_t> &p) { p.back() &= 0xFFFFFF; });
    fails("one row's pads in all rows pass", [&](std::vector<uint64_t> &p) {
      for (size_t i = count; i < p.size(); i++) p[i] = p[i % count];
    });
    if (answers > 1)
      fails("one answer's pads for two answers pass", [&](std::vector<uint64_t> &p) {
        for (size_t r = 0; r < P.R; r++)
          for (int t = 0; t < 3; t++) p[r * count + 3 + t] = p[r * count + t];
      });
    fails("the first masks in the pads' place pass", [&](std::vector<uint64_t> &p) {
      for (size_t i = 0; i < p.size(); i++) p[i] = hf_mask_ref::mask(k.bytes, (uint32_t)(i / count), i % count, full);
    });
    fails("another key's pads pass", [&](std::vector<uint64_t> &p) {
      for (size_t i = 0; i < p.size(); i++) p[i] = hf_mask_ref::pad(k2.bytes, (uint32_t)(i / count), i % count, full);
    });
    pads.host = drawn;
    expect(hf::pads_own_test(P, &k.fast, nullptr, pads) == hf::OWN_PADS, "[own-test-pads] passes without the key's second copy");
    expect(hf::pads_own_test(P, &k.fast, k2.bytes, pads) == hf::OWN_PADS, "[own-test-pads] passes under another second copy");
    expect(P.pad_entry(0) == P.i0 - 3 * (uint64_t)answers && P.pad_entry(count - 1) == P.i0 - 1, "the pads are not the last entries of a row's message");
  }
  printf("hf_pads_host_test: %s, %d checks\n", bad ? "FAIL" : "PASS", checks);
  return bad ? 1 : 0;
}
