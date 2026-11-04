// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// CPU test of hf_mask8() and hf_salt5() (prover/hash_commitment/hf_mask.cuh), which derive a weight
// table's masks (random values encoded with each row of weights) and its Merkle leaves' salts from
// ChaCha20 blocks under the table's 32-byte key. For masks, a block with any of its eight values at
// or above the threshold (`HF_MASK_LIMIT`) is dropped and the next try is made.
//   1. Known answers under public test keys, parsed from toy/mask_rules.py, a Python script that
//      shares no code with the prover and recomputes them when it is run.
//   2. prover/hash_commitment/hf_mask_known.h, which the programs' run-time self-test
//      generator_own_test() uses, has exactly the same known answers, in the same order.
//   3. At five chosen blocks: the values (hc_block_value(), hf_mask_ref::value()), hc_reduce_for_tests()
//      (a value modulo the field's p), and the threshold, the largest multiple of p below 2^64.
// Build and run from prover/: tests/unit.sh both tests/masking_cells_test.cpp <directory>
// The program's exit code: 0: all pass; 1: some differ; 2: toy/mask_rules.py cannot be read.
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_known.h"
#include "hash_commitment/hf_mask_ref.h"
#include <cstdio>
#include <cstring>
#include <string>
#if HC_MASK_LOW_LIMIT
#error "this test is of a build without HC_MASK_LOW_LIMIT"
#endif

static int failed = 0;
static void check(bool good, const std::string &what) {
  printf("%s %s\n", good ? "as wanted" : "DIFFERS  ", what.c_str());
  failed += !good;
}
static hf_key32 key_of_the_tests(int n) {
  hf_key32 key{};
  for (int i = 0; i < HF_SEED_WORDS; i++) key.w[i] = hc_test_seed_word(n, i);
  return key;
}
// Parses `count` numbers of sixteen hexadecimal digits each, separated by single spaces.
static bool numbers(const char *text, int count, uint64_t *out) {
  if (strlen(text) != 17 * (size_t)count - 1) return false;
  for (int i = 0; i < count; i++) {
    unsigned long long v = 0;
    if (sscanf(text + 17 * i, "%16llx", &v) != 1 || (i + 1 < count && text[17 * i + 16] != ' ')) return false;
    out[i] = v;
  }
  return true;
}

int main() {
  FILE *f = fopen("../toy/mask_rules.py", "r");
  if (!f) {
    printf("masking_cells_test: ../toy/mask_rules.py cannot be read: run it from prover/\n");
    return 2;
  }
  constexpr size_t MASKS = sizeof HF_KNOWN_MASKS / sizeof *HF_KNOWN_MASKS,
                   SALTS = sizeof HF_KNOWN_SALTS / sizeof *HF_KNOWN_SALTS;
  size_t masks = 0, salts = 0, far = 0, again = 0;
  bool header = true;
  char line[600], text[300];
  int table = 0; // 1 in KNOWN (for another prime), 2 in KNOWN2; only KNOWN2 is read: BLK_FIELD is 2
  // The reference implementation (prover/hash_commitment/hf_mask_ref.h; printed as "the plain
  // twin") must reproduce every known answer, also those at a lower threshold (k eighths of the
  // full one, k < 8) with a second, third or fourth try. hf_salt5() must reproduce the salts, and
  // hf_mask8() the masks at k = 8 only: without HC_MASK_LOW_LIMIT, as here, it ignores its argument
  // `limit`. hf_mask8() must also return whether it made a later try: three answers at k = 8 need
  // a second try, none a third. "Far" indices (row > 255, block > 65,535, leaf >= 2^32) catch a
  // truncated index.
  while (fgets(line, sizeof line, f)) {
    if (!strncmp(line, "KNOWN = {", 9)) table = 1;
    if (!strncmp(line, "KNOWN2 = {", 10)) table = 2;
    if (table != BLK_FIELD) continue;
    int n = 0, k = 0;
    unsigned r = 0, g = 0, tries = 0;
    unsigned long long leaf = 0;
    uint64_t want[8], got[8];
    uint8_t bytes[32];
    const bool mask_row =
        sscanf(line, " (\"MASK\", %d, %u, %u, %d): (%u, \"%299[0-9a-f ]\"),", &n, &r, &g, &k, &tries, text) == 6;
    if (!mask_row && sscanf(line, " (\"SALT\", %d, %llu): \"%299[0-9a-f ]\",", &n, &leaf, text) != 3) continue;
    const hf_key32 key = key_of_the_tests(n);
    for (int i = 0; i < 32; i++) bytes[i] = (uint8_t)(key.w[i / 4] >> (8 * (i % 4)));
    if (mask_row) {
      const std::string at = "key " + std::to_string(n) + ", row " + std::to_string(r) + ", block " +
                             std::to_string(g) + ", threshold " + std::to_string(k) + " p";
      bool good = numbers(text, 8, want);
      for (int i = 0; good && i < 8; i++) {
        uint32_t took = 0;
        good = hf_mask_ref::mask(bytes, r, 8 * (uint64_t)g + i, hf_mask_ref::limit(k), &took) == want[i] &&
               took == tries;
      }
      check(good, "the plain twin's eight masks and its number of tries: " + at);
      if (k == 8)
        check(hf_mask8(key, r, g, HF_MASK_LIMIT, got) == (tries != 0) && !memcmp(got, want, 64), "hf_mask8(): " + at);
      header = header && masks < MASKS && HF_KNOWN_MASKS[masks].n == n && HF_KNOWN_MASKS[masks].r == r &&
               HF_KNOWN_MASKS[masks].g == g && HF_KNOWN_MASKS[masks].k == k && HF_KNOWN_MASKS[masks].tries == tries &&
               !memcmp(HF_KNOWN_MASKS[masks].m, want, 64);
      masks++, far += r > 255 || g > 65535, again += tries != 0;
    } else {
      const std::string at = "key " + std::to_string(n) + ", leaf " + std::to_string(leaf);
      bool good = numbers(text, HF_SALT_SYMBOLS, want);
      for (int i = 0; good && i < HF_SALT_SYMBOLS; i++) good = hf_mask_ref::salt(bytes, leaf, i) == want[i];
      check(good, "the plain twin's five salt symbols: " + at);
      hf_salt5(key, leaf, got);
      check(!memcmp(got, want, 8 * HF_SALT_SYMBOLS), "hf_salt5(): " + at);
      header = header && salts < SALTS && HF_KNOWN_SALTS[salts].n == n && HF_KNOWN_SALTS[salts].I == leaf &&
               !memcmp(HF_KNOWN_SALTS[salts].s, want, 8 * HF_SALT_SYMBOLS);
      salts++, far += leaf >> 32 != 0;
    }
  }
  fclose(f);
  check(masks >= 20 && salts >= 8 && far >= 8 && again >= 6,
        "the table has at least 20 rows of masks and 8 of salts, 8 of them at far indices and 6 with a later try");
  check(header && masks == MASKS && salts == SALTS, "hf_mask_known.h holds exactly the table's rows, in its order");

  {
    const uint64_t p = hf_mask_ref::P;
    const uint64_t limit = ~(uint64_t)0 / p * p; // the largest multiple of p below 2^64
    uint32_t ones[16], zeros[16] = {}, counted[16], last_good[16], first_bad[16];
    for (int w = 0; w < 16; w++) {
      ones[w] = ~0u, counted[w] = 0x03020100u + 0x04040404u * (uint32_t)w; // byte j is j
      last_good[w] = (uint32_t)(w % 2 ? (limit - 1) >> 32 : limit - 1);    // each value is the threshold less 1
      first_bad[w] = (uint32_t)(w % 2 ? limit >> 32 : limit);              // each value is the threshold
    }
    bool top = true, bottom = true, each = true, under = true, over = true;
    for (int s = 0; s < 8; s++) {
      uint64_t bytes_in_order = 0;
      for (int i = 7; i >= 0; i--) bytes_in_order = 256 * bytes_in_order + (uint64_t)(8 * s + i);
      const auto both = [s](const uint32_t *o, uint64_t value) {
        return hc_block_value(o, s) == value && hf_mask_ref::value(o, s) == value;
      };
      top = top && both(ones, ~(uint64_t)0) && hc_reduce_for_tests(~(uint64_t)0) == ~(uint64_t)0 - limit && ~(uint64_t)0 >= HF_MASK_LIMIT;
      bottom = bottom && both(zeros, 0) && hc_reduce_for_tests(0) == 0;
      each = each && both(counted, bytes_in_order) && hc_reduce_for_tests(bytes_in_order) == bytes_in_order % p;
      under = under && both(last_good, limit - 1) && limit - 1 < HF_MASK_LIMIT && hc_reduce_for_tests(limit - 1) == p - 1;
      over = over && both(first_bad, limit) && !(limit < HF_MASK_LIMIT) && hc_reduce_for_tests(limit) == 0;
    }
    check(top, "a block of all ones: eight times 2^64 - 1, which is not below the threshold and is 2^64 - 1 less the threshold modulo p");
    check(bottom, "a block of zeros: eight times 0");
    check(each,
          "value s is bytes 8 s .. 8 s + 7 of the block, the low byte first, and its mask is that number modulo p");
    check(under, "eight times the largest multiple of p below 2^64, less 1: below the threshold, and p - 1 modulo p");
    check(over, "eight times the largest multiple of p below 2^64: not below the threshold");
  }

  printf("%d DIFFER\n", failed);
  return failed != 0;
}
