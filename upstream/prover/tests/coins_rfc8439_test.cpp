// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks, on the CPU, two implementations of the ChaCha20 block function against the test vectors
// that RFC 8439 publishes. The first is chacha20_block() in prover/common/coin_chacha.h, from which
// the prover program `cp_prove` derives part of its private randomness (`coins` in identifiers).
// The second (a "writing" in messages) is coins::chacha_block() in prover/tests/coins_second_text/.
// With it, prover/tests/coins_known_maker.cpp computes known answers and prints them as
// prover/hiding/coins_known_f2.h. They are for hz::coins_own_test(), a start-up self-test of
// `cp_prove` and the committer program `cp_commit`.
// Neither implementation includes a file of the other, but both follow one source,
// D. J. Bernstein's chacha-merged.c, so the known answers cannot show a fault that both took
// from it: comparing the two shows only that they agree. This test ties both to published values:
// section 2.1.1 for the quarter round; section 2.3.2, and appendix A.1, test vectors 1 to 5, for
// the block function and for coins::chacha_blocks(), which computes several blocks in one call.
// Not covered: chacha20_block() on a GPU, which prover/tests/coin_chacha_gpu_test.cu and
// hz::coins_own_test() check; other ChaCha20 block functions, such as those in
// blocks/coins/chacha.h and prover/hash_commitment/hc_chacha.cuh.
#include "common/coin_chacha.h"
#include "tests/coins_second_text/chacha_block.h"
#include <cstdio>
#include <cstring>

static int failed = 0;
static void expect(bool held, const char *what) {
  if (!held) printf("FAILED: %s\n", what), failed++;
}

// One published vector. `where`: its location in RFC 8439. `state`: the sixteen words that the RFC
// prints as "ChaCha state at the end", typed here. Key and nonce bytes are read as little-endian
// words (section 2.3).
struct Rfc8439Vector {
  const char *where;
  uint8_t key[32];
  uint32_t counter;
  uint8_t nonce[12];
  uint32_t state[16];
};
static const Rfc8439Vector VECTORS[] = {
    {"section 2.3.2",
     {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f},
     1,
     {0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00},
     {0xe4e7f110u, 0x15593bd1u, 0x1fdd0f50u, 0xc47120a3u, 0xc7f4d1c7u, 0x0368c033u, 0x9aaa2204u, 0x4e6cd4c3u,
      0x466482d2u, 0x09aa9f07u, 0x05d7c214u, 0xa2028bd9u, 0xd19c12b5u, 0xb94e16deu, 0xe883d0cbu, 0x4e3c50a2u}},
    {"appendix A.1, test vector 1",
     {0},
     0,
     {0},
     {0xade0b876u, 0x903df1a0u, 0xe56a5d40u, 0x28bd8653u, 0xb819d2bdu, 0x1aed8da0u, 0xccef36a8u, 0xc70d778bu,
      0x7c5941dau, 0x8d485751u, 0x3fe02477u, 0x374ad8b8u, 0xf4b8436au, 0x1ca11815u, 0x69b687c3u, 0x8665eeb2u}},
    {"appendix A.1, test vector 2",
     {0},
     1,
     {0},
     {0xbee7079fu, 0x7a385155u, 0x7c97ba98u, 0x0d082d73u, 0xa0290fcbu, 0x6965e348u, 0x3e53c612u, 0xed7aee32u,
      0x7621b729u, 0x434ee69cu, 0xb03371d5u, 0xd539d874u, 0x281fed31u, 0x45fb0a51u, 0x1f0ae1acu, 0x6f4d794bu}},
    {"appendix A.1, test vector 3", // the key's last byte is 1
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
     1,
     {0},
     {0x2452eb3au, 0x9249f8ecu, 0x8d829d9bu, 0xddd4ceb1u, 0xe8252083u, 0x60818b01u, 0xf38422b8u, 0x5aaa49c9u,
      0xbb00ca8eu, 0xda3ba7b4u, 0xc4b592d1u, 0xfdf2732fu, 0x4436274eu, 0x2561b3c8u, 0xebdd4aa6u, 0xa0136c00u}},
    {"appendix A.1, test vector 4", // the key's second byte is 0xff
     {0, 0xff},
     2,
     {0},
     {0xfb4dd572u, 0x4bc42ef1u, 0xdf922636u, 0x327f1394u, 0xa78dea8fu, 0x5e269039u, 0xa1bebbc1u, 0xcaf09aaeu,
      0xa25ab213u, 0x48a6b46cu, 0x1b9d9bcbu, 0x092c5be6u, 0x546ca624u, 0x1bec45d5u, 0x87f47473u, 0x96f0992eu}},
    {"appendix A.1, test vector 5", // the nonce's last byte is 2
     {0},
     0,
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2},
     {0x374dc6c2u, 0x3736d58cu, 0xb904e24au, 0xcd3f93efu, 0x88228b1au, 0x96a4dfb3u, 0x5b76ab72u, 0xc727ee54u,
      0x0e0e978au, 0xf3145c95u, 0x1b748ea8u, 0xf786c297u, 0x99c28f5fu, 0x628314e8u, 0x398a19fau, 0x6ded1b53u}},
};
static uint32_t word_of(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

int main() {
  { // RFC 8439, 2.1.1
    uint32_t a = 0x11111111u, b = 0x01020304u, c = 0x9b8d6f43u, d = 0x01234567u;
    coin_chacha::quarter_round(a, b, c, d);
    expect(a == 0xea2a92f4u && b == 0xcb1cf8ceu && c == 0x4581472eu && d == 0x5881c4bbu,
           "the programs' quarter round gives the values of section 2.1.1");
    a = 0x11111111u, b = 0x01020304u, c = 0x9b8d6f43u, d = 0x01234567u;
    coins::detail::quarter_round(a, b, c, d);
    expect(a == 0xea2a92f4u && b == 0xcb1cf8ceu && c == 0x4581472eu && d == 0x5881c4bbu,
           "the second writing's quarter round gives the values of section 2.1.1");
  }
  for (const Rfc8439Vector &v : VECTORS) {
    uint32_t key[COINKEYW], out[COINOUTW];
    for (int i = 0; i < COINKEYW; i++) key[i] = word_of(v.key + 4 * i);
    const uint32_t n0 = word_of(v.nonce), n1 = word_of(v.nonce + 4), n2 = word_of(v.nonce + 8);

    chacha20_block(out, key, v.counter, n0, n1, n2);
    if (memcmp(out, v.state, sizeof out)) printf("at %s:\n", v.where);
    expect(!memcmp(out, v.state, sizeof out), "chacha20_block() of common/coin_chacha.h gives the RFC's state at the end");

    const coins::Key second_key = coins::key_from_bytes(v.key);
    const coins::Block block = coins::chacha_block(second_key, v.counter, n0, n1, n2);
    if (memcmp(block.data(), v.state, sizeof v.state)) printf("at %s:\n", v.where);
    expect(!memcmp(block.data(), v.state, sizeof v.state), "the second writing's chacha_block() gives the RFC's state at the end");

    // Sixteen blocks in one call, at consecutive counters (`x0`).
    // Only block AT, at v.counter, is checked.
    constexpr size_t WIDE = 16, AT = 5;
    coins::Word x0[WIDE], x3[WIDE], wide[16][WIDE];
    for (size_t i = 0; i < WIDE; i++) x0[i] = v.counter + (coins::Word)i - (coins::Word)AT, x3[i] = n2;
    coins::chacha_blocks<WIDE>(second_key, x0, n0, n1, x3, wide);
    bool same = true;
    for (int w = 0; w < 16; w++) same &= wide[w][AT] == v.state[w];
    if (!same) printf("at %s:\n", v.where);
    expect(same, "the second writing's chacha_blocks() gives the RFC's state at the end");
  }
  printf(failed ? "coins_rfc8439_test: %d checks FAILED\n" : "coins_rfc8439_test: all checks hold\n", failed);
  return failed ? 1 : 0;
}
