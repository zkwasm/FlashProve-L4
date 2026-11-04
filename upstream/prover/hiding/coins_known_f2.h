// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Known answers of the generator of the prover's coins (common/coin_chacha.h), for the test
// that cp_prove and cp_commit make of that generator when they start [own-test-coins]
// (hiding/coins_own.cuh), and for tests.
//
// tests/coins_known_maker.cpp has printed this file, and it is changed in no other way. From prover/:
//   g++ -std=c++17 -O2 -fno-ipa-modref -DBLK_FIELD=2 -o coins_known_maker tests/coins_known_maker.cpp
//   ./coins_known_maker > hiding/coins_known_f2.h
// tests/coins_known_test.sh makes the file again and compares. The maker computes the answers with a second
// implementation of the coins' rule, the header library in tests/coins_second_text/, but for the conventions that the
// maker's own comments name. That library and the maker include no file of the generator that is tested, so no routine
// of it has a part in a value of this file. The programs do compile copies of some of the library's headers, for
// comparisons at run time (second_writing/README.md, which also says how far the library is independent of them).
//
// The cases were chosen by writers who knew a list of faults that the authors had planted in copies of these sources. So
// that the test stops those faults is no blind trial of it.
//
// NOTHING HERE IS SECRET. Every key below follows from two seeds that are written here, so nothing is hidden under them.
//
// A DIGEST stands for a list of cells. A cell is a coin, a number below p = 2^64 - 2^32 + 1. The digest is the first 32 bytes of
// BLAKE2b-512 (RFC 7693; no key) of the list's bytes, eight bytes a cell, the lowest first. Word w of a digest is its
// bytes 4 w to 4 w + 3, the lowest first.
#pragma once
#include <cstdint>

constexpr int COINS_KNOWN_CHAINS = 2, COINS_KNOWN_CONTEXTS = 12, COINS_KNOWN_LONG_CONTEXTS = 4;
constexpr int COINS_KNOWN_FILL_GROUPS = 3, COINS_KNOWN_LONG_GROUPS = 5, COINS_KNOWN_DIGEST_WORDS = 8;

// FROM A SEED TO THE PROCESS KEYS. The programs derive theirs from other inputs (common/coin_keys.h).
// Chain 0 has digests of n = 8 words, chain 1 of n = 10, whatever the build's own number is. The inputs: seed (test seed
// number 2 or 3 of hash_commitment/hc_format.h), rho, and 20 public words. D and h_0 are words[0 .. n - 1], h_card is
// words[n .. 2 n - 1], and the start of a link is words[0 .. 2 n - 1]. The answers (only leaf's and key_word's routines exist):
//   Kbat          cp::hedged_key(): from the seed, rho and D
//   leaf          cp::draw_leaf(): the draw under Kbat among COINS_KNOWN_LIVE_LEAVES live leaves
//   Kproc         cp::batch_key(): a batch's process key, from Kbat and h_card
//   key_word      coin_key_word(): the word by which two holders of Kproc compare it
//   once          cp::once_key(): the one-time files' process key, from the seed, h_0 and COINS_KNOWN_VERSION
//   link          cp::link_key(): the process key of link COINS_KNOWN_LINK, with rho as the making's fresh bytes
//   leaf_masks    cp::leaf_mask_key(): the mask key of leaf COINS_KNOWN_LINK, with rho as the making's fresh bytes
//   anchor_masks  cp::seed_key() with ANCHOR_MASKS: the anchor's mask key
// A key's 32 bytes are its eight words, each with its lowest byte first.
constexpr uint32_t COINS_KNOWN_VERSION = 530690, COINS_KNOWN_LIVE_LEAVES = 1000, COINS_KNOWN_LINK = 3;
struct CoinsKnownChain {
  int n;
  uint8_t seed[32], rho[32];
  uint32_t words[20];
  uint8_t Kbat[32];
  uint32_t leaf;
  uint8_t Kproc[32];
  uint32_t key_word;
  uint8_t once[32], link[32], leaf_masks[32], anchor_masks[32];
};
constexpr CoinsKnownChain COINS_KNOWN_CHAIN[COINS_KNOWN_CHAINS] = {
    {8,
     /* seed         */ {0xc5, 0x8b, 0xe6, 0xa8, 0x17, 0xf8, 0xf3, 0x23, 0xc2, 0x37, 0x17, 0x43, 0xd6, 0x53, 0x42, 0xba,
                          0x8a, 0x01, 0xc1, 0x52, 0x3a, 0x3a, 0x36, 0x5c, 0x91, 0xb2, 0x9b, 0x59, 0x19, 0xa0, 0x0a, 0x9e},
     /* rho          */ {0xcc, 0x4d, 0x8b, 0xba, 0xc6, 0xf3, 0xf7, 0x00, 0x7a, 0xac, 0x7f, 0x31, 0x66, 0x44, 0x60, 0x93,
                          0x04, 0x16, 0xfe, 0x8b, 0xd0, 0xdf, 0x13, 0xb2, 0x61, 0xf9, 0x57, 0x71, 0x7d, 0x9e, 0x4c, 0xbb},
     /* words        */ {0x3a4c768fu, 0xb141a7d7u, 0x462b7d3fu, 0x6a281f62u, 0x882330d2u, 0xb0387542u, 0x2e88a5dau, 0xc86204ddu,
                          0x3c8bd4cfu, 0x26947f5fu, 0xab4f94fau, 0xcade6a7cu, 0x68097ea0u, 0xd8b5e524u, 0x11d42adeu, 0xef17485cu,
                          0x92ea8bd2u, 0x32f2bfeau, 0x968d91a3u, 0x5bda79ecu},
     /* Kbat         */ {0x5e, 0xe1, 0x39, 0x04, 0x04, 0xdc, 0xf5, 0xf5, 0x1a, 0xe3, 0x78, 0xdf, 0x5f, 0xc8, 0x3a, 0x2e,
                          0x31, 0x7b, 0xe6, 0x63, 0xd6, 0xf7, 0x16, 0xee, 0xf9, 0x8d, 0x9c, 0xd1, 0x6c, 0xc2, 0xe8, 0x3a},
     /* leaf         */ 656u,
     /* Kproc        */ {0x11, 0x1b, 0xf0, 0x61, 0x81, 0x3b, 0x5f, 0x94, 0x77, 0x81, 0xbc, 0xd5, 0x13, 0xc7, 0x5f, 0x12,
                          0x24, 0x1a, 0xd8, 0x8c, 0x32, 0x40, 0x76, 0x38, 0x05, 0xe0, 0xe4, 0xa7, 0xe6, 0x2c, 0xff, 0xc2},
     /* key_word     */ 0x3011ec47u,
     /* once         */ {0xaf, 0x5b, 0xfb, 0x6b, 0x8b, 0xe8, 0x06, 0x74, 0x71, 0x83, 0x4f, 0x9a, 0x52, 0xe1, 0x24, 0xbf,
                          0x9f, 0xf2, 0xfb, 0xfe, 0x8a, 0xfb, 0x49, 0x70, 0x44, 0x6e, 0x57, 0x6a, 0x4f, 0xc7, 0xcf, 0x64},
     /* link         */ {0x60, 0x34, 0xe2, 0x34, 0x20, 0x12, 0xe4, 0x96, 0x39, 0x35, 0x87, 0x71, 0x57, 0xb7, 0x92, 0xfb,
                          0xe3, 0x1c, 0xfd, 0x40, 0x38, 0x35, 0xa6, 0x0f, 0x5b, 0xdf, 0x47, 0x85, 0x74, 0xe8, 0xf6, 0xf2},
     /* leaf_masks   */ {0xd4, 0x6c, 0x49, 0x43, 0x3b, 0xb6, 0x21, 0x2b, 0xfe, 0x70, 0x5b, 0xd2, 0x8a, 0x5d, 0x8d, 0x63,
                          0xba, 0xa6, 0x44, 0x7f, 0xc2, 0x32, 0x43, 0x69, 0x50, 0x51, 0x20, 0xaf, 0x00, 0xb9, 0xe1, 0x0e},
     /* anchor_masks */ {0xc6, 0x9b, 0xd4, 0x28, 0xba, 0x89, 0x7c, 0x60, 0xdf, 0x81, 0x77, 0x6a, 0x50, 0x3f, 0x4b, 0x0c,
                          0x41, 0x6d, 0xcc, 0x75, 0x9f, 0xac, 0xa7, 0x8d, 0x24, 0x3f, 0xd9, 0x68, 0x45, 0x7c, 0x60, 0x63},
    },
    {10,
     /* seed         */ {0x1b, 0xde, 0x04, 0x24, 0xec, 0x14, 0xb5, 0x6d, 0x15, 0xb3, 0xf9, 0x96, 0xd6, 0xeb, 0x19, 0x52,
                          0xc6, 0xa6, 0x58, 0x42, 0x3a, 0xc9, 0x96, 0x6c, 0x04, 0x59, 0xe0, 0x07, 0x33, 0xa6, 0x1d, 0x05},
     /* rho          */ {0x06, 0xf9, 0x5f, 0x79, 0xb1, 0x17, 0x78, 0xb5, 0xe2, 0xef, 0x4c, 0x6a, 0x79, 0x18, 0x88, 0xc1,
                          0x88, 0xb9, 0x6d, 0xee, 0xaa, 0x51, 0x52, 0x25, 0x6b, 0x20, 0x79, 0x3e, 0x23, 0xb8, 0xd0, 0x70},
     /* words        */ {0x56ac86d8u, 0x21ba1f5du, 0xbf3674d4u, 0x86105e27u, 0x0b51f6fdu, 0x0ec460a2u, 0x7c6e2792u, 0xa863b890u,
                          0x8eb60f54u, 0xf07aed59u, 0xec7f4ab3u, 0x306f7653u, 0xa89896e1u, 0xa629d1a3u, 0x0500a0abu, 0x6756d828u,
                          0x327bf781u, 0x92fea58bu, 0xabaf0070u, 0xa4795b62u},
     /* Kbat         */ {0xa3, 0x9b, 0x55, 0x78, 0xd0, 0x5e, 0xdf, 0x71, 0x66, 0x02, 0x68, 0x58, 0xb2, 0x4c, 0xd9, 0xf6,
                          0x8c, 0xfe, 0xc8, 0xc2, 0xb3, 0x93, 0x54, 0x46, 0x83, 0xf6, 0x82, 0x33, 0x03, 0xf4, 0x35, 0x2e},
     /* leaf         */ 558u,
     /* Kproc        */ {0x9b, 0xe4, 0x5c, 0x67, 0x39, 0x69, 0x17, 0xbb, 0x45, 0x8f, 0x43, 0x4e, 0x7a, 0x86, 0x12, 0xa7,
                          0xfc, 0x42, 0xc9, 0x16, 0x8d, 0x49, 0x46, 0x4b, 0x6c, 0x2c, 0xf2, 0x18, 0x3e, 0xc7, 0xd2, 0xa2},
     /* key_word     */ 0x0a6ce046u,
     /* once         */ {0x0b, 0x93, 0x9d, 0x26, 0x32, 0xd7, 0x64, 0x4b, 0x0c, 0xd1, 0xcf, 0x1d, 0xd8, 0x76, 0x26, 0x59,
                          0x76, 0xa9, 0x4f, 0x57, 0x06, 0xb0, 0xf2, 0xba, 0xeb, 0xff, 0xe0, 0x93, 0xf0, 0xa1, 0x7d, 0x79},
     /* link         */ {0xf1, 0xd0, 0x71, 0xb4, 0xc8, 0xa6, 0x18, 0x09, 0x89, 0x45, 0xac, 0x32, 0x98, 0xff, 0xb9, 0x20,
                          0xf5, 0xe9, 0x96, 0x8f, 0x21, 0x32, 0xdb, 0x2c, 0xad, 0x4f, 0x58, 0x98, 0x48, 0x69, 0xb8, 0x25},
     /* leaf_masks   */ {0x2d, 0xa2, 0xec, 0xf4, 0x6a, 0xe3, 0xff, 0x7a, 0x13, 0x19, 0x76, 0x5e, 0x4c, 0xef, 0x01, 0x34,
                          0x94, 0xae, 0x6e, 0x38, 0x29, 0xcd, 0x07, 0xc6, 0x59, 0x52, 0x53, 0x70, 0xf5, 0x04, 0x9a, 0xcf},
     /* anchor_masks */ {0xca, 0x99, 0xef, 0x5e, 0x7b, 0x4d, 0x66, 0xb4, 0x19, 0xa2, 0x4d, 0x2b, 0x58, 0xda, 0xd4, 0x87,
                          0x2a, 0xf2, 0xf8, 0x52, 0xa4, 0x99, 0xdb, 0xd1, 0xcd, 0x65, 0x1c, 0x6a, 0x07, 0xea, 0x78, 0xfe},
    },
};

// THE BLINDING CONTEXTS.
// A context's process key is one of a chain's three, and its key K_ctx follows from that key and the context's counter,
// version word, object and member (coin_context_key(); the nonce is zero). But for context 3, the numbers are those that
// the second implementation gives to the context that is named above the row. Apart from the version word, only
// contexts 0 to 2 and, of the lists below, the slot pads match what cp_prove draws; outside the test at the start no
// program uses the other contexts' objects or the other groups' streams (hiding/coins_own.cuh).
// Word 0 of K_ctx is odd in one of the contexts 0, 1 and 2 and even in another, so that a fault which strikes under one
// value of that bit alone meets a context with it. The bit was chosen because the authors had planted such a fault in a
// copy of these sources; no other property of a key is covered. The maker prints no table otherwise.
enum CoinsKnownKey : int { COINS_KNOWN_KPROC = 0, COINS_KNOWN_ONCE = 1, COINS_KNOWN_LINK_KEY = 2 };
struct CoinsKnownContext {
  int chain;
  CoinsKnownKey key;
  uint64_t counter;
  uint32_t version, object, member;
  uint32_t K_ctx[8];
};
constexpr CoinsKnownContext COINS_KNOWN_CONTEXT[COINS_KNOWN_CONTEXTS] = {
    // 0: a request's proof, slot 0
    {0, COINS_KNOWN_KPROC, 0ull, 530690u, 1u, 0u,
     {0xa3fd6fe2u, 0xd5ccbacdu, 0x47ee8bc0u, 0x9b912227u, 0xa36f1da4u, 0x1e1bdccbu, 0x4e9e2c7cu, 0x2835a525u}},
    // 1: a request's proof, slot 23
    {0, COINS_KNOWN_KPROC, 23ull, 530690u, 1u, 23u,
     {0x16b9ef96u, 0xbd1bec63u, 0xfa8a8080u, 0x8a1f26fbu, 0xba99d92bu, 0xde6818f4u, 0x42591190u, 0xa70b435du}},
    // 2: a request's proof, slot 1
    {1, COINS_KNOWN_KPROC, 1ull, 530690u, 1u, 1u,
     {0x2180bd5bu, 0xdbac8327u, 0x74f3cc23u, 0x600bff5bu, 0xc776389fu, 0xc0489d6eu, 0x4ab83945u, 0x6b9ff037u}},
    // 3: made up: neither a program nor the second implementation has this context
    {1, COINS_KNOWN_KPROC, 21474836487ull, 530690u, 3u, 65537u,
     {0x87fb94b3u, 0xb8ad403au, 0xff6945d4u, 0xb4d362d0u, 0x1d7172d0u, 0x570e2194u, 0xf0ae6906u, 0xf5cd427bu}},
    // 4: the shared opening message of a batch with 64 leaf indices
    {0, COINS_KNOWN_KPROC, 65ull, 530690u, 6u, 0u,
     {0xb17bf24fu, 0x53ad5e89u, 0xaec20da3u, 0x254e8e31u, 0x04aef180u, 0x059d7189u, 0x843f9159u, 0x503ee1deu}},
    // 5: the first opening message of slot 23
    {0, COINS_KNOWN_KPROC, 24ull, 530690u, 7u, 23u,
     {0xe59ad047u, 0x306fa5adu, 0xf6a7817cu, 0xa4a07a07u, 0xeff61e5du, 0x374d2affu, 0x6239d2c0u, 0xbfca8181u}},
    // 6: the second opening message of slot 23, 64 leaf indices
    {0, COINS_KNOWN_KPROC, 89ull, 530690u, 8u, 23u,
     {0xb462a98bu, 0x8a7e4775u, 0x1b39f6f3u, 0x5f1c76abu, 0x80c7221au, 0xc238877au, 0xd184e26cu, 0xbb69fca4u}},
    // 7: the range file
    {0, COINS_KNOWN_ONCE, 1ull, 530691u, 9u, 0u,
     {0x0d003bf1u, 0x7e40ba49u, 0xd5c92840u, 0x8683faa6u, 0x74d7b68du, 0x80abf47fu, 0x22a5abf6u, 0x538c50b6u}},
    // 8: the one-time files' first message
    {0, COINS_KNOWN_ONCE, 2ull, 530691u, 23u, 0u,
     {0x82dda6a2u, 0x62218c7fu, 0xbd7c911du, 0x3ce5d2e8u, 0xb3ebcd71u, 0x3ccd9862u, 0x6e91abfdu, 0xe6021e33u}},
    // 9: the one-time files' shared message
    {0, COINS_KNOWN_ONCE, 3ull, 530691u, 22u, 0u,
     {0xfba7064fu, 0x8cf985bdu, 0x8d9ba716u, 0x2b768cd2u, 0x274d7e88u, 0x8d4015d2u, 0xc7a86f8eu, 0x1917a5c6u}},
    // 10: the one-time files' second message
    {0, COINS_KNOWN_ONCE, 4ull, 530691u, 24u, 0u,
     {0x22b460beu, 0xe3b33c48u, 0x9651002fu, 0x47b2c359u, 0xb0ff3d77u, 0xa7aebfebu, 0x9fce992eu, 0xd52dda09u}},
    // 11: a link
    {0, COINS_KNOWN_LINK_KEY, 1ull, 530690u, 38u, 0u,
     {0xf1545ddcu, 0x16a0e9b2u, 0xb0fc4293u, 0x27dddcb0u, 0x1185f1beu, 0x8bc291ccu, 0xa6017db6u, 0x6e3ea55eu}},
};

// THE STREAMS THAT ARE NUMBERED FROM A BASE: four cells a call (hz::coin4(), hiding/trees.cuh).
// COINS_KNOWN_FILL[context][group] is the digest of the group's list under that context's key. A list holds its cells in
// the order in which they are named here.
//   group 0, slot pads: for gid = 0 to 59, the cells 0 to 3 ns - 1 of stream 4 (4096 + 64 gid) + 1, with
//     ns = COINS_KNOWN_SLOTS[context]. They are the pads of a sub-proof with ns slots: coordinate kappa < 3 of slot y
//     is cell ns kappa + y. Under context 0, ns is above 6584, the largest number of slots in the reference parameter
//     set of a specification that is not in this repository; no ns has been compared with the number of slots at an
//     example configuration (hiding/coins_own.cuh). Every ns is 3 mod 4, so that coordinates 1 and 2 begin at a call's
//     fourth and third cell.
//   group 1, many streams: the first COINS_KNOWN_STARTS[context] cells of the streams 16392, 16393 and 16394 (the
//     crossing-pad commitment's), then of the 96 streams from 31744 on (the opening's).
//   group 2, long runs: the cells 16384 to 32767 of stream 16393, then the 8190 cells from 2^35 + 3 on of stream 31834,
//     which begin at a call's fourth cell and so lie in one call more than 8190 / 4, rounded up.
// No cell of a context is in two of its lists, that of the single draw below among them: a program may draw them all in
// one context, whose ledger lets no cell be drawn twice.
constexpr uint32_t COINS_KNOWN_SLOTS[COINS_KNOWN_CONTEXTS] = {6603, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
constexpr uint32_t COINS_KNOWN_STARTS[COINS_KNOWN_CONTEXTS] = {256, 256, 256, 256, 16384, 256, 256, 256, 256, 256, 256, 256};
constexpr uint32_t COINS_KNOWN_FILL[COINS_KNOWN_CONTEXTS][COINS_KNOWN_FILL_GROUPS][COINS_KNOWN_DIGEST_WORDS] = {
    {
        {0x496c724fu, 0x953cc4f9u, 0x2bc59e26u, 0x49200bd1u, 0xa0b25a44u, 0x1232e2f0u, 0x64907d68u, 0x7066c82du},
        {0xecf8cd83u, 0x6c9edb2du, 0xcb2a5d0bu, 0x503ab88au, 0x121df4c9u, 0x53ace9bcu, 0x98adbdbfu, 0x15c46244u},
        {0x34213488u, 0xd75f63abu, 0x645a4a76u, 0x06ed595eu, 0x1ab41f3bu, 0x9a1e1f6bu, 0xbf125ad0u, 0x4a52e871u},
    },
    {
        {0xdecdee77u, 0x43fe40bbu, 0x45b72ec8u, 0xa0ba4e49u, 0x00af3cb3u, 0xf97be8b5u, 0x457b5b32u, 0x52d32df1u},
        {0x44b52af0u, 0xe3e9b9bau, 0x729a2199u, 0x1ffd331cu, 0xbb7dffb7u, 0x5c452ddeu, 0x8bef1051u, 0x8251188du},
        {0xc5724986u, 0xe0fb40d6u, 0x9c4c1fbeu, 0x94db77dau, 0xcfe33e47u, 0x81b731c9u, 0x1a1781a5u, 0x542e99f9u},
    },
    {
        {0x79548a01u, 0x86b29270u, 0x51c359a8u, 0x3ab70540u, 0x7ed4bf71u, 0xbe46abbeu, 0x62ae23cbu, 0x7ac2d51fu},
        {0xfbca6162u, 0x6e0baf43u, 0x152a396bu, 0xa5c416b4u, 0x94ca44a3u, 0xcfdf705cu, 0xe152d1b7u, 0x24f445f8u},
        {0x68b41f4eu, 0x6cee079eu, 0xb6efd301u, 0x3488f160u, 0x74b870e1u, 0x23cc3346u, 0x4cbd6d74u, 0x8a394ed1u},
    },
    {
        {0x0ba18a20u, 0x627ddfdeu, 0x6e754455u, 0x41d54f5au, 0x8798e858u, 0x18ce8339u, 0x0af406c3u, 0x76682487u},
        {0x4f0f57dbu, 0x543f38aau, 0x2836d8dbu, 0x24e08eddu, 0x8be7ef4cu, 0x56024ec8u, 0x458952a9u, 0xf5635d31u},
        {0x0c8d849du, 0x6632b224u, 0x406af546u, 0x9a460414u, 0xf14733e0u, 0x528d22a3u, 0x739594bfu, 0xdcb3efbdu},
    },
    {
        {0x23e54741u, 0x04aecb8du, 0xaa58ff56u, 0x73f6ea96u, 0x9cff0929u, 0x08742eebu, 0xbff6cf48u, 0xd4fe372bu},
        {0xdae33b5au, 0x2124c0beu, 0x61d8d40eu, 0x447e95b9u, 0x6f40d178u, 0x951ca476u, 0xe26fc415u, 0xa6bf8480u},
        {0xc03ad0f8u, 0xc1f4f152u, 0x7dff1fc4u, 0xbc76075cu, 0xf5f96edfu, 0xd184052bu, 0x5ae0f98fu, 0x3e97c798u},
    },
    {
        {0xfd3d12a1u, 0xce578395u, 0x37719f06u, 0xf3905bb1u, 0x19b74f56u, 0x6c626aa5u, 0xf42f2f81u, 0x69bf8daeu},
        {0xbffe0546u, 0xd10808f8u, 0x035e5ee3u, 0x884780c0u, 0xdb79c400u, 0xb3acdd39u, 0x95cf84c5u, 0x0384fdb7u},
        {0x0e95d750u, 0xe98e28bau, 0xd974972du, 0x22a815e3u, 0xfda5b43au, 0xf9509d6eu, 0x376dd33bu, 0x71feddd5u},
    },
    {
        {0x7b15c636u, 0xc42c160au, 0x20d3cca5u, 0x5002720fu, 0xec230b2au, 0xf04b6679u, 0xe53c3f87u, 0x3112fba5u},
        {0x6f693e4fu, 0xfae6508fu, 0x0fe006d2u, 0xd3be1ce7u, 0x26c4021cu, 0xfe7e189cu, 0xa4ddc18fu, 0x35f3430du},
        {0x3ec91f74u, 0xa391016fu, 0x488818c4u, 0x0966fc98u, 0xba66f536u, 0xc8651114u, 0x468619f7u, 0x9270aeb1u},
    },
    {
        {0xc51f348eu, 0x759ddd5bu, 0x6b4bd63au, 0x9631fd91u, 0xa0ba1edau, 0x84bf1f55u, 0x90ec69d8u, 0x6fb4de8bu},
        {0x57d65c3au, 0x7c426bc5u, 0x7560fe46u, 0xdf82de46u, 0xaf2d0558u, 0x7e923218u, 0x5a6b6413u, 0xc1824646u},
        {0x635356a9u, 0x47898804u, 0xc0916f14u, 0x1c67422au, 0xf3e1d682u, 0xa951e212u, 0x17ba9d82u, 0xabaacfe1u},
    },
    {
        {0x4944e5a7u, 0x19985322u, 0x21256388u, 0x46d752e0u, 0xa34fe63bu, 0xc2401fdau, 0xcd444903u, 0x02c0ce83u},
        {0xf7fcea38u, 0x01c5f30cu, 0x3a7ad057u, 0x50fbe947u, 0xd2691ec8u, 0x148bcba7u, 0xfc77b80cu, 0x5975ff06u},
        {0xf73ffb37u, 0xe443a34bu, 0xebdcdeb0u, 0x07822176u, 0x38ef770du, 0x6d851b0cu, 0x94112197u, 0xa7bb32fbu},
    },
    {
        {0xdf090e7au, 0x712d2394u, 0x3f9d3759u, 0xdf354a2bu, 0x209bcb62u, 0xc36ba580u, 0xe014f812u, 0xfce2402fu},
        {0x503978c8u, 0x0c7682d5u, 0x4a59aef1u, 0xe0a6a6d1u, 0x87389f0cu, 0x65c1aa26u, 0xd9d512a2u, 0xc3cd3506u},
        {0x1901f994u, 0x0241edd2u, 0x65d20e76u, 0x7e7df500u, 0xbdc81b2fu, 0xa5d97a86u, 0x35260133u, 0x2f19f0a1u},
    },
    {
        {0x26b01df3u, 0x7f58550eu, 0x66a01a31u, 0x73da5507u, 0x5f6ae7f9u, 0xab6d8cf4u, 0x3351284eu, 0x7e539fcau},
        {0x1746767du, 0x91878619u, 0x182438e1u, 0x573d7570u, 0x8bc1a006u, 0x082ab51fu, 0x14aca914u, 0xea7ed478u},
        {0x6804619fu, 0xea8c3dbdu, 0x0f7915b7u, 0x86b0a836u, 0x012215a3u, 0x5c0e03e1u, 0xf19fb451u, 0x019255cau},
    },
    {
        {0xc5c62131u, 0xc1f33e58u, 0x59977f3eu, 0xfe5cefd4u, 0x14364565u, 0x625ff8a3u, 0x31e29824u, 0xe8fc87eeu},
        {0x9ddc87f0u, 0x8d3a50ebu, 0x331f9578u, 0x3d2471a2u, 0x87d0c54bu, 0xe4a84689u, 0x2d6410b0u, 0x5e35f618u},
        {0x4afd73e4u, 0x8bfc879du, 0x075ebf41u, 0x1459ca4bu, 0xe4d68df8u, 0x0d9f2673u, 0x7c048f4cu, 0x47707c04u},
    },
};

// A single long draw, under context 6: COINS_KNOWN_WHOLE_STREAM is the digest of the COINS_KNOWN_WHOLE_STREAM_CELLS
// cells from cell COINS_KNOWN_STARTS[6] = 256 on of stream 31761. With the 256 cells of group 1 they are
// the 3112764 cells of the rows of coins of a request's second opening message in the reference parameter set of a
// specification that is not in this repository. That text does not print the number, and nothing here computes it. No
// proof uses this context's object (hiding/coins_own.cuh). In the same set some streams of the shared opening message
// are longer, the longest 27 times: none of them is drawn whole here.
constexpr uint64_t COINS_KNOWN_WHOLE_STREAM_CELLS = 3112508ull;
constexpr uint32_t COINS_KNOWN_WHOLE_STREAM[COINS_KNOWN_DIGEST_WORDS] = {
    0xaaee6d48u, 0xed09060du, 0x415ee612u, 0xe99a02ebu, 0x36a24d3bu, 0x9bc5d9a9u, 0x0fce4855u, 0x2d9a01e2u};

// THE STREAMS OF A REQUEST'S ROW COMMITMENTS.
// Stream (NUM, kind, row) has the number 2^30 + 2^16 NUM + 2^14 kind + row. A call gives five cells in a stream of kind
// 2 and four in every other. COINS_KNOWN_LONG[context][group] is the digest of the group's list under the key of that
// context, one of the first four, with T = COINS_KNOWN_TREES[context].
//   group 0, blinding coordinates: for NUM = 0 to T - 1 and row = 0, 9, 300: the cells 0 to 1023 of (NUM, 0, row).
//   group 1, rows of coins: for NUM = 0 to T - 1: the cells 0 to 1023 of (NUM, 1, 17).
//   group 2, long runs: the cells 0 to 16382 of (2, 1, 0), of (72, 1, 40), of (1, 3, 0) and of (1, 3, 3). Each ends
//     inside a call.
//   group 3, salts: for NUM = 0 to T - 1 and v = 0 to 63: the five cells from 5 I(v) on of (NUM, 2, 0), the salt of
//     position I(v). I(v) = v for v < 56, and I(56 + j) = 2^(20 + 3 j) + j for j < 8.
//   group 4, leaves: for NUM = 0 to T - 1 and I = 0 to 63: the leaf of column I of a tree with 32 rows and 2^6 columns
//     whose columns' hash chains have left zeros. It is the output of the hash's call C(sixteen zero
//     words, [the salt of position I of (NUM, 2, 0) | 32, 6 | NUM, I | zeros], c0 = 2, c1 = 0, l = 0,
//     num = 104 + 2^16 NUM, last = false), where a pair a, b is the 64-bit word a + 2^32 b. The list takes the output's
//     first eight 32-bit words, as four 64-bit words. These are no cells; they show the salts that the leaves took in.
constexpr uint32_t COINS_KNOWN_TREES[COINS_KNOWN_LONG_CONTEXTS] = {320, 8, 8, 8};
constexpr uint32_t COINS_KNOWN_LONG[COINS_KNOWN_LONG_CONTEXTS][COINS_KNOWN_LONG_GROUPS][COINS_KNOWN_DIGEST_WORDS] = {
    {
        {0x2d2306d4u, 0x7227e56bu, 0xb45d3083u, 0x1013bae5u, 0x489aaba0u, 0x83b2bdd0u, 0x20187735u, 0x6d069dc5u},
        {0x97f71085u, 0x1b6e6ac1u, 0xaf8f9f86u, 0xe3e01b4du, 0x97faff05u, 0x3978a852u, 0x5851a4eeu, 0x3ad3e60fu},
        {0x14a367ebu, 0xf9862403u, 0x3ac9ea9du, 0xc7e8c302u, 0x0fce4cfdu, 0x25e5b716u, 0xaf1a8646u, 0xb99d7eb2u},
        {0x8b77679cu, 0xf833fadeu, 0x84717b86u, 0x240810c7u, 0xb4a67c5eu, 0x9c164bffu, 0x447f8b12u, 0x6704a943u},
        {0x5bc4253cu, 0x77d72905u, 0x7ed23d2cu, 0xfe8732b3u, 0xfade9e90u, 0x437f895cu, 0xb8ef3d5cu, 0x1bd9d4d2u},
    },
    {
        {0xa2417630u, 0x14e2729du, 0x61b2e9f8u, 0x780f1a1du, 0xf37f6f71u, 0x7899581au, 0xa73f6db9u, 0x01f8c7cau},
        {0x7e06861du, 0x877e832eu, 0xb14dc10du, 0x303489eau, 0x6d7dfaf6u, 0xb2520c38u, 0x281bf0a6u, 0x47b9f13au},
        {0x0e2d97eeu, 0xacefd16bu, 0x528da0a2u, 0xddc85d6du, 0x3a9e8cd8u, 0x83f2ff0du, 0xd568ab50u, 0x8aedd778u},
        {0x19d4eb40u, 0xd745d53eu, 0xb07c717cu, 0x0c5f0479u, 0xf4022fd4u, 0xe01502d6u, 0xc78b64eau, 0xef32d038u},
        {0x9d3f55b4u, 0x26b9e094u, 0xd8824de6u, 0xb3d7b00eu, 0xe7daac5au, 0xc2e5987au, 0xe612ee76u, 0x960a1959u},
    },
    {
        {0x0012e67cu, 0x86afc3a0u, 0xec4052f5u, 0xeedc1f23u, 0x86af3072u, 0x5c5b823du, 0xdc9dcd48u, 0x6e7ac2cdu},
        {0xd40b472eu, 0x18a4ad6eu, 0x0c902a35u, 0x96ae7d5cu, 0x95f18f7du, 0xaf242a1eu, 0x4d6cd91bu, 0xbb32c9ceu},
        {0x29748ff7u, 0x6a426b94u, 0x94dad278u, 0xf0664365u, 0x29e475d6u, 0x3b3d9014u, 0x73b245e8u, 0x12695537u},
        {0xa69d5d96u, 0x8d3dd34du, 0x6f54ca45u, 0x403eaa49u, 0xc1cc96e8u, 0xd07cd9d4u, 0x6f0b7390u, 0xc45accffu},
        {0xedcf7dd2u, 0xeb0a7eceu, 0xf22cb407u, 0xc2a44587u, 0xb444bc08u, 0x78cd689cu, 0x8da58f94u, 0x2bbc0c81u},
    },
    {
        {0xbd158a0eu, 0x90a5c4f6u, 0xbc9082aau, 0xb94bbe26u, 0x7aa8bd75u, 0x3c33f726u, 0x71fb0f48u, 0x4ff29fd7u},
        {0x861290dau, 0x5752c405u, 0xa4d4d9bau, 0x782b504du, 0xce9b8d85u, 0xd6fa1af3u, 0x63a362f5u, 0x1ce3a9f3u},
        {0x89b1d25cu, 0xfb555973u, 0x748a7d5eu, 0xc5c9115du, 0x7ecf2666u, 0x0b69bf51u, 0xff6ad231u, 0xd3bec78cu},
        {0xbf4a9cf3u, 0xc9ef6e5au, 0x4abf7c97u, 0x87d6f51au, 0x8cac48f6u, 0xe6896643u, 0x510f086du, 0x53905771u},
        {0xdd3cac5fu, 0x94ffaa8bu, 0x39ed0495u, 0xef12645bu, 0xe5e02066u, 0x71e86e57u, 0xddd00bf6u, 0x99a92805u},
    },
};

// A whole row of coins of 2^22 cells, as many as a long row has coordinates (K = 2^22) in the reference parameter set
// of a specification that is not in this repository: the digest of the cells 0 to COINS_KNOWN_WHOLE_ROW_CELLS - 1 of
// (72, 1, 40) under the key of context 0. Only a test reads it (tests/coins_known_host_test.cpp).
constexpr uint64_t COINS_KNOWN_WHOLE_ROW_CELLS = 4194304ull;
constexpr uint32_t COINS_KNOWN_WHOLE_ROW[COINS_KNOWN_DIGEST_WORDS] = {
    0x93647d9bu, 0xa75dfa4eu, 0x5b2d3d4du, 0xdc8fa1cdu, 0x0c63fe92u, 0xa8debabbu, 0xaef50c5cu, 0x378867b0u};
