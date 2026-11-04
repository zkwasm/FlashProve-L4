// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The generator of a weight table's masks and salts: the ChaCha20 block function, and the type of
// its key, the mask seed (other files say a table's key, `hf_key32`; each table has its own key).
// The block function is that of RFC 8439, section 2.3. The RFC's split of the last four state words
// into a counter and a nonce is not used: the caller gives all four, i12 .. i15. These inputs, and
// what is taken from a block, are defined in prover/hash_commitment/hf_mask.cuh. Under a mask seed
// only its hf_mask8() and hf_salt5() and hf_pads.cuh's pad_block() run it. The programs' only other
// call, in a self-test (generator_block_thread() in prover/hash_commitment/hf_mask_own.cuh), is at
// the input of RFC 8439's known answer, under the RFC's key, test_key(0), written in the source.
// No proof is about this function and no verifier runs it. Soundness assumes nothing of it.
// Header only; plain C++ under g++. Host and GPU code share one source, except in hc_rotl32().
//
// The fast path on the GPU. Under nvcc every HD function is force-inlined, and chacha20_block() has
// no loop over rounds, so a kernel can keep the sixteen state words and out[16] in registers (the
// compiler's choice). A kernel takes a table's mask seed as "const __grid_constant__ hf_key32 key".
// Kernel arguments are in the constant bank, so the mask seed's words cost no register and no load,
// and with __grid_constant__ no thread makes a copy in its local memory, which no code could zero
// (NVIDIA, CUDA C++ Programming Guide). A kernel that takes a mask seed by value must have no stack
// frame and no other local memory. A test that is not part of this repository checks this in the
// built programs that it is given.
#pragma once
#include "common/field_host.h"         // HD
#include "hash_commitment/hc_format.h" // HC_P, HC_SEED_WORDS
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define HC_UNROLL _Pragma("unroll")
#else
#define HC_UNROLL
#endif

// The mask seed: eight 32-bit words in a type of their own (hf_seed.h derives `hf_key32` from it).
// chacha20_block() takes a "const hc_seed32 &", so a coin key (a key for private randomness) or any
// other array of words does not compile in the seed's place, and a function that takes words does
// not accept an hc_seed32. Code that names member `w` gives up this protection: outside
// chacha20_block(), `w` is only for filling, comparing and zeroing a seed. Not so for the
// commitment's seed, which cp_commit keeps in a derived type (`Seed` in hf_commit.cu). cp_commit
// writes that seed's `w` to the seed's file and passes it to the second implementation of the key
// derivation (derivation_check::the_seed_is()). Seed::bytes() copies `w` to plain bytes, which
// hf_commit_keys.h loads into a blk::CoinKey to derive coin keys and the tables' keys.
// A mask seed must never be written into a weight table or given to the hash function; no code here
// checks that.
struct hc_seed32 {
  uint32_t w[8];
};
static_assert(sizeof(hc_seed32) == 32 && sizeof(hc_seed32) == 4 * HC_SEED_WORDS &&
                  std::is_trivially_copyable<hc_seed32>::value,
              "the mask seed is its 32 bytes, and can be a kernel's argument");
static_assert(!std::is_convertible<const uint32_t *, hc_seed32>::value &&
                  !std::is_convertible<uint32_t (&)[8], hc_seed32>::value &&
                  !std::is_convertible<uint32_t (&)[16], hc_seed32>::value,
              "no array of words passes for the mask seed");
static_assert(!std::is_convertible<hc_seed32, const uint32_t *>::value &&
                  !std::is_convertible<const hc_seed32 *, const uint32_t *>::value,
              "the mask seed passes for no array of words");
// True if neither hc_seed32 nor CoinKey, nor a pointer to one, converts to the other, and their
// sizes differ. For a static_assert in a file that also sees a coin key's type, as in hf_keeper.cu.
template <class CoinKey> constexpr bool hc_seed_is_no() {
  return !std::is_convertible<hc_seed32, CoinKey>::value && !std::is_convertible<CoinKey, hc_seed32>::value &&
         !std::is_convertible<const hc_seed32 *, const CoinKey *>::value &&
         !std::is_convertible<const CoinKey *, const hc_seed32 *>::value && sizeof(CoinKey) != sizeof(hc_seed32);
}

// RFC 8439, 2.3: the 32 bytes are read as eight little-endian words.
HD void hc_seed_from_bytes(const uint8_t bytes[32], hc_seed32 &seed) {
  HC_UNROLL
  for (int i = 0; i < 8; ++i)
    seed.w[i] = uint32_t(bytes[4 * i]) | (uint32_t(bytes[4 * i + 1]) << 8) | (uint32_t(bytes[4 * i + 2]) << 16) |
                (uint32_t(bytes[4 * i + 3]) << 24);
}

// Rotate left by n, 0 < n < 32.
HD uint32_t hc_rotl32(uint32_t x, int n) {
#if defined(__CUDA_ARCH__)
  return __funnelshift_l(x, x, n);
#else
  return (x << n) | (x >> (32 - n));
#endif
}

// The ChaCha quarter round: replaces the four words by its four results. The arguments must be four
// different objects. The words are secret, so only additions, exclusive ors and rotations by fixed
// counts touch them. Each rotation is a call of hc_rotl32(), the only rotation in this file.
// Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version 20080118, public domain).
// RFC 8439, section 2.1, specifies the same function.
HD void hc_quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  a = a + b, d = hc_rotl32(d ^ a, 16);
  c = c + d, b = hc_rotl32(b ^ c, 12);
  a = a + b, d = hc_rotl32(d ^ a, 8);
  c = c + d, b = hc_rotl32(b ^ c, 7);
}

// One column round and one diagonal round (RFC 8439, 2.3). The order of the eight quarter rounds is that of the round
// loop in D. J. Bernstein's chacha-merged.c.
#define HC_DOUBLE_ROUND()                                                                                              \
  do {                                                                                                                 \
    hc_quarter_round(x0, x4, x8, x12);                                                                                 \
    hc_quarter_round(x1, x5, x9, x13);                                                                                 \
    hc_quarter_round(x2, x6, x10, x14);                                                                                \
    hc_quarter_round(x3, x7, x11, x15);                                                                                \
    hc_quarter_round(x0, x5, x10, x15);                                                                                \
    hc_quarter_round(x1, x6, x11, x12);                                                                                \
    hc_quarter_round(x2, x7, x8, x13);                                                                                 \
    hc_quarter_round(x3, x4, x9, x14);                                                                                 \
  } while (0)

// f(seed; i12, i13, i14, i15): the ChaCha20 block function, sixteen words out.
HD void chacha20_block(const hc_seed32 &seed, uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15,
                       uint32_t out[16]) {
  const uint32_t(&key)[8] = seed.w;
  const uint32_t c0 = 0x61707865u, c1 = 0x3320646eu, c2 = 0x79622d32u, c3 = 0x6b206574u;
  uint32_t x0 = c0, x1 = c1, x2 = c2, x3 = c3;
  uint32_t x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
  uint32_t x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7];
  uint32_t x12 = i12, x13 = i13, x14 = i14, x15 = i15;

  HC_DOUBLE_ROUND(); //  1
  HC_DOUBLE_ROUND(); //  2
  HC_DOUBLE_ROUND(); //  3
  HC_DOUBLE_ROUND(); //  4
  HC_DOUBLE_ROUND(); //  5
  HC_DOUBLE_ROUND(); //  6
  HC_DOUBLE_ROUND(); //  7
  HC_DOUBLE_ROUND(); //  8
  HC_DOUBLE_ROUND(); //  9
  HC_DOUBLE_ROUND(); // 10

  out[0] = x0 + c0, out[1] = x1 + c1, out[2] = x2 + c2, out[3] = x3 + c3;
  out[4] = x4 + key[0], out[5] = x5 + key[1], out[6] = x6 + key[2], out[7] = x7 + key[3];
  out[8] = x8 + key[4], out[9] = x9 + key[5], out[10] = x10 + key[6], out[11] = x11 + key[7];
  out[12] = x12 + i12, out[13] = x13 + i13, out[14] = x14 + i14, out[15] = x15 + i15;
}

#undef HC_DOUBLE_ROUND

// Value number sigma < 8 of a block, 64 bits: v_sigma = o_{2 sigma} + 2^32 o_{2 sigma + 1}.
HD uint64_t hc_block_value(const uint32_t o[16], int sigma) {
  return uint64_t(o[2 * sigma]) | (uint64_t(o[2 * sigma + 1]) << 32);
}

// v mod p, canonical (below p), for any 64-bit v; p is HC_P. field_from_u64() is defined in
// prover/field/field_names.h.
HD uint64_t hc_reduce(uint64_t v) {
  return field_from_u64(v);
}
HD uint64_t hc_reduce_for_tests(uint64_t v) { // same as `hc_reduce`; only tests call it
  return field_from_u64(v);
}
