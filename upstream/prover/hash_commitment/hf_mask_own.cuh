// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Run-time self-checks (`own_test` in identifiers) of a weight table's masks (random values encoded
// with each row) and salts, which hf_mask.cuh generates from the table's making key. If the
// committer program cp_commit and an opening's passes over the weights compute the same wrong masks
// (an all-zero key, an index used as the value, a buffer never filled, a key not passed on), the
// table, the folded row u and the opened columns agree and every verifier accepts, while nothing is
// hidden. So cp_commit and the prover program cp_prove compare with known answers and with
// hf_mask_ref.h, a second implementation, run under a second copy of the key:
//   [own-test-generator] generator_own_test(), before a key is read: hf_mask8() and hf_salt5(), on
//     the host and in a kernel, give the known answers of hf_mask_known.h, some at indices beyond
//     any toy size. It catches r (row) or g (block) ignored, swapped or narrowed, the high word of
//     I (tree leaf) lost, "SALT" for "MASK", truncated values. The ChaCha20 block function itself,
//     in both implementations, gives RFC 8439's known answer (section 2.3.2). Both follow
//     one public source (THIRD_PARTY.md, "ChaCha20"): comparing them misses a fault common to both.
//   [own-test-masks] masks_own_test(), cp_commit only: OWN_PICKS blocks of every row of a buffer.
//   [own-test-salt] salts_own_test(): every salt hashed into a recomputed tree leaf (hf_tree.cuh).
//   [own-test-hidden] own_hidden(): the tier word of the table's commitment record agrees with
//     whether the process has a key; cp_commit only. It does not show that the other checks ran.
//   [own-test-pads] pads_own_test() in hf_pads.cuh: every pad of a table, in each
//     of the four buffers that hf_pads.cuh names, against hf_mask_ref.h.
// The key's second copy (`key_again`). A table's key is derived from the commitment's seed and
// stored in no file. The caller keeps a second copy for [own-test-masks], [own-test-salt] and
// [own-test-pads]: an erased or never-filled working copy, compared with itself, would pass.
// cp_commit derives the second copy from the seed's file, read again (TableKey::derive() in
// hf_commit_keys.h): this also catches an erased seed in memory and a fault in one derivation, but
// no error in the function or in a leaf table's 32 fresh bytes, which both derivations share.
// cp_prove derives the key once, and Unit::leaf_begin() in hf_keeper.cu copies it twice: this
// catches an erased or overwritten copy, but no wrong derivation (that fails [hc-own-node], the
// comparison with stored tree nodes).
// The checks must not be removed. A null key_again fails them (OWN_KEY_FILE): a key without a
// second copy is a failure, never a skip. A failed check only returns an OwnEnd: the caller must
// report own_stop()'s name for it (in brackets above) and stop. A CUDA error exits through CK().
#pragma once
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_known.h"
#include "hash_commitment/hf_mask_ref.h"
#include "common/derivation_check_seam.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace hf {
enum OwnEnd : int {
  OWN_GOOD = 0,
  OWN_GENERATOR = 1,
  OWN_MASKS = 2,
  OWN_SALTS = 3,
  OWN_KEY_FILE = 4,
  OWN_NODE = 5,
  OWN_HIDDEN = 6,
  OWN_PADS = 7
};
static inline const char *own_stop(int end) {
  return end == OWN_GENERATOR  ? HF_OWN_TEST_GENERATOR
         : end == OWN_MASKS    ? HF_OWN_TEST_MASKS
         : end == OWN_SALTS    ? HF_OWN_TEST_SALT
         : end == OWN_KEY_FILE ? HF_OWN_TEST_MASKS // no second copy, or zeros in it
         : end == OWN_NODE     ? HF_OWN_TEST_NODE
         : end == OWN_HIDDEN   ? HF_OWN_TEST_HIDDEN
         : end == OWN_PADS     ? HF_OWN_TEST_PADS
                               : "";
}
// [own-test-hidden]: tier_hidden (the commitment record's tier word is HF_TIER_HIDDEN) must equal
// key_in_hand. Call it just before the first word of a message or file is released or written.
static inline int own_hidden(bool tier_hidden, bool key_in_hand) {
  return tier_hidden == key_in_hand ? OWN_GOOD : OWN_HIDDEN;
}
// Copies the key's second copy, 32 bytes at key_again, to `bytes`. Returns false, with `bytes`
// zero, if key_again is null or all zero. The caller must erase `bytes`.
static inline bool key_again_bytes(const uint8_t *key_again, uint8_t bytes[32]) {
  memset(bytes, 0, 32);
  if (!key_again) return false;
  uint8_t any = 0;
  for (int i = 0; i < 32; i++) any |= bytes[i] = key_again[i];
  return any != 0;
}
static inline hf_key32 test_key(int n) {
  hf_key32 key{};
  for (int i = 0; i < HF_SEED_WORDS; i++) key.w[i] = hc_test_seed_word(n, i);
  return key;
}
// Whether this build can compute a known-answer row with threshold digit k (HfKnownMasks::k): 8 is
// hf_mask8()'s full rejection threshold; only test builds with HC_MASK_LOW_LIMIT read a lower one.
static inline constexpr bool known_row_is_for_this_build(int k) {
  return HC_MASK_LOW_LIMIT || k == 8;
}
// Row x of [own-test-generator], nine 64-bit words at got + 9 * x: for x < mask_rows eight masks
// and whether the first try was rejected, otherwise five salt symbols and zeros.
HD void generator_thread(uint64_t *got, const HfKnownMasks *masks, int mask_rows, const HfKnownSalt *salts,
                         const hf_key32 *keys, int x) {
  uint64_t *o = got + 9 * (size_t)x;
  for (int i = 0; i < 9; i++) o[i] = 0;
  if (x < mask_rows)
    o[8] = hf_mask8(keys[x], masks[x].r, masks[x].g, (uint64_t)masks[x].k * HF_MASK_LIMIT_EIGHTH, o);
  else
    hf_salt5(keys[x], salts[x - mask_rows].I, o);
}
// RFC 8439, section 2.3.2: the block function's known answer. The key is the bytes 00, 01, .., 1f:
// test key 0. RFC8439_LAST: the block counter, 1, and the nonce 00:00:00:09:00:00:00:4a:00:00:00:00
// as three little-endian words. RFC8439_STATE: the "ChaCha state at the end", as the RFC prints it.
constexpr uint32_t RFC8439_LAST[4] = {1, 0x09000000u, 0x4a000000u, 0};
constexpr uint32_t RFC8439_STATE[16] = {0xe4e7f110u, 0x15593bd1u, 0x1fdd0f50u, 0xc47120a3u, 0xc7f4d1c7u, 0x0368c033u,
                                        0x9aaa2204u, 0x4e6cd4c3u, 0x466482d2u, 0x09aa9f07u, 0x05d7c214u, 0xa2028bd9u,
                                        0xd19c12b5u, 0xb94e16deu, 0xe883d0cbu, 0x4e3c50a2u};
// The last row of [own-test-generator]: the RFC's block. It is the programs' only run of
// chacha20_block() (hc_chacha.cuh) at an input that is no mask's, salt's or pad's; its key is
// public and hides nothing. The input repeats RFC8439_LAST, an array that GPU code cannot read.
HD void generator_block_thread(uint64_t *o, const hf_key32 &key) {
  uint32_t block[16];
  chacha20_block(key, 1u, 0x09000000u, 0x4a000000u, 0u, block);
  for (int i = 0; i < 8; i++) o[i] = hc_block_value(block, i);
  o[8] = 0;
}
constexpr int GENERATOR_ROWS = HF_KNOWN_MASKS_ROWS + HF_KNOWN_SALTS_ROWS; // then the RFC's block
static inline bool generator_rows_hold(const uint64_t *got) {
  bool good = true;
  for (int i = 0; i < 8; i++)
    good &= got[9 * GENERATOR_ROWS + i] == ((uint64_t)RFC8439_STATE[2 * i + 1] << 32 | RFC8439_STATE[2 * i]);
  for (int x = 0; x < HF_KNOWN_MASKS_ROWS; x++) {
    if (!known_row_is_for_this_build(HF_KNOWN_MASKS[x].k)) continue;
    for (int i = 0; i < 8; i++) good &= got[9 * x + i] == HF_KNOWN_MASKS[x].m[i];
    good &= (got[9 * x + 8] != 0) == (HF_KNOWN_MASKS[x].tries != 0);
  }
  for (int y = 0; y < HF_KNOWN_SALTS_ROWS; y++)
    for (int i = 0; i < 5; i++) good &= got[9 * (HF_KNOWN_MASKS_ROWS + y) + i] == HF_KNOWN_SALTS[y].s[i];
  return good;
}
static inline std::vector<hf_key32> generator_keys() {
  std::vector<hf_key32> keys;
  for (const HfKnownMasks &row : HF_KNOWN_MASKS) keys.push_back(test_key(row.n));
  for (const HfKnownSalt &row : HF_KNOWN_SALTS) keys.push_back(test_key(row.n));
  keys.push_back(test_key(0)); // the RFC's key
  return keys;
}
// The host half of [own-test-generator]. It also checks that hf_mask_ref.h gives the same rows.
static inline int generator_own_test_on_the_host() {
  const int rows = HF_KNOWN_MASKS_ROWS + HF_KNOWN_SALTS_ROWS;
  const std::vector<hf_key32> keys = generator_keys();
  std::vector<uint64_t> got(9 * ((size_t)rows + 1));
  for (int x = 0; x < rows; x++)
    if (x >= HF_KNOWN_MASKS_ROWS || known_row_is_for_this_build(HF_KNOWN_MASKS[x].k))
      generator_thread(got.data(), HF_KNOWN_MASKS, HF_KNOWN_MASKS_ROWS, HF_KNOWN_SALTS, keys.data(), x);
  generator_block_thread(got.data() + 9 * (size_t)rows, keys[rows]);
  bool good = generator_rows_hold(got.data());
  { // the block function of hf_mask_ref.h
    uint8_t bytes[32];
    uint32_t block[16];
    for (int i = 0; i < 32; i++) bytes[i] = (uint8_t)i;
    hf_mask_ref::block(bytes, RFC8439_LAST, block);
    good &= !memcmp(block, RFC8439_STATE, sizeof block);
  }
  for (int x = 0; x < rows; x++) { // hf_mask_ref.h, all rows: its threshold is always an argument
    uint8_t bytes[32];
    for (int i = 0; i < 32; i++) bytes[i] = (uint8_t)(keys[x].w[i / 4] >> (8 * (i % 4)));
    if (x < HF_KNOWN_MASKS_ROWS) {
      const HfKnownMasks &row = HF_KNOWN_MASKS[x];
      for (int i = 0; i < 8; i++) {
        uint32_t tries;
        good &= hf_mask_ref::mask(bytes, row.r, 8 * (uint64_t)row.g + i, hf_mask_ref::limit(row.k), &tries) == row.m[i];
        good &= tries == row.tries;
      }
    } else {
      const HfKnownSalt &row = HF_KNOWN_SALTS[x - HF_KNOWN_MASKS_ROWS];
      for (int i = 0; i < 5; i++) good &= hf_mask_ref::salt(bytes, row.I, i) == row.s[i];
    }
  }
  return good ? OWN_GOOD : OWN_GENERATOR;
}

// [own-test-salt]. `salts`: host memory, the five 64-bit words hashed into each of `count` tree
// leaves. position(j): leaf j's index. In cp_commit the macro adds a check that exits if it fails.
template <class Position>
static inline int salts_own_test(const uint8_t *key_again, const uint64_t *salts, size_t count, Position position) {
  HELD_TO_THE_DERIVATION_CHECK(table_salts(key_again, salts, count, position))
  uint8_t bytes[32];
  if (!key_again_bytes(key_again, bytes)) return OWN_KEY_FILE;
  uint64_t differs = 0;
  for (size_t j = 0; j < count; j++)
    for (int sigma = 0; sigma < HF_SALT_SYMBOLS; sigma++)
      differs |= salts[HF_SALT_SYMBOLS * j + sigma] ^ hf_mask_ref::salt(bytes, position(j), sigma);
  explicit_bzero(bytes, sizeof bytes);
  return differs ? OWN_SALTS : OWN_GOOD;
}

// Of a row's ng blocks, [own-test-masks] checks only the first two, the last two and OWN_SPREAD
// more, evenly spread; some may coincide. A block is eight consecutive masks. p < OWN_PICKS.
constexpr int OWN_SPREAD = 60, OWN_PICKS = OWN_SPREAD + 4;
HD uint64_t own_pick(uint64_t ng, int p) {
  const uint64_t x = p == 0 ? 0 : p == 1 ? 1 : p == 2 ? ng - 1 : p == 3 ? ng - 2 : (uint64_t)(p - 3) * ng / (OWN_SPREAD + 1);
  return x < ng ? x : 0; // ng = 1: ng - 2 wraps
}
// The host part of masks_own_test(), with its arguments: picked[(r * OWN_PICKS + p) * 8 + sigma] is
// mask sigma of block own_pick(ng, p) in the buffer's row r. Masks with index nm or more must be 0.
static inline int masks_own_compare(const uint8_t *key_again, uint64_t limit, uint32_t r0, uint32_t nr, uint32_t g0,
                                    uint64_t ng, uint64_t nm, const uint64_t *picked) {
  uint8_t bytes[32];
  if (!key_again_bytes(key_again, bytes)) return OWN_KEY_FILE;
  uint64_t differs = 0;
  for (uint32_t r = 0; r < nr; r++)
    for (int p = 0; p < OWN_PICKS; p++)
      for (int sigma = 0; sigma < 8; sigma++) {
        const uint64_t i = 8 * (g0 + own_pick(ng, p)) + sigma;
        const uint64_t want = i < nm ? hf_mask_ref::mask(bytes, r0 + r, i, hf_mask_limit_in_use(limit)) : 0;
        differs |= picked[((size_t)r * OWN_PICKS + p) * 8 + sigma] ^ want;
      }
  explicit_bzero(bytes, sizeof bytes);
  return differs ? OWN_MASKS : OWN_GOOD;
}
} // namespace hf

#ifdef __CUDACC__
namespace hf {
__global__ void k_generator(uint64_t *got, const HfKnownMasks *masks, int mask_rows, const HfKnownSalt *salts,
                            const hf_key32 *keys, int rows) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < rows) generator_thread(got, masks, mask_rows, salts, keys, x);
  if (x == rows) generator_block_thread(got + 9 * (size_t)rows, keys[rows]);
}
// [own-test-generator], complete: the host half, then the same rows in a kernel. Call it at a
// program's start, before the process has a seed or a key. It allocates and frees a few kilobytes
// of GPU memory and waits for st. The only keys that it sends to the GPU are public test keys.
static inline int generator_own_test(cudaStream_t st) {
  if (generator_own_test_on_the_host() != OWN_GOOD) return OWN_GENERATOR;
  const int rows = HF_KNOWN_MASKS_ROWS + HF_KNOWN_SALTS_ROWS;
  const std::vector<hf_key32> keys = generator_keys();
  std::vector<HfKnownMasks> masks(HF_KNOWN_MASKS, HF_KNOWN_MASKS + HF_KNOWN_MASKS_ROWS);
#if !HC_MASK_LOW_LIMIT
  for (HfKnownMasks &row : masks) // a row this build cannot compute becomes row 0; not compared
    if (!known_row_is_for_this_build(row.k)) row = HF_KNOWN_MASKS[0];
#endif
  std::vector<hf_key32> sent = keys;
#if !HC_MASK_LOW_LIMIT
  for (int x = 0; x < HF_KNOWN_MASKS_ROWS; x++)
    if (!known_row_is_for_this_build(HF_KNOWN_MASKS[x].k)) sent[x] = keys[0];
#endif
  uint64_t *got;
  HfKnownMasks *m;
  HfKnownSalt *s;
  hf_key32 *k;
  CK(cudaMalloc(&got, 9 * 8 * (rows + 1)));
  CK(cudaMalloc(&m, sizeof HF_KNOWN_MASKS));
  CK(cudaMalloc(&s, sizeof HF_KNOWN_SALTS));
  CK(cudaMalloc(&k, sizeof(hf_key32) * (rows + 1)));
  CK(cudaMemcpyAsync(m, masks.data(), sizeof HF_KNOWN_MASKS, cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(s, HF_KNOWN_SALTS, sizeof HF_KNOWN_SALTS, cudaMemcpyHostToDevice, st));
  CK(cudaMemcpyAsync(k, sent.data(), sizeof(hf_key32) * (rows + 1), cudaMemcpyHostToDevice, st));
  CK(cudaMemsetAsync(got, 0xFF, 9 * 8 * (rows + 1), st)); // all ones is no row's answer: a lost launch is seen
  k_generator<<<(rows + 1 + 31) / 32, 32, 0, st>>>(got, m, HF_KNOWN_MASKS_ROWS, s, k, rows);
  CK(cudaGetLastError());
  std::vector<uint64_t> home(9 * ((size_t)rows + 1));
  CK(cudaMemcpyAsync(home.data(), got, home.size() * 8, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaFree(got));
  CK(cudaFree(m));
  CK(cudaFree(s));
  CK(cudaFree(k));
  return generator_rows_hold(home.data()) ? OWN_GOOD : OWN_GENERATOR;
}

__global__ void k_own_picks(uint64_t *picked, const uint64_t *__restrict__ buffer, size_t stride, uint64_t ng,
                            uint32_t nr) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nr * OWN_PICKS * 8) return;
  const uint32_t r = t / (OWN_PICKS * 8), p = t / 8 % OWN_PICKS, sigma = t % 8;
  picked[t] = buffer[(size_t)r * stride + 8 * own_pick(ng, (int)p) + sigma];
}
static inline size_t masks_own_room_bytes(uint32_t nr) {
  return (size_t)nr * OWN_PICKS * 64;
}
// [own-test-masks]: buffer[r * stride + j] (GPU) must be mask[r0 + r][8 g0 + j], r < nr, j < 8 ng,
// and 0 from index nm on, as masks_of_rows() writes it. OWN_PICKS blocks of every row are compared
// with hf_mask_ref.h under key_again. `room`: GPU, masks_own_room_bytes(nr) bytes; left zeroed.
// Host cost: nr * OWN_PICKS * 8 runs of hf_mask_ref::block(). Waits for st. Unchecked: nr < 2^23.
static inline int masks_own_test(const uint8_t *key_again, uint64_t limit, uint32_t r0, uint32_t nr, uint32_t g0,
                                 uint64_t ng, uint64_t nm, const uint64_t *buffer, size_t stride, uint64_t *room,
                                 cudaStream_t st) {
  if (!nr || !ng) return OWN_GOOD;
  const uint32_t threads = nr * OWN_PICKS * 8;
  k_own_picks<<<(threads + 127) / 128, 128, 0, st>>>(room, buffer, stride, ng, nr);
  CK(cudaGetLastError());
  std::vector<uint64_t> picked(threads);
  CK(cudaMemcpyAsync(picked.data(), room, (size_t)threads * 8, cudaMemcpyDeviceToHost, st));
  CK(cudaMemsetAsync(room, 0, (size_t)threads * 8, st));
  CK(cudaStreamSynchronize(st));
  const int end = masks_own_compare(key_again, limit, r0, nr, g0, ng, nm, picked.data());
  volatile uint64_t *wipe = picked.data();
  for (size_t w = 0; w < picked.size(); w++) wipe[w] = 0;
  return end;
}
} // namespace hf
#endif // __CUDACC__
