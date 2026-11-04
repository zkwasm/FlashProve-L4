// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The masks and salts of a weight table (all the weights as a matrix of long rows), for host and
// GPU code. A row's masks are the random entries that are encoded with it; each leaf of the table's
// Merkle tree has a salt. Both come from the table's making key (`hf_key32`,
// prover/hash_commitment/hf_seed.h: 32 bytes derived from the commitment's seed) through the
// generator f, the ChaCha20 block function (prover/hash_commitment/hc_chacha.cuh). The four inputs
// after the key are f's last four state words; o_0 .. o_15 are its output words.
//   blk(r, g, try) = f(key; r, g, try, "MASK"),   row r, g < ceil(nm / 8), try = 0, 1, ..
//   v_sigma        = o_(2 sigma) + 2^32 o_(2 sigma + 1), sigma < 8
//   A block is good if all eight v_sigma are below the threshold (`limit` in identifiers), the
//   greatest multiple of the field's modulus p = 2^64 - 2^32 + 1 that fits into 64 bits, which is
//   p itself. try* is the least try whose block is good.
//   mask[r][8 g + sigma] = v_sigma mod p, from blk(r, g, try*)
//   salt symbol sigma < 5 (a field element) of leaf I (of column I) = v_sigma mod p, from the block
//   f(key; I mod 2^32, floor(I / 2^32), 0, "SALT"): the salts have no threshold and no retry.
// A row has nm masks. nm is odd, so the last block always has values with 8 g + sigma >= nm; they
// are dropped. mask[r][i] is coefficient Code::tail_at() + i of row r (blocks/code/code.h).
// Both rules are specified in docs/format/COIN-KEYS.md, "2a. From a key to a random value".
// A test that is not part of this repository compares these functions with a second
// implementation, prover/hash_commitment/hf_mask_ref.h. Compiled by g++, this header has no kernel.
//
// Below the threshold every residue has exactly one pre-image, so uniform blocks give exactly
// uniform masks. A block needs a retry with probability 8 * 2^-32, which a table of the full size
// (2^33.4 blocks) is expected to meet about twenty times. A salt symbol is not exactly uniform
// (2^32 - 1 residues have two pre-images); it has to be hard to guess, no more.
// Under a table's making key, f runs, and must run, only on these two families of inputs and the
// pads' (prover/hash_commitment/hf_pads.cuh); the three differ in the fourth word. Every table has
// a key: draw_table_pads() (hf_pads.cuh) ends the run without one. Which blocks need a retry
// depends on the key, so no retry is printed, written or counted. Masks are hidden values, like the
// weights: the owner of a buffer that these functions fill must erase it on every path. A leaf's
// salt is hidden until its column is opened. Outside such buffers the key, masks and salts exist
// here only in local variables and kernel arguments, not erased.
//
// The threshold is compiled in (HF_MASK_LIMIT): no option, file or variable changes it, and
// hf_mask8() ignores its argument `limit`. Only a test program compiled with HC_MASK_LOW_LIMIT=1,
// which a production build (`SERVING_BUILD`) rejects, uses `limit`. There, to make retries common,
// hf::mask_limit() is k eighths of the threshold if the environment has H_HC_MASK_LIMIT=k, 1 to 7.
// The committer program cp_commit (cp_commit_main() in prover/hash_commitment/hf_commit.cu) accepts
// test variables only in a run on synthetic weights under a test seed.
#pragma once
#include "common/end_name.h"
#include "common/setting.h" // TESTENV, SALTW
#include "hash_commitment/hc_chacha.cuh"
#include "hash_commitment/hf_format.h" // HF_P, HF_MASK_LIMIT, HF_MASK_WORD, HF_SALT_WORD, HF_SALT_SYMBOLS
#include "hash_commitment/hf_seed.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef HC_MASK_LOW_LIMIT
#define HC_MASK_LOW_LIMIT 0
#endif
#if defined(SERVING_BUILD) && HC_MASK_LOW_LIMIT
#error "SERVING_BUILD cannot be combined with HC_MASK_LOW_LIMIT, which builds test programs: the threshold is compiled in"
#endif

constexpr int HF_MASKS_A_BLOCK = HF_BLOCK_MASKS; // one block of the generator is eight masks of one row
static_assert(HF_P == HC_P && HF_MASK_LIMIT == HC_MASK_LIMIT && HF_MASK_LIMIT % HF_P == 0 && 0 - HF_MASK_LIMIT < HF_P,
              "the threshold is the greatest multiple of p that fits into 64 bits");
constexpr uint64_t HF_MASK_LIMIT_EIGHTH = HF_MASK_LIMIT / 8; // what a test's lower thresholds are counted in
static_assert(HF_MASK_WORD == 0x4D41534Bu && HF_SALT_WORD == 0x53414C54u && HF_MASK_WORD != HF_SALT_WORD,
              "\"MASK\" and \"SALT\": the two families of inputs differ in the fourth word");
static_assert(HF_MASKS_A_BLOCK == 8 && HF_SALT_SYMBOLS == 5 && 2 * HF_SALT_SYMBOLS == SALTW && HF_SEED_WORDS == HC_SEED_WORDS,
              "a block is eight masks; a leaf's salt is five symbols, ten words; the key is eight words");

// The threshold that hf_mask8() uses when its caller passes `limit`.
HD constexpr uint64_t hf_mask_limit_in_use(uint64_t limit) {
#if HC_MASK_LOW_LIMIT
  return limit;
#else
  return (void)limit, HF_MASK_LIMIT;
#endif
}
// Ends the run when the block of try number 2^32 - 1 is not good either. On the GPU the kernel
// traps, and the host program ends at the first check, such as CK(), that gets the CUDA error.
HD void hf_mask_no_good_block() {
#ifdef __CUDA_ARCH__
  __trap();
#else
  fprintf(stderr, "%s: no block of the mask generator is good at any of the 2^32 try numbers; no table is made\n",
          HF_STOP_MASK_TRIES);
  exit(4);
#endif
}
// m[sigma] = mask[r][8 g + sigma], sigma < 8, canonical (below p). The caller drops the values with
// 8 g + sigma >= nm. first: the try number to start at, 0 except in a test. Returns whether a retry
// was needed, for tests and a self-test in hf_mask_own.cuh only. Does not check for the zero key.
// Source for the rejection rule: Lemire, "Fast Random Integer Generation in an Interval", arXiv:1805.10941, section
// 3.1 (last paragraph) and section 3.2. Here a whole block of eight values is rejected if one of them is.
HD bool hf_mask8(const hf_key32 &key, uint32_t r, uint32_t g, uint64_t limit, uint64_t m[8], uint32_t first = 0) {
  const uint64_t bound = hf_mask_limit_in_use(limit);
  uint32_t o[16], t = first;
  for (;; ++t) { // a loop, so that the twenty rounds are compiled only once
    chacha20_block(key, r, g, t, HF_MASK_WORD, o);
    bool good = true;
    HC_UNROLL
    for (int s = 0; s < 8; ++s) good &= hc_block_value(o, s) < bound;
    if (good) break;
    if (t == 0xFFFFFFFFu) hf_mask_no_good_block();
  }
  HC_UNROLL
  for (int s = 0; s < 8; ++s) m[s] = hc_reduce(hc_block_value(o, s));
  return t != first;
}
// salt[sigma], sigma < 5: the salt symbols of leaf I, canonical. Does not check for the zero key.
HD void hf_salt5(const hf_key32 &key, uint64_t I, uint64_t salt[HF_SALT_SYMBOLS]) {
  uint32_t o[16];
  chacha20_block(key, (uint32_t)I, (uint32_t)(I >> 32), 0, HF_SALT_WORD, o);
  HC_UNROLL
  for (int s = 0; s < HF_SALT_SYMBOLS; ++s) salt[s] = hc_reduce(hc_block_value(o, s));
}

namespace hf {
// The number of the generator's blocks for a row's nm masks.
static inline constexpr uint64_t mask_blocks(uint64_t nm) {
  return (nm + HF_MASKS_A_BLOCK - 1) / HF_MASKS_A_BLOCK;
}
// The value to pass as `limit`: HF_MASK_LIMIT, unless a test lowers it (never for a null key).
#if HC_MASK_LOW_LIMIT
static inline uint64_t mask_limit(const hf_key32 *key) {
  const char *text = key ? TESTENV("H_HC_MASK_LIMIT") : nullptr;
  if (!text) return HF_MASK_LIMIT;
  if (text[0] < '1' || text[0] > '7' || text[1])
    fprintf(stderr, "usage: the test variable H_HC_MASK_LIMIT is not one digit 1 to 7 (the mask generator then takes a "
                    "block again if a value is that many eighths of the threshold or more)\n");
  else
    return (uint64_t)(text[0] - '0') * HF_MASK_LIMIT_EIGHTH;
  exit(2);
}
#elif defined(SERVING_BUILD)
static inline constexpr uint64_t mask_limit(const hf_key32 *) {
  return HF_MASK_LIMIT;
}
#else
// Every other build: the variable would have no effect, so the program exits if it is set.
static inline uint64_t mask_limit(const hf_key32 *) {
  if (!TESTENV("H_HC_MASK_LIMIT")) return HF_MASK_LIMIT;
  fprintf(stderr, "usage: the test variable H_HC_MASK_LIMIT is set, but this program has the mask generator's threshold "
                  "compiled in and takes no other: unset it, or use a test build (-DHC_MASK_LOW_LIMIT=1)\n");
  exit(2);
}
#endif

// Thread (x, r) of k_masks_of_rows: writes the eight masks of block g0 + x of row r0 + r at
// out + r * stride + 8 x; 0 where the mask's index is nm or more. Unchecked: r0 + r, g0 + x < 2^32.
HD void masks_thread(uint64_t *out, size_t stride, uint32_t r0, uint32_t g0, uint64_t nm, uint64_t limit,
                     const hf_key32 &key, uint64_t x, uint32_t r) {
  uint64_t m[HF_MASKS_A_BLOCK];
  hf_mask8(key, r0 + r, g0 + (uint32_t)x, limit, m);
  uint64_t *o = out + (size_t)r * stride + 8 * x;
  const uint64_t i = 8 * ((uint64_t)g0 + x);
  HC_UNROLL
  for (int sigma = 0; sigma < HF_MASKS_A_BLOCK; sigma++) o[sigma] = i + sigma < nm ? m[sigma] : 0;
}
// Exits if the key is missing or 32 zero bytes. Every host function below calls it first: the
// checks made when a seed is drawn or read from a file cannot catch a pointer to a key erased since
// then. Under the zero key, table and openings agree and every verifier accepts, but anyone can
// compute the masks.
static inline void refuse_the_zero_key(const hf_key32 *key) {
  if (key && !seed_is_zero(*key)) return;
  fprintf(stderr, "%s: no key was handed in, or one of 32 zero bytes: no mask and no salt is made under it\n", HF_STOP_KEY_TEST);
  name_the_end("own-test-zero-key");
  exit(4);
}
// A kernel in another file that calls hf_mask8() or hf_salt5() must get a table's key from here.
static inline hf_key32 key_for_a_launch(const hf_key32 *key) {
  refuse_the_zero_key(key);
  return *key;
}
// masks_of_rows() on the host; only tests call it. out[r * stride + j] = mask[r0 + r][8 g0 + j] for
// r < nr, j < 8 ng; 0 where 8 g0 + j >= nm.
static inline void masks_of_rows_on_the_host(const hf_key32 *key, uint64_t limit, uint32_t r0, uint32_t nr, uint32_t g0,
                                             uint64_t ng, uint64_t nm, uint64_t *out, size_t stride) {
  refuse_the_zero_key(key);
  for (uint32_t r = 0; r < nr; r++)
    for (uint64_t x = 0; x < ng; x++) masks_thread(out, stride, r0, g0, nm, limit, *key, x, r);
}
} // namespace hf

#ifdef __CUDACC__
#include "common/common.h" // CK
namespace hf {
// Fills out, in GPU memory, as masks_of_rows_on_the_host() does: thread (x, r) runs masks_thread().
// Launch: ceil(ng / 128) by nr blocks of 128 threads, nr <= MASKS_ROWS_MOST. The key is
// __grid_constant__ so that no thread makes a local copy, which no code can erase (hc_chacha.cuh).
__global__ void __launch_bounds__(128)
    k_masks_of_rows(uint64_t *__restrict__ out, size_t stride, uint32_t r0, uint32_t g0, uint64_t ng, uint64_t nm,
                    uint64_t limit, const __grid_constant__ hf_key32 key) {
  const uint64_t x = blockIdx.x * (uint64_t)128 + threadIdx.x;
  if (x < ng) masks_thread(out, stride, r0, g0, nm, limit, key, x, blockIdx.y);
}
constexpr uint32_t MASKS_ROWS_MOST = 65535; // the maximum of a grid's second dimension
// Queues k_masks_of_rows on stream st. key: on the host, read before
// return; it reaches the GPU only as a kernel argument.
// out: GPU memory; stride, in 64-bit words, >= 8 ng, unchecked.
static inline void masks_of_rows(const hf_key32 *key, uint64_t limit, uint32_t r0, uint32_t nr, uint32_t g0, uint64_t ng,
                                 uint64_t nm, uint64_t *out, size_t stride, cudaStream_t st) {
  refuse_the_zero_key(key);
  if (!nr || !ng) return;
  for (uint32_t r = 0; r < nr; r += MASKS_ROWS_MOST) {
    const uint32_t now = nr - r < MASKS_ROWS_MOST ? nr - r : MASKS_ROWS_MOST;
    k_masks_of_rows<<<dim3((unsigned)((ng + 127) / 128), now), 128, 0, st>>>(out + (size_t)r * stride, stride, r0 + r,
                                                                            g0, ng, nm, limit, *key);
    CK(cudaGetLastError());
  }
}
} // namespace hf
#endif // __CUDACC__
