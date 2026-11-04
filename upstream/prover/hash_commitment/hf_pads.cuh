// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The pads of a weight table's rows.
//
// A table has a record: for each row, s answers to an out-of-domain sample, published in the
// key file or a leaf file (docs/format/FORMAT.md, R4.8; docs/format/09-key-and-leaf-files.md,
// R13.8). An answer is a linear form of the row's weights, so each answer is masked by one uniform
// element of E, stored in the row's message as three elements of F: three pads. A row has 3 s pads,
// the last entries of its message (HfTableParams::pad_entry()).
//
// Four uses. The pads are encoded and hashed with their row, so they enter three sums: the table as
// cp_commit builds it, the folded row u of an opening, and the columns that an opening recomputes.
// The fourth use is the published answer. All pads are generated on the host. The host adds them to
// the answers and to u; two kernels add them to the table and to the columns, from a copy of the
// host's buffer. No kernel adds a pad to an answer, so a kernel launch that never runs cannot
// publish an unmasked answer.
//
// Every user generates its own pads, into its own buffer. If one buffer served all four and were
// lost (a stray write, an erasure that comes too early), table, u, columns and answers would agree
// on zeros, every check would pass, and the answers would be published unmasked. With separate
// buffers, one lost buffer makes two users disagree. Do not merge the four.
//
// The rule: docs/format/COIN-KEYS.md, "Below a table's commit-run key": the generator runs under
// the table's own making key (that page's commit-run key), on inputs whose fourth word is "PADS".
//
// The self-check [own-test-pads]. Every pad is compared with a second derivation, by other code,
// from the key's second copy (described in hf_mask_own.cuh), and every copy in GPU memory is read
// back and compared with the host's buffer. Each user does so twice: once its buffers are filled,
// and again after their last read, before it passes on anything computed from them. The rule's
// known answers are also checked, once per process, before the first pad is generated.
// Not compared: what the two kernels add. If a launch never runs in all parts of a table, or in an
// opening, the recomputed columns do not hash to the stored nodes [hc-own-node]. If it never runs
// in one part, that shows only when an opened column is in that part. No test skips a launch.
//
// Hidden values: the pads, in host and in GPU memory. The owner of a buffer erases it at every
// return and at every failed self-check of the pads; PartMaker also at a failed [own-test-masks].
// If the process ends elsewhere (a failed CUDA call, a signal), they stay in the dying process's
// memory. The host's buffers are pageable: they are not locked, and copies to and from the GPU pass
// through the driver's staging buffers.
#pragma once
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_own.cuh"
#include "hash_commitment/hf_mask_ref.h"
#include "hash_commitment/hf_pads_known.h"
#include "hash_commitment/hf_table_params.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hf {
static_assert(HF_PAD_WORD == 0x50414453u && HF_PAD_WORD != HF_MASK_WORD && HF_PAD_WORD != HF_SALT_WORD, "\"PADS\": the three families of inputs differ in the fourth word");
// All pads of one table: pad j of row r is host[r * P.pads() + j].
struct TablePads {
  std::vector<uint64_t> host;
  TablePads() = default;
  TablePads(const TablePads &) = delete;
  TablePads &operator=(const TablePads &) = delete;
  void wipe() {
    if (!host.empty()) explicit_bzero(host.data(), host.size() * 8);
    host.clear();
  }
  ~TablePads() { wipe(); }
};

// The four users described in this file's head, in the enum's order: table, answers, u, columns.
enum PadsUser { PADS_FOR_THE_MAKING, PADS_FOR_THE_ANSWERS, PADS_FOR_THE_FOLD, PADS_FOR_THE_COLUMNS };
#ifndef SERVING_BUILD
// Fault injection for tests. Null, or a function that says whether to zero this user's pads, before
// its first comparison or right after it. cp_commit's ROWS_TABLE_FAULT (hf_commit_table.cuh)
// sets the two pointers. A production build has neither.
static bool (*pads_fault)(PadsUser who, bool before_the_comparison) = nullptr;
// Likewise: whether to zero only this user's copy in GPU memory, after its first comparison.
static bool (*pads_fault_on_the_gpu)(PadsUser who) = nullptr;
#endif

// The pads 8 g .. 8 g + 7 of row r: the rule of hf_mask8() (hf_mask.cuh) with the fourth word
// "PADS". All eight values decide whether a block is good, even those that the caller drops.
static inline void pad_block(const hf_key32 &key, uint32_t r, uint32_t g, uint64_t limit, uint64_t m[8]) {
  const uint64_t bound = hf_mask_limit_in_use(limit);
  uint32_t o[16];
  for (uint32_t t = 0;; ++t) {
    chacha20_block(key, r, g, t, HF_PAD_WORD, o);
    bool good = true;
    for (int s = 0; s < 8; ++s) good &= hc_block_value(o, s) < bound;
    if (good) break;
    if (t == 0xFFFFFFFFu) hf_mask_no_good_block();
  }
  for (int s = 0; s < 8; ++s) m[s] = hc_reduce(hc_block_value(o, s));
  explicit_bzero(o, sizeof o);
}
// True if pad_block() gives the known answers of hf_pads_known.h, under the keys written there.
// Some of their blocks need a retry.
static inline bool pads_known_answers_hold() {
  for (const HfKnownPads &row : HF_KNOWN_PADS) {
    hf_key32 key{};
    for (int i = 0; i < 32; i++) {
      const char c[2] = {row.key[2 * i], row.key[2 * i + 1]};
      unsigned byte = 0;
      for (char x : c) byte = 16 * byte + (unsigned)(x <= '9' ? x - '0' : x - 'a' + 10);
      key.w[i / 4] |= (uint32_t)byte << (8 * (i % 4));
    }
    uint64_t m[8], differs = 0;
    pad_block(key, row.r, row.g, HF_MASK_LIMIT, m);
    for (int i = 0; i < 8; i++) differs |= m[i] ^ row.pads[i];
    if (differs) return false;
  }
  return true;
}
// Generates all pads of the table from its key, into pads.host.
// Without a key there are no pads: zeros in their place would publish the answers unmasked, so a
// table with no key ends the run, in every build.
static inline void draw_table_pads(const HfTableParams &P, const hf_key32 *key, TablePads &pads) {
  pads.wipe();
  if (!key) {
    fprintf(stderr, "internal error (%s): the weight table publishes answers for each of its rows, and the program holds no key from which to derive the random values that "
                    "mask them. Nothing was made or opened. This is a fault of the program, not of the input. Report this line\n", HF_OWN_TEST_PADS);
    name_the_end("own-test-pads");
    exit(4);
  }
  refuse_the_zero_key(key);
  const size_t count = (size_t)P.pads();
  const uint64_t limit = mask_limit(key);
  pads.host.assign(P.R * count, 0);
  uint64_t m[8];
  for (size_t r = 0; r < P.R; r++)
    for (size_t j = 0; j < count; j++) {
      if (j % 8 == 0) pad_block(*key, (uint32_t)r, (uint32_t)(j / 8), limit, m);
      pads.host[r * count + j] = m[j % 8];
    }
  explicit_bzero(m, sizeof m);
}
// [own-test-pads] on the host's buffer: compares every pad with the CPU reference version
// (hf_mask_ref.h) under the key's second copy and, in cp_commit, with prover/second_writing/ under
// the key that that library derives itself. Pads that are all 0 fail too. Returns OWN_GOOD or
// OWN_PADS; the caller erases, prints, and ends the run or withholds what it has computed.
static inline int pads_own_test(const HfTableParams &P, const hf_key32 *key, const uint8_t *key_again, const TablePads &pads) {
  const size_t count = (size_t)P.pads();
  uint8_t bytes[32];
  if (!key || pads.host.size() != P.R * count || !key_again_bytes(key_again, bytes)) return OWN_PADS;
#if defined(HC_MASK_LOW_LIMIT) && HC_MASK_LOW_LIMIT
  const uint64_t limit = hf_mask_limit_in_use(mask_limit(key)); // a test program with a lower threshold
#else
  const uint64_t limit = hf_mask_ref::FULL; // the reference version's own constant
#endif
  uint64_t differs = 0, any = 0;
  for (size_t r = 0; r < P.R; r++)
    for (size_t j = 0; j < count; j++) {
      differs |= pads.host[r * count + j] ^ hf_mask_ref::pad(bytes, (uint32_t)r, j, limit);
      any |= pads.host[r * count + j];
    }
  explicit_bzero(bytes, sizeof bytes);
  if (differs || !any) return OWN_PADS;
  bool second_differs = false;
  HELD_TO_THE_DERIVATION_CHECK(table_pads(key_again, pads.host.data(), (uint32_t)P.R, (unsigned)count, limit, &second_differs))
  return second_differs ? OWN_PADS : OWN_GOOD;
}
// Every user calls this first: it generates the pads and compares them. After its last read of the
// pads the user calls pads_own_test() again, and pads_card_test() for a copy in GPU memory, and
// then erases them. Returns OWN_GOOD or OWN_PADS.
static inline int take_table_pads(const HfTableParams &P, const hf_key32 *key, const uint8_t *key_again, PadsUser who, TablePads &pads) {
  static const bool known = pads_known_answers_hold(); // once per process, before the first pad
  if (!known) return OWN_PADS;
  draw_table_pads(P, key, pads);
#ifndef SERVING_BUILD
  if (pads_fault && pads_fault(who, true)) std::fill(pads.host.begin(), pads.host.end(), 0);
#endif
  const int end = pads_own_test(P, key, key_again, pads);
#ifndef SERVING_BUILD
  if (pads_fault && pads_fault(who, false)) std::fill(pads.host.begin(), pads.host.end(), 0);
#else
  (void)who;
#endif
  return end;
}

} // namespace hf

#ifdef __CUDACC__
#include "common/common.h" // CK
#include "common/local_memory.cuh"
namespace hf {
// The pads' part of the folded row u: out[j] = the sum over r < R of fold[r] pad[r][j], for j <
// P.pads(). It belongs at message entry P.pad_entry(j) of u. fold: host memory, R elements of E.
// The result is hidden: the caller erases it.
static inline std::vector<Ext> pads_fold(const TablePads &pads, const HfTableParams &P, const Ext *fold) {
  std::vector<Ext> out((size_t)P.pads(), Ext{});
  for (size_t r = 0; r < P.R; r++)
    for (size_t j = 0; j < out.size(); j++) out[j] = k3_add(out[j], k3_mul(fold[r], ext_base(pads.host[r * out.size() + j])));
  return out;
}
static inline size_t pads_card_bytes(const HfTableParams &P) { return (size_t)P.R * (size_t)P.pads() * 8; }
// Reads back `card`, a copy of the pads in GPU memory, and compares it with the host's buffer.
// Waits for the copy, then returns OWN_GOOD or OWN_PADS.
static inline int pads_card_test(const TablePads &pads, const HfTableParams &P, const uint64_t *card, cudaStream_t st) {
  const size_t bytes = pads_card_bytes(P);
  if (pads.host.size() * 8 != bytes) return OWN_PADS;
  std::vector<uint64_t> back(pads.host.size());
  CK(cudaMemcpyAsync(back.data(), card, bytes, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  uint64_t differs = 0;
  for (size_t i = 0; i < back.size(); i++) differs |= back[i] ^ pads.host[i];
  explicit_bzero(back.data(), bytes);
  return differs ? OWN_PADS : OWN_GOOD;
}
// Copies the pads to `card` (GPU memory, pads_card_bytes(P) bytes, which the caller must erase),
// reads them back and compares. Waits for both copies, then returns OWN_GOOD or OWN_PADS.
static inline int pads_to_card(const TablePads &pads, const HfTableParams &P, PadsUser who, uint64_t *card, cudaStream_t st) {
  if (pads.host.size() * 8 != pads_card_bytes(P)) return OWN_PADS;
  CK(cudaMemcpyAsync(card, pads.host.data(), pads_card_bytes(P), cudaMemcpyHostToDevice, st));
  const int end = pads_card_test(pads, P, card, st);
#ifndef SERVING_BUILD
  if (pads_fault_on_the_gpu && pads_fault_on_the_gpu(who)) CK(cudaMemsetAsync(card, 0, pads_card_bytes(P), st));
#else
  (void)who;
#endif
  return end;
}

// For the code that builds a table (hf_commit_table.cuh). V[t 2^lp + i'] holds the sum over x of
// high[x] M[r0 + t][x 2^lp + i'] without the pads. The kernel adds high[c >> lp] pad[r0 + t][j] at
// i' = c mod 2^lp, where c = first + j is the index of pad j's coefficient. One thread per row.
__global__ void k_pads_into_part(const uint64_t *__restrict__ pads, int count, uint32_t r0, int rows, uint64_t first, int lp,
                                 const uint64_t *__restrict__ high, uint64_t *__restrict__ V) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= rows) return;
  for (int j = 0; j < count; j++) {
    const uint64_t c = first + (uint64_t)j;
    uint64_t &sum = V[((size_t)t << lp) + (c & (((uint64_t)1 << lp) - 1))];
    sum = gf_add(sum, gf_mul(high[c >> lp], pads[(size_t)(r0 + t) * count + j]));
  }
}

// For the columns that an opening recomputes (hf_open_side.cuh). g[z k + r 2^lo + cl] holds the sum
// over y of hi_z(y) W[y][r 2^lo + cl]: the data's part of row r on run z, indexed by the message
// entry e = y 2^lo + cl. The kernel adds hi_z(e >> lo) pad[r][j] at cl = e mod 2^lo, for
// e = entry + j. hi_z(y) is the product of hv[z nb + b] over the set bits b of y, as in
// hf_passes.cuh. One thread per (z, r).
__global__ void k_pads_into_runs(const uint64_t *__restrict__ pads, int count, uint32_t R, uint64_t entry, int lo,
                                 const uint64_t *__restrict__ hv, int nb, int runs, uint64_t *__restrict__ g) {
  const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (x >= (size_t)runs * R) return;
  const size_t z = x / R, r = x % R;
  for (int j = 0; j < count; j++) {
    const uint64_t e = entry + (uint64_t)j, y = e >> lo;
    uint64_t weight = 1;
    for (int b = 0; b < nb; b++)
      if ((y >> b) & 1) weight = gf_mul(weight, hv[z * nb + b]);
    uint64_t &sum = g[((z * R + r) << lo) + (e & (((uint64_t)1 << lo) - 1))];
    sum = gf_add(sum, gf_mul(weight, pads[r * count + j]));
  }
}
// Exits unless both kernels keep pads in registers only: no code can erase a kernel's local memory
// (prover/common/local_memory.cuh). Call it before any pad is generated. It also loads the kernels'
// code, so cp_prove calls it at start-up, outside a batch (hf_side_warm_up() in hf_open_side.cuh).
static inline void pads_kernels_ready() {
  static const bool looked_at = (local_memory::refuse_local_memory(k_pads_into_part, "hf::k_pads_into_part"),
                                 local_memory::refuse_local_memory(k_pads_into_runs, "hf::k_pads_into_runs"), true);
  (void)looked_at;
}
// first: the index of pad 0's coefficient, where the code places message entry P.pad_entry(0).
static inline void pads_into_part(const uint64_t *pads_card, const HfTableParams &P, uint64_t first, uint32_t r0, int rows, int lp,
                                  const uint64_t *high, uint64_t *V, cudaStream_t st) {
  pads_kernels_ready();
  k_pads_into_part<<<(unsigned)((rows + 31) / 32), 32, 0, st>>>(pads_card, (int)P.pads(), r0, rows, first, lp, high, V);
  CK(cudaGetLastError());
}
// hv: GPU memory, [runs][nb], with 2^nb above every message entry's y. g: GPU memory, [runs][k],
// k = R 2^lcl.
static inline void pads_into_runs(const uint64_t *pads_card, const HfTableParams &P, const uint64_t *hv, int nb, int runs, uint64_t *g,
                                  cudaStream_t st) {
  const size_t threads = (size_t)runs * P.R;
  if (!threads) return;
  pads_kernels_ready();
  k_pads_into_runs<<<(unsigned)((threads + 255) / 256), 256, 0, st>>>(pads_card, (int)P.pads(), (uint32_t)P.R, P.pad_entry(0), P.lcl, hv, nb,
                                                                      runs, g);
  CK(cudaGetLastError());
}
} // namespace hf
#endif // __CUDACC__
