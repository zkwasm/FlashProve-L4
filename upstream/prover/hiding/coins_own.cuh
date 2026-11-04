// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// [own-test-coins]: a known-answer self-test of a generator of the prover's private random values
// ("coins" in identifiers). The programs cp_commit and cp_prove call coins_own_test() when they
// start, before any key is set. If an answer differs, the test prints nothing and does not exit:
// the caller does, under the name OWN_TEST_COINS. Include this file after prover/hiding/trees.cuh.
//
// Why. No verifier can tell whether a private random value has the value that the derivation rule
// gives it. If the generator returned zeros, every proof would still verify, what its values were
// to mask would be sent unmasked, and a check that derives a value twice with the same function
// would still pass. So each program first runs the generator under keys that everybody can read and
// compares the output with known answers. The generator is that of prover/common/coin_chacha.h and
// prover/hiding/trees.cuh. From it cp_commit draws nothing and cp_prove only the slot pads
// (Tree::slot_pads(): the one-time masks of a sub-proof's message slots; a sub-proof is the
// argument for one part of the forward pass).
//
// The answers are in prover/hiding/coins_known_f2.h, which prover/tests/coins_known_maker.cpp
// prints: for each test context the context's key, and for each group of streams one BLAKE2b-512
// digest over its values ("cells"). A value's label is its context ("attempt" in function names),
// its stream and its cell in the stream. The answers come from a second implementation of the rule
// (prover/tests/coins_second_text/ with prover/second_writing/), whose ChaCha20 block function is
// specified in RFC 8439 and which includes no file of the generator that is tested here. Of it the
// programs compile only prover/second_writing/, for their run-time comparisons. Both
// implementations follow the same public sources for the ChaCha20 block function and the BLAKE2b
// compression function (THIRD_PARTY.md, "ChaCha20", "BLAKE2b"). So the table checks the calls'
// order and inputs and how a block becomes cells; a fault that both take from those sources or a
// misreading of the rule that both share is not detected. Six answers come from neither and check
// the block function itself: the test vectors that RFC 8439 prints for it (COINS_RFC8439).
// Further limits: prover/second_writing/README.md, "How far it is independent of the programs".
//
// What runs. coins_own_test() calls the functions that a real draw calls. First coin_output(), from
// which every value comes (one call gives four cells), must give the RFC's six blocks, on the host
// and in a kernel, and key32_word_of() the table's check word (32 bits derived from a key) for each
// of its two batch keys (`Kproc`) [own-test-key]. Then each of the twelve test contexts is begun
// (key32_renew() with the test key; COINS_WORD, the label's version word, here one that no program
// sets; set_attempt() with the context's `counter`, `object` and `member`), and in it:
//   - coink() must return the table's key of the context;
//   - Tree::slot_pads() draws the slot pads of sixty sub-proofs, 127 slots each (context 0: 6,603);
//   - fill() draws the first 256 cells (context 4: 16,384) of 99 streams, two more runs and, in
//     context 6, a run of 3.1 million cells in one draw: 3,040 thread blocks, 24 MiB;
//   - rf::hf_cells_differ(), the host's version of the rule, must match a run's first 16,384 cells.
// Every draw goes through k_fill(), coin4() and enter_draw(), which records it. The sizes follow no
// configuration. Apart from the version word, only contexts 0 to 2 (object 1, a request's proof)
// and the slot pads match what cp_prove draws; no program uses the other contexts' objects or the
// other groups' streams.
//
// Not detected (docs/limits-all-rows.tsv, row E18.1.1, says more). An error that shows only in a
// longer draw or a larger launch than the test's; its sizes have not been compared with a
// sub-proof's number of slots at any example configuration.
// And, because the test runs once, before a key is read, under its own keys and stream numbers:
//   - a wrong or stale context key at a request (coin_context_key(), prover/common/coin_chacha.h);
//   - wrong numbers passed to set_attempt(), or wrong stream numbers that the sub-proofs compute;
//   - an error in the copies of the slot pads that are added to the values that they hide;
//   - an error that depends on the real key or arises after the start; a correct value left unused.
// The test does not reach coin_again(): a cell is drawn again with probability 2^-32, and no
// argument of coin4() forces it; prover/tests/coins_field_rule_test.cu calls it directly and finds
// rejected words in a stream. A kernel's inlined copy of a function is tested only if the kernel
// runs here: k_fill() and k_coins_rfc8439() do.
//
// State. While the test runs, the generator's state (static variables of prover/hiding/trees.cuh)
// contains a test key. Call coins_own_test() before the first key32_renew(), when no other thread
// touches that state; called while a key is in force, it exits the program. coins_own_left_clean()
// does not compare CTR and Replay's any, begun, ctr, last, drawn_under, obj and mem, which keep the
// last test context's values until the first real key32_renew() and set_attempt() overwrite them.
//
// Test keys in a production build (SERVING_BUILD). Such a build leaves out key_test(), which sets
// the unit tests' public key, so that no call leads to a key that everybody knows. This test is
// such a call, in every build. Instead: a test key is in the state only while a CoinsKnownAtHand
// exists; once it is erased (key_state == KEY_WIPED), coink() exits the program until a key is set;
// and both programs set each real key by key32_renew(), which erases what it finds (the check
// [own-test-key] then tests the copy just made). An exit during the test: see coins_own_test().
#pragma once
#include "common/blake2b_f.h"
#include "common/coin_chacha.h"
#include "hiding/coins_by_value.cuh"
#include "hiding/coins_known_f2.h"
#include <vector>

namespace hz {
constexpr const char *OWN_TEST_COINS = "own-test-coins";

// True if `want` is the first 32 bytes of BLAKE2b-512 (RFC 7693) of cells[0 .. n - 1], eight bytes
// a cell, low byte first. f2b::C() (prover/common/blake2b_f.h) gets 0 as the counter's high half.
static inline bool coins_known_digest_is(const u64 *cells, size_t n, const u32 want[COINS_KNOWN_DIGEST_WORDS]) {
  constexpr size_t CELLS_A_BLOCK = BLKW / 2;
  u32 cv[SW];
  f2b::first_state(cv);
  const size_t blocks = n ? (n + CELLS_A_BLOCK - 1) / CELLS_A_BLOCK : 1;
  for (size_t b = 0; b < blocks; b++) {
    u32 m[BLKW] = {0};
    for (size_t i = 0; i < CELLS_A_BLOCK && CELLS_A_BLOCK * b + i < n; i++) {
      m[2 * i] = (u32)cells[CELLS_A_BLOCK * b + i];
      m[2 * i + 1] = (u32)(cells[CELLS_A_BLOCK * b + i] >> 32);
    }
    const bool last = b + 1 == blocks;
    const u64 fed = last ? 8 * (u64)n : 4 * (u64)BLKW * (b + 1);
    f2b::C(cv, m, (u32)fed, (u32)(fed >> 32), 0, 0, last);
  }
  u32 differs = 0;
  for (int i = 0; i < COINS_KNOWN_DIGEST_WORDS; i++) differs |= cv[i] ^ want[i];
  return differs == 0;
}

// While an object of this type exists, the table's test context x is the current context of the
// generator's state. The destructor ends the context, erases the test key and restores COINS_WORD.
struct CoinsKnownAtHand {
  const u32 word_before = COINS_WORD;
  explicit CoinsKnownAtHand(int x) {
    if (key_state != KEY_NONE && key_state != KEY_WIPED) key_refuse("the coins' own test was called while a key is in force");
    const CoinsKnownContext &context = COINS_KNOWN_CONTEXT[x];
    const CoinsKnownChain &chain = COINS_KNOWN_CHAIN[context.chain];
    key32_renew(context.key == COINS_KNOWN_KPROC ? chain.Kproc : context.key == COINS_KNOWN_ONCE ? chain.once : chain.link);
    COINS_WORD = context.version;
    set_attempt(context.counter, context.object, context.member);
  }
  ~CoinsKnownAtHand() {
    end_attempt();
    key32_wipe();
    COINS_WORD = word_before;
  }
  CoinsKnownAtHand(const CoinsKnownAtHand &) = delete;
  CoinsKnownAtHand &operator=(const CoinsKnownAtHand &) = delete;
};
// Returns coink()'s key for the current test context x; *good = false if it is not the table's.
static inline CoinK coins_known_key(int x, bool *good) {
  CoinK c = coink(ATTEMPT_STREAMS_FROM); // a context has one key for all its streams
  u32 differs = 0;
  for (int k = 0; k < COINKEYW; k++) differs |= c.k[k] ^ COINS_KNOWN_CONTEXT[x].K_ctx[k];
  if (differs) *good = false;
  return c;
}

// True if the state is clean: the key erased (its check word is that of eight zero words), no
// context current or suspended (`parked`), no draw recorded, COINS_WORD = `word`, its old value.
static inline bool coins_own_left_clean(u32 word) {
  const uint8_t no_key[4 * COINKEYW] = {0};
  const Replay &r = replay_();
  return key_state == KEY_WIPED && key32_word() == key32_word_of(no_key) && !r.set && r.parked.empty() && r.drawn.empty() &&
         COINS_WORD == word;
}

// A run: the cells `first` to first + count - 1 of a stream.
struct CoinsKnownRun {
  u32 stream;
  u64 first;
  size_t count;
};
constexpr int COINS_OWN_WHOLE_STREAM = COINS_KNOWN_FILL_GROUPS, COINS_OWN_GROUPS = COINS_OWN_WHOLE_STREAM + 1;
// Returns the runs of a group under test context x, in the order in which the group's digest takes
// their cells. No cell is in two runs of a context. Group COINS_OWN_WHOLE_STREAM: context 6 only.
static inline std::vector<CoinsKnownRun> coins_known_runs(int x, int group) {
  constexpr u32 CP = 4u * 4096u + 8, OPENING = 4u * (4096u + 64u * 60u); // stream bases, groups 1-3
  const size_t slots = COINS_KNOWN_SLOTS[x], start = COINS_KNOWN_STARTS[x];
  std::vector<CoinsKnownRun> runs;
  if (group == 0) { // the slot pads of the sub-proofs gid = 0 to 59: component kappa of a sub-proof's pads is one run
    for (u32 gid = 0; gid < 60; gid++)
      for (u64 kappa = 0; kappa < 3; kappa++) runs.push_back({4u * (4096u + 64u * gid) + 1, kappa * slots, slots});
  } else if (group == 1) { // the first cells of streams CP to CP + 2 and OPENING to OPENING + 95
    for (u32 stream = CP; stream < CP + 3; stream++) runs.push_back({stream, 0, start});
    for (u32 stream = OPENING; stream < OPENING + 96; stream++) runs.push_back({stream, 0, start});
  } else if (group == 2) { // 16 thread blocks; then from the last cell of call 2^33: a ninth block
    runs.push_back({CP + 1, 16384, 16384});
    runs.push_back({OPENING + 90, ((u64)1 << 35) + 3, 8190});
  } else if (group == COINS_OWN_WHOLE_STREAM && x == 6) { // one long run, after group 1's cells
    runs.push_back({OPENING + 17, start, COINS_KNOWN_WHOLE_STREAM_CELLS});
  }
  return runs;
}
// In the programs a Tree's plane has more cells than slots; the test's planes have this many more.
constexpr size_t COINS_OWN_PLANE_GAP = 5;
// Draws the runs of a group into gpu, in the current context, each run `gap` cells after the end of
// the one before. Group 0: Tree::slot_pads(), gap = COINS_OWN_PLANE_GAP. Others: fill(), gap = 0.
static inline void coins_known_draw(int group, const std::vector<CoinsKnownRun> &runs, u64 *gpu, size_t gap) {
  static_assert(CE == 3, "the table's slot pads have three components");
  for (size_t r = 0; r < runs.size(); r++) {
    if (group != 0) {
      fill(gpu, runs[r].stream, runs[r].first, runs[r].count);
    } else if (r % CE == 0) { // a sub-proof's CE runs are one call, on a tree that has its planes here and nothing else
      Tree pads;
      pads.nm = OWN_TEST_COINS;
      pads.stream = runs[r].stream - 1;
      pads.sh.Kc = runs[r].count + gap;
      pads.ext = gpu;
      pads.slot_pads(runs[r].count);
    }
    gpu += runs[r].count + gap;
  }
}

// RFC 8439's test vectors for the ChaCha20 block function: section 2.3.2, and appendix A.1, test
// vectors 1 to 5. key: the RFC's 32 bytes as eight words, the lowest byte first (section 2.3).
// last: the state's last four words: the RFC's block counter and its nonce's twelve bytes as three
// words. state: the "ChaCha state at the end", as the RFC prints it. The numbers are those of
// RFC 8439; prover/tests/coins_rfc8439_test.cpp has the same vectors, with key and nonce as bytes.
struct CoinsRfc8439Vector {
  u32 key[COINKEYW], last[4], state[COINOUTW];
};
constexpr CoinsRfc8439Vector COINS_RFC8439[] = {
    // section 2.3.2
    {{0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu, 0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu},
     {1, 0x09000000u, 0x4a000000u, 0},
     {0xe4e7f110u, 0x15593bd1u, 0x1fdd0f50u, 0xc47120a3u, 0xc7f4d1c7u, 0x0368c033u, 0x9aaa2204u, 0x4e6cd4c3u,
      0x466482d2u, 0x09aa9f07u, 0x05d7c214u, 0xa2028bd9u, 0xd19c12b5u, 0xb94e16deu, 0xe883d0cbu, 0x4e3c50a2u}},
    // appendix A.1, test vector 1
    {{0},
     {0, 0, 0, 0},
     {0xade0b876u, 0x903df1a0u, 0xe56a5d40u, 0x28bd8653u, 0xb819d2bdu, 0x1aed8da0u, 0xccef36a8u, 0xc70d778bu,
      0x7c5941dau, 0x8d485751u, 0x3fe02477u, 0x374ad8b8u, 0xf4b8436au, 0x1ca11815u, 0x69b687c3u, 0x8665eeb2u}},
    // appendix A.1, test vector 2
    {{0},
     {1, 0, 0, 0},
     {0xbee7079fu, 0x7a385155u, 0x7c97ba98u, 0x0d082d73u, 0xa0290fcbu, 0x6965e348u, 0x3e53c612u, 0xed7aee32u,
      0x7621b729u, 0x434ee69cu, 0xb03371d5u, 0xd539d874u, 0x281fed31u, 0x45fb0a51u, 0x1f0ae1acu, 0x6f4d794bu}},
    // appendix A.1, test vector 3
    {{0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x01000000u},
     {1, 0, 0, 0},
     {0x2452eb3au, 0x9249f8ecu, 0x8d829d9bu, 0xddd4ceb1u, 0xe8252083u, 0x60818b01u, 0xf38422b8u, 0x5aaa49c9u,
      0xbb00ca8eu, 0xda3ba7b4u, 0xc4b592d1u, 0xfdf2732fu, 0x4436274eu, 0x2561b3c8u, 0xebdd4aa6u, 0xa0136c00u}},
    // appendix A.1, test vector 4
    {{0x0000ff00u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
     {2, 0, 0, 0},
     {0xfb4dd572u, 0x4bc42ef1u, 0xdf922636u, 0x327f1394u, 0xa78dea8fu, 0x5e269039u, 0xa1bebbc1u, 0xcaf09aaeu,
      0xa25ab213u, 0x48a6b46cu, 0x1b9d9bcbu, 0x092c5be6u, 0x546ca624u, 0x1bec45d5u, 0x87f47473u, 0x96f0992eu}},
    // appendix A.1, test vector 5
    {{0},
     {0, 0, 0, 0x02000000u},
     {0x374dc6c2u, 0x3736d58cu, 0xb904e24au, 0xcd3f93efu, 0x88228b1au, 0x96a4dfb3u, 0x5b76ab72u, 0xc727ee54u,
      0x0e0e978au, 0xf3145c95u, 0x1b748ea8u, 0xf786c297u, 0x99c28f5fu, 0x628314e8u, 0x398a19fau, 0x6ded1b53u}},
};
constexpr int COINS_RFC8439_VECTORS = sizeof COINS_RFC8439 / sizeof COINS_RFC8439[0];
// out = the block of vector v. coin_output() puts its arguments into the state's last four words in
// the order: low word of `call`, stream, retry, high word of `call`. So any block input can be set.
HD void coins_rfc8439_block(u32 out[COINOUTW], const CoinsRfc8439Vector &v) {
  coin_output(out, v.key, v.last[1], (u64)v.last[3] << 32 | v.last[0], v.last[2]);
}
// Thread x writes the block of vector x < n to out + COINOUTW x. Launch one block of >= n threads.
static __global__ void k_coins_rfc8439(const CoinsRfc8439Vector *v, int n, u32 *out) {
  const int x = threadIdx.x;
  if (x >= n) return;
  u32 o[COINOUTW];
  coins_rfc8439_block(o, v[x]);
  for (int i = 0; i < COINOUTW; i++) out[COINOUTW * x + i] = o[i];
}

// The test that the top of this file describes. It waits for the GPU. It returns 0 if every answer
// is as expected. Otherwise bit COINS_OWN_ANSWERS * x + a is set for each wrong answer a of context
// x (0: its key; 1 to 3: the groups' digests; 4: rf::hf_cells_differ()). Bit 60: the batch keys'
// check words; 61: the whole stream; 62: the state afterwards. Bit 63: RFC 8439's vectors.
// The bits are for prover/tests/coins_own_gpu_test.cu, which prints the wrong answers. cp_commit
// and cp_prove ask only whether the result is 0, and print none of it. Both also run
// cp::keys_own_test() (prover/common/cp_keys_own.h) and, for another generator,
// blk::coins_own_test_card() (prover/engine/stream_coins.cuh).
// The test takes no random bytes from the system, touches no file and frees its GPU buffer.
// A function that the test calls may exit the program: key_refuse(), replay_refuse(), enter_draw()
// and CK() call exit() with their own messages and codes. ~CoinsKnownAtHand() then does not run:
// cp_prove's exit handler still erases the key; cp_commit's state then contains a test key, which
// is public, and no seed or key has been read.
constexpr int COINS_OWN_ANSWERS = 5;
constexpr size_t COINS_OWN_HOST_HOLDS = 16384; // the host checks at most this many cells of a run
static_assert(COINS_KNOWN_CHAINS == 2 && COINS_KNOWN_CONTEXTS == 12 && COINS_KNOWN_FILL_GROUPS == 3 &&
                  COINS_OWN_ANSWERS * COINS_KNOWN_CONTEXTS <= 60 && COINS_KNOWN_SLOTS[0] == 6603 &&
                  COINS_KNOWN_STARTS[4] == 16384 && COINS_KNOWN_STARTS[6] == 256 &&
                  COINS_KNOWN_WHOLE_STREAM_CELLS == 3112508,
              "the table has the sizes that this test is written for: a shorter table would pass a shorter test");
static inline u64 coins_own_test() {
  u64 missed = 0;
  auto hold = [&](int bit, bool holds) { missed |= (u64)!holds << bit; };
  const u32 word_before = COINS_WORD;
  for (const CoinsKnownChain &chain : COINS_KNOWN_CHAIN) hold(60, key32_word_of(chain.Kproc) == chain.key_word);
  size_t most = 0; // the cells of the largest group, counting a gap after each run
  for (int x = 0; x < COINS_KNOWN_CONTEXTS; x++)
    for (int group = 0; group < COINS_OWN_GROUPS; group++) {
      size_t cells = 0;
      for (const CoinsKnownRun &run : coins_known_runs(x, group)) cells += run.count + COINS_OWN_PLANE_GAP;
      most = std::max(most, cells);
    }
  // The parentheses, here and at the end, call the CUDA runtime itself. In cp_prove and cp_commit
  // the bare names are macros, which would add the buffer to the program's records of GPU memory.
  u64 *gpu;
  CK((cudaMalloc)(&gpu, 8 * most));
  std::vector<u64> home(most);
  { // RFC 8439's vectors, in the buffer's first bytes: the table, and after it the kernel's blocks
    static_assert(COINS_RFC8439_VECTORS <= 32, "the launch below has 32 threads");
    CoinsRfc8439Vector *const table = (CoinsRfc8439Vector *)gpu;
    u32 *const blocks = (u32 *)(table + COINS_RFC8439_VECTORS), came[COINS_RFC8439_VECTORS * COINOUTW] = {0};
    CK(cudaMemcpyAsync(table, COINS_RFC8439, sizeof COINS_RFC8439, cudaMemcpyHostToDevice, cudaStreamPerThread));
    // No vector's state is all ones, or all zeros as `came` is: a lost launch or copy is detected.
    CK(cudaMemsetAsync(blocks, 0xFF, sizeof came, cudaStreamPerThread));
    k_coins_rfc8439<<<1, 32, 0, cudaStreamPerThread>>>(table, COINS_RFC8439_VECTORS, blocks);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(came, blocks, sizeof came, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    u32 differs = 0;
    for (int x = 0; x < COINS_RFC8439_VECTORS; x++) {
      u32 o[COINOUTW];
      coins_rfc8439_block(o, COINS_RFC8439[x]);
      for (int i = 0; i < COINOUTW; i++)
        differs |= (o[i] ^ COINS_RFC8439[x].state[i]) | (came[COINOUTW * x + i] ^ COINS_RFC8439[x].state[i]);
    }
    hold(63, differs == 0);
  }
  for (int x = 0; x < COINS_KNOWN_CONTEXTS; x++) {
    const CoinsKnownAtHand at_hand(x);
    bool key_holds = true;
    const CoinK ck = coins_known_key(x, &key_holds);
    hold(COINS_OWN_ANSWERS * x, key_holds);
    for (int group = 0; group < COINS_OWN_GROUPS; group++) {
      const std::vector<CoinsKnownRun> runs = coins_known_runs(x, group);
      if (runs.empty()) continue;
      size_t cells = 0;
      for (const CoinsKnownRun &run : runs) cells += run.count;
      // All ones is not a cell. A launch lost wholly or partly leaves ones, not earlier cells.
      const size_t gap = group == 0 ? COINS_OWN_PLANE_GAP : 0, run_cells = runs[0].count; // with a gap, all runs are equally long
      CK(cudaMemsetAsync(gpu, 0xFF, 8 * (cells + gap * runs.size()), cudaStreamPerThread));
      coins_known_draw(group, runs, gpu, gap);
      if (gap) // the runs without the gaps
        CK(cudaMemcpy2DAsync(home.data(), 8 * run_cells, gpu, 8 * (run_cells + gap), 8 * run_cells, runs.size(),
                             cudaMemcpyDeviceToHost, cudaStreamPerThread));
      else
        CK(cudaMemcpyAsync(home.data(), gpu, 8 * cells, cudaMemcpyDeviceToHost, cudaStreamPerThread));
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      const bool whole_stream = group == COINS_OWN_WHOLE_STREAM;
      hold(whole_stream ? 61 : COINS_OWN_ANSWERS * x + 1 + group,
           coins_known_digest_is(home.data(), cells, whole_stream ? COINS_KNOWN_WHOLE_STREAM : COINS_KNOWN_FILL[x][group]));
      const u64 *came = home.data();
      for (const CoinsKnownRun &run : runs) {
        hold(COINS_OWN_ANSWERS * x + 4,
             ::rf::hf_cells_differ(ck, run.stream, run.first, came, std::min(run.count, COINS_OWN_HOST_HOLDS)) == 0);
        came += run.count;
      }
    }
  }
  CK((cudaFree)(gpu));
  hold(62, coins_own_left_clean(word_before));
  return missed;
}
} // namespace hz
