// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Creates the generators of private random values (`coins` in identifiers and messages) that the
// programs cp_commit (committer) and cp_prove (prover) draw through the library blocks/: random
// tails (the random coefficients encoded with a row), mask rows, pads, salts and others. A
// generator implements `blk::Coins` (blocks/engine/table.h) by the derivation of
// blocks/coins/coins.h and writes to host or GPU memory (`card` in identifiers: GPU);
// engine_coins() is the only function through which the two programs create one.
//
// A caller can get two things wrong. Under one process key (the key of one run) it must create at
// most one generator for each context, the four-word label from which a generator's key is derived
// (see engine_coins()); nothing checks this at run time. It must call EngineCoins::hold() before
// bytes of a proof, or of cp_commit's key file or a leaf file (`entry` in identifiers), leave the
// process: hold() runs the self-checks [own-test-engine-coins] and [own-test-coins-bound].
#pragma once
#include <cstring>
#include <memory>
#include <vector>

#include "common/end_name.h"
#include "common/made_up_alone.h"
#include "common/setting.h" // TESTENV
#include "common/stack_scrub.h"
#include "engine/stream_coins.cuh"
#include "hiding/coins_long.cuh"
#include "second_writing/coin_generator.h"

namespace hz {
namespace lng {
constexpr unsigned ENGINE_COINS_SAMPLES = 4; // of each draw

// The test listing. With COINS_CONTEXTS=<file> in the environment, on synthetic weights
// (`made_up` in identifiers) under a test seed only (prover/common/made_up_alone.h), a generator
// that is destroyed appends to the file its context and its draws, in public numbers: the number
// of its process key (contexts_key_number()), the context's four words and its list of draws
// (`ledger()`: each draw's stream, that is its number; its length; elements or bytes); no bit of a
// key or of a drawn value. A production build does not read the variable.
// prover/tests/coin_contexts_listed.py checks on such listings what no verifier and no self-check
// can see: under one key no context is created twice, no stream is opened twice, and the contexts
// are the expected ones. That covers only what the test suite runs.
// Returns the file's path, or null: no listing.
static inline const char *contexts_file() { return made_up_under_a_tests_seed() ? TESTENV("COINS_CONTEXTS") : nullptr; }
// For a listing: the index of `process` among the keys under which this process has created
// generators, in order of first use. The keys are kept for that, unwiped: under a test seed
// nothing is hidden. Not thread-safe.
static inline long contexts_key_number(const blk::CoinKey &process) {
  static std::vector<blk::CoinKey> seen;
  for (size_t i = 0; i < seen.size(); i++)
    if (!memcmp(seen[i].w, process.w, sizeof process.w)) return (long)i;
  seen.push_back(process);
  return (long)seen.size() - 1;
}

// The generator of one context: `blk::CardStreamCoins` (prover/engine/stream_coins.cuh) and, to
// compare its values with, a second implementation of the same derivation (`second writing` in
// messages): prover/second_writing/coin_generator.h, which includes only standard headers and
// prover/second_writing/chacha_block.h. That file's ChaCha20 block function follows the same
// public source as the one in blocks/coins/chacha.h. Each implementation derives its own context
// key from the process key and keeps only that, so the generator stays usable after the process
// key is wiped (hz::key32_wipe(), prover/hiding/trees.cuh). Destroying the generator wipes both
// context keys. An end by exit(), such as hold()'s own, destroys no generator, and cp_prove's exit
// handler does not wipe these keys.
class EngineCoins : public blk::CardStreamCoins {
public:
  EngineCoins(const blk::CoinKey &process, const blk::CoinContext &c)
      : blk::CardStreamCoins(process, c), process_again_(as_key(process)), second_(process_again_, ::coins::Context{c.counter, c.version, c.object, c.member}), context_(c),
        key_number_(contexts_file() ? contexts_key_number(process) : -1) {
    ::coins::wipe(process_again_);
    sample(ENGINE_COINS_SAMPLES);
  }
  ~EngineCoins() override {
    hold();
    const char *const path = key_number_ >= 0 ? contexts_file() : nullptr; // the test listing
    if (FILE *f = path ? fopen(path, "a") : nullptr) {
      fprintf(f, "context key %ld counter %llu version %u object %u member %u draws %zu\n", key_number_, (unsigned long long)context_.counter, context_.version, context_.object, context_.member, ledger().size());
      for (const blk::CoinDraw &d : ledger()) fprintf(f, "%u %llu %s\n", d.stream, (unsigned long long)d.count, d.elements ? "elements" : "bytes");
      fclose(f);
    }
  }
  // Runs both self-checks, in every run, and calls exit(4) if one fails; in cp_prove an exit
  // handler then chooses the exit code, 9 for a failed self-check (prover/prover/ends.h). Returns
  // the number of samples compared; 0 means that nothing was compared, and the two programs' calls
  // treat it as an internal error. A proof whose masks, pads or salts are zeros verifies like any
  // other, and reveals what it should hide; so code that lets bytes of a proof, a key file or a
  // leaf file out of the process must call hold() first. cp_prove calls it through coins_hold()
  // (prover/prover/chain_coins.cuh); cp_commit, as the last check before the function that builds a
  // key file (prover/hash_commitment/hf_commit_once.cuh) or a leaf file
  // (prover/hash_commitment/hf_entry_make.cuh) returns its bytes. The destructor calls it too, so
  // that no sample goes unread; that is too late to keep a proof back and serves only the report.
  // Unlike the draws, hold() does not check which thread calls it: no other thread may draw during
  // the call.
  //
  // [own-test-engine-coins] Every draw, on the host or into GPU memory, records up to
  // ENGINE_COINS_SAMPLES of the values that it wrote: the first, the last and some between
  // (prover/engine/stream_coins.cuh). hold() compares those recorded since the last call with the
  // second implementation's. The comparison covers the derivation of the context key, the labels
  // (stream, call number, retry) and the values as written.
  // Not seen: a value that is written correctly and then not used, used twice or overwritten; a
  // value that is not sampled; a process key that is wrong when engine_coins() gets it, since both
  // implementations start from it; a fault in ChaCha20's block function that both have from their
  // common source (see EngineCoins). (Other self-checks compare copies of the key:
  // [own-test-batch-keys] in prover/prover/isk.cuh, [own-test-key] in
  // prover/hash_commitment/hf_commit_keys.h. RFC 8439's test vectors check the block functions
  // themselves: coins_own_test_card() (prover/engine/stream_coins.cuh), which both programs call,
  // checks the generator's with the vector of section 2.3.2; only a test,
  // prover/tests/coins_rfc8439_test.cpp, checks the second implementation's, in an identical copy.)
  //
  // [own-test-coins-bound] Right after EngineGpu::commit() (prover/engine/engine_gpu.cuh) has built
  // a Merkle tree, kernels compare, in GPU memory, every value of the main rows' random tails and
  // of the leaves' salts (cp_prove then frees the salts; cp_commit keeps them) with the generator's
  // values, computed again (held_card(), prover/engine/stream_coins.cuh); hold() reads the counts.
  // The tails include R*, the random tail of the main masking row of BASE, the last step of a
  // proximity test: base_prove() (blocks/iop/base_case.h) re-checks that row's message M*, the mask
  // rows' messages and the mask rows' random tails, not R*, and under `EngineCpu`
  // (blocks/engine/engine_cpu.h) nothing re-checks R*.
  // Not seen: values drawn on the host (mask rows, pads), some of which the library compares again
  // with what was drawn, in not_as_drawn() and in send_prove()'s read-back of the pads
  // (blocks/iop/send.h); a kernel that reads a leaf's salt wrongly (at a leaf that no proof opens
  // nothing else sees that) or fewer entries than were drawn, or two rows that point at one row's
  // entries; a value lost after the tree is built, which yields a proof that does not verify; a
  // fault in code that the comparing and drawing kernels share (chacha20_block(), coin_admit()),
  // which only a sample can show; a salt regenerated for an opening (the path check tests it).
  size_t hold() {
    size_t differ = 0, compared = 0;
    compare(differ, compared);
    scrub_the_stack_below(); // the second implementation leaves key and cipher state on the stack
    const blk::CoinsBound b = bound(); // after `compare`, which waits for the draws' CUDA streams
    if (differ) {
      printf("internal error (own-test-engine-coins): %zu of %zu sampled random values of a proof are not those of the rule's second writing. The program or the machine "
             "has a fault; nothing of this run may be used\n", differ, compared);
      name_the_end("own-test-engine-coins");
      exit(4);
    }
    if (!b.good()) {
      printf("internal error (own-test-coins-bound): a self-check of the secret random values in GPU memory failed: %llu of %llu checked values are wrong; drawn %llu, sent to the check %llu, exempt from it %llu "
             "(checked must equal sent; sent + exempt must equal drawn). The program or the machine has a fault; nothing of this run may be used\n", (unsigned long long)b.differ, (unsigned long long)b.compared, (unsigned long long)b.laid, (unsigned long long)b.handed,
             (unsigned long long)b.left);
      name_the_end("own-test-coins-bound");
      exit(4);
    }
    return compared;
  }
  // Returns whether `value` is element `index` of stream `stream` of this generator, according to
  // the second implementation; false if that throws. For a check long after the draw, at the last
  // point before a use in which a lost value would expose something: above all a mask that is the
  // only one on a sent scalar. A drawn 0 passes: the check asks only for the drawn value and
  // excludes no value that a mask can take. It draws and records nothing. The caller decides what
  // to do on false.
  bool is_the_coin(u32 stream, u64 index, const blk::F &value) const {
    const bool same = second_says(stream, index, value);
    scrub_the_stack_below(); // as in hold()
    return same;
  }
  // The same for a whole draw of n elements at `values`: returns how many are not the drawn ones.
  // `left_out`: indices that are not compared, ascending. If the second implementation throws,
  // every compared element counts as different (at least 1).
  u64 not_the_coins(u32 stream, const blk::F *values, u64 n, const std::vector<u64> &left_out) const override { // called by `not_as_drawn` (blocks/iop/send.h)
    const u64 differ = second_counts(stream, values, n, left_out);
    scrub_the_stack_below(); // as in `hold`; once for all n elements
    return differ;
  }

private:
  // Never inlined: see compare().
  __attribute__((noinline)) bool second_says(u32 stream, u64 index, const blk::F &value) const {
    try {
      return second_.cell(stream, index) == value.v;
    } catch (...) {
      return false;
    }
  }
  __attribute__((noinline)) u64 second_counts(u32 stream, const blk::F *values, u64 n, const std::vector<u64> &left_out) const {
    try {
      u64 differ = 0;
      size_t next = 0;
      for (u64 i = 0; i < n; i++)
        if (next < left_out.size() && left_out[next] == i)
          next++;
        else
          differ += second_.cell(stream, i) != values[i].v;
      return differ;
    } catch (...) {
      return n > left_out.size() ? n - left_out.size() : 1;
    }
  }
  // Never inlined, so that what the second implementation leaves on the stack is below hold()'s
  // frame; scrub_the_stack_below() zeroes 32 KiB below it (limits: prover/common/stack_scrub.h).
  // If the second implementation throws (no retry is left for a drawn word that is not a field
  // element; a stream number that it rejects), every sample counts as different (1 if there is
  // none).
  __attribute__((noinline)) void compare(size_t &differ, size_t &compared) {
    std::vector<blk::CoinSample> got = samples();
    try {
      differ = blk::coin_samples_differ(got, [&](u32 stream, bool element, u64 index) { return element ? second_.cell(stream, index) : second_.word(stream, index); });
    } catch (...) {
      differ = got.size() ? got.size() : 1;
    }
    compared = got.size();
    blk::wipe(got.data(), got.size() * sizeof(blk::CoinSample));
  }
  static ::coins::Key as_key(const blk::CoinKey &k) {
    ::coins::Key out;
    for (int i = 0; i < 8; i++) out[i] = k.w[i];
    return out;
  }
  ::coins::Key process_again_; // used only by the constructor, which wipes it
  ::coins::Coins second_;
  const blk::CoinContext context_; // public words; for the test listing
  const long key_number_;          // -1: no listing
};

// Never inlined: see compare().
__attribute__((noinline)) static std::unique_ptr<EngineCoins> engine_coins_made(const blk::CoinKey &process, const blk::CoinContext &c) { return std::make_unique<EngineCoins>(process, c); }
// Creates the generator of the context (counter, VERSION_WORD, object, member) under the process
// key `process`, the key of one run: a batch in cp_prove; in cp_commit the creation of the key
// file or of one leaf file. The caller wipes `process`. A whole run's context has counter and
// member 0; a request's has its attempt number and its slot (prover/prover/chain_coins.cuh).
//
// Under one process key a context gets at most one generator: a second would start again at the
// first stream and give the same values to other messages. That is the caller's duty: each caller
// creates its generators in one place, where it has the run's key. Nothing checks it at run time;
// only a test does (see contexts_file(); docs/limits-all-rows.tsv, row H12).
//
// `object` is ENGINE (a request), ENGINE_BATCH (what a batch's requests share), ENGINE_KEY_FILE or
// ENGINE_ENTRY (a leaf file). The generator of prover/hiding/trees.cuh draws under the same process
// key, which each program also stores in that file's key state (hz::key32_renew()), by the same
// derivation (ChaCha20's block function, the same labels and keys) in other code. Neither sees the
// other's list of draws, and both number streams from 16,384 up: only the object word keeps their
// values apart (static_asserts: prover/hiding/long_rows_words.h; docs/limits-all-rows.tsv, row
// H12.2).
//
// The key is an argument on purpose: nothing here reads it from the key state of
// prover/hiding/trees.cuh. The code that derives a run's key creates the run's generators while it
// has the key, and wipes it, so that later code is given no key to create one from. Otherwise,
// with overlapping batches (`-overlap`), a function of cp_prove that works for a batch that was
// set aside could create a generator under another batch's key, the one then in the state. In a
// test with that fault injected, no self-check and no verifier noticed it.
//
// Limits: the key's bytes remain where the run needs them, in cp_prove in that state and in
// prover/prover/slot_pads_second.cuh. Neither the compiler nor a test of this repository on the
// source text requires that no later function of cp_prove calls engine_coins() or reads the bytes
// from the state or from `Batch::for_the_library`. What ships and bears on a further generator
// (prover/tests/coin_contexts_listed.py; the count of generators in tools/audit/level1_proof.py)
// looks only at runs, never at the sources. No rule covers the copy in
// prover/prover/slot_pads_second.cuh. Of the key, engine_coins() checks only that it is not all
// zeros, not that the calling process derived it: a forked child that holds its parent's key is not
// rejected. cp_prove never calls fork() and
// starts no child process.
// In cp_commit, engine_key() (prover/hash_commitment/hf_commit_coins.h) rejects a forked child.
// cp_commit makes one run at a time, keeps each run's key in the state for [own-test-key], and
// reads it from there through prover/hash_commitment/hf_commit_coins.h, which is not among
// cp_prove's sources and must not be (see its head): neither the compiler nor a test of this
// repository requires that.
static inline std::unique_ptr<EngineCoins> engine_coins(const blk::CoinKey &process, u64 counter, Object object, u32 member) {
  if (object != ENGINE && object != ENGINE_BATCH && object != ENGINE_KEY_FILE && object != ENGINE_ENTRY) {
    printf("internal error: the engine's coins are asked for under an object number of other coins\n");
    exit(4);
  }
  // Eight zero words mean a key that was wiped before this call: every generator would run under
  // zeros, which anyone can reproduce, and both implementations would agree. (A derived key is all
  // zeros with probability 2^-256.)
  u32 any = 0;
  for (const u32 w : process.w) any |= w;
  if (!any) {
    printf("internal error (own-test-engine-key): the random values of a proof are asked for under a key of zeros. The program has a fault; nothing of this run may be used\n");
    name_the_end("own-test-engine-key");
    exit(4);
  }
  auto coins = engine_coins_made(process, blk::CoinContext{counter, (u32)VERSION_WORD, (u32)object, member});
  scrub_the_stack_below(); // the second implementation derived its context key on the stack
  return coins;
}
} // namespace lng
} // namespace hz
