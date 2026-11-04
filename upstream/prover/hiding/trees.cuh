// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* (1) Private randomness (`coins` in names and messages): the 32-byte process key, from which this
   file derives random field elements; blinding contexts (`attempt` in identifiers), each with its
   own counter; and a ledger of draws. The program exits if a recorded value is drawn again: equal
   blinding values on different data leak the data, and no verifier detects it. (2) `Tree`, `TreeN`:
   the shapes, buffers and random values of row commitments (rows with blinding coordinates, encoded,
   their columns under a Merkle tree). Nothing here encodes a row or builds a tree: cp_prove uses
   their shapes, stream numbers and the slot pads (Tree::slot_pads()).
   Never reuse a key and its counters on different data: see CTR. Only set-up allocates GPU memory.
   Include after: prover/hiding/pt_field.h, prover/common/hash.h, prover/common/hash_tree.cuh. */
// Public sources. Each is cited by its tag in square brackets where its idea is used.
// [Seibel] Peter Seibel, Practical Common Lisp, Apress 2005, chapter 6 "Variables", section "Dynamic, a.k.a. Special,
//   Variables", the author's online text https://gigamonkeys.com/book/variables.html
// [Boost.Serialization] Robert Ramey, Boost.Serialization documentation, page "state_saver" (copyright 2002-2004),
//   https://www.boost.org/doc/libs/1_70_0/libs/serialization/doc/state_saver.html
// [CUTLASS layouts] NVIDIA, CUTLASS documentation (4.4.1), "Layouts and Tensors",
//   https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/layout.html
// [Hemo] T. Hemo, K. Jue, E. Rabinovich, G. Roh, R. D. Rothblum, "Jagged Polynomial Commitments (or: How to Stack
//   Multilinears)", Cryptology ePrint Archive 2025/917 (copy dated April 10, 2026), section 1 with Figure 1, and
//   section 1.1 with footnote 4, https://eprint.iacr.org/2025/917
// [CUDA guide] NVIDIA, CUDA C++ Programming Guide, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
// [Nugteren] Cedric Nugteren, Tutorial: OpenCL SGEMM tuning for Kepler, SURFsara 2014, page "Kernel 10: Incomplete
//   tiles and arbitrary matrix sizes", https://cnugteren.github.io/tutorial/pages/page12.html
// [Nugteren 1] The same tutorial, page "Kernel 1: Naive implementation",
//   https://cnugteren.github.io/tutorial/pages/page3.html. Its C = A B is M x N, with sums over K.
// [CUDA best practices] NVIDIA, CUDA C++ Best Practices Guide,
//   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
// [PyTorch] PyTorch documentation 2.8, torch.Tensor.contiguous,
//   https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.contiguous.html
#pragma once
#include "hiding/place.cuh"
#include "common/coin_chacha.h"
#include "common/code_layers.cuh"
#include "hiding/lost_coins_door.cuh"
#include "hiding/lost_copies_door.cuh"
#include <sys/mman.h>
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "compile with --default-stream per-thread: the coins' guard asks about the stream the library launches into"
#endif
#include <map>
#include <string>
#include <vector>
#include <algorithm>
namespace hz { // holds the private randomness, the placement of GPU buffers and the row commitments
typedef uint16_t u16;
typedef Ext T;
typedef FE F;
constexpr int CE = EDEG; // components of an element of E; a row over E is CE real (base-field) rows, component 0 first
constexpr u64 P = FIELD_P; // the build's prime (prover/field/field_names.h)
struct Shape {
  int ln = 0, t = 0, mk = 0;
  size_t n = 0, Ks = 0, Kc = 0;
};
static inline size_t p2c(size_t v) {
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}
static inline int lg2c(size_t x) {
  int l = 0;
  while (((size_t)1 << l) < x) l++;
  return l;
}
/* Returns the smallest t >= 0 with (NUM / 2^LD)^t <= 2^-hb. t sizes a row: mk = a t + 4 or 5 (see
   `Shape`); it also enters the number `bytes` by which shape2() picks a shape. Integers only, so
   every machine gets the same t. Requires 1 <= LD <= 32 and 1 <= NUM <= (7/8) 2^LD (or exits), so t <= 5.2 hb + 1. */
static int tcol(u64 NUM, int LD, int hb) {
  if (hb <= 0) return 0;
  if (!NUM || NUM >> 32 || LD < 1 || LD > 32 || 8 * NUM > 7 * ((u64)1 << LD)) {
    fprintf(stderr, "internal error: tcol(%llu, %d, %d) is outside its range\n", (unsigned long long)NUM, LD, hb);
    exit(2);
  }
  std::vector<u32> p{1}; // NUM^t, least limb first
  for (int t = 1;; t++) {
    u64 c = 0;
    for (auto &x : p) {
      c += (u64)x * NUM;
      x = (u32)c;
      c >>= 32;
    }
    if (c) p.push_back((u32)c);
    const long E = (long)t * LD - hb, len = 32L * (long)(p.size() - 1) + (32 - __builtin_clz(p.back()));
    if (len <= E) return t; // NUM^t < 2^E
    if (len == E + 1) {     // NUM^t = 2^E exactly?
      bool pow2 = (p.back() & (p.back() - 1)) == 0;
      for (size_t i = 0; i + 1 < p.size(); i++) pow2 = pow2 && !p[i];
      if (pow2) return t;
    }
  }
}
/* Overrides for the shape rules. While rw (the largest number of data rows wanted) is not 0,
   `shape2` returns the shape that `shape_flat` finds for rw and the rate 2^-ir. irM (RATE_M):
   `pad_tree_shape` gives a rate Kc / n (row length over code length) of at most 2^-irM. The programs keep
   the defaults. */
// Public sources: a global setting changed for a limited scope and reset by an object's destructor:
// [Seibel], chapter 6, describes such a global, and [Boost.Serialization] (state_saver) describes
// the C++ object; here the caller resets the rule by hand, to its default instead of a saved value.
struct FlatRule {
  size_t rw = 0;
  int ir = 1, irM = 1;
};
static FlatRule flat_rule;
#define RATE_M flat_rule.irM
/* `Tree::setup`'s shape. A row: K data coordinates, zeros up to Ks, mk blinding ones; Kc in all.
   n = 2^ln >= 2048: the smallest at which t and mk reach a fixed point with (Kc << RATE_M) <= n. */
static Shape pad_tree_shape(size_t K, int a, int bits) {
  for (size_t n = std::max<size_t>(2048, p2c(K) << RATE_M);; n *= 2) {
    int t = 1;
    bool ok = true;
    Shape s;
    for (;;) {
      int mk = (a * t + 5) & ~1;
      size_t q = p2c(mk), Ks = (K + q - 1) / q * q, Kc = Ks + mk;
      if ((Kc << RATE_M) > n) {
        ok = false;
        break;
      }
      s.ln = lg2c(n);
      const int t2 = tcol(n + Kc + 1, s.ln + 1, bits);
      s.t = t;
      s.mk = mk;
      s.n = n;
      s.Ks = Ks;
      s.Kc = Kc;
      if (t2 == t) break;
      t = t2;
    }
    if (ok) return s;
  }
}
/* A cell is one random element of the base field F. The generator (prover/common/coin_chacha.h)
   derives it from the process key KEYW, the label of its blinding context (say, a request's proof),
   its stream's number and its own. The label is the context's counter CTR, COINS_WORD, a kind of
   object, a request's number and a zero nonce. This file never asks the operating system for random
   bytes: the caller (the program cp_commit or cp_prove) derives the key (prover/common/coin_keys.h)
   and passes it to `key32_given` or `key32_renew`. So the caller must never use a key twice with
   the same counters on different data. The key is never printed. `key_test` sets the public test
   key, under which nothing is hidden; a production build (SERVING_BUILD) omits it. Every build's
   start-up self-test [own-test-coins] (prover/hiding/coins_own.cuh) sets keys written in the
   source and wipes each. */
static u64 CTR = 0;
// The version word that coink() puts into every context's label. cp_prove sets it to the parameter
// file's (policy.txt), and cp_commit leaves it as it stands here.
// [own-test-coins] sets it for its test and restores it.
static u32 COINS_WORD = VERSION_WORD;
enum KeyState { KEY_NONE, KEY_TEST, KEY_GIVEN, KEY_WIPED };
static KeyState key_state = KEY_NONE;
static u32 KEYW[COINKEYW] = {};
// If set, it gets a sentence from key_refuse(), replay_refuse() or a ledger error before exit();
// cp_prove reports all as `own-test-coin-library`. Pass the two functions only literals or choices
// among them, so that no hidden value enters a message (a test outside this repository checks it).
static void (*refusal_is_told_to)(const char *sentence) = nullptr;
[[noreturn]] static void key_refuse(const char *why) {
  fprintf(stderr, "coins: %s\n", why);
  if (refusal_is_told_to) refusal_is_told_to(why);
  exit(2);
}
/* A child of `fork` inherits its parent's key and counter. If both drew cells, the same blinding
   values would serve two proofs, and no verifier would detect it. The operating system zeroes this
   page in a child, so the word set with the key reads 0 there and the child's first `coink` exits.
   This does not catch a copy of the process or machine made from outside, and it does not cover the
   seed, a batch's fresh bytes (rho) or a mask seed (kM), which never become the process key. */
static volatile u32 *own_process_() {
  static volatile u32 *word = [] {
    void *page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED || madvise(page, 4096, MADV_WIPEONFORK))
      key_refuse("the operating system gave no memory page that a fork wipes: mmap() or madvise(MADV_WIPEONFORK) failed; the latter needs Linux 4.14 or later");
    return (volatile u32 *)page;
  }();
  return word;
}
constexpr u64 TEST_KEY = 7;
#ifndef SERVING_BUILD
static void key_test() {
  if (key_state != KEY_NONE) key_refuse("a key was already chosen; a program chooses its key once");
  for (int k = 0; k < COINKEYW; k++) KEYW[k] = (u32)(mix64(TEST_KEY + k) >> 16);
  *own_process_() = 1;
  key_state = KEY_TEST;
}
#endif
// A stream is a numbered sequence of cells. A context's streams are numbered from here up. A lower
// stream's label has no counter, so under a given key coink() exits for it. No program uses the
// lower streams; unit tests do, under the public test key.
constexpr u32 ATTEMPT_STREAMS_FROM = 4u * 4096;
/* What a kernel needs to derive cells. `coink` makes it, and exits before a key is set and after
   `key32_wipe`. A context has one key for all its streams. CoinK contains that key, so this file's
   kernels that take it (k_fill, k_slot_pads_agree) do so as a const
   __grid_constant__ parameter: a device function can then take a reference to it, and the compiler
   makes no copy of it in thread-local memory, GPU memory that no code here can zero (NVIDIA, CUDA
   C++ Programming Guide, "__grid_constant__"). See k_fill on the built kernels. */
/* See also: a kernel's parameter that is not copied for each thread: NVIDIA, CUDA Programming Guide,
   "C/C++ Language Extensions", "__grid_constant__ Parameters",
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html */
struct CoinK {
  // The context's key, the generator's K_ctx, which coink() derives from the process key and
  // the nine words below. Deriving a cell reads only k; the nine words identify the context.
  u32 k[COINKEYW] = {};
  u32 c0 = 0, c1 = 0;           // the attempt's counter
  u32 n[NONCEW] = {0, 0, 0, 0}; // the label's four nonce words: zero in this format
  u32 version = 0;              // the format's version word: coink() sets it for an attempt's streams; else 0
  // obj, mem: the context's kind of object and the request's number in its batch (`member`)
  u32 obj = 0, mem = 0;
};
// Zeroes the host's copy after the launch for which it was made.
static void forget(CoinK &c) {
  volatile u32 *w = (volatile u32 *)&c;
  for (size_t i = 0; i < sizeof c / 4; i++) w[i] = 0;
}

/* The contexts' state (`parked`: suspended). No (key, counter, stream, cell) may serve two proofs:
   equal blinding values on different data leak the data, and both proofs would still verify. */
struct Replay {
  bool set = false, any = false, begun = false; // begun: `set_attempt` ran under this key
  u64 ctr = 0, last = 0;                        // this attempt's counter; the largest counter used so far
  u32 obj = 0, mem = 0; // this attempt's kind of object and member's number, set by set_attempt()
  // The ledger of draws: stream -> first cell -> one past the last cell, for each range drawn.
  typedef std::map<u32, std::map<u64, u64>> Ledger;
  Ledger drawn;                             // by the current attempt
  u64 drawn_under = 0;                      // the counter under which they were drawn
  std::map<u32, Ledger> drawn_by_committer; // streams below 4 * 4096, by commitment number; cleared by key32_renew()
  struct Parked {
    u32 obj, mem;
    Ledger drawn;
  };
  std::map<u64, Parked> parked; // the suspended attempts, by counter
};
static Replay &replay_() {
  static Replay &r = *new Replay; // never destroyed: key32_wipe() may run in an exit handler, after static destructors
  return r;
}
/* For a test that derives a context's cells again outside the program. Only under the public test
   key, and not in a production build, it appends to the file named by the environment variable
   COINS_LEDGER the current context's counter, object and member, the ranges of cells that it drew
   (numbers only, no values) and how many suspended contexts, if any, it leaves out. Outside tests
   neither `cp_commit` nor `cp_prove` runs with COINS_LEDGER set (prover/common/test_words.h). */
static void ledger_for_a_test_() {
#ifndef SERVING_BUILD
  Replay &r = replay_();
  if (key_state != KEY_TEST || r.drawn.empty()) return;
  if (const char *path = TESTENV("COINS_LEDGER"))
    if (FILE *f = fopen(path, "a")) {
      fprintf(f, "attempt %llu object %u member %u\n", (unsigned long long)r.ctr, r.obj, r.mem);
      for (auto &s : r.drawn)
        for (auto &x : s.second)
          fprintf(f, "%u %llu %llu\n", s.first, (unsigned long long)x.first, (unsigned long long)x.second);
      if (!r.parked.empty()) fprintf(f, "left out: %zu parked attempts\n", r.parked.size());
      fclose(f);
    }
#endif
}
/* Zeroes the key and drops the current context and every suspended one. It may run in
   an exit handler, but not in a signal handler. In an exit handler, call it without
   `end_attempt`: that function can itself call `exit` (on one of its own errors or a
   failed CUDA call), and `exit` called inside `exit` need not return to the wipe. */
static void key32_wipe() {
  ledger_for_a_test_();
  volatile u32 *w = KEYW;
  for (int k = 0; k < COINKEYW; k++) w[k] = 0;
  key_state = KEY_WIPED;
  Replay &r = replay_();
  r.parked.clear();
  r.drawn.clear();
  r.set = false;
}
// Makes these 32 bytes the process key. Exits on 32 zero bytes, which only a bug passes in,
// such as a buffer zeroed too early: under that key anyone could compute every cell, and
// every proof would still verify. Any other constant, public or repeated key is accepted.
static void take_key_(const uint8_t key[4 * COINKEYW]) {
  uint8_t any = 0;
  for (int i = 0; i < 4 * COINKEYW; i++) any |= key[i];
  if (!any) key_refuse("the process key handed in is 32 zero bytes. This is a bug in the program: report this line");
  for (int k = 0; k < COINKEYW; k++)
    KEYW[k] = (u32)key[4 * k] | (u32)key[4 * k + 1] << 8 | (u32)key[4 * k + 2] << 16 | (u32)key[4 * k + 3] << 24;
  *own_process_() = 1;
  key_state = KEY_GIVEN;
}
// Sets the first process key: 32 bytes, eight little-endian words. They are
// copied; the caller must zero its own copy. Exits if a key was set before.
static void key32_given(const uint8_t key[4 * COINKEYW]) {
  if (key_state != KEY_NONE) key_refuse("a key was already chosen; key32_renew() replaces a key");
  take_key_(key);
}
// Replaces the process key, whatever key_state is: zeroes the old key, takes these 32 bytes and
// clears all counters and ledgers, so that the new key's contexts may count from the start again.
// No context draws a nonce, so a key passed in twice gives the same cells again: the caller must
// give each process key one use only (in cp_prove: one batch). That holds only by how the programs
// are written: this file's checks on a key (take_key_(), key32_word()) pass a key given twice.
// Exits if a context is current or suspended: its later cells would come from another key.
static void key32_renew(const uint8_t key[4 * COINKEYW]) {
  Replay &r = replay_();
  if (r.set || !r.parked.empty()) key_refuse("key32_renew() while an attempt is at hand or parked: end it first");
  key32_wipe();
  r.any = r.begun = false;
  r.ctr = r.last = r.drawn_under = CTR = 0;
  r.drawn_by_committer.clear();
  take_key_(key);
}
/* The key's 32-bit check word [own-test-key], defined in prover/common/coin_chacha.h, rule 2. A
   caller compares it with that of the bytes it passed in, to catch a stale key; a wrongly derived
   key passes, and so does a different key about once in 2^32. Treat it as a secret. */
// The check word of this translation unit's KEYW; after key32_wipe(), that of eight zero words.
static u32 key32_word() {
  return coin_key_word(KEYW);
}
// The check word of these 32 bytes, read as in take_key_().
static u32 key32_word_of(const uint8_t key[4 * COINKEYW]) {
  u32 words[COINKEYW];
  for (int k = 0; k < COINKEYW; k++)
    words[k] = (u32)key[4 * k] | (u32)key[4 * k + 1] << 8 | (u32)key[4 * k + 2] << 16 | (u32)key[4 * k + 3] << 24;
  const u32 word = coin_key_word(words);
  volatile u32 *v = words;
  for (int k = 0; k < COINKEYW; k++) v[k] = 0;
  return word;
}
[[noreturn]] static void replay_refuse(const char *m) {
  fprintf(stderr, "coins: %s\n", m);
  if (refusal_is_told_to) refusal_is_told_to(m);
  exit(2);
}
// The id of the CUDA graph capture on the calling thread's stream, or 0. It sees no capture on
// another stream: a cell drawn on a helper thread of a captured product test would be drawn
// again at every launch. No helper thread draws one: product_test.cuh uses no name of hz.
static unsigned long long replay_capture() {
  cudaStreamCaptureStatus s = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  CK(cudaStreamGetCaptureInfo(cudaStreamPerThread, &s, &id));
  return s == cudaStreamCaptureStatusActive ? id : 0;
}
// Sets c.k to the key of the context whose nine words c contains. coink() calls it every
// time: a stored context key could go stale and repeat values in proofs that still verify.
static void set_context_key_(CoinK &c) {
  coin_context_key(c.k, KEYW, c.c0, c.c1, c.version, c.obj, c.n, c.mem);
}
static CoinK coink(u32 stream) {
  CoinK c;
  if (key_state == KEY_NONE)
    key_refuse("no key was chosen; call key32_given() or key32_renew() before the first coin");
  if (key_state == KEY_WIPED) key_refuse("a coin was asked for after the key was wiped");
  if (!*own_process_()) key_refuse("a coin was asked for by a fork's child of the process that chose the key");
  if (stream < ATTEMPT_STREAMS_FROM) { // all nine words of the context are 0, so no counter keeps two draws apart
    if (key_state != KEY_TEST)
      key_refuse("a coin of a stream below 4 * 4096 was asked for under a given key: such a stream belongs to no attempt");
    set_context_key_(c);
    return c;
  }
  /* Every context's label contains the format's version word, so that no key
     gives the same cells to two formats, which number their rows differently. */
  c.version = COINS_WORD;
  Replay &r = replay_();
  if (replay_capture())
    replay_refuse("request coins are being captured into a CUDA graph: the context's key is a kernel argument, "
                  "so every launch of the graph would draw the same coins again");
  if ((key_state == KEY_GIVEN || r.begun) && (!r.set || CTR != r.ctr))
    replay_refuse(
        "request coins with no attempt at hand, or hz::CTR changed by hand: under a secret key, and in any program "
        "that has called set_attempt(), set_attempt() and resume_attempt() alone set the counter");
  if (!r.any || CTR > r.last) r.last = CTR;
  r.any = true;
  c.c0 = (u32)CTR;
  c.c1 = (u32)(CTR >> 32);
  if (r.set && CTR == r.ctr) {
    c.obj = r.obj;
    c.mem = r.mem;
  }
  set_context_key_(c);
  return c;
}
/* Begins a context with the counter ctr, which must exceed every counter used before under this
   key. A current context is dropped silently, unless others are suspended: then the program exits.
   object, member: the kind of object that the context makes (`lng::Object`, coins_long.cuh) and the
   request's number in its batch. Both are in every cell's label, so contexts that differ in either
   derive from different inputs at any counters. Only tests use the defaults. */
static void set_attempt(u64 ctr, u32 object = 0, u32 member = 0) {
  Replay &r = replay_();
  if (key_state != KEY_TEST && key_state != KEY_GIVEN) key_refuse("set_attempt() needs a key");
  if (replay_capture()) replay_refuse("set_attempt() during a capture");
  if (r.set && !r.parked.empty())
    replay_refuse("set_attempt() while an attempt is at hand and others are parked: park or end that one first");
  if (r.any && ctr <= r.last) {
    fprintf(stderr, "coins: counter %llu is not above %llu, the largest used under this key: coins could repeat\n",
            (unsigned long long)ctr, (unsigned long long)r.last);
    exit(2);
  }
  r.any = r.begun = true;
  r.last = r.ctr = CTR = ctr;
  r.obj = object;
  r.mem = member;
  r.drawn.clear();
  r.drawn_under = ctr;
  r.set = true;
}
/* Ends the current context, after waiting for the GPU: in a proof, this file's only wait.
   No cell of a context can be drawn until the next `set_attempt` or `resume_attempt`. */
static void end_attempt() {
  Replay &r = replay_();
  if (replay_capture()) replay_refuse("end_attempt() during a capture");
  if (!r.set) replay_refuse("end_attempt() with no attempt at hand");
  CK(cudaDeviceSynchronize());
  r.set = false;
  ledger_for_a_test_();
  r.drawn.clear();
}
/* Suspends the current context and returns its counter. The context keeps its counter
   and its ledger, but none of its cells can be drawn until `resume_attempt`. */
static u64 park_attempt() {
  Replay &r = replay_();
  if (replay_capture()) replay_refuse("park_attempt() during a capture");
  if (!r.set) replay_refuse("park_attempt() with no attempt at hand");
  Replay::Parked &p = r.parked[r.ctr];
  p.obj = r.obj;
  p.mem = r.mem;
  p.drawn.swap(r.drawn);
  r.drawn.clear();
  r.set = false;
  return r.ctr;
}
// Resumes the suspended context with the counter ctr. Exits if one is current or none has ctr.
static void resume_attempt(u64 ctr) {
  Replay &r = replay_();
  if (key_state != KEY_TEST && key_state != KEY_GIVEN) key_refuse("resume_attempt() needs a key");
  if (replay_capture()) replay_refuse("resume_attempt() during a capture");
  if (r.set) replay_refuse("resume_attempt() while an attempt is at hand: park or end that one first");
  auto it = r.parked.find(ctr);
  if (it == r.parked.end()) {
    fprintf(stderr, "coins: attempt %llu is not parked: it was never begun, has ended, or is already resumed\n",
            (unsigned long long)ctr);
    exit(2);
  }
  r.obj = it->second.obj;
  r.mem = it->second.mem;
  r.drawn.swap(it->second.drawn);
  r.drawn_under = ctr;
  r.parked.erase(it);
  r.set = true;
  r.ctr = CTR = ctr;
}
/* Rejection sampling for cell q of call bk, whose word was rejected: the call is repeated with
   the retry word RETRYQ * k + q, k = 1, 2 and so on; output words 0 and 1 give the word.
   `field_from_word` (prover/field/field_names.h) takes a word's low FIELD_BITS bits and rejects
   them if they are not below p: what it accepts is exactly uniform. At p = 2^64 - 2^32 + 1 it
   rejects about one word in 2^32, so this function runs in practice. */
__device__ __noinline__ u64 coin_again(const CoinK &ck, u32 stream, u64 bk, int q) {
  for (u32 tr = 1;; tr++) {
    u32 o[COINOUTW];
    coin_output(o, ck.k, stream, bk, RETRYQ * tr + (u32)q);
    u64 v;
    if (field_from_word(((u64)o[1] << 32) | o[0], v)) return v;
  }
}
// The generator's output o for call bk of a stream. This file reads only its words 0 to 7.
__device__ __forceinline__ void coin_words(const CoinK &ck, u32 stream, u64 bk, u32 o[COINOUTW]) {
  coin_output(o, ck.k, stream, bk, 0);
}
// The four cells of call bk of a stream: cell q comes from output words 2 q and
// 2 q + 1 by field_from_word() or, if that rejects them, from coin_again().
__device__ __forceinline__ void coin4(const CoinK &ck, u32 stream, u64 bk, u64 v[4]) {
  u32 o[COINOUTW];
  coin_words(ck, stream, bk, o);
  for (int q = 0; q < 4; q++) {
    if (!field_from_word(((u64)o[2 * q + 1] << 32) | o[2 * q], v[q])) v[q] = coin_again(ck, stream, bk, q);
  }
}
// dst[x] = cell i0 + x of the stream for x < n. Launch it with 256 threads a block, one thread for
// each call of the generator (four cells). The kernel takes the context's key as an argument, and
// no copy of the key may reach local memory. So the built kernel must have no stack frame and no
// other local memory, coin_again(), which it calls and whose working words start as the key's, no
// stack frame, and the machine code of both no LDL and no STL instruction (loads from and stores to
// local memory). In the built `cp_prove` and `cp_commit`, a test that is not part of this
// repository requires that this kernel and k_slot_pads_agree() have no stack frame
// and no other local memory. No test checks coin_again()'s stack frame or reads machine code for
// LDL and STL: a script outside this repository has both checks, and only its self-test is run.
__global__ void k_fill(u64 *__restrict__ dst, const __grid_constant__ CoinK ck, u32 stream, u64 i0, size_t n) {
  const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x;
  const int s = (int)(i0 & 3);
  if (4 * t >= n + s) return;
  u64 v[4];
  coin4(ck, stream, (i0 >> 2) + t, v);
  for (int q = 0; q < 4; q++) {
    const size_t y = 4 * t + q;
    if (y >= (size_t)s && y - s < n) dst[y - s] = v[q];
  }
}
/* Records the cells first to first + count - 1 of a stream in the ledger of draws.
   Exits if one of them was drawn before by this context or, for a stream below
   ATTEMPT_STREAMS_FROM, under this key and the same commitment number: see Replay.
   `fill` calls it before it launches k_fill. It detects a second draw, not a second
   use: a row drawn once and read twice passes. For a repeat: `enter_draw_again`.
   The ledger has only what callers enter by `enter_draw`. `coink` returns a context's key to any
   caller, and cells derived with it without such a call are not recorded. How far the ledger
   covers the arrays that the programs draw has not been established. It is not proved that under
   one process key no two blinding values come from the same input of the generator. */
static void enter_draw(u32 stream, u64 first, u64 count, u32 commitment_number = 0) {
  Replay &r = replay_();
  const bool committer = stream < ATTEMPT_STREAMS_FROM;
  // A test program has set hz::CTR directly: the ledger's entries belong to another counter.
  if (!committer && r.drawn_under != CTR) {
    r.drawn.clear();
    r.drawn_under = CTR;
  }
  if (count > ~first) {
    fprintf(stderr, "coins: internal error: %llu cells from cell %llu of stream %u on: cell numbers end at 2^64\n",
            (unsigned long long)count, (unsigned long long)first, stream);
    if (refusal_is_told_to) refusal_is_told_to("the ledger of draws: cell numbers end at 2^64");
    exit(2);
  }
  std::map<u64, u64> &m = (committer ? r.drawn_by_committer[commitment_number] : r.drawn)[stream];
  const auto next = m.upper_bound(first);
  if ((next != m.end() && next->first - first < count) || (next != m.begin() && std::prev(next)->second > first)) {
    fprintf(stderr, "coins: internal error: cells %llu to %llu of stream %u: some of them are drawn twice %s\n",
            (unsigned long long)first, (unsigned long long)(first + count - 1), stream,
            committer ? "in this process" : "in one attempt");
    if (refusal_is_told_to) refusal_is_told_to("the ledger of draws: cells are drawn twice");
    exit(2);
  }
  // Adjacent ranges are merged, so code that draws a row piece by piece adds one entry per row.
  u64 from = first, to = first + count;
  if (next != m.end() && next->first == to) {
    to = next->second;
    m.erase(next);
  }
  const auto before = m.upper_bound(first);
  if (before != m.begin() && std::prev(before)->second == first) from = std::prev(before)->first;
  m[from] = to;
}
/* The ledger's only exemption, a repeat: the same context computes an object again. Exits unless it
   has drawn every cell of the range. The same cells must mask the same data, so the caller must
   compare the result with the first computation's before a byte of it leaves the process.
   Nothing here makes or enforces that comparison: a caller that omits it goes undetected. */
static void enter_draw_again(u32 stream, u64 first, u64 count) {
  Replay &r = replay_();
  const std::map<u64, u64> &m = r.drawn[stream];
  u64 at = first;
  const bool attempt_at_hand = stream >= ATTEMPT_STREAMS_FROM && r.drawn_under == CTR && count <= ~first;
  while (attempt_at_hand && at - first < count) { // adjacent ranges count as one
    const auto next = m.upper_bound(at);
    if (next == m.begin() || std::prev(next)->second <= at) break;
    at = std::prev(next)->second;
  }
  if (!attempt_at_hand || at - first < count) {
    fprintf(stderr,
            "coins: internal error: cells %llu to %llu of stream %u are to be drawn AGAIN or are held to be drawn, but "
            "this "
            "attempt has not drawn all of them before\n",
            (unsigned long long)first, (unsigned long long)(first + count - 1), stream);
    if (refusal_is_told_to) refusal_is_told_to("the ledger of draws: cells that were not drawn are held to be drawn");
    exit(2);
  }
}
/* Exits unless this context has drawn every cell of the range. It checks cell numbers,
   not drawn values. Call it for each row before the row is used: if a blinding row or
   a row of pads had not been drawn, the value that it should hide would be sent
   unmasked, and no verifier would detect it. It draws nothing and reads no cell. */
static void hold_drawn(u32 stream, u64 first, u64 count) {
  if (count) enter_draw_again(stream, first, count);
}
/* The host functions that launch the generator's kernels read the launch status immediately. A
   failed launch would leave the buffer as it was, zero or with another row, where blinding values
   are expected, and the proof would still verify. A later read can miss the error: any call of
   `cudaGetLastError` in between clears it (NVIDIA, CUDA Runtime API reference, Error Handling). */
// fill() for cells that the current context has drawn before: see enter_draw_again().
static void fill_again(u64 *dst, u32 stream, u64 i0, size_t n) {
  if (!n) return;
  CoinK ck = coink(stream);
  enter_draw_again(stream, i0, n);
  const size_t th = (n + (i0 & 3) + 3) / 4;
  k_fill<<<(unsigned)((th + 255) / 256), 256>>>(dst, ck, stream, i0, n);
  forget(ck);
  CK(cudaGetLastError());
}
// dst[x] = cell i0 + x of the stream, for x < n. A first draw: enter_draw() exits on a second.
static void fill(u64 *dst, u32 stream, u64 i0, size_t n) {
  if (!n) return;
  CoinK ck = coink(stream); // coink()'s checks come before the ledger's
  enter_draw(stream, i0, n);
  const size_t th = (n + (i0 & 3) + 3) / 4;
  k_fill<<<(unsigned)((th + 255) / 256), 256>>>(dst, ck, stream, i0, n);
  forget(ck);
  CK(cudaGetLastError());
}
// A sub-proof, the argument for one part of the forward pass (its memory: hz::FORK), has ns slot
// pads: one-time pads in the extension field E. Pad j masks the value in slot j of the sub-proof's
// message and is entry j of the mask row of its segment (its part of the proof). The kernel adds to
// *agree how many cells of an array of the pads (its first argument) equal those derived from the
// key. Component kappa of pad j must be cell kappa * ns + j of the stream (Tree::slot_pads()).
// Launch: 256 threads per block, in x one per generator call (at most ns / 4 + 2); blockIdx.y is
// the component. k_fill()'s rule on local memory applies; see there for what checks it.
__global__ void k_slot_pads_agree(const T *__restrict__ laid, const __grid_constant__ CoinK ck, u32 stream, size_t ns,
                                  unsigned long long *agree) {
  const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x;
  const int kappa = (int)blockIdx.y;
  const u64 i0 = (u64)kappa * ns;
  const int s = (int)(i0 & 3);
  if (4 * t >= ns + s) return;
  u64 v[4];
  coin4(ck, stream, (i0 >> 2) + t, v);
  unsigned long long same = 0;
  for (int q = 0; q < 4; q++) {
    const size_t y = 4 * t + q;
    if (y >= (size_t)s && y - s < ns) same += coord(laid[y - s], kappa) == v[q];
  }
  atomicAdd(agree, same);
}
// Points to a counter in GPU memory: how many cells of slot pads Tree::slot_pads_agree() has
// found equal to the cells derived from the key. `cp_prove` sets the pointer at set-up. At
// a request's start it sets the counter to that request's start value. At the request's
// end it compares what was added with 2 * CE * ns summed over the request's sub-proofs,
// since each pad is checked twice (Prover::finish_request(), prover/prover/request.cuh).
static unsigned long long *slot_pads_agreeing = nullptr;
// From here on, every host function that launches a kernel reads the launch status after
// its last launch and before it returns, and CK() exits on an error. A failed launch leaves
// its output buffer unchanged, so older content would be taken for a row combination.
// The read finds an error that occurs before the kernel starts; it does not
// wait for the kernel. A test that is not part of this repository checks that the reads exist.
// Public sources: [CUDA guide], "Error Checking", and the comment before fill_again().
// The rule for the whole process, with further sources: prover/docs/cuda-status.md, "The rule".

/* While this flag is set, a tree is set up without room for its codewords and without the hash's
   scratch space, so its cw, cv0 and cv1 pointers stay null. The flag is inline so that every
   translation unit shares one copy of it: `Tree::alloc`, which reads it, is a single function
   shared by all of them. */
inline bool g_no_code_words = false;
// NoCodeWords sets the flag for a scope and then restores the earlier value. Nothing creates one.
struct NoCodeWords {
  const bool before;
  NoCodeWords() : before(g_no_code_words) { g_no_code_words = true; }
  ~NoCodeWords() { g_no_code_words = before; }
};
// While this flag is set, a TreeN over one row of elements of E or of 64-bit cells is set up
// without layE or layF. The caller stores the cells elsewhere. NoCells sets the flag for a scope;
// nothing creates one.
inline bool g_no_cells = false;
struct NoCells {
  const bool before;
  explicit NoCells(bool none = true) : before(g_no_cells) { g_no_cells = none; }
  ~NoCells() { g_no_cells = before; }
};
// While a ShapeOnly object exists, a row commitment that is set up gets its numbers (shape, rows,
// streams, tables) and no buffer. This is for an object that is needed only for its numbers and
// that no function of this file commits, such as the objects ZZ and BZ of prover/hiding/tail_f.cuh,
// whose cells ch::moment() commits from other buffers. The caller must neither draw its random
// values nor commit it.
static bool g_shape_only = false;
struct ShapeOnly {
  const bool before = g_shape_only;
  ShapeOnly() { g_shape_only = true; }
  ~ShapeOnly() { g_shape_only = before; }
  ShapeOnly(const ShapeOnly &) = delete;
};
/* A row commitment. Kind 0 (G) has rows of 16-bit cells, kind 1 (U) rows over E, kind 2 (M) no data
   rows, kind 3 (F) rows of 64-bit cells, each any element of F. The rows of cw: R data rows, CE
   for the blinding row B (if hasB), CE for each of nh rows h_i (pads or blinding rows), the zero
   fill and, last, nsalt salt rows: RR rows, a multiple of 8. cw is null under g_no_code_words.
   `mask`, the salt rows of cw and TreeN's `cR` are allocated and neither written nor read; in
   cp_prove, as FORK memory, they are zeroed with all else at the end (ForkMemory::wipe()). */
struct Tree {
  bool shape_only = false;
  std::string nm;
  int kind = 0;
  size_t R = 0, K = 0;
  int a = 1, nh = 0;
  bool hasB = false;
  int nsalt = 0;
  Shape sh;
  u32 stream = 0;
  // The number of the tree's kind in its hash calls (the TREE_ constants of prover/common/hash.h).
  // prover/prover/inputs.cuh sets it for the tree `CP`; nothing reads it.
  u32 tnum = TREE_NONE;
  size_t nx = 0, RR = 0, nsent = 0;
  u64 *mask = nullptr, *ext = nullptr, *cw = nullptr;
  u32 *tree = nullptr;
  const u64 *tw = nullptr;
  u32 *cv0 = nullptr, *cv1 = nullptr;
  // True if the tree was set up under g_no_code_words. No function that reads or writes cw may be
  // called for such a tree; only alloc() tests this flag.
  bool no_code_words = false;
  // Allocates buffers and tables and writes the zero fill.
  void alloc() {
    nx = CE * ((hasB ? 1 : 0) + (size_t)nh);
    nsent = R + nx + nsalt;
    RR = (nsent + FILLC - 1) / FILLC * FILLC;
    if (sh.ln < 11 || sh.ln > 24) {
      fprintf(stderr, "hz: tree %s: code length 2^%d is outside the supported range 2^11 to 2^24\n", nm.c_str(), sh.ln);
      exit(2);
    }
    // A row with its blinding coordinates must fit half the code length: an encoder that takes the upper half of its
    // input as zero would silently drop a longer row's blinding coordinates. Nothing in this file encodes.
    if (2 * sh.Kc > (size_t)1 << sh.ln) {
      fprintf(stderr, "hz: internal error: tree %s: a row of %zu entries is longer than half the code length 2^%d\n",
              nm.c_str(), sh.Kc, sh.ln);
      exit(4);
    }
    const size_t n = sh.n, ch = (RR + 127) / 128;
    tw = row_tree::tables(sh.ln);
    shape_only = g_shape_only;
    if (shape_only) return;
    if (R) hz::alloc(mask, mask_cells() * 8, hz::FORK);
    hz::alloc(ext, ext_cells() * 8, hz::FORK);
    no_code_words = g_no_code_words;
    if (!no_code_words) hz::alloc(cw, RR * n * 8, hz::FORK);
    hz::alloc(tree, n * 8 * DW, hz::FORK);
    if (ch > 1 && !no_code_words) {
      hz::alloc(cv0, ch * n * 4 * DW, hz::FORK);
      hz::alloc(cv1, (ch / 2 + 2) * n * 4 * DW, hz::FORK);
    }
    if (RR > nsent && !no_code_words) CK(cudaMemset(cw + zero_row() * n, 0, (RR - nsent) * n * 8));
  }
  /* The rows of cw after the R + nx codeword rows. The zero fill comes before the salt, so that the
     salt symbols are the last of a column's hash input, which ends at row RR, a multiple of 8. */
  size_t salt_row() const { return RR - (size_t)nsalt; }
  size_t zero_row() const { return R + nx; }
  void setup(const std::string &n_, int kind_, size_t R_, size_t K_, int a_, int nh_, u32 stream_, int bits) {
    nm = n_;
    kind = kind_;
    R = R_;
    K = K_;
    a = a_;
    nh = nh_;
    stream = stream_;
    hasB = kind == 0;
    nsalt = kind == 2 ? SALTW / 2 : 0;
    sh = pad_tree_shape(K, a, bits);
    alloc();
  }
  // Like setup(), but the caller gives the shape sh_, hasB_ and nsalt_.
  void setup_as(const std::string &n_, int kind_, size_t R_, size_t K_, int a_, int nh_, bool hasB_, int nsalt_,
                const Shape &sh_, u32 stream_) {
    nm = n_;
    kind = kind_;
    R = R_;
    K = K_;
    a = a_;
    nh = nh_;
    stream = stream_;
    hasB = hasB_;
    nsalt = nsalt_;
    sh = sh_;
    alloc();
  }
  void release() {
    cudaFree(mask);
    cudaFree(ext);
    cudaFree(cw);
    cudaFree(tree);
    cudaFree(cv0);
    cudaFree(cv1);
    mask = ext = cw = nullptr;
    tree = cv0 = cv1 = nullptr;
  }
  // Public source: Dalal, Hemo, Rabinovich, Rothblum, "VEIL: Lightweight Zero-Knowledge for
  // Hash-Based Multilinear Proof Systems", Cryptology ePrint Archive 2026/683 (version of April 13,
  // 2026). The blinding coordinates follow its section 1.2 ("Trace Queries") and step 2 of its
  // Figures 1 and 8. Here a row has sh.mk of them: in the shapes that this file computes, at least
  // a * t + 4.
  // B is a blinding row as in VEIL's section 1.2 ("The Proximity Test") and step 3 of its Figure 8.
  // A TreeN's row h follows VEIL's Figure 2.
  // Draws a sub-proof's ns slot pads, the only used part of its pad commitment S (a Tree
  // of kind M). They are one-time pads and follow VEIL, Figure 7 (cited above).
  // Component kappa of pad j is cell kappa * ns + j of stream + 1, at h_plane(0, kappa)[j];
  // ns <= sh.Kc, a plane's length, is not checked. Nothing else of S is drawn, neither its
  // row h_1 nor its salt; no tree is built.
  void slot_pads(size_t ns) {
    const lost_coins_door::Draw lost = lost_coins_door::a_draw(nm); // fault injection; test builds only
    for (int kappa = 0; kappa < CE; kappa++) {
      fill((u64 *)h_plane(0, kappa), stream + 1, (u64)kappa * ns, ns);
      lost_coins_door::strike(lost, lost_coins_door::ROWS, (u64 *)h_plane(0, kappa), (u64)kappa * ns, ns);
    }
    // The drawn values are not checked here: slot_pads_held() checks by cell
    // numbers that they were drawn, and slot_pads_agree() compares the values.
  }
  // Exits unless, by cell numbers, the current context has drawn all ns slot pads.
  // Call it before the pads are committed to (ch::host.first(), prover/hiding/moments.cuh).
  void slot_pads_held(size_t ns) const { hold_drawn(stream + 1, 0, (u64)CE * ns); }
  // Checks a sub-proof's slot pads by value. The first argument is the sub-proof's array of
  // the ns pads as elements of E, in GPU memory. It is copied from the planes h_plane(0, kappa),
  // which nothing else reads. The array masks the slots' values, and the segment's mask row is
  // made from it: ChainRun::first() (prover/prover/chain_run.cuh) fetches it to the host and
  // commits to what it fetched.
  //
  // If a faulty GPU skips all or part of a draw or a copy, the cells contain zeros or the
  // previous sub-proof's pads. The slots' values can then be read from the proof file, which a
  // verifier may even accept. Neither the ledger nor the launch status detects that. So a second
  // launch derives every cell again from the key, stores none, and adds the number of equal cells
  // to *slot_pads_agreeing (see there). Nothing is decided here: the request's end compares the sum
  // with the expected number, so a request also gets no proof file if this launch did not run.
  //
  // Call it twice per sub-proof, on the same CUDA stream. First right after the commitment:
  // ch::kept_pads_agree() (prover/hiding/moments.cuh) does that. Then after the pads' last use,
  // before hd::close() (prover/hiding/door.cuh). The caller must not write to the array after the
  // copy: then a cell that was wrong at any use is still wrong at the second check.
  // This check misses an error that both derivations share (a wrong key or stream):
  // ch::masking_pads_hold() (prover/hiding/moments.cuh) compares the same array with a second
  // implementation, which has its own copy of the key and its own stream numbers. A process key
  // that is wrongly derived or given twice passes both checks, and so does a wrong request number,
  // version word or ns: `cp_prove` gives the draw and the second implementation the same ones
  // (further limits: SlotPadsSecond::hold(), prover/prover/slot_pads_second.cuh). Neither check
  // sees an error in the host's fetched copy. After such an error the proof is invalid. Its scalars
  // are still sent under the array's pads, but it is not shown that the invalid proof as a whole
  // gives away no pad.
  // No check in cp_prove notices this: only the verifier program `cp_verify` rejects such a file.
  // This is a limit on hiding under a fault of the machine or the program, and no test here
  // shows how much such a file gives away.
  // For the ledger the check is a repeat (enter_draw_again()).
  void slot_pads_agree(const T *laid, size_t ns) const {
    if (!slot_pads_agreeing) {
      fprintf(stderr, "hz: internal error: tree %s: no word is set up to count the slot pads that agree\n", nm.c_str());
      exit(4);
    }
    CoinK ck = coink(stream + 1);
    enter_draw_again(stream + 1, 0, (u64)CE * ns);
    if (!lost_copies_door::left_out("slot_pads_test")) // fault injection; test builds only
      k_slot_pads_agree<<<dim3((unsigned)((ns / 4 + 2 + 255) / 256), CE), 256>>>(laid, ck, stream + 1, ns,
                                                                                slot_pads_agreeing);
    forget(ck);
    CK(cudaGetLastError());
  }
  /* B and the h_i are rows over E: pads or blinding rows, which follow VEIL (cited above).
     Each is CE planes, `plane_stride` cells apart: plane c has component c of each coordinate. */
  size_t plane_stride() const { return sh.Kc; }
  // Numbers of cells of mask and of ext.
  size_t mask_cells() const { return R * (size_t)sh.mk; }
  size_t ext_cells() const { return nx * sh.Kc; }
  const u64 *h_plane(int i, int c) const { return ext + (size_t)(CE * ((hasB ? 1 : 0) + i) + c) * sh.Kc; }
};
// The shape of a TreeN, an R x K table's row commitment. A code row interleaves 2^x table rows in
// one block of Kpp table columns: Kp = 2^x * Kpp data, then mk blinding coordinates. w, ln: log2 of
// the code's dimension and length n. t: see tcol(). `bytes`: what the shape is chosen by.
struct Shape2 {
  int x = 0, w = 0, t = 0, mk = 0, ln = 0;
  size_t P2 = 0, Kp = 0, Kpp = 0, Kc = 0, R2 = 0, NHr = 0, NH = 0, Rd = 0, Rr = 0, n = 0, bytes = 0;
  // P2: mk rounded up to a power of 2; Kp = 2^w - P2; Kc = Kp + mk, a row's length before encoding.
  // R2 = R / 2^x: the row groups, of 2^x interleaved table rows each. NHr: blocks of Kpp columns
  // that cover K. Rr: base-field data rows, of which Rd are dummy rows (zero data). shapeF()'s
  // `bytes` omits the two sent rows: see shape2().
  // Table cell (r, c) is coordinate colp of code row rc: index rc * Kp + colp in the "committed
  // layout". rowp: the row's number in the padded index space (NH: NHr rounded up to a power of 2).
  // Public source: a layout written as a small function from (row, column) to a place in memory is
  // the layout concept of NVIDIA's CUTLASS [CUTLASS layouts]; here a member function of the shape.
  // Public source: committing only to the data that exists, in an index space padded with zeros
  // that are never stored, is described by [Hemo] as "virtual" zero-padding (section 1) and as
  // "free padding" (footnote 4); here every row group and row has the same shape, so one formula
  // gives both numbers of a row.
  __host__ __device__ void place(size_t r, size_t c, size_t &rowp, size_t &colp, size_t &rc) const {
    const size_t rhi = r >> x, rlo = r & (((size_t)1 << x) - 1), hi = c / Kpp, cm = c % Kpp;
    rowp = rhi * NH + hi;
    colp = (cm << x) + rlo;
    rc = rhi * NHr + hi;
  }
};
// The shape for w and code rate 2^-ir; false if there is none. Of the powers 2^x <= P2 that divide
// R it takes the one with the fewest data rows, the smaller at a tie; Rd dummy rows bring these to
// at least 8. ec: base-field rows per code row. a: openings that mk is sized for. hb: see tcol().
static bool shapeF(size_t R, size_t K, int ec, int a, int hb, int ir, int w, Shape2 &s) {
  s = Shape2();
  s.w = w;
  s.ln = w + ir;
  if (s.ln + 1 > 61 || s.ln + hb > EBITS || s.ln + 1 > 32) return false;
  // The maximum code length is 2^ELNO. A longer code only rules out this w;
  // it does not by itself reject the parameters.
  if (s.ln > ELNO) return false;
  s.n = (size_t)1 << s.ln;
  s.t = tcol((((u64)1 << ir) + 1) * ((u64)1 << w) + 1, w + ir + 1, hb);
  s.mk = (a * s.t + 5) & ~1;
  s.P2 = p2c((size_t)s.mk);
  if (2 * s.P2 > ((size_t)1 << w)) return false;
  s.Kp = ((size_t)1 << w) - s.P2;
  s.Kc = s.Kp + (size_t)s.mk;
  size_t least = 0;
  bool any = false;
  for (int x = 0; ((size_t)1 << x) <= s.P2 && ((size_t)1 << x) <= R; x++) {
    if (R % ((size_t)1 << x)) continue;
    const size_t Kpp = s.Kp >> x, NHr = (K + Kpp - 1) / Kpp, dr = (size_t)ec * (R >> x) * NHr;
    if (any && dr >= least) continue;
    any = true;
    least = dr;
    s.x = x;
    s.Kpp = Kpp;
    s.R2 = R >> x;
    s.NHr = NHr;
    s.NH = p2c(NHr);
  }
  if (!any || least + 8 > ((size_t)1 << 32)) return false;
  s.Rd = least >= 8 ? 0 : 8 - least;
  s.Rr = least + s.Rd;
  s.bytes = (size_t)s.t * (8 * (s.Rr + 2 * CE) + SH2_DIGB * (size_t)s.ln);
  return true;
}
// FlatRule's rule for w: the smallest w for which shapeF() gives at most rw data rows, not counting
// dummy rows, or, if there is none, the smallest w that gives the fewest.
static Shape2 shape_flat(size_t R, size_t K, int ec, int a, int hb, int ir, size_t rw) {
  Shape2 s, fewest;
  bool any = false;
  for (int w = 10; w <= EBITS - 1 - hb && w <= 29; w++) {
    if (!shapeF(R, K, ec, a, hb, ir, w, s)) continue;
    if (s.Rr - s.Rd <= rw) return s;
    if (!any || s.Rr - s.Rd < fewest.Rr - fewest.Rd) fewest = s;
    any = true;
  }
  if (!any) {
    fprintf(stderr, "hz::shape_flat: no shape meets the constraints\n");
    exit(2);
  }
  return fewest;
}
// Parameters: see shapeF(). If flat_rule.rw is not 0, returns shape_flat()'s choice; otherwise the
// w and x with the smallest estimated proof size (`bytes`), at rate 1/2.
static Shape2 shape2(size_t R, size_t K, int ec, int a, int hb) {
  if (flat_rule.rw) return shape_flat(R, K, ec, a, hb, flat_rule.ir, flat_rule.rw);
  Shape2 best;
  bool any = false;
  for (int w = 10; w <= EBITS - 1 - hb && w <= 29; w++)
    for (int x = 0; ((size_t)1 << x) <= R; x++) {
      if (R % ((size_t)1 << x)) continue; // 2^x must divide R, which need not be a power of two
      Shape2 s;
      s.w = w;
      s.x = x;
      s.n = (size_t)1 << (w + 1);
      s.t = tcol(3 * ((u64)1 << w) + 1, w + 2, hb);
      s.mk = (a * s.t + 5) & ~1;
      s.P2 = p2c((size_t)s.mk);
      if (2 * s.P2 > ((size_t)1 << w) || ((size_t)1 << x) > s.P2) continue;
      s.Kp = ((size_t)1 << w) - s.P2;
      s.Kc = s.Kp + (size_t)s.mk;
      s.Kpp = s.Kp >> x;
      s.R2 = R >> x;
      s.NHr = (K + s.Kpp - 1) / s.Kpp;
      s.NH = p2c(s.NHr);
      const size_t dr = (size_t)ec * s.R2 * s.NHr;
      s.Rd = dr >= 8 ? 0 : 8 - dr;
      s.Rr = dr + s.Rd;
      s.ln = w + 1;
      // Two sent rows of Kc elements. Per column: 8 bytes per base-field row (Rr data rows, dummy
      // rows included; CE each for B and h) and a path of w + 1 digests, at SH2_DIGB bytes each.
      s.bytes = 2 * sizeof(T) * s.Kc + (size_t)s.t * (8 * (s.Rr + 2 * CE) + SH2_DIGB * (size_t)(w + 1));
      if (!any || s.bytes < best.bytes) {
        best = s;
        any = true;
      }
    }
  if (!any) {
    fprintf(stderr, "hz::shape2: no shape meets the constraints\n");
    exit(2);
  }
  return best;
}
constexpr int RD_MOST = 9; // maximum number of vectors per call of `rowdotS`
// The table columns on which k_rowdot() reads each vector L[i], as ns ranges ("segments" in
// messages): range s is the columns [c0[s], c0[s] + len[s]) of L[v[s]].
struct RdArg {
  const T *L[RD_MOST] = {};
  int ns = 0;
  int v[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t c0[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         len[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};
// Row dot: out[i * R + r] = sum of L[i][c] * cell(r, c) over the columns c in vector i's ranges,
// for r < R. `lay`: the table's cells in the committed layout.
// Launch: 256 threads per block; blockIdx.x covers r, blockIdx.y is i.
// Public source: one thread for each element of the product: [Nugteren 1]. Here M is the rows r,
// N is the V vectors and K is the columns c. Rows with the same rhi are neighbours in `lay`.
// Summing over a vector's ranges only has public sources of its own: see Ent::c0, c1 and Ent::sv
// in prover/hiding/eqs.cuh.
__global__ void k_rowdot(const u16 *__restrict__ lay, int x, size_t Kpp, size_t Kp, size_t NHr, size_t R, const RdArg A,
                         T *__restrict__ out) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x, i = blockIdx.y;
  if (r >= R) return;
  const size_t rhi = r >> x, rlo = r - (rhi << x);
  T sum = F::zero();
  for (int s = 0; s < A.ns; s++) {
    if (A.v[s] != (int)i) continue;
    size_t hi = A.c0[s] / Kpp, cm = A.c0[s] - hi * Kpp;
    for (size_t c = A.c0[s]; c < A.c0[s] + A.len[s]; c++) {
      sum = F::add(sum, F::muls(A.L[i][c], lay[(rhi * NHr + hi) * Kp + (cm << x) + rlo]));
      if (++cm == Kpp) cm = 0, hi++;
    }
  }
  out[i * R + r] = sum;
}
// A row commitment to a table: R x K 16-bit cells (ec = 1), one row of K elements of E (ec = CE,
// R = 1) or, if `wide`, one row of K 64-bit cells, each any canonical base-field element (ec = 1,
// R = 1). `I` is the row commitment to the shape's Rr rows.
struct TreeN {
  std::string nm;
  size_t R = 0, K = 0;
  int ec = 1, a = 1, nh = 1;
  bool wide =
      false; // ec = 1 with 64-bit cells: the shape is that of ec = 1, the cells are in layF
  Shape2 sh;
  u32 stream = 0;
  size_t nrc = 0, RR = 0, nsent = 0;
  Tree I;
  u16 *lay = nullptr;
  u64 *layF = nullptr;
  T *layE = nullptr, *cR = nullptr;
  static constexpr int VMAX = 4;
  // Checks and stores the parameters, derives the shape and, unless the tree is shape-only (see
  // ShapeOnly), allocates the GPU buffers, `lay` zero-filled; exits with a message on an invalid
  // parameter. nh_ = 0: the blinding row B but no row h, so h_plane() must not be used
  // (not checked). bits: shape2()'s hb. wide_: see `wide`; the caller writes the K cells to layF.
  void setup(const std::string &n_, size_t R_, size_t K_, int ec_, int a_, u32 stream_, int bits, int nh_ = 1,
             bool wide_ = false) {
    nm = n_;
    wide = wide_;
    R = R_;
    K = K_;
    ec = ec_;
    a = a_;
    stream = stream_;
    nh = nh_;
    // Valid row counts: 1 to 256, or a larger multiple of 256.
    if (!R || (R > 256 && R % 256)) {
      fprintf(stderr, "hz: tree %s: the row count %zu is neither at most 256 nor a multiple of 256\n", nm.c_str(), R);
      exit(2);
    }
    if (ec != 1 && ec != CE) {
      fprintf(stderr, "hz: tree %s: ec = %d is neither 1 nor the field's degree\n", nm.c_str(), ec);
      exit(2);
    }
    if (ec == CE && R != 1) {
      fprintf(stderr, "hz: tree %s: rows of elements need R = 1\n", nm.c_str());
      exit(2);
    }
    if (wide && (ec != 1 || R != 1)) {
      fprintf(stderr, "hz: tree %s: 64-bit cells need ec = 1 and R = 1\n", nm.c_str());
      exit(2);
    }
    sh = shape2(R, K, ec, a, bits);
    nrc = sh.R2 * sh.NHr;
    if (sh.x > 13) {
      fprintf(stderr, "hz: tree %s: shape exponent x = log2(R / R2) = %d exceeds the maximum 13\n", nm.c_str(), sh.x);
      exit(2);
    }
    Shape s;
    s.ln = sh.ln;
    s.t = sh.t;
    s.mk = sh.mk;
    s.n = sh.n;
    s.Ks = sh.Kp;
    s.Kc = sh.Kc;
    // Each column ends in SALTW / 2 salt symbols (the number passed to setup_as()),
    // which stream + 2 of the tree would give.
    I.setup_as(nm, wide ? 3 : ec == 1 ? 0 : 1, sh.Rr, sh.Kp, a, nh, true, SALTW / 2, s, stream);
    RR = I.RR;
    nsent = I.nsent;
    if (I.shape_only) return; // a shape-only tree needs just its numbers, no device memory
    if (wide) {
      if (!g_no_cells) hz::alloc(layF, K * 8, hz::FORK);
    } else if (ec == 1) {
      hz::alloc(lay, sh.Rr * sh.Kp * 2, hz::FORK);
      CK(cudaMemset(lay, 0, sh.Rr * sh.Kp * 2));
    } else if (!g_no_cells)
      hz::alloc(layE, K * sizeof(T), hz::FORK);
    hz::alloc(cR, (size_t)VMAX * sh.Rr * sizeof(T), hz::FORK);
  }
  void release() {
    I.release();
    cudaFree(lay);
    cudaFree(layF);
    cudaFree(layE);
    cudaFree(cR);
    lay = nullptr;
    layF = nullptr;
    layE = cR = nullptr;
  }
  // Runs k_rowdot() with one range [c0[i], c1[i]) per vector. L: host array of V <= RD_MOST device
  // pointers; L[i] is the address that element 0 of a vector of K elements would have, but only the
  // elements in the range are read. c0, c1: host arrays. Call it after `lay` has been filled.
  // Only a test that is not part of this repository calls it or rowdotS().
  void rowdot(const T *const *L, const size_t *c0, const size_t *c1, int V, T *out) {
    int sv[RD_MOST];
    for (int i = 0; i < RD_MOST; i++) sv[i] = i;
    rowdotS(L, V, out, V, sv, c0, c1);
  }
  // As rowdot(), with NS <= 16 ranges, given in host arrays: range s is the columns
  // [c0[s], c1[s]) of L[sv[s]]. A vector may have several ranges. They must not overlap, which is
  // not checked: k_rowdot() would count a shared column twice.
  // rowdot() gives one range per vector, so its ranges cannot overlap.
  void rowdotS(const T *const *L, int V, T *out, int NS, const int *sv, const size_t *c0, const size_t *c1) {
    if (ec != 1 || wide || V < 1 || V > RD_MOST || NS < 0 || NS > 16) {
      fprintf(stderr,
              "hz: TreeN::rowdotS: need ec = 1 (16-bit cells), 1 <= V <= %d and 0 <= NS <= 16, got ec = %d, V = "
              "%d, NS = %d\n",
              RD_MOST, ec, V, NS);
      exit(2);
    }
    RdArg A;
    for (int i = 0; i < V; i++) A.L[i] = L[i];
    A.ns = NS;
    for (int s = 0; s < NS; s++) {
      if (sv[s] < 0 || sv[s] >= V || c1[s] > K || c0[s] > c1[s]) {
        fprintf(stderr, "hz: TreeN::rowdotS: segment %d is out of range\n", s);
        exit(2);
      }
      A.v[s] = sv[s];
      A.c0[s] = c0[s];
      A.len[s] = c1[s] - c0[s];
    }
    k_rowdot<<<dim3((unsigned)((R + 255) / 256), (unsigned)V), 256>>>(lay, sh.x, sh.Kpp, sh.Kp, sh.NHr, R, A, out);
    CK(cudaGetLastError());
  }
  size_t plane_stride() const { return I.plane_stride(); }
  const u64 *h_plane(int c) const { return I.h_plane(0, c); }
};
} // namespace hz
