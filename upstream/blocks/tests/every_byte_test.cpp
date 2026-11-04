// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests that the toy verifiers (not the verifier program cp_verify) leave no byte of a complete
// proof unchecked. damage() of blocks/tests/damage.h flips single bits, replaces eight bytes by a
// 64-bit word at or above the field modulus p, cuts the proof short and appends bytes. A verifier
// must accept each honest proof and return false for every damaged copy; a crash fails the test
// too. The proofs are those of blocks/toy/product.h and blocks/toy/segments.h (printed: links), at
// the low error target of small_choice() (blocks/tests/whole_runs.h), 32 bits.
// usage: every_byte_test [bits [stride]]   bits: bits flipped in each byte, one at a time (1 to 8);
//        stride: step between the bytes and lengths tried (1 if only bits is given).
//        BLK_TEST_JOBS: child processes, default 4, capped at the CPUs available.
// `every_byte_test 8 1` flips every bit of every byte, so also bits that a parser might mask off:
// millions of verifications, run by tests/new/run_new.py, stage `bytes`, for its first build
// setting only. Without arguments, as `make test` runs it, the program flips one bit of every 13th
// byte.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tests/whole_runs.h"

using namespace blk;

int main(int argc, char **argv) {
  DamageChoice how;
  how.bits = argc > 1 ? (unsigned)std::atoi(argv[1]) : 1;
  how.stride = argc > 2 ? (size_t)std::atoi(argv[2]) : argc > 1 ? 1 : 13;
  if (how.bits < 1 || how.bits > 8 || how.stride < 1) {
    std::printf("FAILED: usage: every_byte_test [bits 1..8 [stride]]\n");
    return 2;
  }
  // Reader::get() (blocks/iop/message.h) must reject the word 5 + p, a non-canonical encoding of 5
  // (printed: second spelling). Since p = 2^64 - 2^32 + 1, only elements below 2^32 - 1 have one,
  // a random element with probability about 2^-32, so the proofs below may offer none.
  {
    u8 five_and_p[F_BYTES];
    word_to_bytes(5 + P, five_and_p);
    Reader r(five_and_p, F_BYTES);
    F x;
    if (r.get(x)) {
      std::printf("ACCEPTED A SECOND SPELLING: the reader of messages takes the word 5 + p for an element\n");
      return 1;
    }
  }
  size_t proofs = 0;
  DamageCount all;
  for (size_t which = 0; which < WHOLE_RUNS; which++) {
    const std::unique_ptr<WholeRun> run = whole_run(which, small_choice());
    std::string no;
    if (!run->planned(no)) {
      std::printf("FAILED: %s: no such parameters: %s\n", run->name(), no.c_str());
      return 1;
    }
    // prove()'s arbitrary `key` fixes witness and randomness: a failure can be repeated.
    const std::vector<u8> proof = run->prove(40);
    const char *why = nullptr;
    if (!run->verify(proof, &why)) {
      std::printf("FAILED: %s: the honest proof is refused: %s\n", run->name(), why);
      return 1;
    }
    proofs++;
    all += damage(std::string("a proof of ") + run->name(), proof, how, [&](const u8 *bytes, size_t n) { return run->verify(bytes, n); });
    if (all.bad) return 1;
  }
  if (how.only == DamageChoice::WORDS)
    std::printf("refused: %zu proofs with a word at or above p in the place of eight bytes, %zu of them w + p\n", all.words, all.spellings);
  else if (how.only == DamageChoice::CUTS)
    std::printf("refused: %zu proofs cut short\n", all.cuts);
  else
    std::printf("every byte: ok: %zu honest proofs accepted; refused: %s\n", proofs, all.text(how).c_str());
  return 0;
}
