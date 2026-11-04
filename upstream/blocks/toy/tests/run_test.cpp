// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// End-to-end test of the toy proof in blocks/toy/product.h (three committed vectors satisfy
// a[i] * b[i] = c[i]), at a low target so that a run is short: 32 bits a round, not 128, and where
// positions are drawn 8 of the 32 are proof of work.
//
// The honest proof must be accepted. Each damaged version must be rejected (`refused`
// in output): a bit flipped in each stride-th byte in turn (run_test <stride>; 1 = every byte;
// default 29; 0 divides by zero, unchecked), the proof cut short at many lengths, a byte appended.
// Every proof here comes from the honest prover, for a true statement (no call passes `false` to
// prove()), so this test does not show that the verifier rejects a prover that cheats, and no
// test in this repository gives such a prover's proof to toy_verify().
// examples/protocol-toy-false-files/ stores one false proof of this protocol at a smaller size
// (one product is false), made by a test prover that is not part of this repository. Its test,
// stored_false_test.py, gives the file to the second verifier, which is written in Python
// (tests/new/second/toy_check.py) and must reject it.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/engine_cpu.h"
#include "toy/product.h"
#include "toy/toy_coins.h"

using namespace blk;

// Vectors of 2^10 entries; first-stage codewords of length 2^13 (main rows) and 2^9 (mask rows).
static ToyRun small_run() {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 32;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  return toy_size(set, 10, 13, 9, {StageChoice{1, 1, 12, 10}, StageChoice{0, 0, 12, 12}});
}

static std::vector<u8> prove(const ToyRun &run, u64 key, bool true_statement) {
  ToyCoins coins(key);
  Table abc(1, 3, run.shape.codes.base[0].ell);
  coins.fill(abc.a.data(), 2 * abc.len);
  for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
  if (!true_statement) abc.set(2, 5, abc.at(2, 5) + E::one());
  EngineCpu engine;
  Writer proof;
  toy_prove(run, abc, engine, coins, proof);
  return proof.bytes();
}

int main(int argc, char **argv) {
  const ToyRun run = small_run();
  if (run.plan->refused()) {
    std::printf("FAILED: no such parameters: %s\n", run.plan->why_not().c_str());
    return 1;
  }
  const size_t stride = argc > 1 ? (size_t)std::atoi(argv[1]) : 29;
  std::vector<u8> proof = prove(run, 20, true);
  const char *why = nullptr;
  if (!toy_verify(run, proof.data(), proof.size(), &why)) {
    std::printf("FAILED: the honest proof is refused: %s\n", why);
    return 1;
  }
  // toy_verify_file() sizes the run from the public file's text, and must reject the proof if that
  // text is altered.
  if (!toy_verify_file(run.shape.text, proof.data(), proof.size(), &why)) {
    std::printf("FAILED: the honest proof is refused by the verifier that has the public file alone: %s\n", why);
    return 1;
  }
  for (size_t i = 0; i < run.shape.text.size(); i++) {
    std::string other = run.shape.text;
    other[i] = other[i] == '1' ? '2' : '1';
    if (toy_verify_file(other, proof.data(), proof.size(), &why)) {
      std::printf("FAILED: accepted under a public file whose character %zu was changed\n", i);
      return 1;
    }
  }
  if (toy_verify_file(run.shape.text + "samples 1\n", proof.data(), proof.size(), &why) ||
      toy_verify_file(run.shape.text.substr(0, run.shape.text.size() - 1), proof.data(), proof.size(), &why)) {
    std::printf("FAILED: accepted under a public file with a line more or without its last line feed\n");
    return 1;
  }
  size_t flips = 0, cuts = 0;
  for (size_t i = 0; i < proof.size(); i += stride, flips++) {
    const u8 bit = (u8)(1u << (i / stride % 8));
    proof[i] ^= bit;
    if (toy_verify(run, proof.data(), proof.size())) {
      std::printf("ACCEPTED A DAMAGED PROOF: bit %u of byte %zu flipped\n", bit, i);
      return 1;
    }
    proof[i] ^= bit;
  }
  for (size_t len = 0; len < proof.size(); len += 1 + proof.size() / 199, cuts++)
    if (toy_verify(run, proof.data(), len)) {
      std::printf("ACCEPTED A PROOF CUT TO %zu BYTES\n", len);
      return 1;
    }
  if (toy_verify(run, proof.data(), proof.size() - 1)) {
    std::printf("ACCEPTED A PROOF WITHOUT ITS LAST BYTE\n");
    return 1;
  }
  proof.push_back(0);
  if (toy_verify(run, proof.data(), proof.size())) {
    std::printf("ACCEPTED A PROOF WITH A BYTE ADDED\n");
    return 1;
  }
  std::printf("whole run: ok: %zu bytes accepted; %zu flips, %zu cuts and one added byte refused\n", proof.size() - 1, flips, cuts + 1);
  return 0;
}
