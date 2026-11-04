// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the toy segment chain of blocks/toy/segments.h at a low target for a short proof: 32 bits
// a round, not 128, and where positions are drawn 8 of the 32 are proof of work.
// Segment l (`link` in identifiers and output) claims c_l = x_l * b_l entry by entry;
// x_{l+1} = c_l.
// Argument: <stride> (default 61; must be positive, unchecked). The honest proof
// must be accepted, and rejected (`refused` in output) after one bit flip (at every stride-th
// byte), a truncation (at evenly spaced lengths) or an appended byte.
// The prover here is honest: this test does not show that the verifier rejects one that cheats.
// One false proof of this chain at a smaller size, made by a test prover that is not part of this
// repository, is stored in examples/protocol-toy-false-files/, for a verifier in Python alone.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/engine_cpu.h"
#include "toy/segments.h"
#include "toy/toy_coins.h"

using namespace blk;

int main(int argc, char **argv) {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 32;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  // log2 of the lengths: message 10, codeword 13, mask codeword 9
  const SegmentsRun run = segments_size(set, 3, 10, 13, 9, {StageChoice{1, 1, 12, 10}, StageChoice{0, 0, 12, 12}});
  if (run.plan->refused()) {
    std::printf("FAILED: no such parameters: %s\n", run.plan->why_not().c_str());
    return 1;
  }
  ToyCoins coins(31);
  std::vector<F> x(size_t(1) << run.vars());
  std::vector<std::vector<F>> b(run.links(), x);
  coins.fill(x.data(), x.size());
  for (std::vector<F> &v : b) coins.fill(v.data(), v.size());
  EngineCpu engine;
  Writer writer;
  segments_prove(run, x, b, engine, coins, writer);
  std::vector<u8> proof = writer.bytes();
  const char *why = nullptr;
  const bool accepted = segments_verify(run, proof.data(), proof.size(), &why);
  if (!accepted) {
    std::printf("FAILED: the honest proof is refused: %s\n", why);
    return 1;
  }
  const size_t stride = argc > 1 ? (size_t)std::atoi(argv[1]) : 61;
  size_t flips = 0, cuts = 0;
  for (size_t i = 0; i < proof.size(); i += stride, flips++) {
    const u8 bit = (u8)(1u << (i / stride % 8));
    proof[i] ^= bit;
    if (segments_verify(run, proof.data(), proof.size())) {
      std::printf("ACCEPTED A DAMAGED PROOF: bit %u of byte %zu flipped\n", bit, i);
      return 1;
    }
    proof[i] ^= bit;
  }
  for (size_t len = 0; len < proof.size(); len += 1 + proof.size() / 149, cuts++)
    if (segments_verify(run, proof.data(), len)) {
      std::printf("ACCEPTED A PROOF CUT TO %zu BYTES\n", len);
      return 1;
    }
  proof.push_back(0);
  if (segments_verify(run, proof.data(), proof.size())) {
    std::printf("ACCEPTED A PROOF WITH A BYTE ADDED\n");
    return 1;
  }
  std::printf("three links: ok: %zu bytes accepted; %zu flips, %zu cuts and one added byte refused\n", proof.size() - 1, flips, cuts);
  return 0;
}
