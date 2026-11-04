// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests `cp_with_configured` (verifier/src/cp_configured.h) from a program that links the
   verifier's code (cp_verify).
   usage: cp_configured_test <config folder> <bits> [twice]
   From the folder's policy.txt, layout.txt and KL.bin the function builds a `blk::Configured`
   (shapes and round plans), checks its plans' error accounting against <bits> (`asked_bits`) and
   passes it to a callback. The callback prints the configuration digest (a hash of these files),
   the number of hash chains (Fiat-Shamir transcripts) and rounds planned for the key file, the
   leaf file (`entry`) and the proof file, and three sizes. The exit code is 0 only if the call
   returns the callback's value, 7. With `twice` the callback calls the function again, which must
   end the process with "NO: internal-configured-twice". `cp_verify -plan accounting` on the same
   folder prints the same digest ("configuration="), but no test compares the two.
   Build: link with verifier/src/cp_verify.cpp built with -DCP_VERIFY_AS_A_UNIT and the same field
   and digest flags. The row `build.verifier.unit` of SUITE.md has the build commands. No row of
   SUITE.md runs this program. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "cp_configured.h"

static std::string bytes_of(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    printf("cannot read %s\n", path.c_str());
    exit(2);
  }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

int main(int argc, char **argv) {
  if (argc != 3 && !(argc == 4 && !strcmp(argv[3], "twice"))) return printf("usage: cp_configured_test <config folder> <bits> [twice]\n"), 2;
  const std::string folder = argv[1], policy = bytes_of(folder + "/policy.txt"), layout = bytes_of(folder + "/layout.txt"), kl = bytes_of(folder + "/KL.bin");
  std::vector<uint32_t> words(kl.size() / 4);
  memcpy(words.data(), kl.data(), 4 * words.size());
  const double bits = atof(argv[2]);
  const bool twice = argc == 4;
  const int status = cp_with_configured(policy, layout, words, bits, true, [&](const blk::Configured &c) {
    if (!c.refused().empty()) return printf("a refused Configured was handed over: %s\n", c.refused().c_str()), 1;
    blk::u8 by[blk::DIGEST_BYTES];
    blk::digest_to_bytes(c.configuration(), by);
    printf("configuration=");
    for (blk::u8 x : by) printf("%02x", x);
    printf("\n");
    for (const blk::FilePlan *f : {&c.key(), &c.entry(), &c.proof()}) {
      size_t rounds = 0;
      for (const blk::ChainPlan &chain : f->chains) rounds += chain.plan->rounds_.size();
      printf("plan: %s: %zu chains, %zu rounds\n", f->name.c_str(), f->chains.size(), rounds);
    }
    printf("matrix rows %llu, links %zu, pending rows %llu\n", (unsigned long long)c.sized().once.S.matrix_rows, c.fixed().shape.link.size(), (unsigned long long)c.fixed().pending_rows);
    if (twice) return cp_with_configured(policy, layout, words, bits, true, [](const blk::Configured &) { return 0; });
    return 7; // arbitrary; `cp_with_configured` must return it
  });
  printf("returned %d\n", status);
  return status == 7 ? 0 : 1;
}
