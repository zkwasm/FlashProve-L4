// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Writes the parameter bytes of the first queries of a proof file's three hash chains (Fiat-Shamir
// transcripts: first, shared, second) for one configuration (policy.txt, layout.txt, KL.bin). A
// first query is the hash input that starts a chain; its parameters are the items fixed before any
// key file or request is known, so this tool reads and checks no key file, leaf file, request or
// proof. docs/format/read_proof.py, which reads a proof file following
// docs/format/FORMAT.md, takes these bytes as given.
//   first_queries_tool <config folder> <asked bits> <folder that stands> [<stem>]
// writes <stem>.first.parameters (default stem: example), shared.parameters and second.parameters
// into the existing directory <folder that stands>; exit code 0. If an output name exists (also as
// a symbolic link), <stem> contains "/" or <folder that stands> is the empty string, the tool exits
// with code 2: nothing is overwritten, and files that the run has already created stay.
// If cp_with_configured() (verifier/src/cp_configured.h), the verifier's own code, rejects the
// configuration at <asked bits> requested bits, it prints `NO: <name>` and exits with code 1.
// Build from verifier/src: compile cp_verify.cpp with -DCP_VERIFY_AS_A_UNIT into unit.o, then this
// file with -DBLK_FIELD=2 and the unit's hash and digest options as <flags>:
//   g++ -std=c++17 -O2 -fno-ipa-modref -pthread <flags> -I . -I ../../blocks ../tests/first_queries_tool.cpp unit.o
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

#include "cp_configured.h"
#include "frame/check.h"

static std::string fqRead(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fprintf(stderr, "first_queries_tool: cannot read %s\n", path.c_str()), exit(2);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
static void fqWrite(const std::string &path, const std::vector<blk::u8> &b) {
  FILE *const out = fopen(path.c_str(), "wbx"); // x: the file is created here, or the call fails
  if (!out || fwrite(b.data(), 1, b.size(), out) != b.size() || fclose(out)) fprintf(stderr, "first_queries_tool: cannot create or write %s: %s\n", path.c_str(), strerror(errno)), exit(2);
  printf("%s: %zu bytes\n", path.c_str(), b.size());
}
// Fills each `blk::Binder` (blocks/chain/public.h; it collects a first query's items) as
// blk::check_proof() in blocks/frame/check.h does, up to the first statement item, which closes the
// parameters: a `blk::IndexName` (the weight table's root and record) or, in the second chain, the
// digest that seals an earlier chain. The first chain's bytes match a proof only while cp_verify
// passes null to check_proof_if_tested() as the callback `more`, which could add parameters.
int main(int argc, char **argv) {
  if ((argc != 4 && argc != 5) || !*argv[3]) return fprintf(stderr, "usage: first_queries_tool <config folder> <asked bits> <folder that stands> [<stem>]\n"), 2;
  const std::string config = argv[1], to = argv[3], stem = argc == 5 ? argv[4] : "example";
  if (stem.find('/') != std::string::npos) return fprintf(stderr, "first_queries_tool: the stem must not contain \"/\"\n"), 2;
  const std::string kl = fqRead(config + "/KL.bin");
  if (kl.size() % 4) return fprintf(stderr, "first_queries_tool: KL.bin is no whole number of words\n"), 2;
  std::vector<uint32_t> words(kl.size() / 4);
  for (size_t i = 0; i < words.size(); i++)
    words[i] = (uint32_t)(unsigned char)kl[4 * i] | (uint32_t)(unsigned char)kl[4 * i + 1] << 8 | (uint32_t)(unsigned char)kl[4 * i + 2] << 16 | (uint32_t)(unsigned char)kl[4 * i + 3] << 24;
  return cp_with_configured(fqRead(config + "/policy.txt"), fqRead(config + "/layout.txt"), words, atof(argv[2]), true, [&](const blk::Configured &c) {
    {
      blk::Binder q(blk::LABEL_FIRST_CHAIN, c.proof().chains[0].plan);
      (void)blk::bind_request_parameters(q, c);
      fqWrite(to + "/" + stem + ".first.parameters", q.parameter_bytes());
    }
    {
      // bind_shared_chain() without its two statement items, written out because this tool has no
      // `blk::IndexName` to pass. If bind_shared_chain() adds a parameter item, the bytes written
      // here lack it and walker_on_a_world_test.py, which runs this tool, fails. The same holds for
      // the second chain below.
      blk::Binder q(blk::LABEL_PROOF_SHARED_CHAIN, c.proof().chains[1].plan);
      q.configuration(c.fixed().configuration);
      blk::LinPublic x;
      blk::lin_bind_parameters(q, c.fixed().codes, c.fixed().lin, x);
      fqWrite(to + "/shared.parameters", q.parameter_bytes());
    }
    {
      blk::Binder q(blk::LABEL_PROOF_SECOND_CHAIN, c.proof().chains[2].plan);  // bind_second_chain() without the two seals
      q.configuration(c.fixed().configuration);
      fqWrite(to + "/second.parameters", q.parameter_bytes());
    }
    return 0;
  });
}
