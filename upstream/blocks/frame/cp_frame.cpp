// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Planning calls: print the error accounting and file lengths that follow from a parameter file.

     cp_frame -plan id      FILE                      only the file's version word and digest
     cp_frame -plan key     FILE [-rounds] [-fields]  the key file (anchor: first committed table)
     cp_frame -plan leaf    FILE [-rounds] [-fields]  a leaf file (leaf table: a further copy;
                                                      output: `entry`)
     cp_frame -plan request FILE [-rounds] [-fields]  a proof file; accounting of all three files

   A call reads no other file and checks no proof. It plans as committer, prover and verifier do
   (`configure`, configure.h), but on `stand_in_subproofs` (files.h), not a model's sub-proofs, and
   with fewer checks: `configure` can reject a file that passes here. The accounting is about
   soundness only. No verifier can tell from a proof whether its parameters hide the weights, so
   `configure` rejects (output: `REFUSE`) a file that misses a condition of hiding_conditions.h, and
   the line "hiding:" reports the first missed (`request` tests all but `range-checks`, `key` and
   `leaf` fewer).
   Exit code, never 0: 4 `id`, or every round meets the file's target, line `sec` (only with proof
   of work if the line "accounting: THE ROUNDS:" has `HOLD WITH WORK`); 6 some round does not; 7 all
   do, but a hiding condition is missed; 1 file unreadable or rejected; 2 usage. With 4, 6 and 7 the
   last line begins "PLAN ONLY". */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "frame/heads.h"
#include "frame/hiding_conditions.h"
#include "frame/prints.h"
#include "hash/hash.h"

using namespace blk;

int main(int argc, char **argv) {
  bool rounds = false, fields = false;
  for (int i = 4; i < argc; ++i)
    if (!std::strcmp(argv[i], "-rounds")) rounds = true;
    else if (!std::strcmp(argv[i], "-fields")) fields = true;
    else argc = 0;
  const std::string what = argc >= 4 && !std::strcmp(argv[1], "-plan") ? argv[2] : "";
  if (what != "id" && what != "key" && what != "leaf" && what != "request") {
    std::fprintf(stderr, "usage: cp_frame -plan id|key|leaf|request FILE [-rounds] [-fields]\n");
    return 2;
  }
  std::ifstream in(argv[3], std::ios::binary);
  std::stringstream text;
  text << in.rdbuf();
  Parameters par;
  std::string why;
  if (!in || !read_parameters(text.str(), par, why)) {
    std::printf("NO: %s\n", in ? why.c_str() : "parameter-file-unreadable");
    return 1;
  }
  {
    u8 b[DIGEST_BYTES];
    digest_to_bytes(parameter_file_digest(text.str()), b);
    std::printf("parameters: version %s, digest ", version_text(par.version).c_str());
    for (u8 x : b) std::printf("%02x", x);
    std::printf("\n");
  }
  if (what == "id") {
    std::printf("PLAN ONLY: no proof was checked\n");
    return 4;
  }
  const Subproofs sub = stand_in_subproofs();
  std::printf("note: the sub-proofs' shapes are stand-ins that meet known totals of rounds and scalars; sizes of scalars and counts of rounds follow them\n");
  const Sized z = sized(par, sub);
  if (!z.refused.empty()) {
    std::printf("NO: parameters-not-admitted: %s\n", z.refused.c_str());
    return 1;
  }
  const bool holds = print_plan(what, z, sub, rounds, fields);
  std::string missed = what == "request" ? hiding_conditions_missed_in_the_links(z, sub) : "";
  if (missed.empty() && what != "leaf") missed = hiding_conditions_missed_in_the_ranges(z);
  if (missed.empty()) missed = hiding_conditions_missed_on_the_table(z, sub.pending_rows);
  if (missed.empty())
    std::printf("hiding: the file misses none of the conditions that are tested on it here; not looked at: the range checks, whose numbers a planning call has not\n");
  else
    std::printf("hiding: COMMITTER, PROVER AND VERIFIER REFUSE THIS FILE: %s\n", missed.c_str());
  std::printf("PLAN ONLY: no proof was checked\n");
  return !holds ? 6 : missed.empty() ? 4 : 7;
}
