// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «invalid initialization of reference of type». With -DCONTROL it builds.
// tests/new/run_new.py reads the line above: g++, run with LC_ALL=C, must fail and print the quoted
// words. Explained in blocks/tests/must_not_build/README.md.
// Passes check_proof() an `UnprovedTable` (blocks/iop/foldeq.h): a committed table read from the
// first bytes of its key file or leaf file (`entry` in identifiers). The proofs about the table
// that follow them there are not checked. Only check_proof_if_tested() and
// check_entry_if_tested() accept one. Their result type, `IfTested`, has no unconditional yes: any
// value but NO is worth a yes only if separate checks of the table's own files (check_key(),
// check_entry()) pass on the same bytes.
#include "frame/check.h"
using namespace blk;
bool play(const Configured &c, const UnprovedTable &unproved, const std::vector<u8> &request, const E &reply, Reader &file, const LinksCheck &links, const char *&why) {
#ifndef CONTROL
  return check_proof(c, unproved, 0, request, request, reply, file, links, nullptr, why) != ProofVerdict::NO;
#else
  return check_proof_if_tested(c, unproved, 0, request, request, reply, file, links, nullptr, why) != IfTested::NO;
#endif
}
