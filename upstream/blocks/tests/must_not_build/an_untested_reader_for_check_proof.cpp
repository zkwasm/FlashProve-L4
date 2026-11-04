// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «invalid initialization of reference of type». With -DCONTROL it builds.
// Passes check_proof() an untested `IndexKey` (e.g. from IndexCpu::key()), not a `TestedTable`.
#include "frame/check.h"
using namespace blk;
ProofVerdict play(const Configured &c, const IndexKey &untested, const TestedTable &tested, const std::vector<u8> &request, const E &reply, Reader &file, const LinksCheck &links, const char *&why) {
#ifndef CONTROL
  return check_proof(c, untested, 0, request, request, reply, file, links, nullptr, why);
#else
  (void)untested;
  return check_proof(c, tested, 0, request, request, reply, file, links, nullptr, why);
#endif
}
