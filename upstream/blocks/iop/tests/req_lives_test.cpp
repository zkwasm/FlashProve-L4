// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests a request's proof (blocks/iop/req.h) with kept oracle messages of overlapping lives. A
// message of life n is sent again at the code switch (blocks/iop/cs.h) into each of the next n
// segments (one sub-proof each; `links` in code and output). For n = 1, 2, 3, each segment
// followed by at least n others sends such a message of n rows, over the extension E for n = 2,
// otherwise over F. Each segment's equation (rows_toy_equation()) reads every matrix that the
// segment can still read. Its target (right-hand side) is computed from the prover's own matrices
// and masks. An honest proof must be accepted; a false target (first segment) and a verifier's
// changed weight (last segment) must not. Limits: one seed, 24-bit error target.
#include "iop/tests/req_rows_cases.h"

using namespace blk;

int main() {
  for (unsigned links : {2u, 3u, 4u, 5u, 7u}) {
    ReqCase t = {"lives", links, 0, false, 24, 0, 13};
    t.more_kept = {{1, 1}, {2, 2}, {3, 3}};
    const ReqWorld w = req_toy_world(t, 1);
    const ReqLayout layout = rows_toy_layout(w);
    const char *fault = req_fault(w.codes, w.shape);
    if (!fault) fault = req_layout_fault(w.codes, w.shape, layout);
    if (fault) std::printf("%u links: %s\n", links, fault);
    CHECK(!fault, "a lawful configuration or layout is found faulty");
    if (fault) continue;
    size_t most = 0;
    for (const ReqGroup &g : req_groups(w.shape)) most = std::max(most, g.born.size());
    if (links == 7) CHECK(most == 6, "with seven links six kept messages meet at a switch: three of the link before, two and one of those before it");
    EngineCpu engine, again;
    const std::vector<u8> proof = rows_prove(w, layout, 500, E::zero(), engine);
    CHECK(rows_verify(w, layout, 500, proof, false), "an honest proof is refused");
    if (const char *pads = toy_pads_fault([&] {  // each sending's own pad masks its sample's answer
          EngineCpu own;
          (void)rows_prove(w, layout, 500, E::zero(), own);
        })) {
      std::printf("%u links: %s\n", links, pads);
      CHECK(false, "a pad is not drawn, or is drawn once for two places or two runs");
    }
    CHECK(!rows_verify(w, layout, 500, rows_prove(w, layout, 500, E::one(), again), false), "a false target is accepted");
    CHECK(!rows_verify(w, layout, 500, proof, true), "another weight is accepted");
  }
  std::printf("req_lives_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
