// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests ReqRows::row() (blocks/iop/req_rows.h), which turns a sub-proof's equation on named arrays
// into a linear claim (`Row`), by proving and verifying whole synthetic requests at toy size
// (rows_toy_cases()). Each segment of a request (`link` in identifiers and messages: one
// sub-proof's part of the proof) has random matrices and one equation. A layout (`ReqLayout`)
// places the matrices in the rows of the segment's oracle messages (encoded rows, sent as Merkle
// roots). The equation has rank-one weights on each matrix (a product over the bits of one index
// times a written-out vector on the other, both ways round), also on matrices kept from earlier
// segments, and coefficients on mask entries (random elements of the segment's mask row). The
// prover computes the equation's value (`target`) from the matrices, element by element, not from
// the placed rows. So an honest proof tests whether the coefficient vectors that row() makes from
// layout and weights match the prover's placement; the claims go through the segments' closings and
// the last proximity test.
// Must be rejected ("refused" in messages): a value off by one; an honest proof, if the verifier
// has one block of its layout moved or one weight changed. On an invalid layout ("not lawful" in
// messages) a verifier must abort, even if its caller never called req_layout_fault().
#include "chain/tests/death.h"
#include "iop/tests/req_rows_cases.h"

using namespace blk;

// A sub-proof that draws its segments' mask rows itself and also fills the pads' places.
// send_prove() (blocks/iop/send.h), which draws each pad (the random element that masks an
// out-of-domain sample's answer), aborts unless these places are zero.
struct FullMaskProver : RowsSubProver {
  using RowsSubProver::RowsSubProver;
  Table mask_row(size_t l) override {
    for (const ReqMessage &m : w.shape.link[l].messages)
      if (m.shape.mask_rows) {
        Table t = Send::fresh_mask(w.codes, m.shape, coins);
        for (const PadPlace &p : Send::pad(w.codes, m.shape)) t.set_entry(p.index, E::one());
        return t;
      }
    return Table();
  }
};

static std::vector<u8> prove(const ReqWorld &w, const ReqLayout &layout, u64 seed, const E &off) {
  EngineCpu engine;
  return rows_prove(w, layout, seed, off, engine);
}

int main() {
  const std::vector<ReqCase> cases = rows_toy_cases();
  for (const ReqCase &t : cases) {
    const ReqWorld w = req_toy_world(t, 1);
    const ReqLayout layout = rows_toy_layout(w);
    const char *fault = req_layout_fault(w.codes, w.shape, layout);
    if (fault) std::printf("%s: %s\n", t.name, fault);
    CHECK(!fault, "a lawful layout is found faulty");
    if (fault) continue;
    const u64 seed = 500;
    const std::vector<u8> proof = prove(w, layout, seed, E::zero());
    CHECK(rows_verify(w, layout, seed, proof, false), "an honest proof is refused");
    if (const char *pads = toy_pads_fault([&] { (void)prove(w, layout, seed, E::zero()); })) {
      std::printf("%s: %s\n", t.name, pads);
      CHECK(false, "a pad is not drawn, or is drawn once for two places or two runs");
    }
    CHECK(!rows_verify(w, layout, seed, prove(w, layout, seed, E::one()), false), "a false target is accepted");
    CHECK(!rows_verify(w, layout, seed, proof, true), "the verifier has another weight, and accepts");
    for (size_t l = 0; l < layout.link.size(); l++)
      for (size_t a = 0; a < layout.link[l].size(); a++) {
        ReqLayout moved = layout;  // still valid: array B goes to the free first half of its row
        LaidBlock &b = moved.link[l][a].laid.blocks[0];
        if (b.place_bits + 1 != w.codes.base[0].log_ell() || moved.link[l][a].name != "B") continue;
        b.first_place = 0;
        CHECK(!req_layout_fault(w.codes, w.shape, moved), "a lawful layout is found faulty");
        CHECK(!rows_verify(w, moved, seed, proof, false), "the verifier has another layout, and accepts");
      }
    // Invalid layouts: req_layout_fault() must report each.
    auto faulty = [&](void (*change)(ReqLayout &)) {
      ReqLayout bad = layout;
      change(bad);
      CHECK(req_layout_fault(w.codes, w.shape, bad) != nullptr, "a layout that is not lawful passes");
    };
    faulty([](ReqLayout &x) { x.link.pop_back(); });
    faulty([](ReqLayout &x) { x.link[0][0].message = 99; });
    faulty([](ReqLayout &x) { x.link[0][1].name = x.link[0][0].name; });
    faulty([](ReqLayout &x) { x.link[0][1].laid.blocks[0].first_row = 0; });    // on top of another array
    faulty([](ReqLayout &x) { x.link[0][1].laid.blocks[0].first_row = 3; });    // a row that is not sent
    faulty([](ReqLayout &x) { x.link[0][1].laid.blocks[0].first_place += 1; });  // not aligned
    faulty([](ReqLayout &x) { x.link[0][0].laid.size_i = 8; });                 // not covered
    faulty([](ReqLayout &x) { x.link[0].back().laid.blocks.back().first_place = x.link[0].back().laid.size_c; });  // on the pad
    if (&t != &cases[0]) continue;
    // Must abort (first case only): a prover that fills a pad's place, and a verifier started on
    // each of four invalid layouts. `Binder` (blocks/chain/public.h), which takes the public
    // inputs, catches the repeated name; the constructor ReqRows() catches the rest. Both run
    // before row().
    auto started_on = [&](void (*change)(ReqLayout &)) {
      ReqLayout bad = layout;
      change(bad);
      (void)rows_verify(w, bad, seed, proof, false);
    };
    CHECK(ends_with("a pad's place holds something", [&] {
            EngineCpu own;
            (void)rows_prove<FullMaskProver>(w, layout, seed, E::zero(), own);
          }),
          "a link's mask row with something on the pad's place is sent");
    CHECK(ends_with("two arrays of a link have one name", [&] { started_on([](ReqLayout &x) { x.link[0][1].name = x.link[0][0].name; }); }),
          "a layout that contradicts itself is bound");
    CHECK(ends_with("overlap", [&] { started_on([](ReqLayout &x) { x.link[0][1].laid.blocks[0].first_row = 0; }); }), "a verifier runs on a layout with two arrays on one entry");
    CHECK(ends_with("the pad of its oracle message's sample", [&] {
            started_on([](ReqLayout &x) { x.link[0].back().laid.blocks.back().first_place = x.link[0].back().laid.size_c; });
          }),
          "a verifier runs on a layout in which a claim can read a pad");
    CHECK(ends_with("another number of links", [&] { started_on([](ReqLayout &x) { x.link.pop_back(); }); }), "a verifier runs on a layout for another number of links");
  }
  std::printf("req_rows_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
