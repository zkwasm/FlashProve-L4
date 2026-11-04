// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture for ReqRows, used by req_rows_test.cpp and other tests: a layout, equations, a
// prover (on any Engine) and a verifier for the toy requests of req_toy_world() (req_cases.h).
// Each segment of a request (`link` in identifiers) ends in one equation on arrays (named matrices
// in oracle messages' main rows) with weights of rank one: pass rows = 1 to req_toy_world().
#pragma once
#include "iop/req_rows.h"
#include "iop/tests/req_cases.h"

namespace blk {

// Arrays of req_toy_world()'s segments. Oracle messages are found by name: kept cells, where sent,
// shift later messages' numbers.
inline ReqLayout rows_toy_layout(const ReqWorld &w) {
  const unsigned L = w.codes.base[0].log_ell();
  auto block = [](u64 first_i, unsigned bits_i, unsigned bits_c, u64 first_row, unsigned row_bits, u64 first_place, bool i_is_low) {
    LaidBlock b;
    b.first_i = first_i, b.bits_i = bits_i, b.bits_c = bits_c, b.first_row = first_row, b.row_bits = row_bits;
    b.place_bits = bits_i + bits_c - row_bits, b.first_place = first_place, b.i_is_low = i_is_low;
    return b;
  };
  ReqLayout layout;
  for (const ReqLink &link : w.shape.link) {
    std::vector<ReqArray> arrays;
    for (size_t m = 0; m < link.messages.size(); m++) {
      const std::string &name = link.messages[m].shape.name;
      const u64 half = u64(1) << (L - 1);
      if (name == "cells") {
        arrays.push_back(ReqArray{"A", m, LaidMatrix{4, half, {block(0, 2, L - 1, 0, 1, 0, true)}}});        // rows 0 and 1, whole
        arrays.push_back(ReqArray{"B", m, LaidMatrix{1, half, {block(0, 0, L - 1, 2, 0, half, false)}}});    // second half of row 2
      } else if (name == "kept cells" || name == "digits") {  // row i of the matrix in the first half of row i
        arrays.push_back(ReqArray{name == "digits" ? "D" : "K", m, LaidMatrix{2, half, {block(0, 0, L - 1, 0, 0, 0, false), block(1, 0, L - 1, 1, 0, 0, false)}}});
      } else if (name == "more kept cells") {  // row i of the matrix in the first half of row i
        LaidMatrix x{link.messages[m].shape.main_rows, half, {}};
        for (u64 i = 0; i < x.size_i; i++) x.blocks.push_back(block(i, 0, L - 1, i, 0, 0, false));
        arrays.push_back(ReqArray{"M" + std::to_string(m), m, x});
      } else if (name == "reads") {
        arrays.push_back(ReqArray{"R", m, LaidMatrix{1, half, {block(0, 0, L - 1, 0, 0, 0, false)}}});
      }
    }
    layout.link.push_back(arrays);
  }
  return layout;
}

inline std::vector<E> random_elements(TestCoins &c, size_t n) {
  std::vector<E> v(n);
  for (E &x : v) x = c.e();
  return v;
}

// Segment l's equation with public weights from a seed; `target` stays 0. spoil alters one `scale`.
inline ReqEquation rows_toy_equation(const ReqShape &shape, const ReqLayout &layout, size_t l, u64 seed, bool spoil) {
  TestCoins c(seed + 1000 * l);
  ReqEquation eq;
  bool written_rows = false;
  auto add = [&](size_t born, const ReqArray &a) {
    ReqTerm t;
    t.born = born, t.array = a.name, t.scale = c.e();
    const unsigned bits_i = ceil_log2(a.laid.size_i), bits_c = ceil_log2(a.laid.size_c);
    if (written_rows) {
      t.on_rows = IndexWeights::written(std::make_shared<const std::vector<E>>(random_elements(c, a.laid.size_i)));
      t.on_columns = IndexWeights::eq(random_elements(c, bits_c));
    } else {
      t.on_rows = IndexWeights::eq(random_elements(c, bits_i));
      t.on_columns = IndexWeights::written(std::make_shared<const std::vector<E>>(random_elements(c, a.laid.size_c)));
    }
    written_rows = !written_rows;
    eq.terms.push_back(std::move(t));
  };
  for (const ReqArray &a : layout.link[l]) add(l, a);
  const std::vector<ReqGroup> groups = req_groups(shape);
  for (const std::pair<size_t, size_t> &born : groups[l].born)
    for (const ReqArray &a : layout.link[born.first])
      if (a.message == born.second) add(born.first, a);
  eq.on_masks = random_elements(c, 5);
  if (spoil) eq.terms[0].scale += E::one();
  return eq;
}
inline ReqEquation rows_toy_equation(const ReqWorld &w, const ReqLayout &layout, size_t l, u64 seed, bool spoil) {
  return rows_toy_equation(w.shape, layout, l, seed, spoil);
}

inline E weight_at(const IndexWeights &w, u64 j) {
  if (w.values) return (*w.values)[j];
  E x = E::one();
  for (size_t t = 0; t < w.lo.size(); t++) x *= ((j >> t) & 1) ? w.hi[t] : w.lo[t];
  return x;
}

// Places random arrays and computes each equation's `target` with loops of its own, as a check on
// the linear claim (`Row`) that ReqRows::row() builds.
struct RowsSubProver : ReqSubProver {
  const ReqWorld &w;
  const ReqLayout &layout;
  u64 seed;
  Engine &engine;
  Coins &coins;
  TestCoins data;
  E off;  // added to the first segment's `target`
  Public<ReqLayout> bound;  // the layout as bound in the chain's first query; set before the run
  std::unique_ptr<ReqRows<ProverContext>> rows;  // made at the first segment
  std::vector<std::vector<std::vector<E>>> matrix;  // [link][array][i * size_c + c]
  RowsSubProver(const ReqWorld &world, const ReqLayout &lay, u64 s, Engine &e, Coins &c, const E &o)
      : w(world), layout(lay), seed(s), engine(e), coins(c), data(s + 1), off(o), matrix(world.shape.link.size()) {}

  // Pads (each masks a sample's answer) must be 0 here: send_prove() (blocks/iop/send.h) draws them
  // and ends the process if one is nonzero.
  std::vector<TableId> tables(size_t l, unsigned moment) override {
    matrix[l].resize(layout.link[l].size());
    std::vector<TableId> main;
    for (size_t m = 0; m < w.shape.link[l].messages.size(); m++) {
      const OracleShape &o = w.shape.link[l].messages[m].shape;
      if (w.shape.link[l].messages[m].moment != moment) continue;
      Table t(o.main_deg, o.main_rows, w.codes.base[o.code].ell);
      data.fill(t.a.data(), t.a.size());  // also where no array is placed
      for (size_t a = 0; a < layout.link[l].size(); a++) {
        const ReqArray &array = layout.link[l][a];
        if (array.message != m) continue;
        std::vector<E> &values = matrix[l][a];
        values.resize(array.laid.size_i * array.laid.size_c);
        for (E &x : values) x = o.main_deg == 1 ? E::from(data.f()) : data.e();
        for (const LaidBlock &b : array.laid.blocks)
          for (u64 i = 0; i < u64(1) << b.bits_i; i++)
            for (u64 c = 0; c < u64(1) << b.bits_c; c++) {
              const u64 x = b.i_is_low ? (c << b.bits_i) + i : (i << b.bits_c) + c;
              t.set(b.first_row + (x >> b.place_bits), b.first_place + (x & ((u64(1) << b.place_bits) - 1)),
                    values[(b.first_i + i) * array.laid.size_c + b.first_c + c]);
            }
      }
      for (const PadPlace &p : Send::pad(w.codes, o))  // back to 0
        if (p.part == Part::MAIN) t.set_entry(p.index, E::zero());
      main.push_back(engine.put(t));
    }
    return main;
  }
  std::vector<E> plain(size_t l) override { return std::vector<E>(w.shape.link[l].plain, E::from(from_u64(REQ_TOY_REPLY))); }
  // Every second segment's mask row is drawn by the sub-proof, as the GPU prover does, whose
  // kernels mask with it (prover/prover/chain_run.cuh): uniform and fresh, the pad's places left
  // zero. The frame (ReqProver) draws the others, so both branches of ReqProver::first() run.
  Table mask_row(size_t l) override {
    if (l % 2 == 0) return Table();
    for (const ReqMessage &m : w.shape.link[l].messages)
      if (m.shape.mask_rows) return Send::fresh_mask(w.codes, m.shape, coins);
    return Table();
  }
  void run(ProverContext &P, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &) override {
    const size_t l = frame.link();
    if (moment + 1 < w.shape.link[l].moments()) return;
    ReqEquation eq = rows_toy_equation(w, layout, l, seed, false);
    for (const ReqTerm &t : eq.terms) {
      size_t a = 0;
      while (layout.link[t.born][a].name != t.array) a++;
      const LaidMatrix &laid = layout.link[t.born][a].laid;
      for (u64 i = 0; i < laid.size_i; i++)
        for (u64 c = 0; c < laid.size_c; c++)
          eq.target += t.scale * weight_at(t.on_rows, i) * weight_at(t.on_columns, c) * matrix[t.born][a][i * laid.size_c + c];
    }
    for (size_t j = 0; j < eq.on_masks.size(); j++) eq.target += eq.on_masks[j] * P.behind[frame.where(l, 0)].mask.entry(j);
    if (l == 0) eq.target += off;
    P.ch.send(eq.target);
    if (!targets_known(l)) eq.target = E::from(from_u64(0x5EED + l));  // arbitrary, not to be read
    if (!rows) rows = std::make_unique<ReqRows<ProverContext>>(P, frame, bound);
    P.st.add_row(frame.group(), rows->row(eq));
  }
  // Only every third segment gives the frame its claim's true `target`. The others do as the GPU
  // prover, whose kernels never compute it: the frame gets an arbitrary number. A frame that read
  // that number would carry a false claim, which rows_prove() checks for. From three segments on,
  // both ReqProver::close(true) and close(false) run.
  bool targets_known(size_t l) override { return l % 3 == 2; }
};
// The shape comes from the frame, layout and seed from the channel.
struct RowsSubVerifier : ReqSubVerifier {
  const ReqLayout &layout;
  u64 seed;
  bool spoil;
  Public<ReqLayout> bound;
  std::unique_ptr<ReqRows<VerifierContext>> rows;  // made at the first segment
  RowsSubVerifier(const ReqLayout &lay, u64 s, bool sp) : layout(lay), seed(s), spoil(sp) {}
  bool run(VerifierContext &V, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &) override {
    const size_t l = frame.link();
    const ReqShape &shape = frame.shape();
    if (!rows) rows = std::make_unique<ReqRows<VerifierContext>>(V, frame, bound);
    if (moment + 1 < shape.link[l].moments()) return true;
    ReqEquation eq = rows_toy_equation(shape, layout, l, seed, spoil && l + 1 == shape.link.size());
    if (!V.ch.recv(eq.target)) return false;
    V.st.add_row(frame.group(), rows->row(eq));
    return true;
  }
};

// Handles to a run's public inputs, as bound in the hash chain's first query (hashed before any
// message).
struct RowsPublic {
  Public<Codes> codes;
  Public<ReqShape> shape;
  Public<ReqLayout> layout;
  Public<u64> seed;
};
inline RowsPublic rows_toy_bind(Binder &b, const ReqWorld &w, const ReqLayout &layout, u64 seed) {
  RowsPublic p;
  b.configuration(toy_configuration());
  p.codes = b.parameter("codes", w.codes), p.shape = b.parameter("request", w.shape), p.layout = b.parameter("layout", layout);
  p.seed = b.statement("seed of the weights", seed);
  return p;
}

template <class SubProver = RowsSubProver>
inline std::vector<u8> rows_prove(const ReqWorld &w, const ReqLayout &layout, u64 seed, const E &off, Engine &engine) {
  TestCoins coins(toy_coins_seed(seed));
  Writer proof;
  Binder binder(1, w.plan);
  const RowsPublic bound = rows_toy_bind(binder, w, layout, seed);
  ProverRounds ch(binder, toy_random_bytes(seed + 2), proof);
  ProverContext P(bound.codes, ch, engine, coins);
  P.pads = toy_pads;
  SubProver sub(w, ch.get(bound.layout), ch.get(bound.seed), engine, coins, off);
  sub.bound = bound.layout;
  // The claim each close() leaves is checked on the prover's tables, also where the frame got an
  // arbitrary `target`. The proof's bytes cannot tell: none depends on the prover's record of it.
  req_prove(P, bound.shape, sub, [](ReqProver &frame) { BLK_REQUIRE(frame.carried_claim_holds(), "a test: the claim that a closing has left does not hold on the prover's tables"); });
  ch.finish();
  return proof.bytes();
}
// The verifier's side, under run_verifier() (blocks/chain/closed.h): rows_decide() gets the channel
// and the handles of what the first query binds, nothing of the fixture. SPOIL puts another weight
// into the last segment's equation: another program, not another input, hence a template
// parameter, and the first query is the honest one's.
struct RowsInputs {
  const ReqWorld &w;
  const ReqLayout &layout;
  u64 seed;
};
inline RowsPublic rows_toy_bind_inputs(Binder &b, const RowsInputs &in) { return rows_toy_bind(b, in.w, in.layout, in.seed); }
template <bool SPOIL> inline bool rows_decide(VerifierRounds &ch, const RowsPublic &bound, Notes *) {
  VerifierContext V(bound.codes, ch);
  RowsSubVerifier sub(ch.get(bound.layout), ch.get(bound.seed), SPOIL);
  sub.bound = bound.layout;
  return req_verify(V, bound.shape, sub);
}
inline bool rows_verify(const ReqWorld &w, const ReqLayout &layout, u64 seed, const std::vector<u8> &proof, bool spoil) {
  return run_verifier(1, w.plan, RowsInputs{w, layout, seed}, rows_toy_bind_inputs, spoil ? rows_decide<true> : rows_decide<false>, proof.data(), proof.size());
}

// Three of the four cases have a 24-bit ReqCase::target.
inline std::vector<ReqCase> rows_toy_cases() {
  return {
      {"two links", 2, 0, false, 24, 0, 10},
      {"three links, kept for one", 3, 1, false, 24, 0, 10},
      {"four links, kept for two", 4, 2, false, 24, 0, 11},
      {"three links, kept for one, 128 bits", 3, 1, false, 128, 0, 12},
  };
}

}  // namespace blk
