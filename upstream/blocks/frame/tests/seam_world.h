// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture of blocks/frame/tests/seam_test.cpp, blocks/frame/tests/tested_table_test.cpp and
// blocks/frame/handover_proof.cpp: the public files, the prover and the verifier's callback for
// batches of toy requests.
// A segment (`link` in identifiers, strings and the public files) is one sub-proof's part of a
// request's first hash chain. A toy segment multiplies two committed vectors entry by entry
// (blocks/toy/segments.h) and makes CLAIMS linear claims (`TensorRow`) on the matrix M in the
// committed table (`Index` in identifiers). blocks/iop/batt_shown.h reduces these to two claims.
// (U), a `Row` counted in `open_rows`, is on entries of the segment's mask row. (E), pending
// (`pending_rows`) until the request's end, is on M and an entry c of the index-mask message (the
// request's masks for claims on M).
#pragma once
#include <cstdio>
#include <sstream>

#include "engine/engine_cpu.h"
#include "frame/check.h"
#include "frame/tests/toy_file.h"
#include "iop/batt_shown.h"
#include "iop/tests/member_world.h"
#include "toy/segments.h"

using namespace blk;

static const unsigned CLAIMS = 2;  // claims on the matrix per segment; not in the public files

// The parameter file of toy_file() with mask rows of 2^6 instead of 2^5 message entries in the
// segments' group: a segment here also masks its claims' values (`target` in identifiers) and the
// scalars of batt_shown_prove(). slots: the largest batch size, a power of two.
static std::string seam_file(unsigned slots = 2) {
  std::string t = toy_file();
  {
    const std::string two = "\nslots 2\n";
    const size_t at = t.find(two);
    BLK_REQUIRE(at != std::string::npos, "the toy's parameter file has another line for a batch's places");
    t.replace(at, two.size(), "\nslots " + std::to_string(slots) + "\n");
  }
  const std::string old = "group link 1 13 9 13 5\n";
  const size_t at = t.find(old);
  BLK_REQUIRE(at != std::string::npos, "the toy's parameter file has another line for the links' group");
  return t.replace(at, old.size(), "group link 1 13 9 13 6\n");
}
/* The second public file: the numbers of segments, of pending claims (one per segment) and of the
   matrix's rows, then the segments' identifiers. toy_shapes() rejects more than 9 segments.
   `more`: any of three keywords.
   "arrays": the layout lists the segments' vectors as arrays.
   "class": the second segment (the first, if there is only one) has one more oracle message in its
   first prover message, in a leaf class, and so a tree, of its own.
   "later": that segment has one more oracle message at a second commit point (`moment` 1), between
   its sumcheck (`psc` in identifiers) and claims on the matrix.
   The only claims on the two extra messages are those of their out-of-domain samples. */
static std::string toy_model(unsigned links = 3, const std::vector<std::string> &more = {}) {
  std::string t = "links " + std::to_string(links) + "\npending " + std::to_string(links) + "\nmatrix " + "100" + "\nidentifiers";
  for (unsigned l = 1; l <= links; l++) t += " " + std::to_string(l);
  for (const std::string &word : more) t += "\n" + word;
  return t + "\n";
}

// The `ShapesOf` callback for configure() (blocks/frame/configure.h).
static Shapes toy_shapes(const std::vector<std::string> &files, const Parameters &par) {
  Shapes shapes;
  unsigned LINKS = 0, pending = 0, matrix = 0;
  int read = 0;
  bool with_arrays = false, with_class = false, with_later = false;
  if (!par.rest.empty()) return shapes.refused = "the toy's parameter file has lines behind the frame's", shapes;
  if (files.size() != 2 || std::sscanf(files[1].c_str(), "links %u\npending %u\nmatrix %u\nidentifiers%n", &LINKS, &pending, &matrix, &read) != 3 || !read || LINKS > 9 || pending != LINKS)
    return shapes.refused = "the toy's second file is not in its form, or does not leave one row on the table for each link", shapes;
  {
    std::istringstream rest(files[1].substr((size_t)read));
    for (u64 x; rest >> x;) shapes.identifiers.push_back(x);
    rest.clear();
    std::string word;
    while (rest >> word) {
      bool &said = word == "arrays" ? with_arrays : word == "class" ? with_class : with_later;
      if ((word != "arrays" && word != "class" && word != "later") || said) return shapes.refused = "the toy's second file has something else than the words arrays, class, later, each at most once, behind the identifiers", shapes;
      said = true;
    }
  }
  Subproofs &s = shapes.sub;
  const unsigned vars = par.link.log_ell - 1, k_col = par.block_log;
  for (unsigned l = 0; l < LINKS; l++) {
    ReqLink link;
    link.name = "link " + std::to_string(l + 1);
    ReqMessage own, product;
    own.shape.name = link.name + ": b", own.shape.code = 0, own.shape.main_rows = l == 0 ? 2 : 1, own.shape.mask_rows = 1, own.shape.answers = par.answers;
    product.shape.name = link.name + ": c", product.shape.code = 0, product.shape.main_rows = 1, product.shape.answers = par.answers;
    product.life = l + 1 < LINKS ? 1 : 0;
    link.messages = {own, product};
    if (l == (LINKS > 1 ? 1u : 0u))  // in the order of `moment`, which `ReqLink::messages` requires
      for (const bool later : {false, true})
        if (later ? with_later : with_class) {
          ReqMessage further;
          further.shape.name = link.name + (later ? ": a later message" : ": a message in a class of its own"), further.shape.code = 0, further.shape.main_rows = 1, further.shape.answers = par.answers;
          further.leaf_class = later ? 0 : 1, further.moment = later ? 1 : 0;
          link.messages.push_back(further);
        }
    link.plain = l + 1 == LINKS ? 1 : 0;
    s.request.link.push_back(link);
  }
  s.open_rows.assign(LINKS, SEGMENT_ROWS + 1);
  const size_t with_two_moments = with_later ? (LINKS > 1 ? 1 : 0) : LINKS;  // which segment; LINKS: none
  s.plan = [vars, k_col, with_two_moments](Plan &p, size_t l, unsigned moment) {  // the rounds after each commit point's oracle messages
    const std::string where = "link " + std::to_string(l + 1);
    if (moment == 0) segments_plan_link(p, where, vars);
    if (moment == (l == with_two_moments ? 1u : 0u)) {
      plan_scalars(p, "a link's claims' targets", "stand-in", CLAIMS);
      plan_batt_shown(p, where + ": claims on the table", CLAIMS, k_col);
    }
  };
  s.pending_rows = pending;
  if (matrix % 2) return shapes.refused = "the toy's matrix has an odd number of rows; the ranges' pairs of rows want an even one", shapes;
  s.ranges = toy_ranges(matrix);  // the only place that gives the matrix's row count; a table row holds 2^3 of its columns
  shapes.range_checks.assign(LINKS, RangeCheck{(u64)1 << 6, (u64)1 << 8, 128, 41, (u64)1 << 17, {10, 8}});  // stand-ins: no toy segment checks a range, but `hiding_conditions_missed` rejects an empty list
  shapes.layout.link.resize(LINKS);  // without "arrays": an empty list for each segment
  if (with_arrays)  // hashed at the first chain's start (its first query); no toy sub-proof reads arrays (blocks/iop/req_rows.h)
    for (unsigned l = 0; l < LINKS; l++) {  // b: 2^(vars / 2) array rows in one block; c: as many, one half per block
      const unsigned half = vars / 2;
      ReqArray b, c;
      b.name = "b", b.message = 0, b.laid.size_i = u64(1) << half, b.laid.size_c = u64(1) << (vars - half);
      b.laid.blocks = {LaidBlock{0, 0, l == 0 ? u64(1) : u64(0), 0, half, vars - half, 0, vars, true}};
      c.name = "c", c.message = 1, c.laid.size_i = u64(1) << half, c.laid.size_c = u64(1) << (vars - half);
      c.laid.blocks = {LaidBlock{0, 0, 0, 0, half - 1, vars - half, 0, vars - 1, false}, LaidBlock{u64(1) << (half - 1), 0, 0, u64(1) << (vars - 1), half - 1, vars - half, 0, vars - 1, false}};
      shapes.layout.link[l] = {b, c};
    }
  return shapes;
}

// The seed of the forms of a request's claims on the matrix: a public function of its bytes.
static u64 seed_of(const std::vector<u8> &request) {
  u64 x = 7;
  for (u8 b : request) x = x * 131 + b;
  return x % 1000;
}

// Adds (U) to the segment's group g.
static void own_row(Statement &st, GroupId g, BattShown &out) {
  Row u;
  u.name = "the claims on the table leave the link";
  for (MaskTerm<E> &m : out.own_masks) u.add_mask(std::move(m));
  u.target = out.own_target;
  st.add_row(g, std::move(u));
}

// Proves one batch: a proof file for each request, or none if lin_begin() fails: the batch would
// exceed the table's limit of opened columns. The toy segments prove nothing about `replies`.
// M: the matrix on which the claims' values are computed. entry: the committed table's number.
// parts: each file's five parts: header (`head` in identifiers), first chain, place in the binding
// tree (the Merkle tree over the batch's first chains' seals), shared chain, second chain.
using Parts = std::vector<std::vector<u8>>;
static std::vector<std::vector<u8>> prove_batch(const Configured &w, IndexHolder &index, const IndexName &table, const LinMatrix &M, u32 entry,
                                                const std::vector<std::vector<u8>> &requests, const std::vector<E> &replies, u64 coin_seed,
                                                std::vector<PadLook> *pads = nullptr, std::vector<Parts> *parts = nullptr) {
  const LinShape &S = w.fixed().lin;
  TestCoins coins(coin_seed);
  EngineCpu engine;
  struct Request {
    Writer first, second;
    std::unique_ptr<ProverRounds> ch;
    std::unique_ptr<ProverContext> P;
    RequestPublic x;
    LinFirst lin;
    Sealed seal;
  };
  std::vector<Request> req(requests.size());
  LinBinding binding(w.fixed().binding, w.fixed().configuration, coins);
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(index, S, begun)) return {};
  const Summand<E> summand = toy_summand();
  for (unsigned c = 0; c < requests.size(); c++) {
    Request &r = req[c];
    Binder b(LABEL_FIRST_CHAIN, w.proof().chains[0].plan);
    r.x = bind_request_parameters(b, w);
    bind_request_statement(b, r.x, table, entry, requests[c], requests[c]);  // in the toy a request is its own file
    r.ch.reset(new ProverRounds(b, toy_random_bytes(coin_seed + 100 + c), r.first));
    r.P.reset(new ProverContext(r.x.lin.codes, *r.ch, engine, coins));
    r.P->pads = pads;
    ProverContext &P = *r.P;
    ReqProver frame(P, r.x.shape);
    const Codes &codes = P.st.codes();
    const ReqShape &shape = r.ch->get(r.x.shape);
    const unsigned vars = codes.base[shape.code].log_ell() - 1;
    const size_t len = size_t(1) << vars, ell = codes.base[shape.code].ell;
    std::vector<F> x(len);
    coins.fill(x.data(), len);
    std::vector<TensorRow> rows;  // (E) of each segment
    for (size_t l = 0; l < shape.link.size(); l++) {
      std::vector<F> bl(len), cl(len);
      coins.fill(bl.data(), len);
      for (size_t i = 0; i < len; i++) cl[i] = x[i] * bl[i];
      Table own(1, l == 0 ? 2 : 1, ell), product(1, 1, ell);
      for (size_t i = 0; i < len; i++) {
        if (l == 0) own.set(0, i, E::from(x[i]));
        own.set(l == 0 ? 1 : 0, i, E::from(bl[i]));
        product.set(0, i, E::from(cl[i]));
      }
      std::vector<OracleDraft> at_start;
      if (l == 0) at_start.push_back(lin_draft(P, shape.at_start[0].shape, false, shape.at_start[0].leaf_class, {}));
      std::vector<E> plain;
      if (shape.link[l].plain) plain.push_back(replies[c]);
      // An extra message: a row of arbitrary entries.
      auto further = [&]() {
        Table t(1, 1, ell);
        std::vector<F> v(len);
        coins.fill(v.data(), len);
        for (size_t i = 0; i < len; i++) t.set(0, i, E::from(v[i]));
        return engine.put(t);
      };
      std::vector<TableId> at_first = {engine.put(own), engine.put(product)};
      for (size_t m = 2; m < shape.link[l].messages.size(); m++)
        if (shape.link[l].messages[m].moment == 0) at_first.push_back(further());
      const std::vector<OracleId> ids = frame.first(at_first, std::move(at_start), Table(), plain);
      const SegmentPlaces at = segments_places(frame, ids);
      std::vector<E> zeta(vars);
      {
        AlgebraicDraw from = r.ch->close_algebraic(R_ZERO);
        for (E &v : zeta) v = from.draw();
      }
      SumClaim<E> claim = toy_claim(zeta);
      std::vector<std::vector<E>> maps(3, std::vector<E>(len));
      for (size_t i = 0; i < len; i++) maps[0][i] = E::from(x[i]), maps[1][i] = E::from(bl[i]), maps[2][i] = E::from(cl[i]);
      SumcheckCpu<E> inner(summand, claim.vars, std::move(maps));
      std::vector<MaskSlot> fresh = P.st.masks(ids[0]).take(psc_masks(claim, summand));
      const ProverContext *p = &P;
      PscOutput<E> out = psc_prove<E>(*r.ch, std::move(claim), summand, inner, std::move(fresh), [p](u32 oracle, u32 i) { return p->behind[oracle].mask.entry(i); });
      segments_rows(P.st, frame.group(), at, std::move(out));
      if (shape.link[l].moments() > 1) frame.later({further()});
      {  // the segment's claims on the matrix, masked by entries of its own mask row
        [[maybe_unused]] const bool lies = c == 0 && l == 1;
        std::vector<TensorRow> claims;
        std::vector<std::vector<E>> w_of;
        for (size_t j = 0; j < CLAIMS; j++) {
          TensorRow row = member_row_form(S, seed_of(requests[c]), unsigned(l), j);
          LinClaim f;
          f.g = row.g;
          w_of.push_back(lin_w(S, f, M));
          MaskSlot slot = P.st.masks(ids[0]).take();
          row.target = mle_eval(w_of[j].data(), w_of[j].size(), row.psi.data(), S.k_col()) + P.mask_value(slot);
          row.masks.push_back({std::move(slot), E::one()});
          r.ch->send(row.target);
          claims.push_back(std::move(row));
        }
        [[maybe_unused]] E raised = E::zero();
        [[maybe_unused]] const std::vector<E> psi0 = claims[0].psi;
        const BattSumcheck over_w = [&](const Summand<E> &sm, const std::vector<Weight<E>> &v, const std::vector<E> &, const std::vector<std::vector<E>> &) {
          std::unique_ptr<SumcheckProver<E>> cpu(new SumcheckCpu<E>(sm, v, w_of));
          return cpu;
        };
        BattShown left = batt_shown_prove(*r.ch, std::move(claims), P.st.masks(ids[0]), P.st.masks(frame.at_start()[0]),
                                          [p](u32 oracle, u32 i) { return p->behind[oracle].mask.entry(i); }, over_w);
        own_row(P.st, frame.group(), left);
        rows.push_back(std::move(left.row));
      }
      frame.close();
      x = std::move(cl);
    }
    frame.finish();
    const OracleId masks = frame.at_start()[0];
    HeavyCpu heavy(S, M);
    r.lin = member_first_prove(P, r.x.lin, masks, std::move(rows), heavy);
    BLK_REQUIRE(r.ch->seal(R_SEAL, r.seal), "a prover's channel does not refuse");
    r.ch->finish_inside_a_file();
    binding.put(c, r.seal.digest());
  }
  Writer shared_bytes;
  Binder shared_b(LABEL_PROOF_SHARED_CHAIN, w.proof().chains[1].plan);
  const LinPublic shared_x = bind_shared_chain(shared_b, w, table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, coins);
  shared_P.pads = pads;
  LinShared shared;
  lin_shared_prove(shared_P, shared_x, TableTest::LATER, index, std::move(begun), shared);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish_inside_a_file();
  std::vector<std::vector<u8>> files;
  for (unsigned c = 0; c < requests.size(); c++) {
    Request &r = req[c];
    Binder b(LABEL_PROOF_SECOND_CHAIN, w.proof().chains[2].plan);
    bind_second_chain(b, w, r.seal, shared_seal);
    ProverRounds ch(b, toy_random_bytes(coin_seed + 200 + c), r.second);
    ProverContext P = lin_go_on(std::move(*r.P), ch);
    P.pads = pads;
    const LinSecond second = lin_second_prove(P, r.x.lin, r.lin, shared_P, shared);
    lin_second_tests_prove(P, r.x.lin, r.lin, second);
    ch.finish();
    P.st.require_all_tested();
    Writer head, place, file;
    write_head(head, TAG_PROOF, entry);
    binding.write_path(place, c);
    const Parts five = {head.bytes(), r.first.bytes(), place.bytes(), shared_bytes.bytes(), r.second.bytes()};
    for (const std::vector<u8> &part : five) file.put_bytes(part.data(), part.size());
    files.push_back(file.bytes());
    if (parts) parts->push_back(five);
  }
  lin_release(shared_P, shared);
  return files;
}

// The `LinksCheck` callback for check_proof() (blocks/frame/check.h): checks the toy segments and
// collects (E) of each.
static bool toy_links(VerifierContext &V, const RequestPublic &x, LinksResult &out) {
  VerifierRounds &ch = V.ch;
  ReqVerifier frame(V, x.shape);
  const ReqShape &shape = ch.get(x.shape);
  const unsigned vars = V.st.codes().base[shape.code].log_ell() - 1;
  const Summand<E> summand = toy_summand();
  for (size_t l = 0; l < shape.link.size(); l++) {
    std::vector<OracleId> ids;
    if (!frame.first(ids)) return false;
    if (shape.link[l].plain) out.reply = frame.plain()[0], out.replied = true;
    const SegmentPlaces at = segments_places(frame, ids);
    std::vector<E> zeta(vars);
    {
      AlgebraicDraw from = ch.close_algebraic(R_ZERO);
      for (E &v : zeta) v = from.draw();
    }
    SumClaim<E> claim = toy_claim(zeta);
    std::vector<MaskSlot> fresh = V.st.masks(ids[0]).take(psc_masks(claim, summand));
    PscOutput<E> psc;
    if (!psc_verify<E>(ch, std::move(claim), summand, std::move(fresh), psc)) return false;
    segments_rows(V.st, frame.group(), at, std::move(psc));
    if (shape.link[l].moments() > 1) {
      std::vector<OracleId> later;
      if (!frame.later(later)) return false;
    }
    std::vector<TensorRow> claims;
    for (size_t j = 0; j < CLAIMS; j++) {
      TensorRow row = member_row_form(ch.get(x.lin.shape), seed_of(ch.get(x.request)), unsigned(l), j);
      if (!ch.recv(row.target)) return false;
      row.masks.push_back({V.st.masks(ids[0]).take(), E::one()});
      claims.push_back(std::move(row));
    }
    BattShown left;
    if (!batt_shown_verify(ch, std::move(claims), V.st.masks(ids[0]), V.st.masks(frame.at_start()[0]), left)) return false;
    own_row(V.st, frame.group(), left);
    out.pending.push_back(std::move(left.row));
    if (!frame.close()) return false;
  }
  if (!frame.finish()) return false;
  out.index_masks = frame.at_start()[0];
  return true;
}

// The first chain only, by check_proof()'s steps (header, segments, pending claims, seal); the
// reply is not compared. A false (E) passes here: the second chain must reject it (tried in a
// test that is not part of this repository; examples/protocol-toy-false-files/ has one such file).
[[maybe_unused]] static bool first_chain_passes(const Configured &w, const IndexName &table, u32 entry, const std::vector<u8> &request, const std::vector<u8> &bytes) {
  Reader file(bytes.data(), bytes.size());
  u32 made_on = 0;
  if (read_head(file, TAG_PROOF, made_on) || made_on != entry) return false;
  Binder b(LABEL_FIRST_CHAIN, w.proof().chains[0].plan);
  RequestPublic x = bind_request_parameters(b, w);
  bind_request_statement(b, x, table, entry, request, request);
  VerifierSession session;  // for this one chain
  VerifierRounds ch(b, file, session);
  VerifierContext V(x.lin.codes, ch);
  LinksResult r;
  if (!toy_links(V, x, r)) return false;
  LinFirst first;
  if (!member_first_verify(V, x.lin, r.index_masks, std::move(r.pending), first)) return false;
  Sealed seal;
  return ch.seal(R_SEAL, seal) && ch.finish_inside_a_file();
}
