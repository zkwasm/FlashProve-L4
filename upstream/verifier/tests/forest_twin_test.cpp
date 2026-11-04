// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the verifier's lookup for a group of lookup calls (verifier/src/vf_lookup_argument.h) against the
// library's lookup argument by product trees (LKP: every key occurs in a public table), run as the
// prover: blk::lkp_run() in blocks/iop/lkp.h. For each fixture (`World`; the messages say `world`),
// prove() writes a proof about synthetic keys under a synthetic mask row; verify() reads it and
// evaluates the batched linear claims (`Row` in identifiers) on the mask row. Cases: see main().
// Real key: a key that is not padding. Multiplicity: how many real keys are counted at a table row.
// Plane j: bit j of every multiplicity. Slot: the number of a masked scalar and of its mask entry.
//
// A successful run shows that the two implementations agree on format and algebra for these shapes:
// 1 to 16 calls of 2^10 to 2^12 keys, tables of 2^5 to 2^8 rows. It runs no GPU code, no commitment
// and no sub-proof. The proof uses a hash chain (the Fiat-Shamir transcript of blocks/chain/).
// Of the lookup's public inputs, the chain's first query (the hash input that fixes its public
// inputs) has only the plan (the list of rounds), which fixes the numbers of keys and table rows
// (2^v, 2^d) but not their split into trees and calls (dT, rK, kT, cK: struct HForest). Fixing
// the tables, the padding and the rest before the first challenge is the caller's duty, listed in
// verifier/src/vf_lookup_argument.h. No malicious prover runs here: every proof is the honest one,
// and what a case alters is on the verifier's side.
// Exit code: 0 if every check passed, 1 if not, 2 if a fixture does not suit one of its cases.
// Build, from verifier/src:
//   g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
//   -I ../../blocks -I ../../prover ../tests/forest_twin_test.cpp
//   -o <a folder of your own>/forest_twin_test

// VF_WITHOUT_LOOKUPS leaves out verifier/src/vf_lookup_helpers.h, which this test does not use. PSC: the
// library's sumcheck with a hiding polynomial (blocks/iop/psc.h).
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "vf_lookup_argument.h"

#include "iop/lkp.h"
#include "numbers/blocks.h"

namespace blk {
struct SendAccess {  // stands in for SEND (sends and samples oracle messages): makes a receipt
  static SampleReceipt make(u32 oracle, u32 first, u32 count) { return SampleReceipt(oracle, first, count); }
};
}  // namespace blk

// Converts to the library's field element. `Ex`: the verifier's (verifier/src/vf_common.h).
static blk::E eOf(u64 x) { return blk::E::from(blk::from_u64(x)); }
static blk::E eOf(const Ex &x) { return blk::E::from(blk::from_u64(eC(x, 0)), blk::from_u64(eC(x, 1)), blk::from_u64(eC(x, 2))); }
// Synthetic values from a fixed start: each run has the same fixtures.
static u64 rngState = 7;
static u64 madeUp64() {
  rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
  return rngState >> 11;
}

// A lookup call: 2^n keys, at places 0 to 2^n - 1, the last `padding` of them padding keys, and a
// table of 2^k rows. t0: the row that a padding key equals. Row t of the range table has value t.
struct CallShape {
  int n, k;
  bool written; // a written-out table; else the range table
  u64 t0, padding;
};
// A fixture: a group of lookup calls, their keys and the mask row. tau: the written-out tables.
// cl: pointers to `calls`. pub: the numbers of padding keys. forestShape() splits each call's
// keys into trees of 2^dT keys; lid: the cap on dT.
struct World {
  vector<CallShape> shapes;
  vector<vector<u64>> tau;
  vector<Call> calls;
  vector<Call *> cl;
  int lid = 10;
  vector<u64> pub;            // by tree
  vector<vector<u64>> value;  // each call's key vector: the rows' values
  vector<blk::E> mask;
  HForest shape;
};
static void makeWorld(World &w, const vector<CallShape> &shapes, int lid) {
  const size_t NC = shapes.size();
  w.lid = lid;
  w.shapes = shapes, w.tau.resize(NC), w.calls.resize(NC), w.value.resize(NC);
  for (size_t c = 0; c < NC; c++) {
    Call &k = w.calls[c];
    k.name = "call" + std::to_string(c), k.n = shapes[c].n, k.k = shapes[c].k, k.kappa = 5, k.fn = shapes[c].written, k.t0 = shapes[c].t0;
    if (shapes[c].written) {
      for (u64 t = 0; t < 1ull << k.k; t++) w.tau[c].push_back(1000 * (c + 1) + 7 * t);
      k.tau = &w.tau[c];
    }
    w.cl.push_back(&k);
  }
  w.shape = forestShape(w.cl, w.lid);
  w.pub.assign(w.shape.NT, 0);
  for (size_t c = 0; c < NC; c++) {
    const Call &k = w.calls[c];
    const u64 n = 1ull << k.n, padded = shapes[c].padding;
    forestPubRun(w.shape, w.pub, (int)c, n - padded, padded);
    for (u64 i = 0; i < n; i++) w.value[c].push_back(k.tauAt(i < n - padded ? madeUp64() % (1ull << k.k) : k.t0));
  }
  for (size_t i = 0; i < forestElems(w.shape); i++) w.mask.push_back(blk::E::from(blk::from_u64(madeUp64()), blk::from_u64(madeUp64()), blk::from_u64(madeUp64())));
}

// The prover's side for blk::lkp_run(), with the mask row in a plain array.
struct Side {
  blk::ProverRounds &c;
  blk::Statement &st;
  blk::MaskLedger &ledger;
  blk::LkpProver &heavy;
  const vector<blk::E> &mask;
  vector<blk::E> power, plane;
  blk::ProverRounds &ch() { return c; }
  blk::Statement &statement() { return st; }
  void tie_factor(const blk::E &) {}
  blk::MaskValues<blk::E> values() {
    const vector<blk::E> *m = &mask;
    return [m](blk::u32, blk::u32 index) { return (*m)[index]; };
  }
  // plane[j]: the root of plane j's product tree. power[j] = power[j + 1]^2 plane[j], for tie().
  void begin(const blk::E &theta, const blk::E &alpha) {
    heavy.begin(theta, alpha);
    plane = heavy.plane_roots();
    power = plane;
    for (size_t j = plane.size() - 1; j-- > 0;) power[j] = power[j + 1] * power[j + 1] * plane[j];
  }
  bool tie(blk::SumClaim<blk::E> &&claim, const blk::Summand<blk::E> &S, size_t j, blk::PscOutput<blk::E> &out) {
    vector<vector<blk::E>> maps;
    if (j == 0) maps.push_back({heavy.key_roots()[0]});  // the keys are one part
    maps.push_back({power[j + 1]}), maps.push_back({plane[j]});
    blk::SumcheckCpu<blk::E> inner(S, claim.vars, std::move(maps));
    vector<blk::MaskSlot> fresh = ledger.take(blk::psc_masks(claim, S));
    out = blk::psc_prove<blk::E>(c, std::move(claim), S, inner, std::move(fresh), values());
    return true;
  }
  void note(const char *, const blk::E &) {}  // intermediate values; this prover records none
  bool split(blk::SumClaim<blk::E> &&, const blk::Summand<blk::E> &, size_t, blk::PscOutput<blk::E> &) {
    printf("the test's world is wrong: its keys are one part\n"), exit(2);
  }
  // The unnamed flag marks the last key level, whose two inner masks a caller may take from a
  // second oracle message. Here all masks come from one.
  bool psc(blk::SumClaim<blk::E> &&claim, const blk::Summand<blk::E> &S, blk::LkpStep step, unsigned part, unsigned level,
           const vector<unsigned> &order, bool, blk::PscOutput<blk::E> &out) {
    const std::unique_ptr<blk::SumcheckProver<blk::E>> inner = heavy.sumcheck(step, part, level, S, claim.vars, order);
    vector<blk::MaskSlot> fresh = ledger.take(blk::psc_masks(claim, S));
    out = blk::psc_prove<blk::E>(c, std::move(claim), S, *inner, std::move(fresh), values());
    return true;
  }
};

// The lookup's rounds, then the round that batches all claims: those of forestObjects(), the
// three on the planes and the join (the claim on the calls' key vectors).
static blk::Plan planOf(const HForest &h) {
  blk::Plan p(blk::setting_goldilocks());
  forestPlan(p, "group", h);
  blk::plan_bat_sigma(p, "claims", forestLins(h) + 4);
  return p;
}

struct Proved {
  string unfit;  // `blk::lkp_fault`'s message on the public inputs; if set, nothing was proved
  vector<blk::u8> proof;
  vector<u64> bits;  // plane j, row t, part c: j + 32 (c + 2^cK t)
};
static Proved prove(const World &w) {
  const HForest &h = w.shape;
  const blk::Planned plan = planOf(h).finish();
  blk::Writer out;
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{}); // each chain needs a configuration digest: zeros here
  blk::ProverRounds ch(binder, blk::Digest{}, out);
  const size_t trees = (size_t)1 << h.rK, rows = (size_t)1 << h.d(), calls = (size_t)1 << h.cK;
  const vector<vector<u64>> &value = w.value;
  // A statement with one oracle message, as if SEND (blocks/iop/send.h) had run: a main row for the
  // planes, a mask row.
  blk::Codes codes;
  codes.base.push_back(blk::Code{(unsigned)std::max(6 + h.d(), 13), (size_t)32 << h.d(), 1});
  codes.mask.push_back(blk::Code{13, 4096, 1});
  blk::Statement st(codes);
  blk::OracleShape sent;
  sent.name = "planes", sent.code = 0, sent.main_rows = 1, sent.mask_rows = 1, sent.answers = 1;
  const blk::OracleId o = st.sent(sent);
  st.sampled(blk::SendAccess{}, o, {blk::Row{}}, {}, blk::SendAccess::make(o, 0, (blk::u32)w.mask.size()));
  const blk::GroupId group = st.new_group("group", 0, 0);
  st.join(group, o);
  // The public inputs as blk::lkp_run() takes them. Key number y + 2^rK x: place x of tree y. Row
  // number c + 2^cK t: row t of part c. Part c is call c's table, repeated if it is shorter; the
  // parts from NC on repeat the last call's table. shifted: with several calls, a challenge theta
  // moves call c's keys and rows by c theta (the shift).
  blk::LkpPublic fixed;
  fixed.parts = {(unsigned)h.v()}, fixed.d = h.d(), fixed.mu = 5, fixed.g = h.rK, fixed.shifted = h.NC > 1, fixed.read_from = {o};
  auto paddingRow = [&](int c) { return (u64)c + (w.calls[c].t0 << h.cK); };  // row t0 of part c
  for (int c = 0; c < h.NC; c++) fixed.keys.push_back({(u64)h.first[c], (unsigned)h.iota[c], c, 0});
  for (int y = h.NT; y < (int)trees; y++) fixed.keys.push_back({(u64)y, 0, -1, paddingRow(0)});
  auto rowOf = [&](size_t r) {
    const size_t c = std::min(r % calls, (size_t)h.NC - 1);
    return std::pair<size_t, u64>(c, w.cl[c]->tauAt((r / calls) % (1ull << w.cl[c]->k)));
  };
  fixed.h = h.cK;
  for (size_t c = 0; c < calls; c++) {
    const size_t of = std::min(c, (size_t)h.NC - 1);
    vector<blk::E> written;
    for (u64 x : w.tau[of]) written.push_back(eOf(x));
    fixed.tables.push_back({(u64)c, (unsigned)w.calls[of].k, (unsigned)of, written});
  }
  // The padding, as runs of key numbers: the last places of each call, tree by tree. In tree y the
  // places x0 .. 2^dT - 1 have the key numbers y + 2^rK x.
  for (int c = 0; c < h.NC; c++) {
    const u64 D = 1ull << h.dT, real = (1ull << w.shapes[c].n) - w.shapes[c].padding;
    for (u64 b = 0; b < 1ull << h.iota[c]; b++) {
      const u64 x0 = real <= b * D ? 0 : real - b * D;
      if (x0 < D) fixed.padding.push_back({0, (u64)h.first[c] + b + (x0 << h.rK), (unsigned)h.rK, D - x0, paddingRow(c)});
    }
  }
  std::sort(fixed.padding.begin(), fixed.padding.end(), [](const blk::LkpPadding &a, const blk::LkpPadding &b) { return a.first < b.first; });
  fixed.last_key_level_order = blk::lkp_range_order(h.v() - 1, h.rK);
  if (const char *fault = blk::lkp_fault(fixed)) {
    Proved none;
    none.unfit = fault;
    return none;
  }
  // The key vector of all calls, before the shift, and the multiplicities (`count`): a real key is
  // counted at the first row of its call's part that equals it.
  vector<blk::E> keys((size_t)1 << h.v());
  vector<u64> count(rows, 0);
  for (size_t i = 0; i < keys.size(); i++) {
    const size_t y = i % trees, x = i / trees;
    if (y >= (size_t)h.NT) continue;
    size_t c = h.callOf[y];
    const size_t place = ((size_t)h.b((int)y) << h.dT) + x;
    u64 counted = value[c][place];
    keys[i] = eOf(counted);
    if (place >= value[c].size() - w.shapes[c].padding) continue;  // padding
    size_t r = c;
    while (r < rows && rowOf(r).second != counted) r += calls;
    if (r >= rows) printf("the test's world is wrong: a key is in no row\n"), exit(2);
    count[r]++;
  }
  vector<blk::E> bits(rows * 32, blk::E::zero());
  for (size_t r = 0; r < rows; r++)
    for (size_t j = 0; j < 32; j++)
      if ((count[r] >> j) & 1) bits[j + 32 * r] = blk::E::one();
  blk::LkpCpu heavy(fixed, keys, bits);
  Side side{ch, st, st.masks(o), heavy, w.mask, {}, {}};
  blk::LkpOutput left;
  blk::lkp_run(side, fixed, [o](blk::Row &row, blk::Tensor v) { row.add(o, blk::Part::MAIN, std::move(v)); }, group, left);
  (void)left.keys.take();  // the claim on the key vector: `verify` makes its own, the join
  {
    // The number of claims that the batching challenge batches, counted from lkp_run()'s output:
    // the claims that it added and its claim on the keys. require_rows() ends the process unless
    // it equals the plan's number, which comes from forestLins().
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(left.rows.size() + 1);
    (void)from.draw();
  }
  ch.finish();
  Proved p;
  p.proof = out.bytes();
  for (const blk::E &b : bits) p.bits.push_back(b.c[0].v);
  return p;
}

// The multilinear extension of a at the point z; z[0] belongs to the top bit of the index.
static Ex topFirst(vector<Ex> a, const vector<Ex> &z) {
  for (const Ex &c : z) {
    const size_t half = a.size() / 2;
    for (size_t x = 0; x < half; x++) a[x] = a[x] + c * (a[x + half] - a[x]);
    a.resize(half);
  }
  return a[0];
}
// The verifier's side. forestPad() reads the proof through the transcript `Tr`
// (verifier/src/vf_common.h), here a shell over blk::VerifierRounds, which compares each message's
// size, each round's name and each number of challenges with the plan and ends the process if one
// differs (a fault of the program, not of the proof). `HFork` (verifier/src/vf_subproof_engine.h) batches
// the claims into one equation on the mask row.
// Returns "" if the equation holds, else what failed; "refused: " and the rejection name if
// refuse() threw. pub, mask: the verifier's padding numbers and the mask row, which a case may
// alter. forgets: a host (the code that calls vf_lookup_argument.h)
// that does not add 1: the claims on mask entries only, 2: those on the planes, 3: the join.
// countOneMore adds 1 to the sum that only the count claim reads (the multiplicities sum to the
// number of real keys). Here that claim names no slot, so no changed mask entry shows whether it is
// checked. Without this case the test passed with
// the claim dropped.
static string verify(World &w, const Proved &p, const vector<u64> &pub, const vector<blk::E> &mask, int forgets = 0, bool countOneMore = false) {
  const blk::Planned plan = planOf(w.shape).finish();
  blk::Reader reader(p.proof.data(), p.proof.size());
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{}); // each chain needs a configuration digest: zeros here
  blk::VerifierSession session; // one chain, one verdict
  blk::VerifierRounds channel(binder, reader, session);
  Tr tr{channel};
  HFork F;
  const u32 n = (u32)forestElems(w.shape);
  F.slots(n);
  u32 slot = 0;
  HForest h;
  try {
    h = forestPad(F, tr, w.cl, pub, slot, w.lid);
  } catch (const Refusal &r) {
    return "refused: " + r.name;
  }
  if (slot != n) return "the number of slots";
  F.batchBy(trBatchOpen(tr)); // the number of claims is given once all are added: below
  // What the claims read, computed from the data. The planes are a matrix of 2^lm rows, in the
  // layout that forestPlaneClaims() expects.
  const int lm = 7, wd = h.kT + 5 - lm;
  auto bit = [&](u64 t, u64 c, u64 j) { return p.bits[j + 32 * (c + (t << h.cK))]; };
  vector<Ex> phi[3] = {forestPhiB(h, lm, 0), forestPhiB(h, lm, 1), forestPhiP(lm)};
  auto column = [&](int which, u64 col) {
    const u64 c = col >> wd, high = col & ((1ull << wd) - 1);
    Ex s = eZ();
    for (u64 row = 0; row < (1ull << lm); row++) s = s + eS(phi[which][row], bit((high << (lm - 5)) + (row >> 5), c, row & 31));
    return affC(s);
  };
  vector<Aff> KF;
  for (int c = 0; c < h.NC; c++) {
    vector<Ex> keys;
    for (u64 x : w.value[c]) keys.push_back(eB(x));
    KF.push_back(affC(topFirst(keys, w.cl[c]->zk)));
  }
  const u32 before = F.nClaim;
  if (forgets != 1) forestObjects(F, h);
  if (forgets != 1 && F.nClaim - before != forestLins(h)) return "the number of claims";
  if (forgets != 2)
    forestPlaneLins(F, h, lm, [&](int e, u64 col) { return column(e, col); }, [&](u64 col) { return countOneMore && !col ? column(2, col) + affC(eO()) : column(2, col); });
  if (forgets != 3) F.lin(forestJoin(h, KF));
  vector<Ex> coef;
  Ex sum;
  try {
    F.settle(coef, sum, n);
  } catch (const Refusal &r) {
    return "refused: " + r.name;
  }
  trBatchRows(tr, F.nClaim); // the claims added; `settle` has rejected a host that forgets some
  if (!channel.finish() || !session.accepted()) return "the channel";
  for (u32 j = 0; j < n; j++) sum = sum + coef[j] * fromBlkE(mask[j]);
  return isZ(sum) ? "" : "the equation";
}

// For each fixture: the honest proof's equation holds, with the expected numbers of slots and
// claims; HFork::settle() rejects by name a host that omits one of its three groups of claims;
// the equation fails for another number of padding
// keys on the verifier's side, for countOneMore and for each single changed mask entry. Then three
// fixtures (`nothing`) with a call of nothing but padding.
int main() {
  // Shows the field's prime p by arithmetic and not by a name: 0 - 1 = p - 1, in the verifier's
  // functions and in blocks/. The two values are printed, not compared.
  printf("0 - 1 = 0x%llx in the verifier's field, 0x%llx in blocks/\n", (unsigned long long)fsub((u64)0, (u64)1), (unsigned long long)(blk::E::zero() - blk::E::one()).c[0].v);
  refuseThrows = true; // `refuse` throws a `Refusal`, which the test catches where it expects one
  int failures = 0;
  auto expect = [&](bool ok, const char *what) {
    if (!ok) failures++, printf("FAIL: %s\n", what);
  };
  size_t slots = 0, claims = 0;
  struct Spec {
    const char *name;
    int lid;
    vector<CallShape> calls;
  };
  // Name, `lid`, and each call's {n, k, written, t0, padding}.
  vector<Spec> worlds = {
      {"two calls, a range table and a written-out one", 10, {{11, 5, false, 0, 5}, {10, 6, true, 3, 3}}},
      {"three calls: a number of a call is filled up", 10, {{10, 5, false, 2, 4}, {10, 6, true, 0, 1}, {10, 5, true, 31, 6}}},
      {"one call, one tree: no shift, no variable for a tree or a call", 10, {{10, 5, false, 0, 3}}},
      {"one call, a written-out table, no padding", 10, {{10, 6, true, 0, 0}}},
      {"one call of four trees, one of them full of padding", 10, {{12, 5, false, 7, 1500}}},
      {"sixteen calls", 10, {}},
      {"the cap on a tree's depth above the smallest call", 13, {{11, 5, false, 0, 5}, {10, 6, true, 3, 3}}},
      {"a call with two keys that are no padding, first in the list", 10, {{10, 5, false, 2, 1022}, {10, 6, true, 3, 3}}},
      {"a call with two keys that are no padding, last of three", 10, {{10, 5, false, 2, 4}, {11, 6, true, 3, 3}, {10, 5, true, 9, 1022}}},
      {"sizes not sorted in the list", 10, {{10, 5, false, 2, 4}, {12, 6, true, 0, 1}, {11, 5, true, 31, 6}}},
      {"in each call two keys are no padding", 10, {{10, 5, false, 2, 1022}, {10, 6, true, 3, 1022}}},
      {"two calls with one range table", 10, {{10, 5, false, 2, 4}, {10, 5, false, 2, 4}}},
      {"five calls: three numbers filled up; tables of 2^5 to 2^8 rows", 10,
       {{10, 5, false, 2, 4}, {10, 8, true, 200, 4}, {11, 6, false, 2, 0}, {10, 7, true, 2, 4}, {10, 5, true, 2, 4}}},
  };
  for (int c = 0; c < 16; c++) worlds[5].calls.push_back({10, 5 + c % 3, c % 2 == 1, (u64)(c % 7), (u64)(c * 3)});
  for (size_t wi = 0; wi < worlds.size(); wi++) {
    const Spec &spec = worlds[wi];
    const int before = failures;
    World w;
    makeWorld(w, spec.calls, spec.lid);
    const HForest &h = w.shape;
    {
      blk::LkpPublic S;
      S.parts = {(unsigned)h.v()}, S.d = h.d(), S.mu = 5;
      expect(forestElems(h) == blk::lkp_masks(S), "as many slots as blocks' look-up has mask entries");
    }
    const Proved honest = prove(w);
    expect(honest.unfit.empty(), ("blocks takes the public inputs: " + honest.unfit).c_str());
    if (!honest.unfit.empty()) continue;
    string why = verify(w, honest, w.pub, w.mask);
    expect(why.empty(), ("the honest proof: " + why).c_str());
    expect(verify(w, honest, w.pub, w.mask, 1) == "refused: internal-claim-not-added-look-up-mask-claims", "a host that forgets the claims on mask entries");
    expect(verify(w, honest, w.pub, w.mask, 2) == "refused: internal-claim-not-added-look-up-plane-claims", "a host that forgets the claims on the planes");
    expect(verify(w, honest, w.pub, w.mask, 3) == "refused: internal-claim-not-added-look-up-join", "a host that forgets the join");
    // A padding key is a row of its call's table, so a key from outside the table must still be
    // noticed. A real key that equals the padding key passes as padding, rightly: it is a row of
    // the table. Neither is tried here: both need a prover that passes a key off as padding.
    vector<u64> pub = w.pub;
    size_t y = 0;  // a tree with room for another padding key: each call has two real keys or more
    while (y < pub.size() && pub[y] >= (1ull << h.dT)) y++;
    if (y >= pub.size()) printf("the test's world is wrong: no tree has room for a padding key more\n"), exit(2);
    pub[y]++;
    expect(verify(w, honest, pub, w.mask) == "the equation", "another number of padding keys on the verifier's side");
    expect(verify(w, honest, w.pub, w.mask, 0, true) == "the equation", "the multiplicities sum to one more than the number of keys, in the count's form alone");
    vector<blk::E> mask = w.mask;
    int noticed = 0;
    for (size_t j = 0; j < mask.size(); j++) {
      mask[j] += blk::E::one();
      noticed += verify(w, honest, w.pub, mask) == "the equation";
      mask[j] = w.mask[j];
    }
    expect(noticed == (int)mask.size(), "every mask entry stands in the equation");
    slots += forestElems(h), claims += forestLins(h) + 4;
    if (failures != before) printf("  (in the world: %s)\n", spec.name);
  }
  // A call of nothing but padding: an accepted proof would say nothing about it. Both
  // implementations must reject these public inputs before a message is read. call: the call that
  // the verifier names.
  const struct {
    const char *name;
    vector<CallShape> calls;
    const char *call;
  } nothing[] = {{"a call that is all padding, first in the list", {{10, 5, false, 2, 1024}, {10, 6, true, 3, 3}}, "call0"},
                 {"a call of two trees that is all padding, last of three", {{10, 5, false, 2, 4}, {10, 6, true, 3, 3}, {11, 5, true, 9, 2048}}, "call2"},
                 {"all keys are padding", {{10, 5, false, 2, 1024}, {10, 6, true, 3, 1024}}, "call0"}};
  for (const auto &spec : nothing) {
    World w;
    makeWorld(w, spec.calls, 10);
    const Proved none = prove(w);
    expect(none.unfit.find("no key is left that is no padding") != string::npos, (string(spec.name) + ": blocks says: '" + none.unfit + "'").c_str());
    const string why = verify(w, none, w.pub, w.mask);
    expect(why == string("refused: forest-only-padding-") + spec.call, (string(spec.name) + ": this verifier says: '" + why + "'").c_str());
    // With one padding key fewer in the call's last tree, the rejection no longer names this call:
    // another call is all padding, or the verifier goes on to the messages, and `none` has none.
    vector<u64> nearly = w.pub;
    for (size_t y = nearly.size(); y-- > 0;)
      if (w.cl[w.shape.callOf[y]]->name == spec.call) {
        nearly[y]--;
        break;
      }
    const string next = verify(w, none, nearly, w.mask);
    expect(next.find(string("forest-only-padding-") + spec.call) == string::npos, (string(spec.name) + ", one place fewer: this verifier says: '" + next + "'").c_str());
  }
  if (failures)
    printf("forest_twin_test: FAILED (%d checks failed, see above)\n", failures);
  else
    printf("forest_twin_test: ok (%zu worlds: %zu slots, %zu claims; in each, a host that forgets one kind of claims is refused by name, "
           "another padding number and every changed mask entry are noticed; "
           "3 worlds with a call of nothing but padding refused by both texts)\n", worlds.size(), slots, claims);
  return failures ? 1 : 0;  // not the count: 256 would read as 0, and 2 means a faulty fixture
}
