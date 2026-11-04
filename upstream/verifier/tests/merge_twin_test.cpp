// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the verifier's code for the merge against a prover built from the library (blocks/iop) and
// against the weight matrix itself. The merge ends a sub-proof (`Fork` in identifiers: the argument
// for one part of the forward pass) that has linear claims about the committed weight matrix W: it
// batches the claims and binds W's column bits at one point. Under test:
// wsForkEnd() (verifier/src/vf_weights_layout.h), wsForkMerge() and wsForkLin() (verifier/src/vf_weights_merge.h).
//
// Each fixture (`World`) is a small synthetic layout, a synthetic W and claims on some of its units
// (a unit: W's rows for one kind of weight matrix of one layer, the embedding or the output head).
//
// A pass shows that the verifier's code and the library's prover agree on message format and
// algebra for these shapes. The test runs no GPU code, no commitment and no sub-proof before the
// merge.
// Build, from verifier/src:
//   g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
//   -I ../../blocks -I ../../prover -I ../../blocks/iop/tests ../tests/merge_twin_test.cpp
//   -o <a folder of your own>/merge_twin_test
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "vf_weights_layout.h"

#include "iop/batt_shown.h"
#include "numbers/blocks.h"
#include "toy_support.h"

// Converts a field element from the verifier's type to the library's.
static blk::E eOf(const Ex &x) { return blk::E::from(blk::from_u64(eC(x, 0)), blk::from_u64(eC(x, 1)), blk::from_u64(eC(x, 2))); }
static u64 rngState = 23;
static Ex madeUp() {
  Ex x;
  for (int i = 0; i < CE; i++) {
    rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
    eSet(x, i, (rngState >> 3) % FIELD_P);
  }
  return x;
}
static vector<Ex> madeUpList(size_t n) {
  vector<Ex> v(n);
  for (Ex &x : v) x = madeUp();
  return v;
}

// Slots number the proof's scalars in the order sent. The segment is the sub-proof's part of a
// request's proof ("link" in this file's messages).
struct World {
  WsLay lay;
  vector<WsClaim> claims;
  vector<vector<Ex>> W;      // m0 rows of 2^lk columns; in [-128, 127], small like base-256 digits
  vector<vector<Ex>> gamma;  // gamma[a]: claim a's weights on W's rows, without a scale factor
  vector<blk::E> mask;       // the segment's mask row: one entry per slot
  blk::E c;                  // masks the last scalar, fv'; an entry of another oracle message
  u32 slots = 0;
};
// <weights, W[., X]> as a table over the columns.
static vector<Ex> onColumns(const World &w, const vector<vector<Ex>> &W, const vector<Ex> &weights) {
  vector<Ex> t((size_t)1 << w.lay.lk, eZ());
  for (u64 y = 0; y < w.lay.m0; y++)
    if (!isZ(weights[y]))
      for (size_t X = 0; X < t.size(); X++) t[X] = t[X] + weights[y] * W[y][X];
  return t;
}
// The multilinear extension of table a at z; z[0] belongs to the top bit of the index.
static Ex topFirst(vector<Ex> a, const vector<Ex> &z) {
  for (const Ex &c : z) {
    const size_t half = a.size() / 2;
    for (size_t x = 0; x < half; x++) a[x] = a[x] + c * (a[x + half] - a[x]);
    a.resize(half);
  }
  return a[0];
}
static World makeWorld(int lk, bool moe, const vector<u32> &units) {
  World w;
  WsShape s;
  s.D = 32, s.L = 2, s.d = 12, s.dh = 2, s.dkv = 4, s.dq = 8, s.f = 6, s.fe = 4, s.NE = 3, s.V = 5, s.eh = 1;
  wsSetMoe(s, moe);
  w.lay.build(s, (u32)lk);
  w.W.assign(w.lay.m0, vector<Ex>((size_t)1 << lk));
  for (auto &row : w.W)
    for (Ex &x : row) x = eI((i64)(eC(madeUp(), 0) % 256) - 128);
  for (u32 ui : units) {
    WsClaim c;
    c.ui = ui % (u32)w.lay.U.size();
    for (const WsGrp &g : w.lay.U[c.ui].g) {
      WsFac f;
      if (g.cls == 'A') f.AV = madeUpList(g.R);
      else f.HV = madeUpList(g.o), f.pt = madeUpList(g.z);
      c.f.push_back(f);
    }
    w.claims.push_back(c);
    vector<Ex> gam(w.lay.m0, eZ());
    wsGammaAdd(gam, w.lay, wsRowPart(w.lay, c, eO()), eO());  // tested by weights_twin_test.cpp
    w.gamma.push_back(gam);
  }
  w.slots = (u32)(w.claims.size() + wsForkSlots(lk));
  for (u32 j = 0; j < w.slots; j++) w.mask.push_back(eOf(madeUp()));
  w.c = eOf(madeUp());
  return w;
}
static blk::Plan planOf(const World &w) {
  blk::Plan p(blk::setting_goldilocks());
  blk::plan_scalars(p, "the claims' values", "shown before the merge", w.claims.size());
  blk::plan_none(p, blk::R_PSC_VALUES, "shown before the merge");
  blk::plan_batt_shown(p, "merge", w.claims.size(), (unsigned)w.lay.lk);
  blk::plan_bat_sigma(p, "claims", 1);
  return p;
}

// prove() sends each claim's value, masked by an entry of the segment's mask row, and gets the
// batching challenge x (rho_a = x^a). It then runs PSC, the sumcheck with a hiding polynomial
// (blocks/iop/psc.h), over W's column bits on
//   F(X) = sum_a rho_a eps~_a(X) <gamma_a, W[., X]>
// eps~_a(X) is claim a's weight on column X (wsEpsT()); gamma_a is `World::gamma[a]`. Both come
// from the verifier's own code, so an error there can pass unnoticed;
// verifier/tests/weights_twin_test.cpp compares that code with the prover's and the definition.
// PSC's last scalar is fv' = F(rm) + beta c: rm is the sumcheck's point, beta its last challenge.
// The last round batches the sub-proof's remaining claims: here one.
// Plan, summand, challenge x and order of masks are those of batt_shown_prove()
// (blocks/iop/batt_shown.h). That function does not fit: its claims weight the columns by
// eq(psi, X) only, but a claim here can have an explicit vector (`WsFac::AV`) for an index group
// that goes entirely to the columns.
enum class Cheat { NONE, OTHER_MATRIX };
static vector<blk::u8> prove(const World &w, Cheat cheat) {
  const blk::Planned plan = planOf(w).finish();
  blk::Writer out;
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{});  // every chain needs one; all zeros on both sides
  blk::ProverRounds ch(binder, blk::Digest{}, out);
  const size_t J = w.claims.size();
  const int lk = w.lay.lk;
  blk::MaskLedger own(blk::SendAccess::make(0, 0, w.slots)), index(blk::SendAccess::make(1, 0, 1));
  const blk::MaskValues<blk::E> masks = [&w](blk::u32 oracle, blk::u32 at) { return oracle ? w.c : w.mask.at(at); };
  vector<vector<Ex>> held = w.W;
  if (cheat == Cheat::OTHER_MATRIX) held[w.lay.U[w.claims[0].ui].off][0] = held[w.lay.U[w.claims[0].ui].off][0] + eO();
  vector<blk::MaskSlot> under = own.take(J);
  vector<blk::E> shown(J);
  vector<vector<Ex>> table;
  for (size_t a = 0; a < J; a++) {
    // On `held`: under OTHER_MATRIX the proof is self-consistent; only verify()'s (E) can fail.
    table.push_back(onColumns(w, held, w.gamma[a]));
    Ex v = eZ();
    for (size_t X = 0; X < table[a].size(); X++) {
      vector<Ex> bits(lk);
      for (int i = 0; i < lk; i++) bits[i] = eB((X >> (lk - 1 - i)) & 1);
      v = v + wsEpsT(w.lay, w.claims[a], bits) * table[a][X];
    }
    shown[a] = eOf(v) + w.mask[a];
    ch.send(shown[a]);
  }
  ch.close_empty(blk::R_PSC_VALUES);
  const blk::E x = blk::batt_draw(ch, J);
  blk::SumClaim<blk::E> claim;
  claim.vars.assign(lk, blk::Weight<blk::E>::plain(3));
  claim.target = blk::E::zero();
  vector<Ex> rho;
  blk::E power = blk::E::one();
  for (size_t a = 0; a < J; a++, power = power * x) {
    claim.target += power * shown[a];
    claim.masks.push_back({std::move(under[a]), power});
    rho.push_back(fromBlkE(power));
  }
  const blk::ToyMap<blk::E> F = [&](const vector<blk::E> &z) {
    vector<Ex> point;
    for (const blk::E &q : z) point.push_back(fromBlkE(q));
    Ex f = eZ();
    for (size_t a = 0; a < J; a++) f = f + rho[a] * wsEpsT(w.lay, w.claims[a], point) * topFirst(table[a], point);
    return eOf(f);
  };
  const blk::Summand<blk::E> S = blk::batt_shown_summand();
  blk::DefinitionSumcheck<blk::E> inner(claim.vars, S, {F});
  (void)blk::psc_prove<blk::E>(ch, std::move(claim), S, inner, blk::batt_shown_slots((unsigned)lk, own, index), masks);
  {
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(1);
    (void)from.draw();
  }
  ch.finish();
  return out.bytes();
}
// Returns "" or the name of the first check that failed. claims, W, mask, c: the verifier's view,
// which main() alters. The functions under test read the proof through `Tr::one`
// (verifier/src/vf_common.h): the library's VerifierRounds, which aborts if a message's size, a
// round's name or a number of challenges differs from the plan.
//   (U) The sub-proof's one remaining claim, on the segment's mask entries only, holds. The entry
//       at the slot of fv' has coefficient 0: it masks nothing.
//   (E) fv' = <Gam, W[., rm]> + beta c, with wsGammaAdd() computing the row weights Gam from the
//       returned row parts: the pending claim that the sub-proof leaves on W ("the row that leaves
//       the sub-proof" in the messages) is true of the matrix.
// Also checked: the merge uses as many slots as wsForkSlots() returns and leaves exactly one claim.
// Not checked, though soundness needs it: that c's oracle message is sent before beta is sampled.
// That is left to the frame, the part of a request's proof outside the sub-proofs.
static string verify(const World &w, const vector<blk::u8> &proof, const vector<WsClaim> &claims, const vector<vector<Ex>> &W, const vector<blk::E> &mask, const blk::E &c) {
  const blk::Planned plan = planOf(w).finish();
  blk::Reader reader(proof.data(), proof.size());
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{});
  blk::VerifierSession session;
  blk::VerifierRounds channel(binder, reader, session);
  Tr tr{channel};
  HFork F;
  F.slots(w.slots);
  u32 slot = 0;
  vector<u32> sv;
  WsForkOut o;
  vector<Ex> coef;
  Ex sum;
  try {
    const vector<Ex> shown = trRead(tr, claims.size());
    trEmpty(tr, RoundName::PSC_VALUES);
    for (const Ex &v : shown) F.hide(slot, v, eO()), sv.push_back(slot++);
    o = wsForkEnd(F, tr, slot, w.lay, claims, sv);
    if (slot != w.slots) return "the number of slots";
    F.batchBy(trBatchOpen(tr));
    wsForkLin(F, o);
    if (F.nClaim != 1) return "the number of claims";
    F.settle(coef, sum, w.slots);
    trBatchRows(tr, F.nClaim);
  } catch (const Refusal &r) {
    return "refused: " + r.name;
  }
  if (!channel.finish() || !session.accepted()) return "the channel";
  if (!isZ(coef[o.sFv])) return "the link's entry under the number of fv' has a coefficient";
  for (u32 j = 0; j < w.slots; j++) sum = sum + coef[j] * fromBlkE(mask[j]);
  if (!isZ(sum)) return "(U)";
  vector<Ex> Gam(w.lay.m0, eZ());
  for (const WsRecClaim &r : o.rec) wsGammaAdd(Gam, w.lay, r, eO());
  return o.fvp == topFirst(onColumns(w, W, Gam), o.rm) + o.beta * fromBlkE(c) ? "" : "(E)";
}

// Checks each fixture's honest proof, then single changes. verify() must fail at (U) for a false
// claim value or a changed mask entry that masks something; at (E) for a changed W (either side),
// c or claim order; by name for a misshapen claim; anywhere for a changed scalar.
int main() {
  printf("0 - 1 = 0x%llx in the verifier's field, 0x%llx in blocks/\n", (unsigned long long)fsub((u64)0, (u64)1), (unsigned long long)(blk::E::zero() - blk::E::one()).c[0].v);
  refuseThrows = true;
  int failures = 0;
  string world;
  auto expect = [&](bool ok, const string &what) {
    if (!ok) failures++, printf("FAIL: %s: %s\n", world.c_str(), what.c_str());
  };
  struct Spec {
    const char *name;
    int lk;
    bool moe;
    vector<u32> units;
  };
  const vector<Spec> specs = {
      {"one claim", 3, false, {1}},
      {"two claims on one unit", 3, false, {2, 2}},
      {"a claim on a unit whose whole-index groups go to the columns", 4, false, {0, 3}},
      {"expert layers; five claims", 4, true, {0, 7, 8, 10, 11}},
      {"embedding and head", 3, true, {1000000, 1000001, 5}},  // placeholders: see the loop below
      {"one column bit", 1, true, {0, 4, 9}},
      {"five column bits", 5, false, {0, 1, 2, 3}},
  };
  size_t worlds = 0, scalars = 0, entries = 0, claims = 0;
  for (Spec spec : specs) {
    world = spec.name;
    if (spec.units[0] >= 1000000) {  // the last two units of the layout
      World probe = makeWorld(spec.lk, spec.moe, {0});
      spec.units[0] = (u32)probe.lay.U.size() - 2, spec.units[1] = (u32)probe.lay.U.size() - 1;
    }
    const World w = makeWorld(spec.lk, spec.moe, spec.units);
    worlds++, claims += w.claims.size();
    const vector<blk::u8> honest = prove(w, Cheat::NONE);
    expect(honest.size() == blk::DIGEST_BYTES + (size_t)w.slots * blk::E_BYTES, "the proof is the chain's random bytes and one scalar for each slot");
    const string why = verify(w, honest, w.claims, w.W, w.mask, w.c);
    expect(why.empty(), "the honest proof: " + why);
    if (!why.empty()) continue;
    expect(verify(w, prove(w, Cheat::OTHER_MATRIX), w.claims, w.W, w.mask, w.c) == "(E)", "a prover that is consistent on a matrix which differs in one cell: (U) holds, the row that leaves is false");
    expect(verify(w, honest, w.claims, w.W, w.mask, w.c + blk::E::one()) == "(E)", "another c");
    {
      vector<vector<Ex>> other = w.W;
      const u64 y = w.lay.U[w.claims[0].ui].off;
      other[y][1] = other[y][1] + eO();
      expect(verify(w, honest, w.claims, other, w.mask, w.c) == "(E)", "another matrix on the verifier's side");
    }
    if (w.claims.size() > 1 && w.claims[0].ui != w.claims[1].ui) {
      vector<WsClaim> crossed = w.claims;
      std::swap(crossed[0], crossed[1]);
      expect(verify(w, honest, crossed, w.W, w.mask, w.c) == "(E)", "two claims in each other's place on the verifier's side");
    }
    {
      vector<WsClaim> misshapen = w.claims;
      WsFac &f = misshapen[0].f[0];
      (f.AV.empty() ? f.HV : f.AV).push_back(eO());
      expect(verify(w, honest, misshapen, w.W, w.mask, w.c) == "refused: internal-claim-factors", "a claim with a factor of another length than its group's is refused by name");
    }
    int noticed = 0;
    for (u32 j = 0; j < w.slots; j++) {
      vector<blk::E> mask = w.mask;
      mask[j] += blk::E::one();
      noticed += verify(w, honest, w.claims, w.W, mask, w.c) == "(U)";
    }
    expect(noticed == (int)w.slots - 1, "every mask entry of the link stands in (U), but the one under the number of fv'");
    entries += w.slots - 1;
    for (u32 j = 0; j < w.slots; j++) {
      vector<blk::u8> other = honest;
      other[blk::DIGEST_BYTES + (size_t)j * blk::E_BYTES] ^= 1;
      expect(!verify(w, other, w.claims, w.W, w.mask, w.c).empty(), "scalar " + std::to_string(j) + " changed");
      scalars++;
    }
  }
  if (failures) return printf("merge_twin_test: FAILED (%d checks)\n", failures), 1;
  printf("merge_twin_test: ok (%zu worlds, %zu claims; in each the row that leaves the sub-proof is true of the matrix; another matrix on either side, another c, claims exchanged, %zu changed mask entries and %zu changed scalars noticed)\n",
         worlds, claims, entries, scalars);
  return 0;
}
