// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the verifier's functions for the token sumcheck (verifier/src/vf_sumcheck_token.h) and for two types
// of sumcheck in the attention sub-proof (`Fork` in identifiers): SC3, on products of two maps,
// and SCD, whose rounds differ in degree (verifier/src/vf_sumcheck_attention.h). The proofs come from psc_prove()
// (blocks/iop/psc.h), the library's prover for PSC, the sumcheck with a hiding polynomial. Both
// sides run on a hash chain (`Tr`, for transcript, in identifiers) under planOf()'s plan.
//
// Each test case (`World`) is one sumcheck on maps, the multilinear extensions of synthetic tables
// (`madeUp` in identifiers). Its start is the claimed sum; its end values are the maps' values at
// the challenge point; a message slot (`slot`) is the index of a sent scalar and of its entry in
// the prover's mask row. The honest proof must pass, with as many slots as the verifier's functions
// announce. These must fail: a prover on another table, any one
// mask entry or scalar changed, another public value on the verifier's side. A pass shows that
// verifier/src and blocks/ agree on message format and algebra for these shapes. No GPU code,
// commitment or sub-proof runs: that a sub-proof ties each end value to the right map is untested.
// Build, from verifier/src: g++ -std=c++17 -O2 -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
//   -I ../../blocks -I ../../prover -I ../../blocks/iop/tests ../tests/sumchecks_twin_test.cpp
//   -o <a folder of your own>/sumchecks_twin_test
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "vf_sumcheck_attention.h"

#include "attn/plan_rounds.h"
#include "hiding/token_plan.h"
#include "iop/psc.h"
#include "numbers/blocks.h"
#include "toy_support.h"

// `E`: the library's extension-field element; `Ex`: the verifier's.
using E = blk::E;
static E eOf(const Ex &x) { return E::from(blk::from_u64(eC(x, 0)), blk::from_u64(eC(x, 1)), blk::from_u64(eC(x, 2))); }
static u64 rngState = 41;
static E madeUp() {
  Ex x;
  for (int i = 0; i < CE; i++) {
    rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
    eSet(x, i, (rngState >> 3) % FIELD_P);
  }
  return eOf(x);
}
static vector<E> madeUpList(size_t n) {
  vector<E> v(n);
  for (E &x : v) x = madeUp();
  return v;
}
static vector<Ex> exList(const vector<E> &v) {
  vector<Ex> o;
  for (const E &x : v) o.push_back(fromBlkE(x));
  return o;
}
static E eq1(const E &a, const E &b) { return a * b + (E::one() - a) * (E::one() - b); }
static E eqAll(const vector<E> &a, const E *b) {
  E p = E::one();
  for (size_t i = 0; i < a.size(); i++) p = p * eq1(a[i], b[i]);
  return p;
}
// Multilinear extension of `table` in the variables `on`; on[0] goes with the index's top bit.
static blk::ToyMap<E> mapOn(vector<E> table, vector<int> on) {
  return [table, on](const vector<E> &z) {
    vector<E> a = table;
    for (int v : on) {
      const size_t half = a.size() / 2;
      for (size_t x = 0; x < half; x++) a[x] = a[x] + z[v] * (a[x + half] - a[x]);
      a.resize(half);
    }
    return a[0];
  };
}
static vector<int> range(int first, int n) {
  vector<int> v;
  for (int i = 0; i < n; i++) v.push_back(first + i);
  return v;
}

// What `World::read` returns: the claim that the sumcheck leaves, the first end value's slot, the
// challenges.
struct Read {
  Aff rowU;
  u32 sEnd = 0;
  vector<Ex> r;
};
struct World {
  string name;
  // Not tried: a start combining several slots, as in AttentionSubProof::run() (verifier/src/vf_attention_check.h).
  bool hiddenStart = true;  // hidden, sent alone before the hiding polynomial's sum; else public 0
  std::function<void(blk::Plan &)> plan;  // the start's scalar, if any, and the sumcheck
  // `Weight`'s integer (`d`): a round polynomial's number of coefficients (degree + 1).
  vector<blk::Weight<E>> vars;
  blk::Summand<E> S;
  vector<vector<E>> table;  // the shown maps' tables
  vector<vector<int>> on;   // and the variables that each depends on
  u32 slots = 0;            // as the verifier's functions announce them
  // `bent`: if 1, the verifier's side uses another public value (equality point, weight, summand)
  std::function<Read(HFork &, Tr &, u32 &, int bent)> read;
  vector<E> mask;
};
// The plan (the list of the chain's rounds), built with the programs' planning functions
// (prover/hiding/token_plan.h, prover/attn/plan_rounds.h). blk::VerifierRounds ends the process if
// a round's name, message size or number of challenges differs from it.
static blk::Plan planOf(const World &w) {
  blk::Plan p(blk::setting_goldilocks());
  w.plan(p);
  blk::plan_bat_sigma(p, "claims", 1 + w.table.size());
  return p;
}
enum class Cheat { NONE, OTHER_TABLE };
// The library's prover. DefinitionSumcheck (blocks/iop/tests/toy_support.h) computes each round
// polynomial from the sum's definition. `OTHER_TABLE`: a consistent run on a changed table; what
// then fails is the claim on that map's end value.
static vector<blk::u8> prove(const World &w, Cheat cheat) {
  const blk::Planned plan = planOf(w).finish();
  blk::Writer out;
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{});  // configuration digest, required: zeros on both sides
  blk::ProverRounds ch(binder, blk::Digest{}, out);
  blk::MaskLedger ledger(blk::SendAccess::make(0, 0, (blk::u32)w.mask.size()));  // by the library's own count (`main`), not by what the verifier's functions announce
  vector<blk::ToyMap<E>> maps;
  for (size_t i = 0; i < w.table.size(); i++) {
    vector<E> t = w.table[i];
    if (cheat == Cheat::OTHER_TABLE && i == 0) t[0] += E::one();
    maps.push_back(mapOn(t, w.on[i]));
  }
  blk::DefinitionSumcheck<E> inner(w.vars, w.S, maps);
  blk::SumClaim<E> claim;
  claim.vars = w.vars;
  claim.target = E::zero();
  if (w.hiddenStart) {
    vector<blk::MaskSlot> under = ledger.take(1);
    claim.target = inner.sum() + w.mask[0];
    claim.masks.push_back({std::move(under[0]), E::one()});
    ch.send(claim.target);
  }
  const size_t fresh = blk::psc_masks(claim, w.S);
  (void)blk::psc_prove<E>(ch, std::move(claim), w.S, inner, ledger.take(fresh), blk::mask_row(w.mask));
  {
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(1 + w.table.size());
    (void)from.draw();
  }
  ch.finish();
  return out.bytes();
}
// The verifier's side. The transcript `tr` wraps the chain's channel (`Tr::one`), which reads
// the proof. verify() adds the claim that the sumcheck leaves and, for each end value, the claim
// that it is its map's value at the challenge point. Returns "" if the batched equation holds on
// `mask`, else what failed. Rounds test no equation (the verifier fills in the unsent
// coefficient), so a well-formed false proof fails at "the equation".
static string verify(const World &w, const vector<blk::u8> &proof, const vector<E> &mask, int bent = 0) {
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
  vector<Ex> coef;
  Ex sum;
  try {
    const Read o = w.read(F, tr, slot, bent);
    if (slot != w.slots) return "the number of slots";
    F.batchBy(trBatchOpen(tr));
    F.lin(o.rowU);
    vector<E> point;
    for (const Ex &x : o.r) point.push_back(eOf(x));
    for (size_t i = 0; i < w.table.size(); i++) F.lin(affSlot(o.sEnd + (u32)i) - affC(fromBlkE(mapOn(w.table[i], w.on[i])(point))));  // in a sub-proof: a linear claim on committed cells
    F.settle(coef, sum, w.slots);
    trBatchRows(tr, F.nClaim);
  } catch (const Refusal &r) {
    return "refused: " + r.name;
  }
  if (!channel.finish() || !session.accepted()) return "the channel";
  for (u32 j = 0; j < w.slots; j++) sum = sum + coef[j] * fromBlkE(mask[j]);
  return isZ(sum) ? "" : "the equation";
}

static void tables(World &w, const vector<vector<int>> &on) {
  w.on = on;
  for (const vector<int> &v : on) w.table.push_back(madeUpList((size_t)1 << v.size()));
}
// A token sumcheck: start = sum_t eq(P, t) s(t) over t < 2^lt, in lt rounds. s (`summand`) is a
// public polynomial of degree `degree` in the values at t of `shown` maps.
static World tokenWorld(const string &name, int lt, u32 shown, int degree, std::function<E(const vector<E> &)> summand) {
  World w;
  w.name = name;
  const vector<E> P = madeUpList(lt);
  for (const E &z : P) w.vars.push_back(blk::Weight<E>::equality(z, 3));
  w.S = blk::Summand<E>{shown, 0, (unsigned)degree, [summand](const vector<E> &, const vector<E> &y) { return vector<E>{summand(y)}; }};
  tables(w, vector<vector<int>>(shown, range(0, lt)));
  w.plan = [lt, shown, degree](blk::Plan &p) { blk::plan_token_sum(p, "token sum-check", lt, shown, degree); };
  w.slots = (u32)tscSlots(lt, shown, degree);
  w.read = [P, lt, shown, degree, summand](HFork &F, Tr &tr, u32 &slot, int bent) {
    vector<Ex> point = exList(P);
    if (bent) point[0] = point[0] + eO();
    TokenSumcheckState o = tscIn(F, tr, slot, point.data(), lt, shown, degree);
    vector<E> y;
    for (u32 i = 0; i < shown; i++) y.push_back(eOf(F.shown(o.sEnd + i)));
    return Read{pscRow(F, o.p, fromBlkE(summand(y))), o.sEnd, o.r};
  };
  return w;
}
// The form used for the output head (verifier/src/vf_output_head_check.inc): tscRounds() runs lt rounds over
// the tokens; the caller then runs `more` rounds over an index that is summed plainly, as rounds of
// the form `RoundForm::EQ2` (verifier/src/vf_sumcheck_receive.h) at the coordinate 1/2, each of which gives
// half the plain sum.
static World headWorld(int lt, int more) {
  World w;
  w.name = "token sum-check that goes on over an index summed plainly (the head's)";
  const vector<E> P = madeUpList(lt);
  const E half = blk::inv(E::one() + E::one());
  for (const E &z : P) w.vars.push_back(blk::Weight<E>::equality(z, 3));
  for (int i = 0; i < more; i++) w.vars.push_back(blk::Weight<E>::equality(half, 3));
  w.S = blk::Summand<E>{2, 0, 2, [](const vector<E> &, const vector<E> &y) { return vector<E>{y[0] * y[1]}; }};
  tables(w, {range(0, lt + more), range(0, lt + more)});
  w.plan = [lt, more](blk::Plan &p) { blk::plan_token_sum(p, "the head's", lt, 2, 2, more); };
  w.slots = (u32)tscSlots(lt, 2, 2, more);
  w.read = [P, half, lt, more](HFork &F, Tr &tr, u32 &slot, int bent) {
    const vector<Ex> point = exList(P);
    TokenSumcheckState o = tscRounds(F, tr, slot, point.data(), lt);
    for (int i = 0; i < more; i++) o.r.push_back(roundIn(F, tr, slot, RoundForm::EQ2, o.p, fromBlkE(half) + (bent && !i ? eO() : eZ())));
    o.sEnd = pscEnds(F, tr, slot, o.p, 2, 2);
    return Read{pscRow(F, o.p, F.shown(o.sEnd) * F.shown(o.sEnd + 1)), o.sEnd, o.r};
  };
  return w;
}
// SC3 (round form `SUM3`): sum_b sum_x W_b(x) A_b(x) B_b(x) over nb pairs (A_b, B_b), with public
// W_b(x) = om_b eq(t, x). `bits`: the start is the public 0, as for attention's index bits:
// A in {0, 1}, B = A - 1.
static World sc3World(const string &name, int nv, size_t nb, bool bits) {
  World w;
  w.name = name;
  w.hiddenStart = !bits;
  const vector<E> t = madeUpList(nv), om = madeUpList(nb);
  w.vars.assign(nv, blk::Weight<E>::plain(4));
  w.S = blk::Summand<E>{(unsigned)(2 * nb), 0, 2, [t, om, nb](const vector<E> &x, const vector<E> &y) {
                          E s = E::zero();
                          for (size_t b = 0; b < nb; b++) s += om[b] * y[2 * b] * y[2 * b + 1];
                          return vector<E>{eqAll(t, x.data()) * s};
                        }};
  tables(w, vector<vector<int>>(2 * nb, range(0, nv)));
  if (bits)
    for (size_t b = 0; b < nb; b++)
      for (size_t x = 0; x < w.table[2 * b].size(); x++) {
        w.table[2 * b][x] = (x * 5 + b) % 3 == 0 ? E::one() : E::zero();
        w.table[2 * b + 1][x] = w.table[2 * b][x] - E::one();
      }
  w.plan = [nv, nb, bits](blk::Plan &p) {
    if (!bits) blk::plan_scalars(p, "the start", "SC3", 1);
    blk::plan_psc(p, "SC3", blk::attention_sc3(nv, (unsigned)nb));
  };
  w.slots = (u32)((bits ? 0 : 1) + sc3Slots(nv, nb));
  w.read = [t, om, nv, nb, bits](HFork &F, Tr &tr, u32 &slot, int bent) {
    const Aff start = bits ? affC(eZ()) : affSlot(F.claimsIn(tr, slot, 1));
    Degree3SumcheckState o = sc3In(F, tr, slot, nv, nb, start);
    vector<E> r;
    for (const Ex &x : o.r) r.push_back(eOf(x));
    vector<Ex> W;
    for (size_t b = 0; b < nb; b++) W.push_back(fromBlkE(om[b] * eqAll(t, r.data())) + (bent && !b ? eO() : eZ()));
    sc3Row(F, o, W, "test");
    if (!(sc3A(o, nb - 1).s.size() == 1 && sc3A(o, nb - 1).s[0].first == o.sEnd + 2 * (nb - 1) && sc3B(o, nb - 1).s[0].first == o.sEnd + 2 * (nb - 1) + 1)) refuse("the-test-sc3-slots");
    return Read{o.rowU, o.sEnd, o.r};
  };
  return w;
}
// SCD (round form `NODES`: each round has its own degree), attention's "key at the index".
// AttentionSubProof::run() computes the summand inline, so this test uses a copy of it, not the verifier's
// code. In attention IB_(b,k) is bit k of an index, so the product picks KC_lam(b) at j = that
// index; here the tables hold arbitrary field elements.
// Variables: j (lt bits), g (lgk bits), i (lt bits); lam(b) = b / HA;
//   sum eq(ri, i) eq(rg, g) sum_b om_b KC_lam(b)(j, g) prod_k eq1(IB_(b,k)(g, i), j_(lt - 1 - k)).
static World scdWorld(int lt, int lgk, size_t Lam, size_t HA) {
  World w;
  w.name = "SCD: the key at the index";
  const size_t NP = Lam * HA;
  const vector<E> ri = madeUpList(lt), rg = madeUpList(lgk), om = madeUpList(NP);
  for (int j = 0; j < 2 * lt + lgk; j++) w.vars.push_back(blk::Weight<E>::plain((unsigned)(j < lt ? 3 : j < lt + lgk ? lt + 3 : lt + 2)));
  auto summand = [=](const E *pj, const E *pg, const E *pi, const vector<E> &y) {
    E s = E::zero();
    for (size_t b = 0; b < NP; b++) {
      E t = om[b] * y[b / HA];
      for (int k = 0; k < lt; k++) t = t * eq1(y[Lam + b * lt + k], pj[lt - 1 - k]);
      s += t;
    }
    return eqAll(ri, pi) * eqAll(rg, pg) * s;
  };
  w.S = blk::Summand<E>{(unsigned)(Lam + NP * lt), 0, (unsigned)(lt + 1), [=](const vector<E> &x, const vector<E> &y) { return vector<E>{summand(x.data(), x.data() + lt, x.data() + lt + lgk, y)}; }};
  vector<vector<int>> on(Lam, range(0, lt + lgk));            // KC_lam(j, g)
  on.insert(on.end(), NP * lt, range(lt, lgk + lt));          // IB_(b,k)(g, i)
  tables(w, on);
  w.plan = [=](blk::Plan &p) {
    blk::plan_scalars(p, "the start", "SCD", 1);
    blk::plan_psc(p, "SCD", blk::attention_scd(lt, lgk, (unsigned)NP, (unsigned)Lam));
  };
  vector<int> degrees;
  for (int j = 0; j < 2 * lt + lgk; j++) degrees.push_back(j < lt ? 2 : j < lt + lgk ? lt + 2 : lt + 1);  // AttentionSubProof::scdDegrees()
  w.slots = (u32)(1 + sumcheckSlotsDegrees(degrees, Lam + NP * lt, lt + 1));
  w.read = [=](HFork &F, Tr &tr, u32 &slot, int bent) {
    const Aff start = affSlot(F.claimsIn(tr, slot, 1));
    VariedDegreeSumcheckState o = scdIn(F, tr, slot, degrees, lt + 1, Lam + NP * (size_t)lt, start);
    vector<E> y, r;
    for (size_t i = 0; i < Lam + NP * lt; i++) y.push_back(eOf(F.shown(o.sEnd + (u32)i)));
    for (const Ex &x : o.r) r.push_back(eOf(x));
    o.rowU = pscRow(F, o.p, fromBlkE(summand(r.data(), r.data() + lt, r.data() + lt + lgk, y)) + (bent ? eO() : eZ()));
    return Read{o.rowU, o.sEnd, o.r};
  };
  return w;
}

int main() {
  printf("0 - 1 = 0x%llx in the verifier's field, 0x%llx in blocks/\n", (unsigned long long)fsub((u64)0, (u64)1), (unsigned long long)(E::zero() - E::one()).c[0].v);
  refuseThrows = true;
  int failures = 0;
  string world;
  auto expect = [&](bool ok, const string &what) {
    if (!ok) failures++, printf("FAIL: %s: %s\n", world.c_str(), what.c_str());
  };
  const E c2 = madeUp(), w0 = madeUp(), w1 = madeUp(), w2 = madeUp();
  vector<World> worlds;
  worlds.push_back(tokenWorld("token sum-check: a product of two maps", 3, 2, 2, [](const vector<E> &y) { return y[0] * y[1]; }));
  worlds.push_back(tokenWorld("token sum-check: a sum of two products", 3, 4, 2, [c2](const vector<E> &y) { return y[0] * y[1] + c2 * y[2] * y[3]; }));
  worlds.push_back(tokenWorld("token sum-check: three maps under public weights", 2, 3, 1, [w0, w1, w2](const vector<E> &y) { return w0 * y[0] + w1 * y[1] + w2 * y[2]; }));
  worlds.push_back(tokenWorld("token sum-check: one round", 1, 2, 2, [](const vector<E> &y) { return y[0] * y[1]; }));
  worlds.push_back(headWorld(2, 2));
  worlds.push_back(sc3World("SC3: one pair", 3, 1, false));
  worlds.push_back(sc3World("SC3: three pairs", 2, 3, false));
  worlds.push_back(sc3World("SC3 from the public start 0: the index bits", 3, 2, true));
  worlds.push_back(scdWorld(2, 1, 1, 1));
  worlds.push_back(scdWorld(2, 1, 2, 2));
  size_t scalars = 0, entries = 0;
  for (World &w : worlds) {
    world = w.name;
    {
      blk::SumClaim<E> shape;
      shape.vars = w.vars;
      w.mask = madeUpList((w.hiddenStart ? 1 : 0) + blk::psc_masks(shape, w.S));
    }
    expect(w.mask.size() == w.slots, "the slots that the verifier's routines announce are the mask entries that the library's prover takes");
    if (w.mask.size() != w.slots) continue;
    const vector<blk::u8> honest = prove(w, Cheat::NONE);
    expect(honest.size() == blk::DIGEST_BYTES + (size_t)w.slots * blk::E_BYTES, "the proof is the chain's random bytes and one scalar for each slot");
    const string why = verify(w, honest, w.mask);
    expect(why.empty(), "the honest proof: " + why);
    if (!why.empty()) continue;
    expect(verify(w, prove(w, Cheat::OTHER_TABLE), w.mask) == "the equation", "a prover on another table");
    expect(verify(w, honest, w.mask, 1) == "the equation", "another public input on the verifier's side");
    int noticed = 0;
    for (u32 j = 0; j < w.slots; j++) {
      vector<E> mask = w.mask;
      mask[j] += E::one();
      noticed += verify(w, honest, mask) == "the equation";
    }
    expect(noticed == (int)w.slots, "every mask entry stands in the equation");
    entries += w.slots;
    for (u32 j = 0; j < w.slots; j++) {
      vector<blk::u8> other = honest;
      other[blk::DIGEST_BYTES + (size_t)j * blk::E_BYTES] ^= 1;
      expect(!verify(w, other, w.mask).empty(), "scalar " + std::to_string(j) + " changed");
      scalars++;
    }
  }
  if (failures) return printf("sumchecks_twin_test: FAILED (%d checks)\n", failures), 1;
  printf("sumchecks_twin_test: ok (%zu worlds: 5 token sum-checks, 3 SC3, 2 SCD; in each another table and another public input noticed; in all %zu changed mask entries and %zu changed scalars)\n", worlds.size(), entries, scalars);
  return 0;
}
