// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the verifier's functions for the layers of a product tree (readProductLevelRounds(), handDown() and
// layerObject(); in verifier/src/: vf_subproof_engine.h, vf_sumcheck_psc.h, vf_sumcheck_receive.h) on proofs from the library
// blocks/. A layer is one PSC, the sumcheck with a hiding polynomial (blocks/iop/psc.h), then one
// LINE, which merges the claims on a map's two halves (blocks/iop/line.h). Both sides run on a hash
// chain (the verifier's side through `Tr`, for transcript), whose channel compares each message's
// size and each round's name and number of draws with planOf()'s plan.
//
// The verifier's claims add up to one equation in the leaves and the mask row, which has one entry
// per sent scalar. On trees of 2 to 32 synthetic leaves (madeUp()) the equation must hold for the
// prover's proof and fail for a false root or with any one scalar or mask entry changed. Four
// misuses of `HFork`, a sub-proof's state, must be rejected by name. A pass shows that both sides
// agree on the message format for this shape and orderDefault()'s order of rounds. No GPU code,
// commitment or whole sub-proof runs.
// Build, from verifier/src: g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8
//   -DBLK_FIELD=2 -DBLK_HASH=1 -I . -I ../../blocks -I ../../prover -I ../../blocks/iop/tests
//   ../tests/psc_twin_test.cpp -o <a folder of your own>/psc_twin_test

// Leaves out verifier/src/vf_lookup_helpers.h, unused here.
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"

#include "toy_support.h"

#include "iop/psc.h"
#include "numbers/blocks.h"

using E = blk::E;
static E eOf(const Ex &x) { return E::from(blk::from_u64(eC(x, 0)), blk::from_u64(eC(x, 1)), blk::from_u64(eC(x, 2))); }
static u64 rngState = 1;
static E madeUp() {
  Ex x;
  for (int i = 0; i < CE; i++) {
    rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
    eSet(x, i, (rngState >> 3) % FIELD_P);
  }
  return eOf(x);
}
// The multilinear extension of the table a at z; z[0] belongs to the index's top bit.
static E topFirst(vector<E> a, const vector<E> &z) {
  for (const E &c : z) {
    const size_t h = a.size() / 2;
    for (size_t x = 0; x < h; x++) a[x] = a[x] + c * (a[x + h] - a[x]);
    a.resize(h);
  }
  return a[0];
}
static blk::ToyMap<E> mapOf(vector<E> a) {
  return [a](const vector<E> &z) { return topFirst(a, z); };
}

// One tree and its mask row. `slots`: the number of sent scalars (message slots, `slot`).
struct World {
  int n = 0;
  vector<vector<E>> level;  // level[i]: 2^i entries; entry x = entries x and x + 2^i of level i + 1 multiplied
  vector<E> mask;
  u32 slots = 0;
};
static World makeWorld(int n) {
  World w;
  w.n = n;
  vector<E> leaves((size_t)1 << n);
  for (E &x : leaves) x = madeUp();
  w.level.push_back(leaves);
  while (w.level.front().size() > 1) {
    const vector<E> &b = w.level.front();
    vector<E> up(b.size() / 2);
    for (size_t x = 0; x < up.size(); x++) up[x] = b[x] * b[x + up.size()];
    w.level.insert(w.level.begin(), up);
  }
  for (int i = 0; i < n; i++) w.slots += (u32)layerSlots(i);
  w.mask.resize(w.slots);
  for (E &x : w.mask) x = madeUp();
  return w;
}
// The layers' rounds, then the challenge that batches the claims: one per layer, one on the leaves.
static blk::Plan planOf(const World &w) {
  blk::Plan p(blk::setting_goldilocks());
  blk::plan_product_tree(p, "layers", (unsigned)w.n, 0);
  blk::plan_bat_sigma(p, "claims", (blk::u64)w.n + 1);
  return p;
}
// Passes zeros as the chain's random bytes (its salt).
static vector<blk::u8> prove(const World &w, E root) {
  const blk::Planned plan = planOf(w).finish();
  blk::Writer out;
  blk::Binder binder(6, plan);
  (void)binder.configuration(blk::Digest{});  // required of every chain; zeros on both sides here
  blk::ProverRounds ch(binder, blk::Digest{}, out);
  blk::MaskLedger ledger(blk::SendAccess::make(0, 0, w.slots));
  const blk::Summand<E> S{2, 0, 2, [](const vector<E> &, const vector<E> &y) { return vector<E>{y[0] * y[1]}; }};
  blk::PointClaim<E> c{{}, {}, root};
  for (int i = 0; i < w.n; i++) {
    blk::SumClaim<E> claim = blk::SumClaim<E>::at_point(std::move(c), vector<unsigned>(i, 3));
    const vector<E> &b = w.level[i + 1];
    blk::DefinitionSumcheck<E> inner(claim.vars, S, {mapOf(vector<E>(b.begin(), b.begin() + b.size() / 2)),
                                                      mapOf(vector<E>(b.begin() + b.size() / 2, b.end()))});
    const size_t fresh = blk::psc_masks(claim, S);
    blk::PscOutput<E> out = blk::psc_prove(ch, std::move(claim), S, inner, ledger.take(fresh), blk::mask_row(w.mask));
    c = blk::line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
    // line() appends its challenge; the verifier puts it first (the top bit, bound first).
    std::rotate(c.point.begin(), c.point.end() - 1, c.point.end());
  }
  {
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows((blk::u64)w.n + 1);
    (void)from.draw();
  }
  ch.finish();
  return out.bytes();
}
// Returns "" if the equation holds, else what failed. `tr` only wraps `channel`.
static string verify(const World &w, E root, const vector<blk::u8> &proof, const vector<E> &mask) {
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
  vector<Ex> rho, coef;
  Ex sum;
  try {
    Aff C = affC(fromBlkE(root));
    vector<ProductLevel> layers;
    for (int i = 0; i < w.n; i++) {
      layers.push_back(readProductLevelRounds(F, tr, slot, i, rho, C, orderDefault(i)));
      handDown(layers.back(), rho, C);
    }
    if (slot != w.slots) return "the number of slots";
    F.batchBy(trBatchOpen(tr));
    for (const ProductLevel &o : layers) layerObject(F, o);
    vector<E> z;
    for (const Ex &x : rho) z.push_back(eOf(x));
    F.lin(C - affC(fromBlkE(topFirst(w.level[w.n], z))));  // a sub-proof ties C to committed data
    F.settle(coef, sum, w.slots);
    trBatchRows(tr, F.nClaim);  // the number of claims added, compared with the plan's
  } catch (const Refusal &r) {
    return "refused: " + r.name;
  }
  if (!channel.finish() || !session.accepted()) return "the channel";
  for (u32 j = 0; j < w.slots; j++) sum = sum + coef[j] * fromBlkE(mask[j]);
  return isZ(sum) ? "" : "the equation";
}

int main() {
  // 0 - 1 = p - 1 shows which prime the verifier's functions and blocks/ use. Not compared here.
  printf("0 - 1 = 0x%llx in the verifier's field, 0x%llx in blocks/\n", (unsigned long long)fsub((u64)0, (u64)1), (unsigned long long)(blk::E::zero() - blk::E::one()).c[0].v);
  int failures = 0, changed = 0;
  auto expect = [&](bool ok, const char *what) {
    if (!ok) failures++, printf("FAIL: %s\n", what);
  };
  refuseThrows = true;
  const size_t head = blk::DIGEST_BYTES, scalar = 8 * (size_t)CE;  // a proof: the chain's random bytes, then the scalars, eight bytes a coordinate
  for (int n = 1; n <= 5; n++) {
    const World w = makeWorld(n);
    const E root = w.level[0][0];
    const vector<blk::u8> proof = prove(w, root);
    expect(proof.size() == head + (size_t)w.slots * scalar, "the proof has one scalar for each slot");
    expect(verify(w, root, proof, w.mask).empty(), "the honest proof");
    expect(verify(w, root + E::one(), proof, w.mask) == "the equation", "a false root is noticed");
    for (u32 j = 0; j < w.slots; j++) {
      vector<blk::u8> other = proof;
      other[head + (size_t)j * scalar] ^= 1;  // the low bit of the scalar's first coordinate
      changed++;
      expect(verify(w, root, other, w.mask) == "the equation", "a changed scalar is noticed");
      vector<E> mask = w.mask;
      mask[j] = mask[j] + E::one();
      expect(verify(w, root, proof, mask) == "the equation", "another mask entry is noticed");
    }
  }
  {
    HFork F;
    F.slots(4);
    F.hide(1, eO(), eO());
    string name;
    try {
      F.hide(1, eO(), eO());
    } catch (const Refusal &r) {
      name = r.name;
    }
    expect(name == "internal-slot-twice", "a number is used once");
    name.clear();
    try {
      (void)F.publicPart(affSlot(2));
    } catch (const Refusal &r) {
      name = r.name;
    }
    expect(name == "internal-slot-order", "a form names only what has been shown");
    name.clear();
    try {
      F.lin(affC(eO()));
    } catch (const Refusal &r) {
      name = r.name;
    }
    expect(name == "internal-f-draws", "no claim is added before the batching challenge is drawn");
    name.clear();
    try {
      vector<Ex> coef;
      Ex sum;
      F.settle(coef, sum, 3);  // slots 0 and 2 are still empty
    } catch (const Refusal &r) {
      name = r.name;
    }
    expect(name == "internal-slot-unused", "a slot under whose number nothing was put is found at the end");
  }
  printf("psc_twin_test: %s (trees of 2 to 32 leaves; %d changed scalars and as many changed mask entries noticed)\n",
         failures ? "FAILED" : "ok", changed);
  return failures ? 1 : 0;
}
