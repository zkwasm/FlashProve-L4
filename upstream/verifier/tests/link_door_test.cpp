// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests how a sub-proof's verifier drives `mb::ChainCheck` (verifier/src/vf_request_chain.h), the check of
   a request's proof outside the sub-proofs, through verifier/src/vf_subproof_segment.h.

   Sub-proof (`HFork`): the argument for one part of the forward pass; synthetic here, see `Seen`.
   Segment (`link` in identifiers and output): one sub-proof's part of a request's proof. Array: a
   named matrix of committed values, placed by the layout (`blk::ReqLayout`). Commit point
   (`moment`, from 0): a prover message of a segment that commits arrays. Tree (`HTree`): the
   verifier's entry for one array. Virtual row (`HRow`): weights on the array's rows and columns.

   The sub-proof's verifier calls `rootInH` at each commit point after the segment's first,
   `t2Batch` for the batching challenge, and `doorH` to pass on its one equation and close the
   segment. The prover is that of blocks/iop/req.h, on synthetic requests of toy size
   (blocks/iop/tests/req_rows_cases.h), some with arrays kept for one or two later segments.

   An honest proof must be accepted and a false value rejected (`refuse` in identifiers, "refused"
   in output). A verifier with one fault of `Bend` in one segment must reject the honest proof under
   the name given in `main`. Because verifiers with other weights or mask coefficients reject it,
   acceptance means that both sides agree on the bit order of a row index, on the signs and on the
   order of the mask entries.

   Build, from verifier/src:
     g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1
       -I . -I ../../blocks -I ../../prover ../tests/link_door_test.cpp */
#include "vf_subproof_segment.h"

#include "iop/tests/req_rows_cases.h"

static int bad = 0;

// The synthetic sub-proof of segment l has one claim for each array a that the segment can read:
//   <u_a (x) w_a, array a> + <u'_a (x) w'_a, array a> + k_a = v_a,
// with public weights and a public constant derived from a seed; the two outer products (x) are the
// two virtual rows of the array's tree. After the last commit point the prover sends all v_a in one
// message, each as v_a + entry a of the segment's mask row (committed random values); powers of one
// challenge batch the claims.
//
// An array that a segment can read; `born`: the segment that committed it. seenBy() gives both
// sides one order: the segment's own arrays in the layout's order, then earlier segments' arrays
// that are kept for it.
struct Seen {
  size_t born;
  const blk::ReqArray *array;
};
static vector<Seen> seenBy(const blk::ReqWorld &w, const blk::ReqLayout &layout, size_t l) {
  vector<Seen> v;
  for (const blk::ReqArray &a : layout.link[l]) v.push_back({l, &a});
  const std::vector<blk::ReqGroup> groups = blk::req_groups(w.shape);
  for (const std::pair<size_t, size_t> &born : groups[l].born)
    for (const blk::ReqArray &a : layout.link[born.first])
      if (a.message == born.second) v.push_back({born.first, &a});
  return v;
}
// The public weights of one product of a claim: `rows` (x) `columns`. `kind` of the row vector:
// 0, an equality vector on all rows; 1, all ones; 2, a plain vector without a product form; 3, the
// powers of point[0]; 4, as 0 on the upper half of the rows.
struct Weights {
  int kind = 0;
  vector<blk::E> point;     // kind 0 or 4: coordinate 0 goes with the top bit, as in `eqTab`
  blk::u64 first = 0;       // kinds 0 and 4
  vector<blk::E> rows, columns;  // written out
};
// which: 0 or 1, the virtual row.
static Weights weightsOf(const Seen &s, size_t l, size_t a, blk::u64 seed, int which) {
  blk::TestCoins c(seed + 100 * l + 2 * a + (blk::u64)which);
  const blk::u64 R = s.array->laid.size_i, K = s.array->laid.size_c;
  Weights w;
  w.kind = (int)((a + l + 2 * (size_t)which) % 5);
  w.columns = blk::random_elements(c, K);
  w.rows.assign(R, blk::E::zero());
  if (w.kind == 0 || w.kind == 4) {
    const unsigned bits = blk::ceil_log2(R), np = bits - (unsigned)(w.kind == 4 && bits > 0);
    w.point = blk::random_elements(c, np);
    w.first = np < bits ? (blk::u64)1 << np : 0;
    for (blk::u64 r = 0; r < ((blk::u64)1 << np) && w.first + r < R; r++) {
      blk::E x = blk::E::one();
      for (unsigned t = 0; t < np; t++) x *= ((r >> (np - 1 - t)) & 1) ? w.point[t] : blk::E::one() - w.point[t];
      w.rows[w.first + r] = x;
    }
  } else if (w.kind == 1)
    w.rows.assign(R, blk::E::one());
  else if (w.kind == 3) {
    w.point = blk::random_elements(c, 1);
    w.rows[0] = blk::E::one();
    for (blk::u64 r = 1; r < R; r++) w.rows[r] = w.rows[r - 1] * w.point[0];
  } else
    w.rows = blk::random_elements(c, R);
  return w;
}

static blk::E constantOf(size_t l, size_t a, blk::u64 seed) {
  blk::TestCoins c(seed + 7000 + 100 * l + a);
  return c.e();
}
// The library's toy layout plus an array "Q", so that the last commit point commits two arrays.
static blk::ReqLayout layoutOf(const blk::ReqWorld &w) {
  blk::ReqLayout layout = blk::rows_toy_layout(w);
  const unsigned L = w.codes.base[0].log_ell();
  for (size_t l = 0; l < w.shape.link.size(); l++)
    for (size_t m = 0; m < w.shape.link[l].messages.size(); m++)
      if (w.shape.link[l].messages[m].shape.name == "multiplicities") {
        blk::LaidBlock b;
        b.bits_c = b.place_bits = L - 1;
        layout.link[l].push_back(blk::ReqArray{"Q", m, blk::LaidMatrix{1, (blk::u64)1 << (L - 1), {b}}});
      }
  return layout;
}

static blk::ReqSubPlan planOf(const blk::ReqWorld *&w, const blk::ReqLayout *&layout) {
  return [&w, &layout](blk::Plan &p, size_t l, unsigned moment) {
    if (moment + 1 < w->shape.link[l].moments()) return;
    const size_t n = seenBy(*w, *layout, l).size();
    blk::plan_scalars(p, "values", w->shape.link[l].name, n);
    blk::plan_bat_sigma(p, w->shape.link[l].name, n);
  };
}

// ---- the prover's side ----
// Calls no function of verifier/src. A nonzero `off` makes the first value of segment 0 false.
struct DoorProver : blk::RowsSubProver {
  using blk::RowsSubProver::RowsSubProver;
  void run(blk::ProverContext &P, const blk::ReqFrame &frame, unsigned moment, const std::vector<blk::OracleId> &) override {
    const size_t l = frame.link();
    if (moment + 1 < w.shape.link[l].moments()) return;
    const vector<Seen> seen = seenBy(w, layout, l);
    vector<blk::E> shown;
    for (size_t a = 0; a < seen.size(); a++) {
      size_t at = 0;
      while (&layout.link[seen[a].born][at] != seen[a].array) at++;
      const blk::LaidMatrix &laid = seen[a].array->laid;
      blk::E v = constantOf(l, a, seed);
      for (int which = 0; which < 2; which++) {
        const Weights x = weightsOf(seen[a], l, a, seed, which);
        for (blk::u64 i = 0; i < laid.size_i; i++)
          for (blk::u64 c = 0; c < laid.size_c; c++) v += x.rows[i] * x.columns[c] * matrix[seen[a].born][at][i * laid.size_c + c];
      }
      if (l == 0 && a == 0) v += off;
      shown.push_back(v + P.behind[frame.where(l, 0)].mask.entry(a));
      P.ch.send(shown.back());
    }
    blk::AlgebraicDraw from = P.ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(seen.size());
    const blk::E theta = from.draw();
    // sum_a theta^a (the two products on array a + mask entry a) = sum_a theta^a (shown_a - k_a).
    // Each product gives one term for each row of its array.
    blk::ReqEquation eq;
    blk::E power = blk::E::one();
    for (size_t a = 0; a < seen.size(); a++, power *= theta) {
     for (int which = 0; which < 2; which++) {
      const Weights x = weightsOf(seen[a], l, a, seed, which);
      const unsigned bits = blk::ceil_log2(seen[a].array->laid.size_i);
      auto columns = std::make_shared<const std::vector<blk::E>>(x.columns);
      for (blk::u64 i = 0; i < x.rows.size(); i++) {
        blk::ReqTerm t;
        t.born = seen[a].born, t.array = seen[a].array->name, t.scale = power * x.rows[i];
        for (unsigned b = 0; b < bits; b++) {
          const bool one = (i >> b) & 1;
          t.on_rows.lo.push_back(one ? blk::E::zero() : blk::E::one()), t.on_rows.hi.push_back(one ? blk::E::one() : blk::E::zero());
        }
        t.on_columns = blk::IndexWeights::written(columns);
        eq.terms.push_back(std::move(t));
      }
     }
      eq.on_masks.push_back(power);
      eq.target += power * (shown[a] - constantOf(l, a, seed));
    }
    if (!rows) rows = std::make_unique<blk::ReqRows<blk::ProverContext>>(P, frame, bound);
    P.st.add_row(frame.group(), rows->row(eq));
  }
};

// ---- the verifier's side ----
enum class Bend {
  NONE,
  ROW_WEIGHT,     // other row weights on virtual row 0 of the first array that `bendRows` can alter
  COLUMN_WEIGHT,  // another weight on a column
  MASK_SIGN,      // the hidden value is taken as shown + mask
  FORM_APART,     // the product form is not the written-out vector
  NO_COMMITMENT,  // `subProof` leaves out commit point 2
  OTHER_ARRAY,    // at commit point 1 it names the arrays of commit point 2
  ONE_OF_TWO,     // at commit point 2 it names one of the two arrays
  NAMED_TWICE,    // .. or one of them twice
  CONSTANT,       // another constant in a claim
  SECOND_ROW,     // another weight on a tree's second virtual row
  WINDOW,         // a virtual row with a window of columns
  KEPT_ELSEWHERE, // `N2::outside`: a tree whose weights are kept outside the rows
  NO_BATCHING,    // it closes without the batching
  TWICE,          // it closes twice
  OUTSIDE,        // weight on a tree outside the list `Tail2::nt`
  NO_CLOSE        // it returns without closing
};
static Ex exOf(const blk::E &x) { return fromBlkE(x); }
static vector<Ex> exOf(const vector<blk::E> &v) {
  vector<Ex> o;
  for (const blk::E &x : v) o.push_back(fromBlkE(x));
  return o;
}
// Changes x to another row vector of the same kind, product form and written-out vector alike.
// Returns false if it cannot.
static bool bendRows(Weights &x) {
  const size_t np = x.point.size();
  if ((x.kind == 0 || x.kind == 4) && np) {
    std::swap(x.point.front(), x.point.back());  // another bit order, if np > 1
    x.point.front() += blk::E::one();
    for (blk::u64 r = 0; r < ((blk::u64)1 << np) && x.first + r < x.rows.size(); r++) {
      blk::E e = blk::E::one();
      for (size_t t = 0; t < np; t++) e *= ((r >> (np - 1 - t)) & 1) ? x.point[t] : blk::E::one() - x.point[t];
      x.rows[x.first + r] = e;
    }
    return true;
  }
  if (x.kind == 3 && x.rows.size() > 1) {
    x.point[0] += blk::E::one();
    for (size_t r = 1; r < x.rows.size(); r++) x.rows[r] = x.rows[r - 1] * x.point[0];
    return true;
  }
  if (x.kind == 2) return x.rows[0] += blk::E::one(), true;
  return false;
}
static void commitment(HFork &F, Tr &tr, const vector<int> &t) {
  if (t.size() == 1)
    rootInH(F, tr, {t[0]});
  else if (t.size() == 2)
    rootInH(F, tr, {t[0], t[1]});
  else
    refuse("test-shape");
}
// Checks segment l's sub-proof with the fault `bend`. It fills its list of trees itself, without
// t2AddN() or t2Plan(), then makes the calls that the real sub-proofs' verifiers make.
static void subProof(const blk::ReqWorld &w, const blk::ReqLayout &layout, size_t l, blk::u64 seed, Bend bend) {
  const vector<Seen> seen = seenBy(w, layout, l);
  HFork F;
  F.G_ = "toy:";
  Tail2 T;
  T.link = true;
  vector<Weights> W[2];
  for (size_t a = 0; a < seen.size(); a++) {
    HTree X;
    X.name = seen[a].array->name;
    X.kind = 'N';
    X.R = seen[a].array->laid.size_i, X.K = seen[a].array->laid.size_c;
    X.rows.resize(2);
    F.trees.push_back(X);
    T.nt.push_back(N2());
    for (int which = 0; which < 2; which++) W[which].push_back(weightsOf(seen[a], l, a, seed, which));
  }
  if (bend == Bend::OUTSIDE) {
    HTree X;
    X.name = "outside", X.kind = 'N', X.R = 1, X.K = 2;
    X.rows.resize(1);
    F.trees.push_back(X);
  }
  T.planned = true;
  F.slots((u32)seen.size());
  F.planClaimsF = (u32)seen.size();
  Tr tr{currentSegment->V->ch};
  // This segment's arrays that are committed at commit point `moment`, as indices into `seen`.
  auto at = [&](unsigned moment) {
    vector<int> t;
    for (size_t a = 0; a < seen.size(); a++)
      if (seen[a].born == l && w.shape.link[l].messages[seen[a].array->message].moment == moment) t.push_back((int)a);
    return t;
  };
  vector<int> two = at(2);
  if (at(1).size() != 1 || two.size() != 2) refuse("test-shape");
  commitment(F, tr, at(bend == Bend::OTHER_ARRAY ? 2 : 1));
  if (bend == Bend::ONE_OF_TWO) two.pop_back();
  if (bend == Bend::NAMED_TWICE) two[1] = two[0];
  if (bend != Bend::NO_COMMITMENT) commitment(F, tr, two);
  u32 slot = 0;
  if (bend == Bend::NO_COMMITMENT)
    slot = (u32)seen.size();  // it goes straight on to its batching
  else if (bend == Bend::MASK_SIGN) {
    const vector<Ex> v = trRead(tr, seen.size());
    for (u32 i = 0; i < v.size(); i++) F.hide(i, v[i], eZ() - eO());
    slot = (u32)v.size();
  } else
    F.claimsIn(tr, slot, (u32)seen.size());
  if (bend == Bend::NO_BATCHING) {
    doorH(F, T, tr, 0);
    return;
  }
  t2Batch(F, T, tr, slot);
  if (bend == Bend::OUTSIDE) F.trees.back().rows[0].L.assign(2, eO());
  if (bend == Bend::ROW_WEIGHT || bend == Bend::SECOND_ROW) {
    bool bent = false;
    for (Weights &x : W[bend == Bend::SECOND_ROW])
      if (!bent) bent = bendRows(x);
    if (!bent) refuse("test-shape");
  }
  if (bend == Bend::COLUMN_WEIGHT) W[0][1].columns.back() += blk::E::one();
  vector<vector<Ex>> columns[2];
  for (int which = 0; which < 2; which++)
    for (const Weights &x : W[which]) columns[which].push_back(exOf(x.columns));
  for (size_t a = 0; a < seen.size(); a++) {
    vector<Run2> runs;
    for (int which = 0; which < 2; which++) {
      const Weights &x = W[which][a];
      HRow &row = F.trees[a].rows[(size_t)which];
      row.coef = exOf(x.rows);
      if (x.kind == 0 || x.kind == 4) row.rowForm = true, row.rowPoint = exOf(x.point), row.rowFirst = x.first;
      if (x.kind == 1) row.rowForm = row.rowOnes = true;
      if (x.kind == 3) {
        row.rowForm = true;
        Ex square = exOf(x.point[0]);
        for (unsigned t = 0; t < blk::ceil_log2(F.trees[a].R); t++, square = square * square) row.rowLo.push_back(eO()), row.rowHi.push_back(square);
      }
      Run2 r;
      r.tree = (u32)a, r.row = (u32)which, r.first = 0, r.coef = eO(), r.w = &columns[which][a], r.w0 = 0, r.n = columns[which][a].size();
      runs.push_back(r);
    }
    const blk::E k = constantOf(l, a, seed) + (bend == Bend::CONSTANT && a == 0 ? blk::E::one() : blk::E::zero());
    t2Lin(F, affC(exOf(k)) - affSlot((u32)a), runs);
  }
  if (bend == Bend::FORM_APART) {
    bool bent = false;
    for (HTree &X : F.trees)
      for (HRow &row : X.rows)
        if (!bent && row.rowForm) row.coef.back() = row.coef.back() + eO(), bent = true;
    if (!bent) refuse("test-shape");
  }
  if (bend == Bend::WINDOW) F.trees[0].rows[0].L0 = 1;
  if (bend == Bend::KEPT_ELSEWHERE) T.nt[0].outside = true;
  if (bend == Bend::NO_CLOSE) return;
  doorH(F, T, tr, 0);
  if (bend == Bend::TWICE) doorH(F, T, tr, 0);
}
// Returns "" if the proof is accepted; otherwise the rejection name, "not-closed", or "no" if a
// call of `check` or `channel` here returned false.
static string verify(const blk::ReqWorld &w, const blk::ReqLayout &layout, blk::u64 seed, const std::vector<blk::u8> &proof, Bend bend, size_t bentLink) {
  blk::Binder binder(1, w.plan);
  const blk::RowsPublic bound = blk::rows_toy_bind(binder, w, layout, seed);
  blk::Reader reader(proof.data(), proof.size());
  blk::VerifierSession session; // one verdict, one chain
  blk::VerifierRounds channel(binder, reader, session);
  blk::VerifierContext V(bound.codes, channel);
  mb::ChainCheck check(V, mb::ChainShape{bound.shape, bound.layout});
  refuseThrows = true;
  try {
    for (size_t l = 0; l < w.shape.link.size(); l++) {
      if (!check.first()) return "no";
      vector<std::pair<size_t, string>> arrayOf;
      for (const Seen &s : seenBy(w, layout, l)) arrayOf.push_back({s.born, s.array->name});
      LinkCtx C;
      C.check = &check, C.V = &V, C.link = l, C.arrayOf = &arrayOf, C.layout = &channel.get(bound.layout);
      currentSegment = &C;
      subProof(w, channel.get(bound.layout), l, channel.get(bound.seed), l == bentLink ? bend : Bend::NONE);
      currentSegment = nullptr;
      if (!C.handed || !C.done) return "not-closed";
    }
  } catch (Refusal &r) {
    currentSegment = nullptr;
    return r.name;
  }
  if (!check.finish()) return "no";
  return channel.finish() ? "" : "no";
}

int main() {
  struct Want {
    Bend bend;
    const char *what, *name;
  };
  const Want wants[] = {
      {Bend::ROW_WEIGHT, "another row weight", "chain-refused"},       {Bend::COLUMN_WEIGHT, "another column weight", "chain-refused"},
      {Bend::MASK_SIGN, "another sign of a mask", "chain-refused"},    {Bend::FORM_APART, "a form apart from its vector", "toy:internal-row-form"},
      {Bend::NO_COMMITMENT, "a commitment left out", "toy:plan-link"}, {Bend::OTHER_ARRAY, "another array at a commitment", "toy:plan-link"},
      {Bend::ONE_OF_TWO, "one of a moment's two arrays", "toy:plan-link"}, {Bend::NAMED_TWICE, "an array named twice", "toy:plan-link"},
      {Bend::CONSTANT, "another constant", "chain-refused"},           {Bend::SECOND_ROW, "another weight on a second virtual row", "chain-refused"},
      {Bend::WINDOW, "a window of columns", "toy:internal-coef-"},     {Bend::KEPT_ELSEWHERE, "weights kept elsewhere", "toy:internal-t2-order"},
      {Bend::NO_BATCHING, "closed before the batching", "toy:internal-t2-order"},
      {Bend::TWICE, "closed twice", "toy:internal-t2-order"},          {Bend::OUTSIDE, "weight outside the list", "toy:internal-f-door"},
      {Bend::NO_CLOSE, "not closed", "not-closed"},
  };
  size_t runs = 0;
  for (const blk::ReqCase &t : blk::rows_toy_cases()) {
    const blk::ReqWorld *world = nullptr;
    const blk::ReqLayout *lay = nullptr;
    // The sub-proof's plan needs the layout, and the layout the shape: a first fixture (`ReqWorld`)
    // gives the shape, the second has the plan.
    const blk::ReqWorld shapes = blk::req_toy_world(t, 1);
    const blk::ReqLayout layout = layoutOf(shapes);
    world = &shapes, lay = &layout;
    const blk::ReqWorld w = blk::req_toy_world(t, 1, planOf(world, lay));
    blk::EngineCpu engine, again;
    const std::vector<blk::u8> proof = blk::rows_prove<DoorProver>(w, layout, 900, blk::E::zero(), engine);
    const string honest = verify(w, layout, 900, proof, Bend::NONE, 0);
    if (!honest.empty()) bad++, printf("an honest proof is refused (%s): %s\n", honest.c_str(), t.name);
    const string lie = verify(w, layout, 900, blk::rows_prove<DoorProver>(w, layout, 900, blk::E::one(), again), Bend::NONE, 0);
    if (lie.empty()) bad++, printf("A FALSE VALUE IS ACCEPTED: %s\n", t.name);
    runs += 2;
    for (const Want &x : wants)
      for (size_t l = 0; l < w.shape.link.size(); l++, runs++) {
        const string got = verify(w, layout, 900, proof, x.bend, l);
        // A false equation is tested only in ChainCheck::finish(): close() folds it into a claim
        // that is carried on. So "no" is accepted for "chain-refused". A name that ends in a hyphen
        // is a prefix.
        const string name = x.name;
        const bool fits = got == name || (name == "chain-refused" && got == "no") || (name.back() == '-' && !got.rfind(name, 0));
        if (!fits) bad++, printf("%s, link %zu: got '%s', expected '%s': %s\n", x.what, l + 1, got.c_str(), x.name, t.name);
      }
  }
  printf("link_door_test: %zu runs: %s\n", runs, bad ? "FAILED" : "ok");
  return bad ? 1 : 0;
}
