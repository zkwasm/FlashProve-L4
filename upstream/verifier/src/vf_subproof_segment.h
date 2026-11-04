// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* What the functions that check a sub-proof share for its committed arrays. A sub-proof (`fork` in
   identifiers) is the argument for one part of the forward pass; it runs as one segment (`link` in
   names and output) of a request's first chain, the request's own hash chain up to its seal. The
   frame (blk::ReqVerifier, blocks/iop/req.h) is the proof outside the sub-proofs: roots, samples,
   codeword positions, closings, proximity tests; mb::ChainCheck (verifier/src/vf_request_chain.h) walks it
   call by call. A sub-proof lists its arrays ("trees": HFork::trees, verifier/src/vf_subproof_engine.h, each
   with an N2 in Tail2::nt), adds up its linear claims as weights on them, and passes the frame one
   equation. Virtual row F.trees[i].rows[v] puts the weight coef[r] * L[c] on entry (r, c) of array
   i; the weights are in the challenge field E (`Ex`, verifier/src/vf_common.h).
   Order of calls, enforced by flags only in part: t2AddN, t2AddP, t2Plan (in the sub-proof's plan,
   before the proof is parsed); rootInH at each later commit point (`moment`); t2Batch after the
   last message (it draws the batching element theta; claim number c, from 0, gets kappa = theta^c);
   the claims ("the walk"); doorH. */
#pragma once
#include "vf_subproof_engine.h"
#include "vf_smallness_packing.h"
#include "vf_request_chain.h"
#include <functional>
#include <new>
#include <stdexcept>
// What the list keeps of an array beside its HTree.
struct N2 {
  u32 ec = 1, a = 1; // ec: 1 (base field) or CE (field E). a: 1 + the other sub-proofs that read it
  bool foreign = false; // an earlier sub-proof, the owner, has committed to it
  bool outside = false; // no function sets it; doorH ends without a verdict if it is set
  // P: a symbol tree (vf_smallness_packing.h), the array of the symbols of one matrix of the approximate range
  // check (`tail` in identifiers), of the shape p. t2AddP sets P and p.
  bool P = false;
  ShP p;
};
/* Compares a reader's plan with the owner's plan in what they record of a foreign array. On a
   mismatch its callers (vf_layers_plan.inc, vf_output_head_plan.inc) end without a verdict (see Tail2): it is a
   fault of this program, never of a proof. */
static bool t2SameShape(const N2 &a, const HTree &A, const N2 &b, const HTree &B) {
  return A.R == B.R && A.K == B.K && a.ec == b.ec && a.a == b.a && a.P == b.P && (!a.P || a.p.same(b.p));
}
// The list of arrays. Its flags end some calls out of order without a verdict, where otherwise a
// check would cover nothing. The names given to refuse() are internal-t2-order and
// internal-slot-count (t2Batch also tests `link`, under plan-link, which is a NO). A checking
// call of cp_verify reports a name with internal-, also after a sub-proof's prefix
// (G0:internal-t2-order), as NOT CHECKED (status 5), unless another of the call's stages (key,
// leaf, request) has ended NO. A planning call (-plan), which makes the list too, ends NO: <name>
// (status 1). planned: t2Plan has run. walked: doorH has begun. A "slot" numbers a scalar that
// the sub-proof sends and its mask entry, one random element of the segment's mask row.
struct Tail2 {
  bool link = false; // HTailF2::addTreesF sets it before the first array; functions below need it
  vector<N2> nt;
  u32 used = 0; // the slots that the sub-proof's messages have taken (t2Batch)
  bool planned = false, walked = false;
};
/* Adds an array of R rows and K columns with nv virtual rows; returns its index in F.trees. The
   caller sets the virtual rows' coef before doorH and, with `foreign`, passes the owner's `a`. */
static u32 t2AddN(HFork &F, Tail2 &T, const string &name, u64 R, u64 K, u32 ec, u32 a, bool foreign, u32 nv) {
  if (!T.link || T.planned || F.trees.size() != T.nt.size()) refuse(F.G_ + "internal-t2-order");
  HTree X;
  X.name = name;
  X.kind = 'N';
  X.R = R;
  X.K = K;
  X.owner = !foreign;
  X.rows.resize(nv);
  N2 n;
  n.ec = ec;
  n.a = a;
  n.foreign = foreign;
  F.trees.push_back(std::move(X));
  T.nt.push_back(std::move(n));
  return (u32)F.trees.size() - 1;
}
/* Adds the symbol tree of one matrix of the range check: m cell rows, three to a symbol, give m' =
   ceil(m / 3) rows; k columns. Its one virtual row has full-length weights, so that the checks can
   cover every row and column: coef, powers of the batching element (powersF); L, from tieRight.
   w must be 0 (plan-w-zero). */
static u32 t2AddP(HFork &F, Tail2 &T, const string &name, u64 m, u64 k, u32 w, u32 a, bool foreign) {
  if (!T.link || T.planned || F.trees.size() != T.nt.size()) refuse(F.G_ + "internal-t2-order");
  if (w) refuse("plan-w-zero");
  N2 n;
  n.ec = 1;
  n.a = a;
  n.foreign = foreign;
  n.P = true;
  {
    ShP &q = n.p;
    q.m = m;
    q.k = k;
    q.mS = (m + 2) / 3;
    q.J = 1;
    q.cJ = 0;
    q.R = q.Rr = q.mS;
    q.Rd = 0;
    q.aX = clog2(q.mS);
    q.K1 = q.Kc = k;
    q.a = a;
  }
  HTree X;
  X.name = name;
  X.kind = 'P';
  X.R = n.p.Rr;
  X.K = n.p.Kc;
  X.owner = !foreign;
  X.rows.resize(1);
  F.trees.push_back(std::move(X));
  T.nt.push_back(std::move(n));
  return (u32)F.trees.size() - 1;
}
// Closes the list to further arrays.
static void t2Plan(HFork &F, Tail2 &T) {
  if (!T.link || T.planned || F.trees.size() != T.nt.size()) refuse(F.G_ + "internal-t2-order");
  T.planned = true;
}
/* Gives each virtual row of the listed arrays K column weights L, all zero, unless it has K
   already. A failed allocation is no fault of the proof: std::bad_alloc reaches
   `main`, which prints NOT CHECKED, or becomes internal-exception-parFor in a job of parFor. */
static void t2FillL(HFork &F, const Tail2 &T) {
  for (size_t ti = 0; ti < T.nt.size(); ti++)
    for (auto &r : F.trees[ti].rows)
      if (r.L.size() != F.trees[ti].K) {
        try {
          r.L.assign(F.trees[ti].K, eZ());
        } catch (const std::length_error &) {
          throw std::bad_alloc(); // a vector longer than the library allows: out of memory too
        }
      }
}

/* What the function that checks a sub-proof knows of its segment. It runs straight through and
   calls the frame where its prover does: rootInH at each later commit point, doorH at the end.
   The driver, runLinksH (vf_segments.inc), makes the segment's first call, ChainCheck::first, before
   it starts the function, and rejects (plan-link) if the function returns with `handed` or `done`
   not set. */
struct LinkCtx {
  mb::ChainCheck *check = nullptr; // walks the frame; the present segment is this sub-proof's
  blk::VerifierContext *V = nullptr; // its context; V->ch is the channel of the proof's first chain
  size_t link = 0;                 // counted from 0
  // For each listed tree: the segment that committed to it and its name in that segment's layout.
  const vector<std::pair<size_t, string>> *arrayOf = nullptr;
  const blk::ReqLayout *layout = nullptr; // as bound in the first chain's first query
  unsigned moments = 1;            // the segment's commit points so far
  bool handed = false;             // t2Batch has drawn the last challenge; rootInH then rejects
  bool done = false;               // the segment is closed
  // For the log: terms; rows without a product form; written-out weights.
  size_t terms = 0, rowsCut = 0, written = 0;
  // Seconds in ChainCheck::later(); doorH before close(); close().
  double sLater = 0, sDoor = 0, sClose = 0;
};
static thread_local LinkCtx *currentSegment = nullptr;
// Draws the element whose powers batch the sub-proof's linear claims.
// used: the slots that its messages have taken; must equal F.ns.
static void t2Batch(HFork &F, Tail2 &T, Tr &tr, u32 used) {
  if (!T.planned || used != F.ns) refuse(F.G_ + "internal-slot-count");
  /* The challenge drawn below, the sub-proof's last transcript operation, must follow the segment's
     last commit point. Tested first: a draw out of the planned order terminates the program. */
  if (!T.link || !currentSegment || currentSegment->handed || currentSegment->moments != currentSegment->check->frame().shape().link[currentSegment->link].moments())
    refuse(F.G_ + "plan-link");
  t2FillL(F, T);
  T.used = used;
  F.batchBy(trBatchOpen(tr)); // doorH reports how many claims its powers have batched
  if (!F.planClaimsF) refuse(F.G_ + "internal-f-draws");
  F.kapMapF.assign(F.planClaimsF, 0);
  currentSegment->handed = true;
}
/* One run of entries in a linear claim. Under one kappa t2Lin adds the claim
       a + sum over the runs and i < n of coef * w[w0 + i] * entry(tree, row, first + i) = 0.
   a is an affine form (`Aff`, vf_subproof_engine.h); entry(t, v, c) is the sum over the rows r of array t of
   rows[v].coef[r] * the array's entry (r, c). */
struct Run2 {
  u32 tree = 0, row = 0;
  u64 first = 0;
  Ex coef = eO();
  const vector<Ex> *w = nullptr;
  u64 w0 = 0, n = 0;
};
static void t2Lin(HFork &F, const Aff &a, const vector<Run2> &runs) {
  const u32 number = F.nClaim;
  const Ex k = F.kap();
  F.push(k, a);
  if (F.valuesF) { // for tests: is this claim true on the prover's data? (../tests/claims_report.h)
    HFork::TestClaimValueParts value = F.valueOf(a);
    for (auto &r : runs) {
      if (!r.w || r.w0 > r.w->size() || r.n > r.w->size() - r.w0) refuse(F.G_ + "internal-L-index");
      for (u64 i = 0; i < r.n; i++) F.take(value, r.tree, r.row, r.first + i, r.coef * (*r.w)[r.w0 + i]);
    }
    F.judged(number, k, std::move(value), a);
  }
  for (auto &r : runs) {
    if (r.tree >= F.trees.size() || r.row >= F.trees[r.tree].rows.size()) refuse(F.G_ + "internal-L-row");
    HRow &R_ = F.trees[r.tree].rows[r.row];
    vector<Ex> &L = R_.L;
    // r.first is a column of the entire array, but L covers only the window that starts at R_.L0.
    if (!r.w || r.w0 > r.w->size() || r.n > r.w->size() - r.w0 || r.first < R_.L0 || r.first - R_.L0 > L.size() ||
        r.n > L.size() - (r.first - R_.L0))
      refuse(F.G_ + "internal-L-index");
    const Ex kc = k * r.coef;
    const Ex *w = r.w->data() + r.w0;
    Ex *Lp = L.data() + (r.first - R_.L0);
    // Each i has its own entry of L, so the blocks of one run are independent; the runs are done
    // one after another.
    parFor((r.n + ENTRIES_PER_PARALLEL_JOB - 1) / ENTRIES_PER_PARALLEL_JOB, [&](u64 b) {
      for (u64 i = b * ENTRIES_PER_PARALLEL_JOB; i < r.n && i < (b + 1) * ENTRIES_PER_PARALLEL_JOB; i++)
        if (!isZ(w[i])) Lp[i] = Lp[i] + kc * w[i];
    });
  }
}

// Gives lead * eq(zp, idx) at consecutive indices idx; eq is the multilinear equality polynomial
// of the point zp and the n low bits of idx. Only the tiling self-test of a sub-proof uses it
// (tileSelfTestF, vf_smallness_test_members.inc). With the walk it shares zp and field arithmetic, no more.
// pre[v] is lead times the product of the factors of the first v bits of idx, the highest of the n
// bits first, in the order in which the plain product multiplies them. Bits of idx from n up have
// no effect. next() steps to idx + 1: it rebuilds the products from the highest bit that changed
// on and keeps those above it.
// lead is 1, or the factor by which the calling code would multiply eq; it must be a value that
// the self-test has itself recomputed (from its own kappa), never one that the walk also reads.
// This is not checked.
struct EqRun {
  const Ex *zp = nullptr;
  int n = 0;
  u64 idx = 0;
  vector<Ex> pre, om;
  EqRun(const Ex *zp_, int n_, u64 idx_, Ex lead = eO())
      : zp(zp_), n(n_), idx(idx_), pre((size_t)n_ + 1), om((size_t)n_) {
    for (int v = 0; v < n; v++) om[v] = eO() - zp[v];
    pre[0] = lead;
    from(0);
  }
  inline __attribute__((always_inline)) void from(int v0) {
    Ex *p = pre.data();
    const Ex *o = om.data();
    for (int v = v0; v < n; v++) p[v + 1] = p[v] * (((idx >> (n - 1 - v)) & 1) ? zp[v] : o[v]);
  }
  Ex at() const { return pre[n]; }
  void next() {
    const int j = 63 - __builtin_clzll(idx ^ (idx + 1));
    idx++;
    from(j >= n ? 0 : n - 1 - j);
  }
};

// ==== the context of a segment, and what a sub-proof passes to the frame ====
// currentSegment is set by the driver (runLinksH, vf_segments.inc) for the sub-proof that runs on this thread.
// toBlkE gives an element of E in the library's type, blk::E, component by component.
static blk::E toBlkE(const Ex &x) {
  blk::E e;
  for (int k = 0; k < CE; k++) e.c[k].v = eC(x, k);
  return e;
}
// After a call of ChainCheck returned false: prints the channel's reason, if any, and rejects.
[[noreturn]] static void linkRefused(const Tr &tr) {
  chainHeld(tr.one);
  refuse("chain-refused"); // ChainCheck itself returned false: a call that was not due
}
/* Called at the segment's next commit point, where the prover commits to further arrays. `trees`
   must name exactly the layout's arrays there, so that no challenge that should follow an array's
   commitment can precede it. Else plan-link. */
static void rootInH(HFork &F, Tr &tr, std::initializer_list<int> trees) {
  static const u32 received[DW] = {0}; // marks HTree::root as committed; the frame has the root
  if (!currentSegment || currentSegment->handed || currentSegment->done) refuse(F.G_ + "plan-link");
  const unsigned moment = currentSegment->moments;
  const double tLater = now();
  if (!currentSegment->check->later()) linkRefused(tr);
  currentSegment->sLater += now() - tLater;
  currentSegment->moments++;
  const blk::ReqLayout &layout = *currentSegment->layout;
  const vector<blk::ReqMessage> &messages = currentSegment->check->frame().shape().link[currentSegment->link].messages;
  size_t named = 0;
  for (int t : trees) {
    if (t < 0 || (size_t)t >= currentSegment->arrayOf->size() || F.trees[(size_t)t].root) refuse(F.G_ + "plan-link");
    const std::pair<size_t, string> &at = (*currentSegment->arrayOf)[(size_t)t];
    const blk::ReqArray *a = at.first == currentSegment->link ? layout.find(at.first, at.second) : nullptr;
    if (!a || messages[a->message].moment != moment) refuse(F.G_ + "plan-link");
    F.rootTaken((size_t)t, received);
    named++;
  }
  size_t laid = 0;
  for (const blk::ReqArray &a : layout.link[currentSegment->link]) laid += messages[a.message].moment == moment;
  if (named != laid) refuse(F.G_ + "plan-link");
}
/* A virtual row's weights on the R rows in the form that the frame takes: a product over the bits
   of the row's number, bit 0 first (HRow::rowForm). coef, which self-tests read, is compared
   with it entry by entry (internal-row-form). */
static blk::IndexWeights rowWeights(const HFork &F, const HTree &X, const HRow &r) {
  const unsigned bits = blk::ceil_log2(X.R), np = (unsigned)r.rowPoint.size();
  const bool given = !r.rowLo.empty() || !r.rowHi.empty();
  if (np > bits || ((r.rowOnes || given) && (np || r.rowFirst)) || (r.rowFirst & ((1ull << np) - 1)) || r.rowFirst >= X.R ||
      (given && (r.rowOnes || r.rowLo.size() != bits || r.rowHi.size() != bits)))
    refuse(F.G_ + "internal-row-form");
  blk::IndexWeights w;
  for (unsigned t = 0; t < bits; t++) {
    if (given)
      w.lo.push_back(toBlkE(r.rowLo[t])), w.hi.push_back(toBlkE(r.rowHi[t]));
    else if (r.rowOnes)
      w.lo.push_back(blk::E::one()), w.hi.push_back(blk::E::one());
    else if (t < np) {
      const blk::E z = toBlkE(r.rowPoint[np - 1 - t]);
      w.lo.push_back(blk::E::one() - z), w.hi.push_back(z);
    } else {
      const bool one = (r.rowFirst >> t) & 1;
      w.lo.push_back(one ? blk::E::zero() : blk::E::one()), w.hi.push_back(one ? blk::E::one() : blk::E::zero());
    }
  }
  vector<blk::E> all(1, blk::E::one());
  for (unsigned t = 0; t < bits; t++) {
    all.resize(2 * all.size());
    for (size_t i = all.size() / 2; i-- > 0;) all[i + all.size() / 2] = all[i] * w.hi[t], all[i] = all[i] * w.lo[t];
  }
  for (u64 i = 0; i < X.R; i++)
    if (!(all[i] == toBlkE(r.coef[i]))) refuse(F.G_ + "internal-row-form");
  return w;
}
/* Called after the walk, in which the claims are added up. They have become one equation,
       sum over the trees' virtual rows of <coef (x) L, array> + <LS, the mask row> + c0 = 0.
   It goes to the frame as rank-one terms on named arrays, coefficients on the first entries of the
   segment's mask row and the public value -c0; ChainCheck::close makes it a linear claim on the
   segment's group and closes the segment.
   equationsLaid: how many powers of theta, after the claims', the range check's equations take up:
   one for each equation number, unused numbers included (HTailF2::equationsLaidF and stepF,
   vf_smallness_test_members.inc). It is a count only in part: the equations of the range check's reads are counted
   as they are added; those of its matrices are not counted, but computed from the matrices by the
   plan's formula, written a second time, so trBatchRows below cannot notice a matrix equation
   that was never added. */
static void doorH(HFork &F, Tail2 &T, Tr &tr, u64 equationsLaid) {
  if (!T.link || !currentSegment || T.walked || currentSegment->done) refuse(F.G_ + "internal-t2-order");
  const double tDoor = now();
  T.walked = true;
  // From here on kapTake() ends without a verdict.
  F.doorShutF = true;
  if (!currentSegment->handed) refuse(F.G_ + "internal-t2-order");
  if (!F.planClaimsF || F.nClaim != F.planClaimsF) refuse(F.G_ + "internal-f-door");
  // Ends without a verdict unless every kappa of the plan was taken: exactly once, as kapTake() refuses a second take.
  F.kapAllTakenF();
  // What is passed on is the weights on the listed arrays, LS and c0; a weight anywhere else would
  // be dropped without a test.
  for (size_t ti = T.nt.size(); ti < F.trees.size(); ti++)
    for (const HRow &r : F.trees[ti].rows)
      for (const Ex &x : r.L)
        if (!isZ(x)) refuse(F.G_ + "internal-f-door");
  if (currentSegment->arrayOf->size() != T.nt.size()) refuse(F.G_ + "plan-link");
  blk::ReqEquation eq;
  for (size_t ti = 0; ti < T.nt.size(); ti++) {
    if (T.nt[ti].outside) refuse(F.G_ + "internal-t2-order"); // weights that are kept outside the virtual rows could not be passed on
    const HTree &X = F.trees[ti];
    for (const HRow &r : X.rows) {
      if (r.coef.size() != X.R || r.L.size() != X.K || r.L0) refuse(F.G_ + "internal-coef-" + X.name); // no tree of a segment has a window of columns
      auto L = std::make_shared<vector<blk::E>>(X.K);
      for (u64 c = 0; c < X.K; c++) (*L)[c] = toBlkE(r.L[c]);
      blk::ReqTerm t;
      t.born = (*currentSegment->arrayOf)[ti].first, t.array = (*currentSegment->arrayOf)[ti].second;
      t.on_columns = blk::IndexWeights::written(std::move(L));
      currentSegment->written += X.K;
      if (r.rowForm) {
        t.on_rows = rowWeights(F, X, r);
        eq.terms.push_back(std::move(t));
        continue;
      }
      currentSegment->rowsCut++;
      // A term takes only one written-out factor; here both are: one term per row with coef not 0.
      for (u64 i = 0; i < X.R; i++) {
        if (isZ(r.coef[i])) continue;
        t.scale = toBlkE(r.coef[i]);
        t.on_rows = blk::IndexWeights();
        for (unsigned b = 0; b < blk::ceil_log2(X.R); b++) {
          const bool one = (i >> b) & 1;
          t.on_rows.lo.push_back(one ? blk::E::zero() : blk::E::one()), t.on_rows.hi.push_back(one ? blk::E::one() : blk::E::zero());
        }
        eq.terms.push_back(t);
      }
    }
  }
  // The equation on the mask row itself: LS[j] is mask entry j's coefficient, c0 the constant.
  vector<Ex> LS;
  Ex c0 = eZ();
  F.settle(LS, c0, T.used);
  for (const Ex &x : LS) eq.on_masks.push_back(toBlkE(x));
  eq.target = toBlkE(eZ() - c0);
  currentSegment->terms = eq.terms.size();
  const double tClose = now();
  currentSegment->sDoor = tClose - tDoor;
  /* The channel terminates the program unless the batching round's error was computed for exactly
     this number of claims and equations. That round is still the last one closed: the functions
     that check the sub-proofs do not use the transcript between t2Batch's draw and this line. */
  trBatchRows(tr, (u64)F.nClaim + equationsLaid);
  if (!currentSegment->check->close(eq)) linkRefused(tr);
  currentSegment->sClose = now() - tClose;
  currentSegment->done = true;
}
