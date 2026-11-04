// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* For tests only: compares the verifier's coefficient vectors for a sub-proof (`fork` in
 * identifiers) with the prover's. The verifier program cp_verify does not include this file: a
 * checker in this folder does.
 * A sub-proof's claims are batched into one linear equation: claim number j is scaled by theta^j
 * (theta: a challenge). Prover and verifier number the claims in separate code; the comparison
 * tests that they agree.
 * The equation's weight on cell (r, c) of an array of R rows and K columns (HTree, a "tree") is
 * the sum, over the tree's virtual rows (HRow in vf_subproof_engine.h), of coef[r] * L[c]. The prover calls
 * a virtual row a pair, and coef cv. */
#pragma once
#include <map>
// One virtual row. pair: its number in its tree. coef and L point into the HFork.
struct TestVirtualRowVectors {
  string array;
  u32 pair = 0;
  const vector<Ex> *coef = nullptr, *L = nullptr;
  u64 L0 = 0, K = 0; // L contains columns L0 .. L0 + L->size() - 1 of the array's K
};
// Lists F's virtual rows, tree by tree. names: null (use each tree's own name), or for each tree
// the name of its array in the prover's file, or "" to skip the tree. If `names` and F.trees
// differ in size, refuse() (vf_common.h) exits or throws.
static vector<TestVirtualRowVectors> forkVectors(const HFork &F, const vector<string> *names = nullptr) {
  vector<TestVirtualRowVectors> out;
  if (names && names->size() != F.trees.size()) refuse(F.G_ + "internal-claim-values");
  for (size_t t = 0; t < F.trees.size(); t++) {
    const HTree &X = F.trees[t];
    if (names && (*names)[t].empty()) continue;
    for (size_t r = 0; r < X.rows.size(); r++) out.push_back({names ? (*names)[t] : X.name, (u32)r, &X.rows[r].coef, &X.rows[r].L, X.rows[r].L0, X.K});
  }
  return out;
}
/* Returns "" if the two sides agree, else a message on the first fault or difference.
 * f: positioned where `put_door_vectors` began to write (format: prover/hiding/door_vectors.cuh).
 * Compares the linear form on each array, not the lists of vectors. The two sides may list their
 * arrays in different orders. One side may have one pair where the other has two virtual rows with
 * the same cv: on the approximate range check's array ZZ, for each block of its rows, the verifier
 * has one virtual row for each of the two lookup calls (low and high digits' columns) and the
 * prover one pair for both (verifier/src/vf_smallness_test_members.inc, prover/hiding/tail_f.cuh). Either side may
 * list a virtual row on which no claim puts a weight.
 * So within an array, on each side, the L of all virtual rows with the same cv are added, and the
 * sums are compared column by column. If only one side has a cv, its sum must be 0. Lists that give
 * the same form only through a linear dependence among different cv are reported as different.
 * Not compared: the equation's coefficients on hidden values (`HFork::LS`) and its constant
 * (`HFork::c0`). */
static string vectorsDifference(const HFork &F, FILE *f, const vector<string> *names = nullptr) {
  auto word = [f](u64 &x) { return fread(&x, 8, 1, f) == 1; };
  auto element = [&](Ex &x) {
    u64 w[CE];
    for (int k = 0; k < CE; k++)
      if (!word(w[k]) || w[k] >= FIELD_P) return false;
    x = Ex{w[0], w[1], w[2]};
    return true;
  };
  struct Gathered {
    vector<Ex> cv, theirs, mine; // prover's and verifier's sums of L; empty: no row with this cv
    string first;                // the first row with this cv, for the message
  };
  std::map<string, vector<Gathered>> arrays;
  auto same = [](const vector<Ex> &a, const vector<Ex> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
      if (!(a[i] == b[i])) return false;
    return true;
  };
  auto at = [&](const string &array, const vector<Ex> &cv, const string &who) -> Gathered & {
    vector<Gathered> &list = arrays[array];
    for (Gathered &g : list)
      if (same(g.cv, cv)) return g;
    list.push_back({cv, {}, {}, who});
    return list.back();
  };
  auto add = [](vector<Ex> &sum, u64 K, u64 first, const vector<Ex> &L) {
    if (sum.empty()) sum.assign(K, eZ());
    if (sum.size() != K || first > K || L.size() > K - first) return false;
    for (size_t c = 0; c < L.size(); c++) sum[first + c] = sum[first + c] + L[c];
    return true;
  };
  u64 n = 0;
  if (!word(n) || n > (1u << 20)) return "the file holds no vectors";
  for (u64 i = 0; i < n; i++) {
    u64 len = 0, pair = 0, R = 0, K = 0;
    if (!word(len) || len > 64) return "the file is short or no such file";
    string name(len, ' ');
    if (fread(&name[0], 1, len, f) != len || !word(pair) || !word(R) || !word(K) || R > (1ull << 32) || K > (1ull << 32)) return "the file is short";
    vector<Ex> cv(R), L(K);
    for (Ex &x : cv)
      if (!element(x)) return "the file is short, or holds a word that is no element";
    for (Ex &x : L)
      if (!element(x)) return "the file is short, or holds a word that is no element";
    if (!add(at(name, cv, "the prover's pair " + std::to_string(pair)).theirs, K, 0, L)) return name + ": pairs with other numbers of columns";
  }
  for (const TestVirtualRowVectors &v : forkVectors(F, names))
    if (!add(at(v.array, *v.coef, "the verifier's virtual row " + std::to_string(v.pair)).mine, v.K, v.L0, *v.L)) return v.array + ": rows with other numbers of columns";
  for (const auto &a : arrays)
    for (const Gathered &g : a.second) {
      const vector<Ex> &one = g.theirs.empty() ? g.mine : g.theirs;
      if (!g.theirs.empty() && !g.mine.empty() && g.theirs.size() != g.mine.size()) return a.first + ": other numbers of columns";
      for (size_t c = 0; c < one.size(); c++) {
        const Ex t = g.theirs.empty() ? eZ() : g.theirs[c], m = g.mine.empty() ? eZ() : g.mine[c];
        if (!(t == m))
          return a.first + ", the rows' weights of " + g.first + ": the weights on the columns differ at column " + std::to_string(c) +
                 (g.theirs.empty() ? " (the prover has no pair with these weights on the rows)" : g.mine.empty() ? " (the verifier has no virtual row with these weights on the rows)" : "");
      }
    }
  return "";
}
