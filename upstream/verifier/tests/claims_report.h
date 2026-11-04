// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests: lists which claims of a sub-proof are false on the prover's data. A rejection alone
// does not show whether the intended check or a later one caught a false statement.
// A sub-proof (`HFork`, verifier/src/vf_subproof_engine.h) covers one part of the forward pass; a claim
// states that an affine form (`Aff`) is 0. The verifier's form decides what is proved, so the
// verifier evaluates it, on the prover's values. After claimsLoad(), HFork::lin() and t2Lin()
// (verifier/src/vf_subproof_segment.h) record each added claim; HFork::section() names the claims that
// follow it (verifier/src/vf_lookup_argument.h names the lookup's claims). No decision of the verifier reads
// the list. If a claim reads a row, column or slot that the sub-proof or the values lack
// (HFork::take(), judgeAll()), refuse() gets a name ending in `internal-claim-values`, which the two
// checkers that load values catch. The verifier program, `cp_verify`, as this repository builds it,
// loads no values and never gets there. A file that does not fit the sub-proof's arrays only makes
// claimsLoad() return the reason. Synthetic data only: the values are linear combinations of the
// committed arrays' rows, and the masks of the scalars that the sub-proof sends, so their file
// reveals committed data.
// `cp_verify` does not include this file: the checkers in this folder do.
#pragma once
#include <map>
// Reads a file in the format of put_door_values() (prover/hiding/door_values.cuh), which only a
// test build of the prover and a test program call: virtual rows, each a linear combination of a
// committed array's rows, and the mask row. That row has one entry for each slot, that is, for each
// scalar that the sub-proof sends, masked or not; at the slots of a sumcheck's unmasked scalars the
// entries are its hiding polynomial's coefficients (`Psc`, verifier/src/vf_sumcheck_psc.h). Call
// claimsLoad() before the first claim is added (checked).
// Returns "", or why no values were loaded. names[t] is the file's name for the array of
// `F.trees[t]`; a tree named "" gets no values, and a claim that reads it is refused under that
// name (`internal-claim-values`). HFork::judgeAll() finds a virtual row's values (`TestProverVirtualRow`:
// weights `cv`, entries `D`, one for each column) by `cv`, not by the file's pair number.
static string claimsLoad(HFork &F, FILE *f, const vector<string> &names) {
  auto word = [f](u64 &x) { return fread(&x, 8, 1, f) == 1; };
  auto element = [&](Ex &x) {
    u64 w[CE];
    for (int k = 0; k < CE; k++)
      if (!word(w[k]) || w[k] >= FIELD_P) return false;
    x = Ex{w[0], w[1], w[2]};
    return true;
  };
  if (names.size() != F.trees.size()) return "one name for each tree";
  if (F.nClaim) return "claims have been added already";
  std::map<string, vector<HFork::TestProverValues::TestProverVirtualRow>> theirs;
  u64 n = 0;
  if (!word(n) || n > (1u << 20)) return "the file holds no values";
  for (u64 i = 0; i < n; i++) {
    u64 len = 0, pair = 0, R = 0, K = 0;
    if (!word(len) || len > 64) return "the file is short or no such file";
    string name(len, ' ');
    if (fread(&name[0], 1, len, f) != len || !word(pair) || !word(R) || !word(K) || R > (1ull << 32) || K > (1ull << 32)) return "the file is short";
    HFork::TestProverValues::TestProverVirtualRow t;
    t.cv.resize(R), t.D.resize(K);
    for (Ex &x : t.cv)
      if (!element(x)) return "the file is short, or holds a word that is no element";
    for (Ex &x : t.D)
      if (!element(x)) return "the file is short, or holds a word that is no element";
    theirs[name].push_back(std::move(t));
  }
  auto values = std::make_shared<HFork::TestProverValues>();
  if (!word(n) || n > (1u << 28)) return "the file is short";
  values->mask.resize(n);
  for (Ex &x : values->mask)
    if (!element(x)) return "the file is short, or holds a word that is no element";
  if (values->mask.size() < F.ns) return "the mask row has fewer entries than the sub-proof has slots";
  values->pairs.resize(F.trees.size());
  for (size_t t = 0; t < F.trees.size(); t++) {
    if (names[t].empty()) continue;
    auto list = theirs.find(names[t]);
    if (list == theirs.end()) return "the verifier has the array " + names[t] + ", the prover has not";
    for (const auto &p : list->second)
      if (p.cv.size() != F.trees[t].R || p.D.size() != F.trees[t].K) return names[t] + ": other numbers of rows or columns";
    values->pairs[t] = std::move(list->second);
    theirs.erase(list);
  }
  if (!theirs.empty()) return "the prover has the array " + theirs.begin()->first + ", the verifier has not";
  F.valuesF = values;
  return "";
}
