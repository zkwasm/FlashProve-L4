// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Unit test of aforkShapeHid() (verifier/src/vf_attention_check.h). From the public parameters (`ACfg`) of
// the attention sub-proof (`AFork`), the part of a proof that covers attention, that function
// derives the shape of the sub-proof's lookup forest: the product trees of its lookup calls. A
// lookup call claims that all its keys occur in a public table. The expected values are the APUB
// and AREFUSE lines of verifier/tests/forest_weights.txt, which
// verifier/tests/known_answers/forest_values.py computes without the verifier's code.
// verifier/tests/hc_parts_test.sh builds and runs this program.
//
// Not tested: anything that reads a proof; how the verifier, cp_verify, fills `ACfg` from a
// configuration's files; a call's table (ACall::tau, Call::fn, Call::t0); where its keys are placed
// (ACall::pcs); HForest::rK, cK and callOf; the padding counts (HForest::pub) of the other
// forests: the approximate range check's (verifier/src/vf_smallness_test_members.inc) and those of the layer and
// output-head sub-proofs (verifier/src/cp_verify.cpp). cp_verify computes both sub-proofs' counts
// only during runStage(), which needs an entire configuration.
#define main cp_verify_main
#include "cp_verify.cpp"
#undef main
#include "flat_known.h"

// Fails the test: prints "NO: internal-hc-forest-attention-test-<what>" and exits with code 1.
static void bad(const string &what) {
  refuse("internal-hc-forest-attention-test-" + what);
}
static vector<string> cutAt(const string &s, char at) {
  vector<string> o(1);
  for (char c : s)
    if (c == at)
      o.emplace_back();
    else
      o.back() += c;
  return o;
}
// Fills A.cf from a line's fields and sets the global LIDB, the cap on a tree's depth dT (the
// `lidb` line of the parameter file, `policy.txt`). Field `other` holds that file's lines rx, lmS,
// la, lmbA, lmh and lmo. The compared numbers do not depend on ACfg's fields S, NAs, wcQ and Ty,
// which are chosen so that AFork::shape() accepts them, not as cp_verify sets them. There Ty, a
// bound in the no-overflow checks, is 2^18 (runStage() in verifier/src/cp_verify.cpp), so the
// checks that contain it (af-no-wrap-N1 to N3, N5 and N6 in verifier/src/vf_attention_shape.h) pass more
// easily here.
static void fill(AFork &A, const KnownLine &k) {
  static const std::map<string, int> segment = {{"RM", sRM}, {"RD", sRD}, {"FM", sFM}, {"FD", sFD}, {"WL", sWL}, {"WH", sWH}, {"IX", sIX}};
  const string at = "-" + k.arg.at("name");
  {
    const vector<u64> sh = knownList(k.arg.at("sh")), other = knownList(k.arg.at("other"));
    if (other.size() != 6 || sh.empty() || (sh.size() & (sh.size() - 1))) bad("malformed-line" + at);
    ACfg &c = A.cf;
    c.lt = (int)k.n("lt"), c.lB = (int)k.n("lB"), c.lgk = (int)k.n("lgk"), c.ldh = (int)k.n("ldh"), c.ll = clog2(sh.size());
    c.H = k.n("HA") << c.lgk;
    c.S = 12, c.rx = (int)other[0], c.kFD = (int)k.n("kFD"), c.lmS = (int)other[1], c.la = (int)other[2], c.lmb = (int)other[3];
    c.nmin = (int)k.n("nmin");
    c.NAs = 1;
    c.wcH = (1ull << other[4]) >> c.lt, c.wcO = c.wcQ = (1ull << other[5]) >> c.lt;
    c.Ty = 1000;
    for (u64 x : sh) c.sh.push_back((u32)x);
    for (auto &l : cutAt(k.arg.at("list"), ';')) {
      const vector<string> f = cutAt(l, ':');
      if (f.size() != 3) bad("malformed-line" + at);
      ACfg::Line L;
      L.name = f[0], L.kind = f[1];
      for (auto &s : cutAt(f[2], '+')) L.segs.push_back(segment.at(s));
      c.lines.push_back(L);
    }
    LIDB = (int)k.n("lid");
  }
}
// The sub-proof's calls form one forest of NT product trees of 2^dT keys each. Call c has 2^n_c
// keys, of which those from index tot_c on are padding, and a table of 2^k_c rows. Its keys fill
// 2^iota_c trees, from tree first_c on. Tree y has pub_y padding keys.
//
// An AREFUSE line gives the expected rejection name. For an APUB line the test compares:
//   - the calls that AFork::shape() expands ACfg::lines into: names in list order, tot_c, n_c, k_c;
//   - dT = min(LIDB, min n_c), kT = max k_c, NT, first_c, iota_c = n_c - dT and every pub_y;
//   - the number of columns of bA, the matrix of the bits of the multiplicities (how many
//     non-padding keys hit each table row).
// If aforkShapeHid() rejects an APUB line, the program ends with
// "NO: <the verifier's rejection name>".
// In one APUB line ACfg::sh (each layer's attention shift: sh in `KL.bin`) has two distinct values,
// so the call lines KR and KFD each become two calls. Except for `sh`, the line `full_prompt`
// has the values of prover/examples/target_full_f2/policy.txt; no program compares them.
int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: hc_forest_attention_test <forest_weights.txt>\n");
    return 2;
  }
  size_t sets = 0, trees = 0, allPadding = 0, noPadding = 0, moved = 0, capBinds = 0, filled = 0, refusals = 0;
  for (auto &k : knownLines(argv[1], "APUB")) {
    const string at = "-" + k.arg.at("name");
    std::map<string, string> want;
    for (auto &x : cutAt(k.answer, ' ')) {
      const size_t eq = x.find('=');
      if (eq == string::npos) bad("malformed-line" + at);
      want[x.substr(0, eq)] = x.substr(eq + 1);
    }
    AFork A;
    fill(A, k);
    const HForest h = aforkShapeHid(A);
    const vector<string> names = cutAt(want.at("calls"), ',');
    const vector<u64> tot = knownList(want.at("tot")), n = knownList(want.at("n")), kc = knownList(want.at("k")), first = knownList(want.at("first")),
                      pub = knownList(want.at("pub"));
    if (A.calls.size() != names.size() || tot.size() != names.size() || n.size() != names.size() || kc.size() != names.size() ||
        first.size() != names.size())
      bad("calls" + at);
    for (size_t q = 0; q < names.size(); q++) { // what shape() makes of the list
      if (A.calls[q].c.name != names[q]) bad("name" + at);
      if (A.calls[q].tot != tot[q]) bad("tot" + at);
      if ((u64)A.calls[q].c.n != n[q]) bad("n" + at);
      if ((u64)A.calls[q].c.k != kc[q]) bad("k" + at);
      filled += tot[q] == 1ull << n[q];
    }
    if ((u64)h.dT != std::stoull(want.at("dT")) || (u64)h.kT != std::stoull(want.at("kT")) || (u64)h.NT != std::stoull(want.at("NT")) ||
        pub.size() != (size_t)h.NT)
      bad("forest" + at);
    capBinds += k.n("lid") < *std::min_element(n.begin(), n.end());
    for (size_t q = 0; q < names.size(); q++) {
      if ((u64)h.first[q] != first[q]) bad("first" + at);
      if ((u64)A.calls[q].c.iota != n[q] - std::stoull(want.at("dT"))) bad("iota" + at);
      moved += q && first[q] < first[q - 1];
    }
    if (h.pub.size() != pub.size()) bad("pub-length" + at);
    for (size_t y = 0; y < pub.size(); y++) {
      if (h.pub[y] != pub[y]) bad("pub-tree-" + std::to_string(y) + at);
      allPadding += pub[y] == 1ull << h.dT;
      noPadding += !pub[y];
    }
    if (A.kbA != std::stoull(want.at("columns"))) bad("columns" + at);
    sets++, trees += pub.size();
  }
  for (auto &k : knownLines(argv[1], "AREFUSE")) {
    AFork A;
    fill(A, k);
    string got;
    refuseThrows = true;
    try {
      (void)aforkShapeHid(A);
    } catch (const Refusal &r) {
      got = r.name;
    }
    refuseThrows = false;
    if (got != k.answer) bad("refusal-" + k.arg.at("name") + "-expected-" + k.answer + "-got-" + (got.empty() ? "none" : got));
    refusals++;
  }
  // The input must contain the cases that the test is meant to cover. moved: a call whose trees
  // come before those of the call listed before it (forestShape() in verifier/src/vf_lookup_argument.h
  // places larger calls first); capBinds: LIDB lowers dT; filled: a call without padding.
  if (sets != 8 || trees != 89 || allPadding != 15 || noPadding != 40 || moved != 6 || capBinds != 1 || filled != 1 || refusals != 5) {
    printf("hc_forest_attention_test: found (expected): sets %zu (8), trees %zu (89), all padding %zu (15), no padding %zu (40), moved %zu (6), "
           "the cap binds %zu (1), a call filled %zu (1), refusals %zu (5)\n",
           sets, trees, allPadding, noPadding, moved, capBinds, filled, refusals);
    bad("the-files-counts");
  }
  printf("hc_forest_attention_test: %zu sets, %zu trees, %zu refusals: all tests passed\n", sets, trees, refusals);
  return 0;
}
