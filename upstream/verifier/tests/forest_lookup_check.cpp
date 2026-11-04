// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test program: runs the verifier's side of the lookup argument for a group of lookup calls
// (verifier/src/vf_lookup_argument.h; `forest` in identifiers, because the keys are arranged as trees of
// equal size) on scalars that a GPU prover has sent. prover/tests/forest_lookup_test.cu writes the
// files in the input directory `<folder>` (struct File); main()'s comment lists what each group
// must satisfy.
// A successful run shows that the GPU's lookup and the verifier's agree in format and algebra on
// these groups. It runs no commitment and none of the sub-proofs that use the lookup. The prover's
// test and this one run on a hash chain (the Fiat-Shamir transcript of blocks/chain/; `Tr::one`,
// verifier/src/vf_common.h). The chain's first query (the hash input that fixes its public inputs)
// contains the plan (the list of rounds) but none of the group's other public inputs. The plan does
// not fix the group's shape: it has v and d, not dT, rK, kT, cK (struct HForest).
// Build, from verifier/src:
//   g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
//       -I ../../blocks -I ../../prover ../tests/forest_lookup_check.cpp
// Run: a.out <folder> [refuse]
// VF_WITHOUT_LOOKUPS leaves out verifier/src/vf_lookup_helpers.h, which this test does not use.
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "vf_lookup_argument.h"
#include "coefficient_vectors.h"
#include "claims_report.h"

#include "chain/public.h"
#include "numbers/blocks.h"

// One group's input: lookup_<i>.bin and, for `values`, lookup_<i>.bin.values.
//   lid      the cap on a tree's depth (forestShape())
//   n        the number of slots: of sent scalars and of mask entries
//   tau      a call's table if it is written out; empty for a range table (row t has value t)
//   keys     a call's keys, as row numbers of its table
//   padding  how many of a call's last keys are padding keys
//   bits     the planes, one byte each: bit j of the multiplicity of row t of call c's table is at
//            ((c << kT) + t) * 32 + j; the largest table has 2^kT rows
//   walk     the coefficients on the mask row that the prover's `hw::Walk` (prover/hiding/walk.cuh)
//            has added up
//   proof    the chain's salt, then the sent scalars; main() checks this. The prover's test passes
//            a salt of zeros, not random bytes.
//   vectors  `hw::Walk`'s coefficient vectors on planes and keys
//   values   the virtual rows (linear combinations of an array's rows) and the mask row
struct File {
  int lid = 0;
  u32 n = 0;
  vector<Call> calls;
  vector<vector<u64>> tau;
  vector<vector<uint16_t>> keys;
  vector<u64> padding;
  vector<u8> bits;
  vector<Ex> sent, mask, walk;
  vector<u8> proof;
  int lm = 0;         // the prover arranges the planes as a matrix of 2^lm rows
  vector<u8> vectors; // written by `hd::put_door_vectors` (prover/hiding/door_vectors.cuh)
  vector<u8> values;  // written by `hd::put_door_values` (prover/hiding/door_values.cuh)
};
static u64 get64(FILE *f) {
  u64 x;
  if (fread(&x, 8, 1, f) != 1) printf("the file is short\n"), exit(2);
  return x;
}
static Ex getE(FILE *f) {
  Ex x;
  for (int i = 0; i < CE; i++) eSet(x, i, get64(f));
  return x;
}
template <class X> static void getAll(FILE *f, vector<X> &v) {
  if (!v.empty() && fread(v.data(), sizeof(X), v.size(), f) != v.size()) printf("the file is short\n"), exit(2);
}
// The multilinear extension of a at z; z[0] belongs to the top bit of the index.
static Ex topFirst(vector<Ex> a, const vector<Ex> &z) {
  for (const Ex &c : z) {
    const size_t half = a.size() / 2;
    for (size_t x = 0; x < half; x++) a[x] = a[x] + c * (a[x + half] - a[x]);
    a.resize(half);
  }
  return a[0];
}
// Runs the verifier's lookup on `sent`. Returns "" if all checks pass, else what failed.
// forestPad() must read exactly the w.n scalars. It leaves linear claims on mask entries, planes
// and keys.
// morePadding: count one more padding key in one tree. No call becomes all padding, which
// forestPad() rejects: the prover's test gives each call at least three other keys.
// falseClaims, if given, receives the false claims' names (modes 0 and 3).
// entries is the mode: what the claims read of planes and keys, and what is checked.
//   0  Values computed from w.bits and w.keys. The batched equation (the sum of the claims, claim
//      i, from 0, times the i-th power of the batching challenge) must hold on the mask row. With
//      compareWalk, its coefficients on the mask row must equal w.walk.
//   1  Entries of named arrays' virtual rows. The coefficient vectors on these arrays must agree
//      with w.vectors (vectorsDifference(), coefficient_vectors.h). The reason: the prover's and
//      the verifier's sources each write out the order of the claims, and a claim's weight goes by
//      its number.
//   2  Negative control for 1: the join (the claim on the key vectors, forestJoin()) is added
//      before the three claims on the planes, not after, so the vectors must differ.
//   3  The verifier's report of false claims (claims_report.h) on w.values. falseClaims is
//      required and receives its list, under the names of mode 0.
static string verify(File &w, const vector<Ex> &sent, bool morePadding, bool compareWalk, string *falseClaims = nullptr, int entries = 0) {
  vector<Call *> cl;
  for (Call &c : w.calls) cl.push_back(&c);
  const HForest shape = forestShape(cl, w.lid);
  if (forestElems(shape) != w.n) return "the number of slots";
  vector<u64> pub(shape.NT, 0);
  for (size_t c = 0; c < cl.size(); c++) forestPubRun(shape, pub, (int)c, (1ull << cl[c]->n) - w.padding[c], w.padding[c]);
  if (morePadding) { // one more in the first tree that is not all padding
    size_t y = 0;
    while (y < pub.size() && pub[y] >= (1ull << shape.dT)) y++;
    if (y >= pub.size()) return "the test's group is nothing but padding";
    pub[y]++;
  }
  vector<u32> words;
  for (const Ex &x : sent)
    for (int i = 0; i < CE; i++) words.push_back((u32)eC(x, i)), words.push_back((u32)(eC(x, i) >> 32));
  HFork F;
  F.slots(w.n);
  // `tr` wraps a hash chain (functions: verifier/src/vf_transcript_channel.h): the chain reads the
  // scalars itself, under a lookup's plan. The proof is the chain's salt
  // and then the scalars, eight bytes per coordinate; rebuilding it from `sent` lets a
  // changed scalar reach the chain. A w.proof too short for the salt is a fault of the test, never
  // a rejection (main()).
  if (w.proof.size() < blk::DIGEST_BYTES) throw Refusal{"the proof's bytes are missing"};
  vector<u8> bytes(w.proof.begin(), w.proof.begin() + blk::DIGEST_BYTES);
  for (u32 x : words)
    for (int i = 0; i < 4; i++) bytes.push_back((u8)(x >> (8 * i)));
  blk::Plan plan(blk::setting_goldilocks());
  forestPlan(plan, "group", shape);
  blk::plan_bat_sigma(plan, "claims", forestLins(shape) + 4);
  blk::Binder binder(6, plan.finish());
  blk::Digest configuration{}; // required; a test value, as in prover/tests/forest_lookup_test.cu
  configuration.w[0] = 0x6c6b7570u;
  (void)binder.configuration(configuration);
  blk::Reader reader(bytes.data(), bytes.size());
  blk::VerifierSession session; // one chain, one verdict; it outlives the channel
  blk::VerifierRounds channel(binder, reader, session);
  Tr tr{channel};
  u32 slot = 0;
  HForest h = forestPad(F, tr, cl, pub, slot, w.lid);
  if (slot != w.n) return "scalars left unread";
  F.batchBy(trBatchOpen(tr)); // the chain gets the number of claims once all are added, below
  const int lm = entries ? w.lm : 7, wd = h.kT + 5 - lm;
  auto bit = [&](u64 t, u64 c, u64 j) { return (u64)w.bits[((c << h.kT) + t) * 32 + j]; };
  const vector<Ex> phi[3] = {forestPhiB(h, lm, 0), forestPhiB(h, lm, 1), forestPhiP(lm)};
  if (entries) { // the arrays, named and ordered as in prover/tests/forest_lookup_test.cu
    HTree planes;
    planes.name = "planes", planes.R = 1ull << lm, planes.K = (u64)h.NC << wd;
    for (int i = 0; i < 3; i++) {
      HRow r;
      r.coef = phi[i], r.L.assign(planes.K, eZ());
      planes.rows.push_back(r);
    }
    F.trees.push_back(planes);
    for (size_t c = 0; c < cl.size(); c++) {
      HTree keys;
      keys.name = "keys " + std::to_string(c), keys.R = 1, keys.K = 1ull << cl[c]->n;
      HRow r;
      r.coef = {eO()}, r.L.assign(keys.K, eZ());
      keys.rows.push_back(r);
      F.trees.push_back(keys);
    }
  }
  auto column = [&](int which, u64 col) {
    if (entries) return affEnt(0, (u32)which, col);
    const u64 c = col >> wd, high = col & ((1ull << wd) - 1);
    Ex s = eZ();
    for (u64 row = 0; row < (1ull << lm); row++)
      if (bit((high << (lm - 5)) + (row >> 5), c, row & 31)) s = s + phi[which][row];
    return affC(s);
  };
  vector<Aff> KF;
  for (size_t c = 0; c < cl.size(); c++) {
    if (entries) {
      const vector<Ex> weight = eqTab(cl[c]->zk.data(), (int)cl[c]->zk.size());
      Aff form;
      for (u64 i = 0; i < weight.size(); i++) {
        Ent x;
        x.tree = (u32)(1 + c), x.row = 0, x.idx = i, x.coef = weight[i];
        form.e.push_back(x);
      }
      KF.push_back(std::move(form));
      continue;
    }
    vector<Ex> values;
    for (uint16_t key : w.keys[c]) values.push_back(eB(cl[c]->tauAt(key)));
    KF.push_back(affC(topFirst(std::move(values), cl[c]->zk)));
  }
  if (entries == 3) {
    vector<string> names = {"planes"};
    for (size_t c = 0; c < cl.size(); c++) names.push_back("keys " + std::to_string(c));
    FILE *f = fmemopen(w.values.data(), w.values.size(), "rb");
    if (!f) return "the values cannot be read";
    const string fault = claimsLoad(F, f, names);
    fclose(f);
    if (!fault.empty()) return "the values: " + fault;
  }
  forestObjects(F, h);
  const vector<Aff> onPlanes = forestPlaneClaims(F, h, lm, [&](int e, u64 col) { return column(e, col); }, [&](u64 col) { return column(2, col); });
  const Aff join = forestJoin(h, KF);
  if (entries == 2) F.lin(join);
  if (onPlanes.size() != 3) return "not three claims on the planes";
  for (size_t i = 0; i < 3; i++) F.section(FOREST_PLANE_SECTIONS[i]), F.lin(onPlanes[i]);
  forestPlanesAdded(h);
  F.section(FOREST_JOIN_SECTION);
  if (entries != 2) F.lin(join);
  trBatchRows(tr, F.nClaim); // the number of claims actually added
  if (!channel.finish() || !session.accepted()) return "the end of the proof";
  if (entries == 3) {
    F.judgeAll();
    for (const HFork::TestClaimFoundFalse &x : F.falseF) *falseClaims += x.section.substr(string("look-up: ").size()) + "; ";
    return isZ(F.valuesSumF) == F.falseF.empty() ? "" : "the report's list and the equation's value on the data disagree";
  }
  if (entries) {
    FILE *f = fmemopen(w.vectors.data(), w.vectors.size(), "rb");
    if (!f) return "the vectors cannot be read";
    const string differs = vectorsDifference(F, f);
    fclose(f);
    return differs;
  }
  if (falseClaims) {
    // Slot j stands for the value vp[j] - mf[j] * mask[j] (vp: sent; mf: public factor).
    auto holds = [&](const Aff &a) {
      Ex s = a.c;
      for (auto &x : a.s) s = s + x.second * (F.vp[x.first] - F.mf[x.first] * w.mask[x.first]);
      return a.e.empty() && isZ(s);
    };
    auto name = [&](size_t i) {
      const size_t v = (size_t)h.v(), d = (size_t)h.d();
      return i < 31 ? "tie " + std::to_string(i) : i < 31 + v ? "key level " + std::to_string(i - 31) : i < 31 + v + d ? "table level " + std::to_string(i - 31 - v) : string("bits");
    };
    for (size_t i = 0; i < h.rows.size(); i++)
      if (!holds(h.rows[i])) *falseClaims += name(i) + "; ";
    const char *planes[3] = {"leaves; ", "planes at the bits' point; ", "number of keys; "};
    for (int i = 0; i < 3; i++)
      if (!holds(onPlanes[i])) *falseClaims += planes[i];
    if (!holds(join)) *falseClaims += "join; ";
  }
  vector<Ex> coef;
  Ex sum;
  F.settle(coef, sum, w.n);
  for (u32 j = 0; j < w.n; j++) sum = sum + coef[j] * w.mask[j];
  if (!isZ(sum)) return "the equation";
  if (compareWalk)
    for (u32 j = 0; j < w.n; j++)
      if (!(coef[j] == w.walk[j])) return "the walk's coefficient of mask entry " + std::to_string(j);
  return "";
}

// Each group must pass verify() in modes 0 (with compareWalk), 1 and 3 with no claim false. It must
// fail in mode 2 and, in mode 0, with 1 added to any one of 40 evenly spaced scalars, a key
// changed, a bit of a multiplicity flipped, or one more padding key. w.proof must be exactly the
// chain's salt and the sent scalars.
// With `refuse` the files are a malicious prover's (printed: cheating prover's), and each group
// must be rejected (printed: refused): by a thrown `Refusal`, or by verify()'s result, and then
// modes 0 and 3 must name the same false claims. A w.proof that is not the salt and the scalars
// is a fault of the test, not a rejection: the group fails.
// Exit code: 0 if every group passed, 1 if not, 2 for bad usage or unreadable input.
int main(int argc, char **argv) {
  if (argc < 2) return printf("usage: forest_lookup_check <folder> [refuse]\n"), 2;
  const bool refuse_all = argc > 2 && string(argv[2]) == "refuse";
  refuseThrows = true; // a rejection is caught below and reported for its group
  int groups = 0, failed = 0, expected = 0;
  {
    // A group without a file failed in the prover's test and counts as failed here.
    FILE *f = fopen((string(argv[1]) + "/lookup_groups.txt").c_str(), "r");
    if (!f || fscanf(f, "%d", &expected) != 1 || expected < 1) return printf("forest_lookup_check: the folder does not say how many groups were run (lookup_groups.txt)\n"), 2;
    fclose(f);
  }
  for (int gi = 0; gi < expected; gi++) {
    const string name = string(argv[1]) + "/lookup_" + std::to_string(gi) + ".bin";
    FILE *f = fopen(name.c_str(), "rb");
    if (!f) {
      printf("the look-up of group %d: FAILS: it has no file\n", gi);
      groups++, failed++;
      continue;
    }
    File w;
    const u64 NC = get64(f);
    w.lid = (int)get64(f), w.n = (u32)get64(f);
    w.calls.resize(NC), w.tau.resize(NC), w.keys.resize(NC), w.padding.resize(NC);
    int kT = 0;
    for (u64 c = 0; c < NC; c++) {
      Call &k = w.calls[c];
      k.name = "call" + std::to_string(c), k.kappa = 5;
      k.n = (int)get64(f), k.k = (int)get64(f), k.fn = (int)get64(f), k.t0 = get64(f), w.padding[c] = get64(f);
      if (k.fn) w.tau[c].resize(1ull << k.k), getAll(f, w.tau[c]), k.tau = &w.tau[c];
      w.keys[c].resize(1ull << k.n), getAll(f, w.keys[c]);
      kT = std::max(kT, k.k);
    }
    w.bits.resize(NC << (kT + 5)), getAll(f, w.bits);
    for (vector<Ex> *v : {&w.sent, &w.mask, &w.walk})
      for (u32 i = 0; i < w.n; i++) v->push_back(getE(f));
    u64 proofBytes = 0;
    if (fread(&proofBytes, 8, 1, f) != 1) proofBytes = 0;
    w.proof.resize(proofBytes), getAll(f, w.proof);
    w.lm = (int)get64(f);
    for (int byte; (byte = fgetc(f)) != EOF;) w.vectors.push_back((u8)byte);
    fclose(f);
    if (FILE *f2 = fopen((name + ".values").c_str(), "rb")) {
      for (int byte; (byte = fgetc(f2)) != EOF;) w.values.push_back((u8)byte);
      fclose(f2);
    }
    string why;
    try {
      { // nothing but the salt and the scalars; a file without them fails here
        bool same = w.proof.size() == blk::DIGEST_BYTES + 8 * (size_t)CE * w.n;
        for (u32 i = 0; i < w.n && same; i++)
          for (int k = 0; k < CE; k++) {
            u64 x = 0;
            memcpy(&x, w.proof.data() + blk::DIGEST_BYTES + 8 * ((size_t)CE * i + k), 8);
            same = same && x == eC(w.sent[i], k);
          }
        if (!same) throw Refusal{"the proof's bytes are not the sent scalars"};
      }
      string falseClaims;
      why = verify(w, w.sent, 0, !refuse_all, &falseClaims);
      if (!refuse_all && why.empty() && !falseClaims.empty()) why = "the equation holds, but single claims are false: " + falseClaims;
      if (!refuse_all && why.empty()) {
        why = verify(w, w.sent, false, false, nullptr, 1);
        if (!why.empty()) why = "the two sides' coefficient vectors: " + why;
        else if (verify(w, w.sent, false, false, nullptr, 2).empty()) why = "the coefficient vectors are the same with two claims in each other's place: the comparison is blind";
      }
      // Mode 3 must list exactly the claims that mode 0 found false.
      string reported;
      const string report_fault = verify(w, w.sent, false, false, &reported, 3);
      const bool report_agrees = report_fault.empty() && reported == falseClaims;
      if (!refuse_all && why.empty() && !report_agrees) why = "the report of false claims: " + (report_fault.empty() ? "it names: " + reported : report_fault);
      if (refuse_all) {
        printf("the look-up of group %d (%u slots), a cheating prover's: %s%s; false: %s%s\n", gi, w.n, why.empty() ? "ACCEPTED" : "refused by ", why.c_str(),
               falseClaims.c_str(), report_agrees ? "" : (" FAILS: THE REPORT OF FALSE CLAIMS SAYS OTHERWISE: " + report_fault + " " + reported).c_str());
        groups++, failed += why.empty() || !report_agrees;
        continue;
      }
      for (u32 i = 0; i < 40 && why.empty(); i++) {
        vector<Ex> other = w.sent;
        const u32 at = (u32)((u64)i * (w.n - 1) / 39);
        other[at] = other[at] + eO();
        if (verify(w, other, 0, false).empty()) why = "a changed scalar is accepted: number " + std::to_string(at);
      }
      if (why.empty()) {
        w.keys[0][5] ^= 1;
        if (verify(w, w.sent, 0, false).empty()) why = "a changed key is accepted";
        w.keys[0][5] ^= 1;
        w.bits[40] ^= 1;
        if (why.empty() && verify(w, w.sent, 0, false).empty()) why = "a flipped bit of a multiplicity is accepted";
        w.bits[40] ^= 1;
        if (why.empty() && verify(w, w.sent, 1, false).empty()) why = "another number of padding keys is accepted";
      }
    } catch (const Refusal &r) {
      why = "refused: " + r.name;
      if (refuse_all) {
        // A fault of the program ("internal-..") or of this test does not count as a rejection.
        const bool own_fault = r.name.find("internal-") != string::npos || r.name.find("the proof's bytes") != string::npos;
        printf("the look-up of group %d (%u slots), a cheating prover's: %s%s\n", gi, w.n, own_fault ? "FAILS: a fault of the program or the test: " : "", why.c_str());
        groups++, failed += own_fault;
        continue;
      }
    }
    printf("the look-up of group %d (%u slots, on the chain): %s%s\n", gi, w.n, why.empty() ? "ok" : "FAILS: ", why.c_str());
    groups++, failed += !why.empty();
  }
  printf(refuse_all ? "forest_lookup_check: %d of %d cheating provers' files refused\n" : "forest_lookup_check: %d of %d groups passed\n", groups - failed, groups);
  return failed || !groups;
}
