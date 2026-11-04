// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the verifier's functions for product layers (readProductLevelRounds(), handDown(), layerObject(), the
// batching in `HFork`) on proofs made on a GPU. A product layer is a sumcheck with a hiding
// polynomial (PSC) for sum_x eq(rho, x) L[x] R[x] = C. prover/tests/psc_layer_test.cu writes, for
// mu = 1 .. 17, a file with the first layer's C, the true value of the claim that the second layer
// passes on, the mask row, the coefficients on it that the prover's walk (`hw::Walk`) adds up, and
// a proof of two consecutive layers (mu and mu + 1 variables). The functions read the proof on its
// hash chain (`Tr::one`), under the plan of prover/tests/psc_layer_plan.h. Checked: the proof ends
// where the plan ends; the equation that batches the claims holds on the mask row, and its
// coefficients are the walk's, which together catch a difference in challenges, batching
// coefficients or entry numbers; every copy of the proof with one scalar raised by 1 is rejected
// (`refuse` in identifiers). No other false proof is tried here. The test shows nothing about
// hiding: in these files the mask row is synthetic and the chain's random bytes are all 0.
//
// Build: prover/tests/psc_rows.sh checks <new directory>. By hand, from verifier/src:
//   g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
//   -I ../../blocks -I ../../prover ../tests/psc_layer_check.cpp
// Run: a.out <directory>
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "tests/psc_layer_plan.h" // of the prover's tree

static Ex take(FILE *f, bool &ok) {
  Ex x;
  for (int i = 0; i < CE; i++) {
    u64 c = 0;
    if (fread(&c, 8, 1, f) != 1 || c >= FIELD_P) ok = false, c = 0;
    eSet(x, i, c);
  }
  return x;
}
// Returns "" if the functions accept the proof, else what failed. walk: null, or the coefficients to compare this run's with.
static string verify(int mu, u32 n, const Ex &claim, const Ex &next, const vector<blk::u8> &bytes, const vector<Ex> &mask,
                     const vector<Ex> *walk) {
  HFork F;
  F.slots(n);
  blk::Plan plan(blk::setting_of_this_build());
  plan_psc_layer_test(plan, (unsigned)mu);
  blk::Binder binder(PSC_LAYER_TEST_LABEL, plan.finish());
  (void)binder.configuration(psc_layer_test_configuration());
  blk::Reader reader(bytes.data(), bytes.size());
  blk::VerifierSession session; // one chain, one verdict; it outlives the channel
  blk::VerifierRounds channel(binder, reader, session);
  Tr tr{channel};
  vector<Ex> rho = trPoint(tr, RoundName::ZERO, mu);
  u32 slot = 0;
  Aff C = affC(claim);
  vector<ProductLevel> layers;
  for (int m = mu; m <= mu + 1; m++) {
    layers.push_back(readProductLevelRounds(F, tr, slot, m, rho, C, orderDefault(m)));
    handDown(layers.back(), rho, C);
  }
  F.batchBy(trBatchOpen(tr));
  for (const ProductLevel &o : layers) layerObject(F, o);
  F.lin(C - affC(next));
  trBatchRows(tr, F.nClaim); // the number of claims that this run has listed, one power of the challenge each; the plan's last round states the same number
  if (!channel.finish()) return "the proof does not end where the plan ends";
  if (!session.accepted()) return "the chain did not accept the proof";
  vector<Ex> coef;
  Ex sum;
  F.settle(coef, sum, n);
  string why;
  for (u32 j = 0; j < n; j++) {
    sum = sum + coef[j] * mask[j];
    if (walk && why.empty() && !(coef[j] == (*walk)[j])) why = "the walk has another coefficient on mask entry " + std::to_string(j);
  }
  if (why.empty() && !isZ(sum)) why = "the equation does not hold";
  if (why.empty() && slot != n) why = "the number of slots";
  return why;
}
int main(int argc, char **argv) {
  if (argc != 2) return printf("usage: psc_layer_check <directory>\n"), 2;
  refuseThrows = true; // a function's refuse() then throws, and the rejection is caught below
  int cases = 0, failed = 0, changed = 0;
  for (int mu = 1; mu <= 17; mu++) {
    const string name = string(argv[1]) + "/layers_" + std::to_string(mu) + ".bin";
    FILE *f = fopen(name.c_str(), "rb");
    cases++;
    if (!f) {
      failed++, printf("%d and %d variables: FAILS: no file %s\n", mu, mu + 1, name.c_str());
      continue;
    }
    const u32 n = (u32)(layerSlots(mu) + layerSlots(mu + 1));
    const size_t head = blk::DIGEST_BYTES; // the chain's random bytes; the scalars follow them
    u64 sizes[3] = {0, 0, 0};
    bool ok = fread(sizes, 8, 3, f) == 3 && sizes[0] == (u64)mu && sizes[1] == n && sizes[2] == head + (u64)n * blk::E_BYTES;
    const Ex claim = take(f, ok), next = take(f, ok);
    vector<Ex> mask(n), walk(n);
    for (Ex &x : mask) x = take(f, ok);
    for (Ex &x : walk) x = take(f, ok);
    vector<blk::u8> bytes(ok ? sizes[2] : 0);
    if (ok && fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) ok = false;
    fclose(f);
    string why = ok ? "" : "the file is short, is of another shape or holds a word that is no field element";
    try {
      if (why.empty()) why = verify(mu, n, claim, next, bytes, mask, &walk);
    } catch (const Refusal &r) {
      why = "rejected: " + r.name;
    }
    for (u32 j = 0; j < n && why.empty(); j++) {
      vector<blk::u8> other = bytes;
      blk::E x;
      if (!blk::E_from_bytes(other.data() + head + j * blk::E_BYTES, x)) why = "scalar " + std::to_string(j) + " is not in canonical form";
      blk::E_to_bytes(x + blk::E::one(), other.data() + head + j * blk::E_BYTES);
      changed++;
      try {
        if (why.empty() && verify(mu, n, claim, next, other, mask, nullptr).empty()) why = "a changed scalar is accepted: number " + std::to_string(j);
      } catch (const Refusal &r) { // a rejection, as wanted, unless its name says that the program itself is at fault
        if (r.name.rfind("internal", 0) == 0) why = "with scalar " + std::to_string(j) + " changed: " + r.name;
      }
    }
    failed += !why.empty();
    printf("%d and %d variables: %s%s\n", mu, mu + 1, why.empty() ? "ok" : "FAILS: ", why.c_str());
  }
  printf("psc_layer_check: %d of %d cases passed (%d scalars changed one at a time)\n", cases - failed, cases, changed);
  return failed;
}
