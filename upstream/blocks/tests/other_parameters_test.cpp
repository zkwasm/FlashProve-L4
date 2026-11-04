// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests that a verifier rejects (printed: "refused") an honest proof made under other parameters,
// such as a lower target (the bits that every round's error must reach) or less proof of work
// (`work`). Runs both toy protocols through `WholeRun` (blocks/tests/whole_runs.h): "product"
// (blocks/toy/product.h) and "links" (blocks/toy/segments.h: a chain of segments, `links` in
// identifiers). A side is a prover and a verifier under one set of parameters. A crossing gives one
// side's proof to the other side's verifier, in a child process: the verifier must return false,
// not end the process.
// 1. Each entry of variations() makes a small change to the base, small_choice(). Each side must
//    accept its own proof, so that no rejection comes from a broken set-up; both crossings must be
//    rejected, unless the protocol stays the same (see main()).
// 2. Each toy protocol's proof is given to the other's verifier.
// 3. crossings_of_one_run(): the two sides differ only in the digest of their public file.
// Any line "FAILED: ..." or "ACCEPTED ..." makes main() return 1. Toy size only, with targets of 24
// to 40 bits; the verifier program cp_verify is not run here.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tests/whole_runs.h"

using namespace blk;

namespace {

struct Variation {
  const char *name;
  bool counting_only; // may leave the protocol as it is
  bool links_only;    // applies to the toy protocol "links" only
  std::function<void(RunChoice &)> change;
};

std::vector<Variation> variations() {
  auto work = [](ErrorKind k, unsigned bits) { return [k, bits](RunChoice &c) { c.set.set_work(k, Scope::REQUEST, bits); }; };
  return {
      {"target 40 for 32", false, false, [](RunChoice &c) { c.set.target = 40; }},
      {"target 24 for 32", false, false, [](RunChoice &c) { c.set.target = 24; }},
      {"work on positions 7 for 8", false, false, work(ErrorKind::QUERY, 7)},
      {"work on positions 9 for 8", false, false, work(ErrorKind::QUERY, 9)},
      {"no work on positions", false, false, work(ErrorKind::QUERY, 0)},
      {"work 4 on algebraic rounds", false, false, work(ErrorKind::FIELD, 4)},
      {"work 4 on the seeds of samples", false, false, work(ErrorKind::SAMPLE, 4)},
      {"work 4 on closeness rounds", false, false, work(ErrorKind::CLOSENESS, 4)},
      {"salts eight bytes longer", false, false, [](RunChoice &c) { c.set.salt_bytes += 8; }},
      {"salts sixteen bytes longer", false, false, [](RunChoice &c) { c.set.salt_bytes += 16; }},
      {"nonces of 7 bytes for 8", false, false, [](RunChoice &c) { c.set.nonce_bytes = 7; }},
      {"two answers to a sample", false, false, [](RunChoice &c) { c.set.samples = 2; }},
      {"vectors twice as long", false, false, [](RunChoice &c) { c.log_len += 1; }},
      {"first code half as long (twice the rate)", false, false, [](RunChoice &c) { c.log_n -= 1; }},
      {"first mask code twice as long", false, false, [](RunChoice &c) { c.mask_log_n += 1; }},
      {"first code and first mask code twice as long (half the rates)", false, false, [](RunChoice &c) { c.log_n += 1, c.mask_log_n += 1; }},
      {"second stage's code twice as long", false, false, [](RunChoice &c) { c.further[0].log_n += 1; }},
      {"second stage's code half as long", false, false, [](RunChoice &c) { c.further[0].log_n -= 1; }},
      {"second stage's mask code twice as long", false, false, [](RunChoice &c) { c.further[0].mask_log_n += 1; }},
      {"second stage's mask code half as long", false, false, [](RunChoice &c) { c.further[0].mask_log_n -= 1; }},
      {"second stage cuts the message in four", false, false, [](RunChoice &c) { c.further[0].k = 2; }},
      {"last stage's code twice as long", false, false, [](RunChoice &c) { c.further[1].log_n += 1; }},
      {"last stage's mask code half as long", false, false, [](RunChoice &c) { c.further[1].mask_log_n -= 1; }},
      {"one stage fewer", false, false, [](RunChoice &c) { c.further.erase(c.further.begin()); }},
      {"one stage more", false, false, [](RunChoice &c) { c.further.insert(c.further.begin(), StageChoice{1, 1, 12, 10}); }},
      {"two links for three", false, true, [](RunChoice &c) { c.links = 2; }},
      {"four links for three", false, true, [](RunChoice &c) { c.links = 4; }},
      {"two requests share a session", true, false, [](RunChoice &c) { c.set.instances = 2; }},
  };
}

// Returns 0 if rejected, 1 if accepted, -1 if a signal ended the child process.
int judged_in_child(const WholeRun &verifier, const std::vector<u8> &proof) {
  return in_child([&] { return verifier.verify(proof) ? 1 : 0; });
}

// Crossings between two sides that differ only in their public file, the text from which
// verify_from_text() plans a run. another_text() appends a byte to one side's file that changes no
// number of the run, so only the file's digest can cause a rejection. The crossings go through
// verify(), which takes the run as this test built it: verify_from_text() accepts a public file
// only as toy_text() (blocks/toy/product.h) writes it.
// The digest is hashed in two places. It is an item of the parameters in the first query, the
// input with which the hash chain (the Fiat-Shamir transcript) starts, so the challenges differ.
// The hash of every inner vertex of the Merkle trees starts from it, so no opening leads to the
// tree's root. Either is enough, so this test cannot detect the loss of one, and no test in this
// repository takes the digest out of one place to see the crossings still rejected.
// Returns the number of crossings; adds failures to `faults`.
size_t crossings_of_one_run(const RunChoice &choice, size_t &faults) {
  size_t played = 0;
  for (size_t which = 0; which < WHOLE_RUNS; which++) {
    const std::unique_ptr<WholeRun> base = whole_run(which, choice), other = whole_run(which, choice);
    other->another_text();
    const std::vector<u8> base_proof = base->prove(61), other_proof = other->prove(61);
    const char *why = nullptr;
    if (!base->verify(base_proof, &why) || !other->verify(other_proof, &why)) {
      std::printf("FAILED: %s: an honest proof is refused by its own verifier: %s\n", base->name(), why);
      faults++;
      continue;
    }
    // verify_from_text() must accept the exact text, and reject the longer one with either proof.
    if (in_child([&] { return base->verify_from_text("", base_proof) ? 1 : 0; }) != 1) {
      std::printf("FAILED: %s: the verifier that takes the public file's text refuses the honest proof\n", base->name());
      faults++;
    }
    for (const std::vector<u8> *proof : {&base_proof, &other_proof}) {
      const int verdict = in_child([&] { return base->verify_from_text("\n", *proof) ? 1 : 0; });
      if (verdict == 0) continue;
      std::printf("%s: %s: the public file's text with one byte more, before the verifier that takes the text\n",
                  verdict == 1 ? "ACCEPTED A CHEATING PROVER: A TEXT IN A SECOND SPELLING" : "FAILED: THE VERIFIER ENDED THE PROCESS", base->name());
      faults++;
    }
    for (const bool forth : {true, false}) {
      const int verdict = judged_in_child(forth ? *other : *base, forth ? base_proof : other_proof);
      played++;
      if (verdict == 0) continue;
      std::printf("%s: %s: %s\n", verdict == 1 ? "ACCEPTED A CHEATING PROVER: A PROOF MADE UNDER ANOTHER PUBLIC FILE" : "FAILED: THE VERIFIER ENDED THE PROCESS", base->name(),
                  forth ? "the true file's proof before the other's verifier" : "the other file's proof before the true verifier");
      faults++;
    }
  }
  return played;
}

} // namespace

int main() {
  size_t crossings = 0, one_protocol = 0, faults = 0;
  std::string same_names;
  const RunChoice base_choice = small_choice();
  std::vector<std::vector<u8>> base_proofs;
  for (size_t which = 0; which < WHOLE_RUNS; which++) {
    const std::unique_ptr<WholeRun> base = whole_run(which, base_choice);
    std::string no;
    if (!base->planned(no)) {
      std::printf("FAILED: %s: no such parameters: %s\n", base->name(), no.c_str());
      return 1;
    }
    const std::vector<u8> base_proof = base->prove(60);
    base_proofs.push_back(base_proof);
    const char *why = nullptr;
    if (!base->verify(base_proof, &why)) {
      std::printf("FAILED: %s: the honest proof is refused: %s\n", base->name(), why);
      return 1;
    }
    for (const Variation &v : variations()) {
      if (v.links_only && which != 1) continue;
      RunChoice choice = base_choice;
      v.change(choice);
      const std::unique_ptr<WholeRun> other = whole_run(which, choice);
      if (!other->planned(no)) {
        std::printf("FAILED: %s: %s: no such parameters, so the variation shows nothing: %s\n", base->name(), v.name, no.c_str());
        faults++;
        continue;
      }
      const std::vector<u8> other_proof = other->prove(60);
      if (!other->verify(other_proof, &why)) {
        std::printf("FAILED: %s: %s: the variation's own honest proof is refused: %s\n", base->name(), v.name, why);
        faults++;
        continue;
      }
      // bound() is the parameters' digest in the first query. Only the three variations that change
      // how errors are counted (`counting_only`) may leave it unchanged. The protocol is then the
      // same: no crossing is made, and prove(60), whose key seeds the witness and the private
      // randomness, must give the same proof.
      if (base->bound() == other->bound()) {
        if (!v.counting_only || other_proof != base_proof) {
          std::printf("FAILED: %s: %s: the first query of the hash chain does not tell the two sets of parameters apart%s\n", base->name(), v.name,
                      other_proof != base_proof ? ", and their proofs differ" : "");
          faults++;
        } else {
          one_protocol++;
          if (which == 0) same_names += std::string(same_names.empty() ? "" : "; ") + v.name;
        }
        continue;
      }
      const struct {
        const WholeRun &verifier;
        const std::vector<u8> &proof;
        const char *what;
      } cross[2] = {{*other, base_proof, "the base's proof before the variation's verifier"}, {*base, other_proof, "the variation's proof before the base's verifier"}};
      for (const auto &x : cross) {
        const int verdict = judged_in_child(x.verifier, x.proof);
        crossings++;
        if (verdict == 0) continue;
        std::printf("%s: %s: %s: %s\n", verdict == 1 ? "ACCEPTED A PROOF MADE AT OTHER PARAMETERS" : "FAILED: THE VERIFIER ENDED THE PROCESS", base->name(), v.name, x.what);
        faults++;
      }
    }
  }
  // `1 - which` needs WHOLE_RUNS == 2; nothing checks that.
  for (size_t which = 0; which < WHOLE_RUNS; which++) {
    const std::unique_ptr<WholeRun> verifier = whole_run(which, base_choice);
    const int verdict = judged_in_child(*verifier, base_proofs[1 - which]);
    crossings++;
    if (verdict == 0) continue;
    std::printf("%s: the other whole run's proof before the verifier of %s\n", verdict == 1 ? "ACCEPTED A PROOF OF ANOTHER PROTOCOL" : "FAILED: THE VERIFIER ENDED THE PROCESS", verifier->name());
    faults++;
  }
  const size_t silent = crossings_of_one_run(base_choice, faults);
  if (faults) return 1;
  std::printf("other parameters: ok: %zu crossings refused, %zu of them between public files from which the same run follows; %zu times a way of counting moved no number of the protocol (%s)\n",
              crossings + silent, silent, one_protocol, same_names.empty() ? "none" : same_names.c_str());
  return 0;
}
