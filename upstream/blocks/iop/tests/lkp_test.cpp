// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests of LKP, the lookup argument of blocks/iop/lkp.h: every key of a vector occurs in a public
// table. Most runs (prove() and verify() in blocks/iop/tests/lkp_world.h) are complete proofs, at
// a 32-bit target to keep them short: one oracle message (encoded rows, sent as Merkle roots)
// holds the keys, the bits of the multiplicities (how many keys equal each table row) and a mask
// row of random values; LKP runs; the caller turns LKP's claim on the key vector into a linear
// claim on the sent keys; the proximity test, which takes in all linear claims, ends the run.
// A linear claim (a linear form of what was sent has a public value) is a `Row` in identifiers and
// a "row" in the output, which also says "row" for a row of the lookup table:
// a table row in these comments. A test fixture (`World`) is one set of LKP's public inputs.
//
// This test checks honest proofs, proofs with one bit flipped (at up to 300 offsets of one
// proof per fixture), and the faulty public inputs and callers on which the process must abort.
// No malicious prover runs in this test.
// Each counted run must be rejected ("refused" in the output).
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "iop/tests/lkp_world.h"

using namespace blk;
using namespace lkp_world;

// LKP's terms, defined in blocks/iop/lkp.h:
//   the tie    the sumchecks for Z = a * prod_j G_j^(2^j): the keys' root Z is the product of
//              alpha - key for a challenge alpha, G_j is plane j's root, a the padding's factor
//   level      Z and each G_j are roots of product trees, proved by one sumcheck per level
//   plane j    bit j of every multiplicity; of the 2^mu plane slots only the live ones are sent
//   padding    key indices ("places" in the output) declared to hold a named table row; the
//              multiplicities do not count them
//   part       one of several key vectors under one alpha and one table
//   function   one of several lookup tables joined in one table, with its own keys
//   the shift  moves function f's keys and table rows by f * theta, to keep the functions apart
//   tree       the keys whose indices agree in their low g bits; a function owns a block of trees
// The fixtures of `worlds()`, which this file takes by index:
//   0  one range table
//   1  two functions of different sizes under the shift; a tree of padding
//   2  many keys; the last key level's rounds in the order of lkp_range_order()
//   3  padding inside a tree; a written-out table stored twice
//   4  six live planes of eight slots
//   5  the keys in three parts of different sizes
//   6  the table part's number in the low bit of a table row's number; a short table repeated

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// `FALSE_ROWS`: the names of the verifier's claims that must be false of what the prover sent
// (false_rows() in lkp_world.h): none, because the prover here is honest. The proofs that this file
// expects to be rejected are honest ones, with one bit flipped or checked under other padding.
// Malicious provers, one to a build, belong to a test that is not part of this repository. Across
// its builds each type of claim is the only false one in some run, except `LKP number of keys` and
// `LKP parts, step k`: no build isolates either, and none makes `LKP parts, step 0` false.
static const char *const FALSE_ROWS = "";

// Public inputs under which LKP would show nothing about some keys: all key indices of the lookup,
// of a part or of a function's trees are declared as padding.
[[maybe_unused]] static std::vector<World> only_padding_worlds() {
  const std::vector<World> w = worlds();
  std::vector<World> v;
  v.push_back(w[0]), v.back().name = "one run of padding over all keys", v.back().fixed.padding = {{0, 0, 0, 32, 0}};
  v.push_back(w[0]), v.back().name = "the one key block is padding alone", v.back().fixed.keys = {{0, 0, -1, 0}};
  v.push_back(w[5]), v.back().name = "the second of three parts is all padding";
  v.back().fixed.padding.clear();
  v.back().fixed.padding.push_back({1, 0, 0, u64(1) << w[5].fixed.parts[1], 0});
  v.push_back(w[1]), v.back().name = "the second function's tree is all padding", v.back().fixed.padding = {{0, 2, 2, 8, 7}};
  return v;
}

// Runs f in a child process. Returns true if SIGABRT ended the child and stderr contains `text`.
static bool aborts_saying(const char *text, const std::function<void()> &f) {
  int fd[2];
  if (pipe(fd)) return false;
  std::fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    dup2(fd[1], 2), close(fd[0]), close(fd[1]);
    f();
    _exit(0);
  }
  close(fd[1]);
  std::string said;
  char buffer[512];
  for (ssize_t n; (n = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(fd[0]);
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  const bool as_foretold = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && said.find(text) != std::string::npos;
  if (!as_foretold) std::printf("  (the child %s; its last words: %s)\n", WIFSIGNALED(status) ? "was ended by a signal" : "ended by itself", said.empty() ? "none" : said.substr(said.size() > 300 ? said.size() - 300 : 0).c_str());
  return as_foretold;
}
// Runs the honest prover on fixture w with the public inputs altered by `change`. The process must
// abort with a message that contains `text`: in the Binder (blocks/chain/public.h), which writes
// public inputs into the first query, if lkp_fault() reports the fault, otherwise in LKP.
// second_unsampled: the prover also sends a second oracle message, of which no out-of-domain
// sample (blocks/iop/send.h) is taken.
static void stops(const char *text, World w, const std::function<void(LkpPublic &)> &change, bool second_unsampled = false) {
  const LookupRun run = size_run(w);
  LkpPublic changed = w.fixed;
  change(changed);
  handed_in = &changed;
  if (!aborts_saying(text, [&] { Seen s; prove(run, 40, s, second_unsampled); })) failures++, std::printf("FAILED: LKP did not stop with the words: %s\n", text);
  handed_in = nullptr;
}

int main() {
  size_t runs = 0, refused = 0;
  for (const World &w : worlds()) {
    const LookupRun run = size_run(w);
    if (run.plan->refused()) {
      std::printf("FAILED: %s: no such parameters: %s\n", w.name, run.plan->why_not().c_str());
      return 1;
    }
    Seen earlier;
    for (u64 key = 40; key < 42 + (w.fixed.parts.size() > 1 ? 1 : 0); key++) {
      Seen p, v;
      const std::vector<u8> proof = prove(run, key, p);
      const char *why = "";
      const bool accepted = verify(run, proof, v, &why);
      std::string foretold = FALSE_ROWS;
      if (false_rows(p, v) != foretold) std::printf("  %s: false rows: %s\n", w.name, false_rows(p, v).c_str());
      CHECK(v.rows.size() == rows_left(w) && false_rows(p, v) == foretold, "other rows are false on what was sent than this prover should leave");
      CHECK(accepted, "an honest proof is refused");
      if (!accepted) std::printf("  %s: %s\n", w.name, why);
      CHECK(same_rows(p, v), "prover and verifier leave LKP with other rows");
      CHECK(p.masks_used == lkp_masks(w.fixed), "LKP uses another number of mask entries than it announces");
      if (key == 41) {
        bool points_differ = true;
        for (size_t i = 0; i < p.key_point.size(); i++) points_differ = points_differ && p.key_point[i] != earlier.key_point[i];
        CHECK(p.alpha != earlier.alpha && (!w.fixed.shifted || p.theta != earlier.theta) && points_differ, "two runs with other random bytes have a challenge in common");
      }
      earlier = p;
      if (key == 40)
        for (size_t i = 0; i < proof.size(); i += proof.size() / 300 + 1) {
          std::vector<u8> changed = proof;
          changed[i] ^= (u8)(1u << (i % 8));
          Seen s;
          runs++;
          if (!verify(run, changed, s)) refused++;
          else std::printf("ACCEPTED a proof with byte %zu changed: %s\n", i, w.name);
        }
    }
  }
  {
    const std::vector<World> w = worlds();
    stops("two tables overlap", w[3], [](LkpPublic &p) { p.tables[1].first = 0; });
    stops("the table parts are not numbered", w[6], [](LkpPublic &p) { std::swap(p.tables[0], p.tables[1]); });
    stops("do not cover all rows", w[6], [](LkpPublic &p) { p.tables.pop_back(); });
    stops("do not cover all rows", w[3], [](LkpPublic &p) { p.tables.pop_back(); });
    stops("two blocks of tree numbers overlap", w[1], [](LkpPublic &p) { p.keys[1].first = 0; });
    stops("do not cover all tree numbers", w[1], [](LkpPublic &p) { p.keys.pop_back(); });
    stops("names a variable twice", w[2], [](LkpPublic &p) { p.last_key_level_order[1] = p.last_key_level_order[0]; });
    stops("names a variable twice", w[2], [](LkpPublic &p) { p.last_key_level_order[1] = 1000; });
    stops("names a variable twice", w[5], [](LkpPublic &p) { p.last_key_level_order = {2}; });  // the part of eight keys has the variables 0, 1
    // lkp_cubes() covers a run of padding exactly: each index of the run is in one cube (`LkpCube`:
    // the indices i with (i & mask) == value), no other index in any. Tried for all runs in a part
    // of 2^6 keys.
    for (unsigned step = 0; step < 6; step++)
      for (u64 first = 0; first < 64; first++)
        for (u64 length = 1; first + ((length - 1) << step) < 64; length++) {
          LkpPublic p = w[0].fixed;
          p.parts = {6}, p.padding = {{0, first, step, length, 0}};
          const std::vector<LkpCube> cubes = lkp_cubes(p.padding[0], 6);
          bool same = cubes.size() <= 12;
          for (u64 i = 0; i < 64; i++) {
            unsigned in = 0;
            for (const LkpCube &c : cubes) in += (i & c.mask) == c.value;
            same = same && in == (lkp_is_padding(p, 0, i) ? 1u : 0u) && lkp_is_padding(p, 0, i) == (i >= first && (i - first) % (u64(1) << step) == 0 && (i - first) >> step < length);
          }
          CHECK(same && lkp_padding_count(p) == length, "a run's cubes, its places and its count differ");
        }
    {  // LKP's weights on the keys: eq(point, i) at a real index, 0 at padding; a function's share
      const LookupRun run = size_run(w[1]);
      Seen p;
      prove(run, 40, p);
      const Row &k = p.rows.back();
      const LkpPublic &f = w[1].fixed;
      bool right = k.forms.size() == 1;
      for (u64 i = 0; right && i < 32; i++) {
        E eq = E::one();
        for (unsigned t = 0; t < 5; t++) eq = eq * (((i >> t) & 1) ? p.key_point[t] : E::one() - p.key_point[t]);
        const bool real = lkp_block_of(f, i).function >= 0 && !lkp_is_padding(f, 0, i);
        right = k.forms[0].c.at(2 * i) == (real ? eq : E::zero());
      }
      CHECK(right, "the weights that LKP hands on are not the equality vector on the real places and 0 on the padding");
      Form all;  // the same weights from the definitions; `own`: function 1's share (keys 2 + 4 x)
      all.add(Tensor::eq(p.key_point.data(), 5));
      all.add(lkp_cube_tensor(lkp_cube(f, f.keys[2]), p.key_point, -E::one()));
      for (const LkpCube &c : lkp_cubes(f.padding[0], 5)) all.add(lkp_cube_tensor(c, p.key_point, -E::one()));
      const Form own = lkp_weights_of_block(f, all, 1);
      bool cut = own.fits(3);
      for (u64 x = 0; cut && x < 8; x++) cut = own.at(x) == all.at(2 + 4 * x);
      CHECK(cut, "a function's own share of the weights differs from the weights at its key numbers");
    }
    {  // The verifier's public inputs declare other padding than the prover's: another first query.
      const LookupRun run = size_run(w[3]);
      Seen p, v;
      const std::vector<u8> proof = prove(run, 40, p);
      World other = w[3];
      other.fixed.padding[1] = {0, 30, 0, 2, 7};
      LookupRun others = run;
      others.world = other;
      CHECK(!verify(others, proof, v), "a verifier that is bound to other places of padding accepts");
    }
    stops("does not lie in its part", w[3], [](LkpPublic &p) { p.padding = {{0, 29, 0, 4, 7}}; });
    stops("does not lie in its part", w[3], [](LkpPublic &p) { p.padding = {{0, 29, 0, 0, 7}}; });
    stops("does not lie in its part", w[3], [](LkpPublic &p) { p.padding = {{0, 1, 0, u64(1) << 63, 7}}; });
    stops("does not lie in its part", w[3], [](LkpPublic &p) { p.padding = {{0, 1, 5, 1, 7}}; });
    stops("a part that is not there", w[3], [](LkpPublic &p) { p.padding[0].part = 1; });
    stops("is no row of the table", w[3], [](LkpPublic &p) { p.padding[1].row = 8; });
    stops("is no row of the table", w[1], [](LkpPublic &p) { p.keys[2].padding_row = 8; });
    stops("not in increasing order", w[3], [](LkpPublic &p) { std::swap(p.padding[0], p.padding[1]); });
    stops("not in increasing order", w[3], [](LkpPublic &p) { p.padding[1] = p.padding[0]; });
    stops("two runs of padding meet", w[3], [](LkpPublic &p) { p.padding = {{0, 5, 0, 25, 3}, {0, 29, 0, 3, 7}}; });
    stops("two runs of padding meet", w[3], [](LkpPublic &p) { p.padding = {{0, 1, 2, 8, 3}, {0, 28, 0, 2, 7}}; });   // 1, 5, .., 29 and 28, 29
    stops("two runs of padding meet", w[3], [](LkpPublic &p) { p.padding = {{0, 4, 0, 9, 3}, {0, 8, 3, 2, 7}}; });    // 4 .. 12 and 8, 16
    stops("leaves its tree", w[1], [](LkpPublic &p) { p.padding[0].log_step = 1; });
    stops("lies in a tree of padding alone", w[1], [](LkpPublic &p) { p.padding[0].first = 27; });
    stops("too few bits for the number of keys", w[0], [](LkpPublic &p) { p.parts = {8}; });
    stops("no part of keys", w[0], [](LkpPublic &p) { p.parts.clear(); });
    stops("too few bits for the number of keys", w[4], [](LkpPublic &p) { p.live = 5; });
    stops("too few bits for the number of keys", w[5], [](LkpPublic &p) { p.parts = {7, 7}; });
    stops("as many as there are or more", w[4], [](LkpPublic &p) { p.live = 9; });
    stops("as many as there are or more", w[4], [](LkpPublic &p) { p.live = 8; });  // 0 means all
    stops("keys in several parts are served for one function", w[5], [](LkpPublic &p) { p.shifted = true; });
    stops("several functions share a table without the shift", w[1], [](LkpPublic &p) { p.shifted = false; });
    stops("could wrap around in the field", w[0], [](LkpPublic &p) { p.d = 0xFFFFFFF8u; });
    // lkp_fault() called directly: the calls of stops() above leave some of its conditions, or
    // halves of conditions, untested. The direct calls were added because thirteen checks, or
    // halves of checks, could be taken out of blocks/iop/lkp.h and every test still passed.
    auto faults = [&](const char *text, const World &world, const std::function<void(LkpPublic &)> &change) {
      LkpPublic changed = world.fixed;
      CHECK(lkp_fault(changed) == nullptr, "a world of this test is unfit");
      change(changed);
      const char *fault = lkp_fault(changed);
      if (!fault || !std::strstr(fault, text)) failures++, std::printf("FAILED: lkp_fault says '%s', not: %s\n", fault ? fault : "(fit)", text);
    };
    faults("it is not said from which oracle messages", w[0], [](LkpPublic &p) { p.read_from.clear(); });
    faults("the plane slots are not 2, 4, .. or 64", w[0], [](LkpPublic &p) { p.mu = 0; });
    faults("the plane slots are not 2, 4, .. or 64", w[0], [](LkpPublic &p) { p.mu = 7; });
    faults("fewer than two of them are live", w[0], [](LkpPublic &p) { p.mu = 1, p.live = 1; });
    faults("could wrap around in the field", w[0], [](LkpPublic &p) { p.mu = 6; });  // 64 planes
    faults("could wrap around in the field", w[0], [](LkpPublic &p) { p.d = F_BITS - 8; });  // d alone is below the field's bits; d + 8 planes is not
    faults("could wrap around in the field", w[0], [](LkpPublic &p) { p.d = F_BITS; });
    faults("no key in a part", w[0], [](LkpPublic &p) { p.parts = {0}; });
    faults("too few bits for the number of keys", w[5], [](LkpPublic &p) { p.parts = {4, 8, 5}; });  // one part alone has as many bits as there are planes
    faults("keys in several parts are served for one function", w[5], [](LkpPublic &p) { p.g = 1; });
    faults("a tree's number has more bits than a key's", w[0], [](LkpPublic &p) { p.g = 6; });
    faults("no aligned block of tree numbers", w[1], [](LkpPublic &p) { p.keys[0].first = 1; });
    faults("no aligned block of tree numbers", w[1], [](LkpPublic &p) { p.keys[0].log_trees = 3; });
    faults("no aligned block of tree numbers", w[1], [](LkpPublic &p) { p.keys[1].first = 6; });
    faults("several functions share a table without the shift", w[1], [](LkpPublic &p) { p.shifted = false, p.keys[1].function = 0; });  // the tables still name function 1
    faults("several functions share a table without the shift", w[1], [](LkpPublic &p) { p.shifted = false, p.tables[1].function = 0; });  // the keys do
    faults("a table part's number has more bits than a row's", w[6], [](LkpPublic &p) { p.h = 4; });
    faults("a part's table has more rows than a part", w[6], [](LkpPublic &p) { p.tables[0].log_rows = 3; });
    faults("has not the announced number of rows", w[6], [](LkpPublic &p) { p.tables[1].rows.pop_back(); });
    faults("has not the announced number of rows", w[3], [](LkpPublic &p) { p.tables[0].rows.pop_back(); });
    faults("has not the announced number of rows", w[3], [](LkpPublic &p) { p.tables[0].rows.clear(), p.tables[0].rows.push_back(e_of(7)); });
    faults("no aligned block of the rows", w[3], [](LkpPublic &p) { p.tables[1].first = 2; });
    faults("no aligned block of the rows", w[3], [](LkpPublic &p) { p.tables[0].log_rows = 4; });
    faults("no aligned block of the rows", w[3], [](LkpPublic &p) { p.tables[1].first = 8; });
    faults("more than 4096 runs of padding", w[2], [](LkpPublic &p) { p.padding.assign(4097, LkpPadding{0, 0, 0, 1, 0}); });
    for (const World &void_world : only_padding_worlds()) {  // all padding: lookup, part, function
      const char *fault = lkp_fault(void_world.fixed);
      if (!fault || !std::strstr(fault, "no key is left that is no padding")) failures++, std::printf("FAILED: lkp_fault says '%s' of: %s\n", fault ? fault : "(fit)", void_world.name);
    }
    faults("no key is left that is no padding", w[0], [](LkpPublic &p) { p.padding = {{0, 0, 1, 16, 0}, {0, 1, 1, 16, 0}}; });  // two runs that together cover the part
    {  // one index short of that: no fault
      LkpPublic nearly = w[0].fixed;
      nearly.padding = {{0, 0, 1, 16, 0}, {0, 1, 1, 15, 0}};
      CHECK(lkp_fault(nearly) == nullptr, "a look-up with one real key left was refused");
    }
    stops("has no answered sample", w[0], [](LkpPublic &p) { p.read_from = {0, 1}; }, true);
    key_masks_from_second = true;
    stops("gives mask entries has no answered sample", w[0], [](LkpPublic &) {}, true);
    key_masks_from_second = false;
    const LookupRun run = size_run(w[0]);
    Seen p;
    const std::vector<u8> proof = prove(run, 40, p);
    CHECK(aborts_saying("the claim on the key vector was dropped", [&] { Seen s; verify(run, proof, s, nullptr, true); }), "a caller dropped LKP's claim on the keys and nothing stopped it");
  }
  {  // `small_world` in a run that ends after LKP (`block_run`), as in blocks/tests/hiding/:
     // the prover finishes for each witness tried, and the proof's bytes depend on the witness and
     // on the private randomness. No verifier runs: without a proximity test its verdict would mean
     // nothing. The prover's own claims are true of what it sent.
    const World w = small_world();
    CHECK(lkp_fault(w.fixed) == nullptr, "the small world is unfit");
    const LookupRun run = block_run(w, 7, 7);
    std::vector<std::vector<u8>> proofs;
    for (u64 witness : {u64(0), u64(1), u64(1000), u64(1001), u64(1002)}) {
      ToyCoins coins(500);
      Writer proof;
      Seen seen;
      prove(run, lkp_witness(w.fixed, witness), coins, proof, seen);
      proofs.push_back(proof.bytes());
      CHECK(!proofs.back().empty() && proofs.back().size() == proofs[0].size(), "the small world: a witness moves the proof's length");
      for (size_t i = 0; i + 1 < proofs.size(); i++) CHECK(proofs[i] != proofs.back(), "the small world: two witnesses give one proof under the same coins");
      CHECK(false_rows(seen, seen).empty(), "the small world: a row of the honest prover's is false on what it has sent");
    }
    Seen one, other;
    const std::vector<u8> a = prove(run, 501, one), b = prove(run, 1501, other);  // same witness
    CHECK(a != b && a.size() == b.size(), "the small world: the coins move nothing");
    std::printf("  %s: %zu bytes, %zu mask entries\n", w.name, a.size(), lkp_masks(w.fixed));
    // Keys and multiplicities in two oracle messages, both sent and sampled, again in a run that
    // ends after LKP. With both in LkpPublic::read_from the prover finishes and its claims are
    // true; with the second missing, lkp_run() aborts at its first claim on the multiplicities.
    World two = w;
    two.fixed.read_from = {0, 1};
    {
      const LookupRun both = block_run(two, 7, 7, 2, true);
      Seen seen;
      const std::vector<u8> proof = prove(both, 40, seen);
      CHECK(!proof.empty() && proof.size() > a.size() && false_rows(seen, seen).empty(), "two messages: the run does not go through, or a row of the honest prover's is false");
    }
    two.fixed.read_from = {0};
    const LookupRun one_declared = block_run(two, 7, 7, 2, true);
    CHECK(aborts_saying("is not among those declared as read", [&] { Seen s; prove(one_declared, 40, s); }), "LKP made a row on a message that is not declared as read");
  }
  CHECK(refused == runs && runs > 0, "something that had to be refused was accepted");
  std::printf("lkp_test: %s (%zu of %zu refused)\n", failures ? "FAILED" : "ok", refused, runs);
  return failures ? 1 : 0;
}
