// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU half of a two-program test of the lookup forest (prover/hiding/forest.cuh), which proves all
// lookup calls of a group as one lookup. A lookup call shows that every one of its keys occurs in a
// public table. For each of ten groups (listed in main()) this program generates synthetic keys and
// a mask row (one mask entry per sent scalar), counts the multiplicities, runs the prover, and
// batches the lookup's linear claims (prover/hiding/walk_forest.cuh), equations on the prover's
// data. For the i-th group that it runs, from 0, it writes `lookup_<i>.bin` (layout: where main()
// writes it) and `lookup_<i>.bin.values` (the data that the claims are about, for the verifier's
// report of false claims); at the end, `lookup_groups.txt` (the number of groups run). A group that
// fails a check here gets no file; the checker then counts it as failed.
// The checker, verifier/tests/forest_lookup_check.cpp, runs the verifier's functions on the files:
// it alone decides whether the scalars are a valid proof. This program checks only the prover's
// failure bits (prover/common/slot_flags.h), that an alteration took effect, that the rounds run
// are the planned ones, and what the comment at main() lists by build.
// At build time: four structs that the forest's kernels take by value have no member without an
// initial value.
// Build and run, from prover/: ./build.sh forest_lookup_test; build/forest_lookup_test <folder>
// [key | multiplicity | crossed | repeated]. One GPU, no weights. <folder> is a new, empty folder
// of your own: the test removes files named `lookup_<i>.bin`, `lookup_<i>.bin.values` and
// `lookup_groups.txt` in it. In <folder>, which must exist, files and links that have the three
// names above are replaced. Nothing is written outside it.
#include "hiding/product_test.cuh"

#include "hiding/forest.cuh"

#include "hiding/walk_forest.cuh"
#include "hiding/door_vectors.cuh"
#include "hiding/door_values.cuh"

#include "chain/public.h"
#include "numbers/blocks.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

// Kernels take these four structs by value, so a launch copies every member to the GPU, unused list
// entries included. A declaration compiles only if every member of its type has an initial value.
constexpr fr::Eqs default_eqs;
constexpr fr::Cuts default_cuts;
constexpr ft::Trees default_trees;
constexpr EqJobs default_eq_jobs;
static_assert(default_eqs.n == 0 && default_cuts.NC == 0, "an initial value must be 0");
static_assert(default_trees.NT == 0 && default_eq_jobs.n == 0, "an initial value must be 0");

// A synthetic extension-field element (`T`), determined by h.
// The mask row consists of these, so it is fixed, not random.
static T made_up(u64 h) {
  const u64 w[3] = {mix64(h + 1) % FIELD_P, mix64(h + 2) % FIELD_P, mix64(h + 3) % FIELD_P}; // FIELD_P: the field's prime, 2^64 - 2^32 + 1
  return {w[0], w[1], w[2]};
}
// One lookup call: 2^n keys, a table of 2^k rows.
struct CallShape {
  int n, k;
  bool written; // true: explicit row values (`tau`); false: the range table (row t holds t)
  u32 t0;       // the padding key's row
  u32 padding;  // number of padding keys; they are the call's last keys
};
// The keys of a group's calls are cut into trees of 2^dT keys, dT = min(lid, the smallest n).
struct Group {
  const char *name;
  int lid;
  std::vector<CallShape> calls;
};
__global__ void k_other_key(uint16_t *key, uint16_t value) {
  *key = value;
}
__global__ void k_other_counts(u32 *up, u32 *down) {
  *up += 1, *down -= 1;
}
static void put(FILE *f, const void *p, size_t bytes) {
  if (fwrite(p, 1, bytes, f) != bytes) printf("cannot write\n"), exit(2);
}
static void put64(FILE *f, u64 x) { put(f, &x, 8); }
static void put(FILE *f, const T &x) {
  for (int k = 0; k < EDEG; k++) put64(f, coord(x, k));
}

// Arguments: the output directory (<folder>) and optionally one way to alter the prover's data
// (`cheat`). A group to which the way does not apply is skipped and not counted.
// Exit code: the number of failed groups. Among the errors that end the run early, a CUDA error
// also gives 1, and a usage error, a directory that cannot be opened or a failed fwrite() in put()
// gives 2. hd::put_door_vectors() and hd::put_door_values() do not check their writes.
// Three ways make a malicious prover, whose files the checker must reject (give it "refuse" after
// the directory). Each acts after the multiplicities are counted. The prover goes on consistently
// with the altered data and must raise the failure bit OWN_TEST and no other: its own check that
// the product trees over the keys (key side) and the table rows (table side) match at the root.
//   key           Key 5 of the first call becomes `beyond`, a number outside that call's range
//                 table. The replaced key stays counted.
//   multiplicity  One row's count is raised by one and another's lowered by one.
//   crossed       As for key, and the count moves to row `beyond` of another call's range table.
//                 Without the shift between the calls' tables (call c runs with the challenge
//                 alpha - c theta) the two sides would still match.
// The fourth way is valid, but this program refuses the word at its start, with exit code 2: the
// build that its message asks for is a test build that is not part of this repository.
//   repeated      A key of the first call is counted at the second copy of its row (a call's table
//                 is repeated up to the group's 2^kT rows), which is still a row of that call's
//                 table. The table side's roots include only the first copy's counts
//                 (ft::k_root_parts(), prover/hiding/forest_table.cuh). No test in this repository
//                 shows that the checker accepts such a count.
// One build variant, a preprocessor macro that no target of prover/build.sh defines:
//   SLOT_COUNT=1   Also checks that every message slot was written once and the next four never.
// The three ways alter the prover's data before it proves. A malicious prover that alters what it
// sends (for example, it swaps a product layer's two end values) is not part of this repository, so
// no test in it gives the checker such a prover's files. sc_door::cheat_from_the_environment() and
// cheat_again(), called below, are that prover's hooks: empty (prover/hiding/sumcheck_door.cuh).
// The transcript is a hash chain (blocks/chain/chain.h) under a plan, the list of its rounds
// (plan_lookup(), blocks/numbers/blocks.h). The CPU lends the chain to the kernels for the lookup
// and the claims' batching challenge; Stretch::end() (prover/hiding/transcript_shell.cuh) replays
// the rounds against the plan by name and ends the process on a difference, except for what
// take_back() itself exempts (a lent round's rows). The proof's bytes go into the file. Each group
// is proved once; if PADDING_TOLD is set in the environment, with `told` (below). Runs without and
// with PADDING_TOLD must write the same files; this program does not compare them.
int main(int argc, char **argv) {
  if (argc < 2) return printf("usage: forest_lookup_test <folder> [key | multiplicity | crossed | repeated]\n"), 2;
  // The directory is opened once. Whatever file or link in it has one of this test's three names
  // (`lookup_<i>.bin`, `lookup_<i>.bin.values`, `lookup_groups.txt`), for each i that this run
  // uses, is removed and, unless its group fails, created again: both through the directory's
  // descriptor, by file name alone. With O_EXCL, made_new() fails if anything has the name already,
  // even a symbolic link. So nothing outside the directory is written, whatever another account
  // puts into it during the run. Any "/" and "/." at the path's end are dropped first: with one of
  // these after a symbolic link's name, open() follows the link in spite of O_NOFOLLOW.
  std::string word = argv[1]; // the trimmed path
  while (word.size() > 1 && (word.back() == '/' || (word.size() > 2 && !word.compare(word.size() - 2, 2, "/.")))) word.pop_back();
  const int folder = open(word.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (folder < 0)
    return printf("forest_lookup_test: cannot open the folder %s: %s (it must be an existing directory, not a symbolic link)\n", argv[1], strerror(errno)), 2;
  const auto made_new = [folder](const std::string &name) -> FILE * {
    const int fd = openat(folder, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
    return fd < 0 ? nullptr : fdopen(fd, "wb");
  };
  const std::string cheat = argc > 2 ? argv[2] : "";
  if (cheat == "repeated") return printf("forest_lookup_test: the word repeated wants a build with DOOR_CHEAT=1: only there are the table's further copies walked\n"), 2;
  printf("0 - 1 = 0x%llx\n", (unsigned long long)coord(F::sub(F::zero(), F::one()), 0)); // p - 1, computed by the field's own code: 0xffffffff00000000
  sc_door::cheat_from_the_environment();
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&g_st));
  pt_alloc = [](void **p, size_t n, int) { return cudaMalloc(p, n); };
  const long long distance = (long long)PAD_STRIDE;
  CK(cudaMemcpyToSymbol(PADOFF, &distance, 8));
  // msg has parts of PAD_STRIDE elements: the message slots (slot i gets the i-th sent scalar), the
  // mask row, scratch and, with SLOT_COUNT, the number of writes to each slot.
  const size_t planes = (3 + SLOT_COUNT) * PAD_STRIDE;
  T *msg;
  Tr *tr;
  CK(cudaMalloc(&msg, planes * sizeof(T)));
  CK(cudaMalloc(&tr, sizeof(Tr)));
  // The verifier rejects a call that is all padding (forestPubAsk(), verifier/src/vf_lookup_argument.h):
  // the calls that are nearly all padding have three other keys.
  std::vector<Group> groups = {
      {"one call", 17, {{17, 8, false, 0, 9}}},
      {"two calls, a tree of padding alone", 17, {{18, 8, false, 3, 70000}, {17, 10, true, 5, 11}}},
      {"three calls, a call's number filled up", 17, {{17, 9, false, 0, 0}, {17, 16, false, 0, 1000}, {17, 7, true, 1, 2}}},
      {"four trees of one call", 17, {{19, 12, false, 7, 300000}}},
      {"sixteen calls", 17, {}},
      {"the cap on a tree's depth above the smallest call", 20, {{18, 8, false, 3, 70000}, {17, 10, true, 5, 11}}},
      {"five calls, the second and the fourth all padding but three keys", 17,
       {{17, 9, false, 0, 3}, {18, 10, true, 5, (1u << 18) - 3}, {17, 7, false, 1, 2}, {17, 11, false, 6, (1u << 17) - 3}, {17, 8, true, 200, 100000}}},
      {"sizes not sorted in the list", 17, {{17, 9, false, 0, 3}, {19, 10, true, 5, 200000}, {18, 7, false, 1, 2}}},
      {"four calls, each all padding but three keys", 17,
       {{17, 8, false, 3, (1u << 17) - 3}, {17, 9, true, 4, (1u << 17) - 3}, {17, 7, false, 5, (1u << 17) - 3}, {17, 7, false, 6, (1u << 17) - 3}}},
      {"three calls, each all padding but three keys, and a tree of padding alone", 17,
       {{17, 8, false, 3, (1u << 17) - 3}, {17, 9, true, 4, (1u << 17) - 3}, {17, 7, false, 5, (1u << 17) - 3}}},
  };
  for (int c = 0; c < 16; c++) groups[4].calls.push_back({17, 7 + (c * 5) % 10, c % 2 == 1, (u32)(c % 7), (u32)(c * 8000)});
  int failed = 0, run = 0; // run: the groups run (not skipped) so far; it numbers their files
  for (size_t gi = 0; gi < groups.size(); gi++) {
    const Group &g = groups[gi];
    const int NC = (int)g.calls.size();
    // Every way uses key 5 of the first call, and `multiplicity` also its key 6: that
    // call needs a range table and seven keys that are not padding.
    if (!cheat.empty() && (g.calls[0].written || ((size_t)1 << g.calls[0].n) - g.calls[0].padding < 7)) continue;
    std::vector<ForestCall> list(NC);
    std::vector<std::vector<u64>> tau(NC);
    std::vector<std::vector<uint16_t>> keys(NC);
    std::vector<uint16_t *> keys_dev(NC);
    for (int c = 0; c < NC; c++) {
      const CallShape &s = g.calls[c];
      list[c].n = s.n, list[c].k = s.k, list[c].fn = s.written ? 1 : 0, list[c].t0 = s.t0;
      if (s.written) {
        for (u64 t = 0; t < 1ull << s.k; t++) tau[c].push_back(mix64(1000 * gi + 10 * c + (t << 8)) >> 5); // below 2^59
        u64 *d;
        CK(cudaMalloc(&d, 8 << s.k));
        CK(cudaMemcpy(d, tau[c].data(), 8 << s.k, cudaMemcpyHostToDevice));
        list[c].tau = d;
      }
      const size_t n = (size_t)1 << s.n;
      keys[c].resize(n);
      for (size_t i = 0; i < n; i++) keys[c][i] = (uint16_t)(i < n - s.padding ? mix64(77 * gi + c + (i << 4)) & ((1u << s.k) - 1) : s.t0);
      CK(cudaMalloc(&keys_dev[c], 2 * n));
      CK(cudaMemcpy(keys_dev[c], keys[c].data(), 2 * n, cudaMemcpyHostToDevice));
    }
    Forest f;
    f.init(list, g.lid);
    const ForestPlan &P = f.plan;
    const size_t n = P.nmsg();
    // pub[y]: the number of padding keys in tree y.
    std::vector<u32> pub(P.NR, 0);
    std::vector<u32 *> pub_dev(NC);
    std::vector<std::vector<size_t>> padded(NC); // per tree, from where both halves are padding
    for (int c = 0; c < NC; c++) {
      const size_t tree = (size_t)1 << P.dT;
      size_t left = g.calls[c].padding;
      for (int b = (1 << P.iota[c]) - 1; b >= 0 && left; b--) pub[P.first[c] + b] = (u32)std::min(left, tree), left -= pub[P.first[c] + b];
      CK(cudaMalloc(&pub_dev[c], 4 << P.iota[c]));
      CK(cudaMemcpy(pub_dev[c], pub.data() + P.first[c], 4 << P.iota[c], cudaMemcpyHostToDevice));
      ForestRuns runs;
      runs.add(((size_t)1 << g.calls[c].n) - g.calls[c].padding, (size_t)1 << g.calls[c].n);
      padded[c] = forest_padded_from(P.dT, P.iota[c], runs, std::vector<u32>(pub.begin() + P.first[c], pub.begin() + P.first[c] + (1 << P.iota[c])));
    }
    std::vector<T> mask(n);
    for (size_t i = 0; i < n; i++) mask[i] = made_up(0x5000000ull * (gi + 1) + i);
    // One byte per multiplicity bit: bit j of row t of call c at ((c << kT) + t) * 32 + j.
    std::vector<u8> bits(P.plane_bits());
    const uint16_t beyond = (uint16_t)((1u << g.calls[0].k) + 44);
    if (!cheat.empty() && keys[0][5] == keys[0][6]) return printf("the test's groups do not fit its cheats\n"), 2;
    int other = -1; // crossed: another call with a range table that has the row `beyond`
    for (int c = 1; c < NC && other < 0; c++)
      if (!g.calls[c].written && (1u << g.calls[c].k) > beyond) other = c;
    if ((cheat == "repeated" && P.kT <= g.calls[0].k) || (cheat == "crossed" && other < 0)) {
      printf("the look-up of a group, %s: passed over: it has no place for this way\n", g.name);
      continue;
    }
    blk::Plan plan(blk::setting_goldilocks());
    {
      blk::LookupSize x;
      x.v = P.v(), x.d = P.d(), x.mu = 5, x.shifted = NC > 1;
      blk::plan_lookup(plan, "group", x);
      // Claims: 31 tie the roots, one per level (v + d), one on the bits, three on the planes,
      // the join.
      blk::plan_bat_sigma(plan, "claims", 36 + P.v() + P.d());
    }
    blk::Binder binder(6, plan.finish());
    blk::Digest configuration{}; // a chain needs one: arbitrary here, the same in the checker
    configuration.w[0] = 0x6c6b7570u;
    (void)binder.configuration(configuration);
    blk::Writer proof;
    blk::ProverRounds channel(binder, blk::Digest{}, proof);
    Stretch stretch;
    stretch.init(4096, 4096);
    // told: the key side is given `padded` and may skip the padding that it covers; otherwise it
    // reads every key.
    auto prove = [&](bool told) {
      CK(cudaMemset(msg, 0, planes * sizeof(T)));
      CK(cudaMemcpy(msg + PAD_STRIDE, mask.data(), n * sizeof(T), cudaMemcpyHostToDevice));
      stretch.begin(channel, g_st);
      tr = stretch.tr;
      CK(cudaMemcpy(keys_dev[0], keys[0].data(), 2 * keys[0].size(), cudaMemcpyHostToDevice));
      // The memset and copies above are not ordered with g_st (the build uses per-thread default
      // streams), and a cudaMemcpy() from a vector's pageable memory, or a cudaMemset(), can
      // return before the GPU's memory holds the new bytes: wait before any launch on g_st.
      CK(cudaDeviceSynchronize());
      for (int c = 0; c < NC; c++) f.counts(c, keys_dev[c], pub_dev[c], told ? padded[c].data() : nullptr);
      if (cheat == "key" || cheat == "crossed") k_other_key<<<1, 1, 0, g_st>>>(keys_dev[0] + 5, beyond);
      if (cheat == "crossed") k_other_counts<<<1, 1, 0, g_st>>>(f.table.cnt + ((size_t)other << P.kT) + beyond, f.table.cnt + keys[0][5]);
      if (cheat == "multiplicity") k_other_counts<<<1, 1, 0, g_st>>>(f.table.cnt + keys[0][5], f.table.cnt + keys[0][6]);
      sc_door::cheat_again();
      f.planes();
      CK(cudaMemcpyAsync(bits.data(), f.bits(0), bits.size(), cudaMemcpyDeviceToHost, g_st));
      CK(cudaStreamSynchronize(g_st));
      f.prove(tr, msg);
      CK(cudaStreamSynchronize(g_st));
      CK(cudaDeviceSynchronize());
      std::vector<T> sent(n);
      CK(cudaMemcpy(sent.data(), msg, n * sizeof(T), cudaMemcpyDeviceToHost));
      return sent;
    };
    const std::vector<T> sent = prove(getenv("PADDING_TOLD") != nullptr);
    std::string why;
    if (cheat == "repeated") { // an honest run counts nothing at a row's second copy
      bool counted = false;
      for (int j = 0; j < 32; j++) counted = counted || bits[((size_t)keys[0][5] + ((size_t)1 << g.calls[0].k)) * 32 + j];
      if (!counted) why = "nothing is counted at the second copy of the row: the way altered nothing";
    }
    const unsigned flags = take_live_flags();
    if (cheat.empty() || cheat == "repeated" ? flags != 0 : flags != (unsigned)OWN_TEST) why = "the failure bits are " + std::to_string(flags);
#if SLOT_COUNT
    if (why.empty()) {
      std::vector<T> counts(n + 4);
      CK(cudaMemcpy(counts.data(), msg + 3 * PAD_STRIDE, counts.size() * sizeof(T), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < counts.size() && why.empty(); i++)
        if (coord(counts[i], 0) != (i < n ? 1u : 0u)) why = "slot " + std::to_string(i) + " was written " + std::to_string(coord(counts[i], 0)) + " times";
    }
#endif
    std::vector<T> ls(n);
    const std::string name = "lookup_" + std::to_string(run++) + ".bin";
    unlinkat(folder, name.c_str(), 0), unlinkat(folder, (name + ".values").c_str(), 0); // no stale file if the group fails
    if (why.empty()) {
      // The lookup's linear claims, added to a `hw::Walk` (prover/hiding/walk.cuh: it batches
      // claims into one equation) in the verifier's order, as a sub-proof (the argument for one
      // part of the forward pass) adds them. A space is an array of variables. Space 0 is the
      // slots: LS0 gets the batched claims' coefficients on the mask row.
      hw::Walk W;
      size_t all_keys = 0;
      for (int c = 0; c < NC; c++) all_keys += (size_t)1 << list[c].n;
      W.init(all_keys + (1 << 16)); // pool: an equality vector per call's keys, and small vectors
      W.scratch = msg + 2 * PAD_STRIDE;
      T *LS0;
      CK(cudaMalloc(&LS0, n * sizeof(T)));
      CK(cudaMemset(LS0, 0, n * sizeof(T)));
      hw::Space slots_space;
      slots_space.lam = LS0, slots_space.len = n, slots_space.neg = 1;
      W.sp.push_back(slots_space);
      // The multiplicity bits (array "planes": plane j has bit j of each multiplicity) form a
      // matrix of 2^lm rows; space 1 holds three virtual rows, the matrix's rows summed under each
      // of phi's three weight vectors. Space 2 + c is call c's key vector. `door` lists these
      // arrays by name with their coefficient vectors.
      const int lm = std::min(P.kT + 5, 9);
      const size_t columns = (size_t)NC << (P.kT + 5 - lm), rows = (size_t)1 << lm;
      auto cleared = [](size_t entries) {
        T *p;
        CK(cudaMalloc(&p, entries * sizeof(T)));
        CK(cudaMemset(p, 0, entries * sizeof(T)));
        return p;
      };
      std::vector<T *> held = {cleared(3 * columns), cleared(3 * rows), cleared(1)};
      T *on_columns = held[0], *phi = held[1], *one = held[2];
      {
        const T unit = F::one();
        CK(cudaMemcpy(one, &unit, sizeof(T), cudaMemcpyHostToDevice));
        hw::Space s;
        s.lam = on_columns, s.len = 3 * columns;
        W.sp.push_back(s);
      }
      std::vector<hd::Term> door;
      std::vector<void *> arrays; // GPU arrays that the terms of `door` point to
      {
        hd::Term t;
        t.nm = "planes", t.kind = hd::CELLS16, t.R = rows, t.K = columns;
        // The matrix's cells, for the report of false claims. Bit j of the multiplicity of row t of
        // call c is in row (low lm - 5 bits of t | j), column (c | the other bits of t).
        const int wd = P.kT + 5 - lm;
        std::vector<uint16_t> cells(rows * columns);
        for (size_t row = 0; row < rows; row++)
          for (size_t col = 0; col < columns; col++) {
            const size_t c = col >> wd, high = col & (((size_t)1 << wd) - 1), tt = (high << (lm - 5)) + (row >> 5);
            cells[row * columns + col] = bits[((c << P.kT) + tt) * 32 + (row & 31)];
          }
        uint16_t *there;
        CK(cudaMalloc(&there, 2 * cells.size()));
        CK(cudaMemcpy(there, cells.data(), 2 * cells.size(), cudaMemcpyHostToDevice));
        t.data = there, t.ld = columns;
        arrays.push_back(there);
        for (int i = 0; i < 3; i++) {
          hd::Pair p;
          p.cv = phi + i * rows, p.L = on_columns + i * columns, p.c0 = 0, p.c1 = columns;
          t.v.push_back(p);
        }
        door.push_back(t);
      }
      std::vector<hw::AF> KF(NC);
      for (int c = 0; c < NC; c++) {
        const size_t entries = (size_t)1 << list[c].n;
        held.push_back(cleared(entries));
        hw::Space s;
        s.lam = held.back(), s.len = entries;
        W.sp.push_back(s);
        KF[c] = hw::rng(2 + c, 0, 1, entries, W.eq(f.claims->zk[c], list[c].n));
        hd::Term t;
        t.nm = "keys " + std::to_string(c), t.kind = hd::ELEMENTS, t.R = 1, t.K = entries;
        // The prover's keys as field elements: each row number becomes that row's value.
        std::vector<T> values(entries);
        for (size_t i = 0; i < entries; i++) {
          const uint16_t key = c == 0 && i == 5 && (cheat == "key" || cheat == "crossed") ? beyond : keys[c][i];
          values[i] = F::small(tau[c].empty() ? (u64)key : tau[c][key]);
        }
        T *there;
        CK(cudaMalloc(&there, entries * sizeof(T)));
        CK(cudaMemcpy(there, values.data(), entries * sizeof(T), cudaMemcpyHostToDevice));
        t.data = there;
        arrays.push_back(there);
        hd::Pair p;
        p.cv = one, p.L = held.back(), p.c0 = 0, p.c1 = entries;
        t.v.push_back(p);
        door.push_back(t);
      }
      for (int e = 0; e < 2; e++) hw::k_forest_phi_bits<<<8, 128, 0, g_st>>>(f.claims->zb[e], P.kT, P.cK, lm, phi + e * rows);
      hw::k_forest_phi_powers<<<8, 128, 0, g_st>>>(lm, phi + 2 * rows);
      const hw::ForestForms forms = hw::walk_forest(W, f, 0);
      hw::walk_forest_planes(W, f, forms, 1, 0, columns, 2 * columns, lm);
      W.lin(hw::forest_join(f, forms, KF));
      W.seal();
      W.finish(tr, LS0);
      CK(cudaDeviceSynchronize());
      const std::vector<hd::DoorVector> vectors = hd::door_vectors(door);
      const std::vector<hd::DoorValues> rows_values = hd::door_values(door, 0);
      for (T *p : held) cudaFree(p);
      for (void *p : arrays) cudaFree(p);
      (void)stretch.end(channel, g_st);
      channel.finish();
      CK(cudaMemcpy(ls.data(), LS0, n * sizeof(T), cudaMemcpyDeviceToHost));
      cudaFree(LS0);
      // Layout of `lookup_<i>.bin`: numbers are 64-bit words in the machine's byte order (an
      // element: EDEG words), keys 16 bits, multiplicity bits a byte each.
      FILE *o = made_new(name);
      if (!o) why = "cannot write " + name + ": " + strerror(errno);
      else {
        put64(o, (u64)NC), put64(o, (u64)g.lid), put64(o, (u64)n);
        for (int c = 0; c < NC; c++) {
          put64(o, (u64)list[c].n), put64(o, (u64)list[c].k), put64(o, (u64)list[c].fn), put64(o, list[c].t0), put64(o, g.calls[c].padding);
          put(o, tau[c].data(), 8 * tau[c].size());
          std::vector<uint16_t> held = keys[c]; // what the prover holds as its keys
          if (c == 0 && (cheat == "key" || cheat == "crossed")) held[5] = beyond;
          put(o, held.data(), 2 * held.size());
        }
        put(o, bits.data(), bits.size());
        for (const T &x : sent) put(o, x);
        for (const T &x : mask) put(o, x);
        for (const T &x : ls) put(o, x);
        put64(o, proof.size()), put(o, proof.data(), proof.size());
        put64(o, (u64)lm);
        hd::put_door_vectors(o, vectors);
        fclose(o);
        FILE *o2 = made_new(name + ".values"); // format: prover/hiding/door_values.cuh
        if (!o2) why = "cannot write " + name + ".values: " + strerror(errno);
        else hd::put_door_values(o2, rows_values, mask), fclose(o2);
      }
    }
    printf("the look-up of a group, %s (v %d, d %d, %zu slots): %s%s\n", g.name, P.v(), P.d(), n, why.empty() ? "run and written" : "FAILS: ",
           why.c_str());
    fflush(stdout);
    failed += !why.empty();
  }
  {
    // How many files the checker must find.
    unlinkat(folder, "lookup_groups.txt", 0); // removes an earlier run's file
    FILE *o = made_new("lookup_groups.txt");
    if (!o) return printf("cannot write the number of groups\n"), 2;
    fprintf(o, "%d\n", run);
    fclose(o);
  }
  printf("forest_lookup_test: %d of %d groups run and written\n", run - failed, run);
  return failed;
}
