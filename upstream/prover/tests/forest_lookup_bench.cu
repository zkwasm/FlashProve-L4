// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Benchmark of the lookup forest (prover/hiding/forest.cuh): the GPU time of Forest::prove(), which
// proves a group of lookup calls as one lookup, on synthetic keys.
//   forest_lookup_bench <layers | attention | head | tail> [reb=<2..5>] [runs=<n>] [lanes=<n>]
// reb: tree levels rebuilt from the keys, not stored.
// It proves the group in two ways: with `forest_fused` false, the key side's Gram passes (each
// serves three sumcheck rounds; prover/hiding/gram_products.cuh) write digits to GPU memory for
// cublasLtMatmul; with true, they use the kernels of prover/hiding/forest_fused*.cuh, except in the
// first pass over a level that only reb=4 or 5 rebuilds. For each way it prints the mean and least
// time of a proof launched as a CUDA graph, the phases' times and a digest of the sent scalars.
// The exit code is 1 if the digests differ or a failure bit is set. Built with -DSLOT_COUNT=1
// (prover/tests/unit.sh --slot-count), it also fails unless every message slot (a sent scalar's
// index) was written exactly once. It does not check that the scalars are a valid proof; on small
// groups, prover/tests/forest_lookup_test.cu with verifier/tests/forest_lookup_check.cpp does.
// Build in prover/: ./build.sh forest_lookup_bench. One GPU, no weights; besides the printed
// scratch memory, the keys take 2 bytes each.
#include "hiding/product_test.cuh"

#include "hiding/forest.cuh"

#include "chain/public.h"
#include "numbers/blocks.h"
#include <memory>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {
enum Table { RANGE, GENERAL, PACKED, BYTE_LIMBS, SHIFTED };
// One lookup call: 2^n keys in blocks (four in `layers`, else one), a table of 2^k rows.
struct Shape {
  int n, k;
  Table table;
  size_t columns; // a block's first 1,024 x `columns` keys are used, the rest are padding; 0: none
  int fn() const { return table == RANGE ? 0 : table == BYTE_LIMBS ? 2 : 1; }
  u32 t0() const { return table == RANGE ? 0 : 1u << (k - 1); }
};
// Meant to be the groups of the full-size configuration (prover/examples/target_full_f2), with its
// padding; nothing here checks that.
std::vector<Shape> group_of(const std::string &g) {
  if (g == "layers")
    return {{27, 16, RANGE, 32768}, {27, 16, RANGE, 31744}, {27, 16, RANGE, 17792}, {28, 8, RANGE, 60998},
            {27, 16, PACKED, 17536}, {27, 16, SHIFTED, 16385}, {26, 16, PACKED, 16384}, {26, 16, RANGE, 16384}};
  if (g == "head")
    return {{28, 16, RANGE, 212288}, {28, 16, RANGE, 202048}, {28, 8, RANGE, 202168}, {27, 16, PACKED, 131072},
            {27, 16, PACKED, 70976}, {27, 16, PACKED, 5120}};
  if (g == "attention") return {{27, 16, BYTE_LIMBS, 92160}, {27, 16, GENERAL, 92160}, {26, 16, RANGE, 41120}, {25, 16, BYTE_LIMBS, 20480}};
  if (g == "tail") return {{27, 10, RANGE, 0}, {27, 8, RANGE, 0}};
  printf("forest_lookup_bench: no such group: %s\n", g.c_str());
  exit(2);
}
template <class Fn> void in_threads(size_t parts, const Fn &fn) {
  std::vector<std::thread> all;
  const size_t n = std::min<size_t>(parts, std::max(1u, std::thread::hardware_concurrency()));
  for (size_t w = 0; w < n; w++)
    all.emplace_back([&, w] {
      for (size_t p = w; p < parts; p += n) fn(p);
    });
  for (auto &t : all) t.join();
}
struct Marks {
  std::vector<std::string> names;
  std::vector<cudaEvent_t> events;
  size_t at = 0;
  bool on = false;
} marks;
bool capturing() {
  cudaStreamCaptureStatus s;
  CK(cudaStreamIsCapturing(g_st, &s));
  return s == cudaStreamCaptureStatusActive;
}
// Hook for `forest_mark`. Acts only while `marks.on`, which main() sets from the captured proof on.
// The counted runs only replay that capture (forest_prove_graph()), and every replay stamps the same events anew.
// Public source: an event as a graph node, by cudaEventRecordExternal: NVIDIA, CUDA 13.0,
// cuda_runtime_api.h, the comment at cudaEventRecordWithFlags().
void mark(const char *name) {
  if (!marks.on) return;
  if (marks.at == marks.events.size()) {
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    marks.events.push_back(e), marks.names.push_back(name);
  }
  CK(cudaEventRecordWithFlags(marks.events[marks.at++], g_st, capturing() ? cudaEventRecordExternal : cudaEventRecordDefault));
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 2) return printf("usage: forest_lookup_bench <layers | attention | head | tail> [reb=n] [runs=n] [lanes=n]\n"), 2;
  std::map<std::string, int> word;
  for (int i = 2; i < argc; i++) {
    const std::string w = argv[i];
    const size_t eq = w.find('=');
    word[w.substr(0, eq)] = eq == std::string::npos ? 1 : atoi(w.c_str() + eq + 1);
  }
  auto number = [&](const char *w, int otherwise) { return word.count(w) ? word.at(w) : otherwise; };
  const std::string group = argv[1];
  const int reb = number("reb", 3), runs = number("runs", 5);
  forest_lanes = number("lanes", 2);
  CK(cudaSetDevice(0));
  CK(cudaStreamCreate(&g_st));
  pt_alloc = [](void **p, size_t n, int) { return cudaMalloc(p, n); };
  const long long distance = (long long)PAD_STRIDE;
  CK(cudaMemcpyToSymbol(PADOFF, &distance, 8));
#ifndef SLOT_COUNT
#define SLOT_COUNT 0
#endif
  // `msg`: planes of PAD_STRIDE elements: slots, masks, scratch, write counters (SLOT_COUNT).
  const size_t planes = (3 + SLOT_COUNT) * PAD_STRIDE;
  T *msg;
  Tr *tr;
  CK(cudaMalloc(&msg, planes * sizeof(T)));
  CK(cudaMalloc(&tr, sizeof(Tr)));
  const std::vector<Shape> shapes = group_of(group);
  const int NC = (int)shapes.size();
  std::vector<ForestCall> list(NC);
  std::vector<uint16_t *> keys_dev(NC);
  std::vector<ForestRuns> padding(NC);
  for (int c = 0; c < NC; c++) {
    const Shape &s = shapes[c];
    const size_t N = (size_t)1 << s.n, rows = (size_t)1 << s.k, block = (size_t)1 << (group == "layers" ? s.n - 2 : s.n),
                 used = s.columns ? s.columns << 10 : block;
    list[c].n = s.n, list[c].k = s.k, list[c].fn = s.fn(), list[c].t0 = s.t0();
    if (s.table != RANGE) {
      std::vector<u64> tau(rows);
      for (size_t t = 0; t < rows; t++) {
        const u64 z = mix64(3001 + c + 3 * t);
        tau[t] = s.table == GENERAL   ? z >> 4
                 : s.table == PACKED  ? t + ((z & 65535) << 28)
                 : s.table == SHIFTED ? t + (((z & 0xFFFFF) | 1) << 21)
                                      : (t & 255) + ((u64)(t >> 8) << 20) + ((z & 1) << 40);
      }
      u64 *d;
      CK(cudaMalloc(&d, 8 * rows));
      CK(cudaMemcpy(d, tau.data(), 8 * rows, cudaMemcpyHostToDevice));
      list[c].tau = d;
    }
    std::vector<uint16_t> keys(N);
    in_threads(N >> 16, [&](size_t part) {
      for (size_t i = part << 16; i < (part + 1) << 16; i++) keys[i] = (uint16_t)(i % block >= used ? s.t0() : mix64(4001 + c + i) % rows);
    });
    for (size_t b = 0; b < N; b += block) padding[c].add(b + used, b + block);
    CK(cudaMalloc(&keys_dev[c], 2 * N));
    CK(cudaMemcpy(keys_dev[c], keys.data(), 2 * N, cudaMemcpyHostToDevice));
  }
  Forest f;
  f.init(list, 28, 18, reb);
  const ForestPlan &P = f.plan;
  const size_t n = P.nmsg();
  std::vector<u32 *> pub_dev(NC);
  std::vector<std::vector<size_t>> padded(NC);
  for (int c = 0; c < NC; c++) {
    const size_t tree = (size_t)1 << P.dT;
    std::vector<u32> pub((size_t)1 << P.iota[c], 0);
    for (const auto &r : padding[c].run)
      for (size_t at = r.first; at < r.second;) {
        const size_t end = std::min(r.second, (at / tree + 1) * tree);
        pub[at / tree] += (u32)(end - at), at = end;
      }
    padded[c] = forest_padded_from(P.dT, P.iota[c], padding[c], pub);
    CK(cudaMalloc(&pub_dev[c], 4 << P.iota[c]));
    CK(cudaMemcpy(pub_dev[c], pub.data(), 4 << P.iota[c], cudaMemcpyHostToDevice));
  }
  printf("%s: %d calls, %d trees of 2^%d keys (%d with those of padding alone), v %d, d %d; %zu slots; %d rebuilt levels, %d lanes; scratch %.3f GB\n",
         group.c_str(), NC, P.NR, P.dT, P.NT, P.v(), P.d(), n, reb, forest_lanes, f.at.total / 1e9);
  std::vector<T> mask(n);
  const u64 low60 = (1ull << 60) - 1; // synthetic masks; 60-bit coordinates, below the 64-bit prime
  for (size_t i = 0; i < n; i++) mask[i] = T{mix64(7 * i + 1) & low60, mix64(7 * i + 2) & low60, mix64(7 * i + 3) & low60};
  forest_mark = mark;
  PtGraph graphs;
  // Before each proof the host starts a hash chain (the Fiat-Shamir transcript; `Tr` on the GPU)
  // under the plan of one lookup and lends it to the kernels. After the timed interval
  // Stretch::end() (prover/hiding/transcript_shell.cuh) takes it back: the host repeats every round
  // against the plan and ends the process if anything differs.
  blk::Plan plan(blk::setting_goldilocks());
  {
    blk::LookupSize x;
    x.v = P.v(), x.d = P.d(), x.mu = 5, x.shifted = NC > 1;
    blk::plan_lookup(plan, "group", x);
  }
  blk::Binder binder(6, plan.finish());
  blk::Digest configuration{}; // a chain needs one: arbitrary here
  configuration.w[0] = 0x6c6b7570u;
  (void)binder.configuration(configuration);
  std::unique_ptr<blk::Writer> proof;
  std::unique_ptr<blk::ProverRounds> channel;
  Stretch stretch;
  stretch.init(4096, 4096);
  tr = stretch.tr;
  // One proof's set-up, not timed.
  auto before = [&] {
    CK(cudaMemsetAsync(msg, 0, PAD_STRIDE * sizeof(T), g_st));
    CK(cudaMemcpyAsync(msg + PAD_STRIDE, mask.data(), n * sizeof(T), cudaMemcpyHostToDevice, g_st));
    if (SLOT_COUNT) CK(cudaMemsetAsync(msg + 3 * PAD_STRIDE, 0, PAD_STRIDE * sizeof(T), g_st));
    proof.reset(new blk::Writer), channel.reset(new blk::ProverRounds(binder, blk::Digest{}, *proof));
    stretch.begin(*channel, g_st);
    for (int c = 0; c < NC; c++) f.counts(c, keys_dev[c], pub_dev[c], padded[c].data());
    f.planes();
    CK(cudaStreamSynchronize(g_st));
  };
  u64 digest[2] = {0, 0};
  int failed = 0;
  for (int fused = 0; fused < 2; fused++) {
    forest_fused = fused != 0;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    float sum = 0, least = 1e30f;
    std::vector<double> phase;
    for (int r = -2; r < runs; r++) { // not counted: a direct run (-2), then the captured one (-1)
      before();
      marks.on = r >= -1, marks.at = 0;
      CK(cudaEventRecord(a, g_st));
      forest_prove_graph(graphs, f, tr, msg);
      CK(cudaEventRecord(b, g_st));
      CK(cudaStreamSynchronize(g_st));
      (void)stretch.end(*channel, g_st);
      channel->finish();
      if (r < 0) continue;
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      sum += ms, least = std::min(least, ms);
      phase.resize(marks.events.size() ? marks.events.size() - 1 : 0, 0);
      for (size_t i = 0; i + 1 < marks.events.size(); i++) {
        CK(cudaEventElapsedTime(&ms, marks.events[i], marks.events[i + 1]));
        phase[i] += ms;
      }
    }
    std::vector<T> sent(n);
    CK(cudaMemcpy(sent.data(), msg, n * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
      for (int k = 0; k < EDEG; k++) digest[fused] = mix64(digest[fused] ^ coord(sent[i], k));
    const unsigned flags = take_live_flags();
    failed += flags != 0;
#if SLOT_COUNT
    { // the last proof's counters: n slots written once, the next 4 never
      std::vector<T> counts(n + 4);
      CK(cudaMemcpy(counts.data(), msg + 3 * PAD_STRIDE, counts.size() * sizeof(T), cudaMemcpyDeviceToHost));
      size_t wrong = 0;
      for (size_t i = 0; i < counts.size(); i++) wrong += coord(counts[i], 0) != (i < n ? 1u : 0u);
      printf("%s, fused=%d: every one of the %zu slots was written once and none beyond: %s\n", group.c_str(), fused, n, wrong ? "FAILS" : "ok");
      failed += wrong != 0;
    }
#endif
    printf("TIME %s, %s: a proof as a CUDA graph: mean %.3f ms, least %.3f ms of %d runs (counts apart); failure bits %u; digest of the %zu sent "
           "scalars %016llx\n", group.c_str(), fused ? "the kernels of forest_fused*.cuh" : "digits through GPU memory", sum / runs, least, runs, flags, n,
           (unsigned long long)digest[fused]);
    for (size_t i = 0; i < phase.size(); i++) printf("PHASE %s fused=%d: %8.3f ms  %s\n", group.c_str(), fused, phase[i] / runs, marks.names[i].c_str());
    fflush(stdout);
  }
  failed += digest[0] != digest[1];
  printf("forest_lookup_bench: the two ways give %s scalars; %s\n", digest[0] == digest[1] ? "the same" : "DIFFERENT", failed ? "FAILS" : "ok");
  return failed != 0;
}
