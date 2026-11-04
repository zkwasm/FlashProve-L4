// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the LINK proof (blocks/iop/link.h; `link` in identifiers and messages), which is to show
// that a leaf table (`further`, `copy`) contains the same weights W as the anchor (`first`), the
// commitment's first weight table, under other masks and salts. Runs on a GPU (`card` in
// identifiers), on random cells, at toy sizes and at the error target `fhb` (16 bits in the default
// shapes, 131 in shape 9). As in the committer program `cp_commit`, make_table() builds each
// table's tree, and prover/hash_commitment/hf_open_side.cuh folds rows and opens columns for the
// HfKeeper through which LINK reads a table. But `cp_commit` calls blk::make_entry(); only this
// test includes prover/hash_commitment/hf_make_link.h (set-up, proof and, on the CPU, check).
//   K1  a proof for two tables of one W is accepted and is no longer than planned;
//   K2  each table takes two passes over the weights (`walk`), and no failure flag is raised;
//   K3  for shapes < 7, EngineCpu and EngineGpu (blk::Engine: encoding, hashing, folds) write the
//       same bytes from the same private randomness (`coins` in identifiers);
//   K4  a proof made by the rules for a leaf table with one cell changed fails the proximity test;
//   K5  a proof is rejected ("refused" in the test's failure messages) if the verifier has another
//       leaf table's root;
//   K6  the tables have records (stored sample answers, each masked by pads). Each table has its
//       own pads, so LINK compares a row's message only up to its first pad. The test computes the
//       records on the CPU, with the pads from hf_mask_ref::pad(). A proof is also rejected if one
//       answer of the leaf table's record is changed, the same for prover and verifier.
// Arguments: numbers of toy shapes, HfParams::toy(); default 1 5 7; SUITE.md adds 9. Build and run:
//   prover/tests/unit.sh both hash_commitment/tests/hf_link_side_test.cu <directory>
// which compiles with nvcc and writes the test's output to <directory>/hf_link_side_test.log. Last
// line on success: "hf_link_side_test: all passed"; exit code 0, or 1 if a check failed.
#define BLK_TESTED_TABLE_FOR_A_TEST  // enables `on_a_tests_word` (blocks/iop/foldeq.h): see `run`
#include "engine/engine_cpu.h"
#include "engine/engine_gpu.cuh"
#include "iop/tests/iopp_support.h"

#include "common/setting.h"
#include "weights_commitment/lent_memory.h"
#include "weights_commitment/common.cuh"
#include "hash_commitment/hf_commit_table.cuh"
#include "hash_commitment/hf_open_side.cuh"
#include "hash_commitment/hf_make_link.h"
#include <chrono>
#include <cstring>
#include <random>
#include <string>
using namespace hf;
typedef uint64_t u64_;

static int g_failed = 0;
static void expect(bool ok, const std::string &what) {
  if (ok) return;
  g_failed++;
  printf("FAILED: %s\n", what.c_str());
  fflush(stdout);
}
// The code from here through side_of() is hf_lin_side_test.cu's, except `World`'s constructor and
// that `World` has no member `record` here.

// In parentheses, (cudaMalloc) and (cudaFree) are the CUDA runtime's functions, not the macros of
// these names in prover/weights_commitment/lent_memory.h.

// Returns and clears the failure flags raised on the host and the GPU (prover/common/slot_flags.h).
static unsigned take_the_flags() {
  CK(cudaDeviceSynchronize());
  return take_live_flags();
}
struct RandomCells : RowCells {
  const HfParams &P;
  size_t stride;
  int8_t *M = nullptr;
  RandomCells(const HfParams &P_, const std::vector<int8_t> &host, size_t stride_) : P(P_), stride(stride_) {
    CK((cudaMalloc)((void **)&M, host.size()));
    CK(cudaMemcpy(M, host.data(), host.size(), cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
  }
  const int8_t *rows(uint32_t r0, int, size_t *s, cudaStream_t) override { return *s = stride, M + r0 * stride; }
  int most() const override { return (int)P.R; }
  size_t bytes() const override { return P.R * stride; }
  void release() override {
    if (M) CK((cudaFree)(M));
    M = nullptr;
  }
};
struct World {
  const HfParams &P;
  rf::LayW Y;
  std::vector<int8_t> W;     // W[y * k + c]
  std::vector<int8_t> cells; // M: W's cells as the table's rows, at `stride`
  size_t stride = 0;
  int8_t *panel = nullptr; // GPU: panel[c * pitch + y], all rows of W
  size_t pitch = 0;
  hf_key32 key{};
  uint8_t key_bytes[32];   // the key's second copy (HfHeld::key_again)
  uint8_t other_bytes[32]; // another key's
  int walks = 0;
  // rng comes by value, so the `World`s of one shape draw the same W.
  // copy: another key, so other masks and salts. one_cell: one cell of W is changed.
  World(const HfParams &P_, std::mt19937_64 rng, bool copy, bool one_cell) : P(P_) {
    Y.lk = P.lk, Y.k = P.k, Y.m0 = Y.mW = P.m0;
    W.resize(P.m0 * P.k);
    for (auto &b : W) b = (int8_t)(rng() & 255);
    if (one_cell) W[5 * P.k + 3] ^= 1;
    stride = ((size_t)P.l + 255) & ~(size_t)255;
    cells.assign(((P.R + 15) & ~(size_t)15) * stride, 77);
    for (size_t y = 0; y < P.m0; y++)
      for (size_t c = 0; c < P.k; c++) cells[P.row_of(y, c) * stride + P.coef_of(y, c)] = W[y * P.k + c];
    pitch = (P.m0 + 15) / 16 * 16;
    std::vector<int8_t> turned(P.k * pitch, 0);
    for (size_t y = 0; y < P.m0; y++)
      for (size_t c = 0; c < P.k; c++) turned[c * pitch + y] = W[y * P.k + c];
    CK((cudaMalloc)((void **)&panel, turned.size()));
    CK(cudaMemcpy(panel, turned.data(), turned.size(), cudaMemcpyHostToDevice));
    for (u32 &w : key.w) w = (u32)rng() ^ (copy ? 0x5a5a5a5au : 0u);
    memcpy(key_bytes, key.w, 32);
    hf_key32 other = key;
    other.w[3] ^= 1;
    memcpy(other_bytes, other.w, 32);
  }
  ~World() {
    CK((cudaFree)(panel));
  }
  // One pass over the weights: gives W to units.cells() as up to three "units" (runs of whole rows
  // that begin at multiples of 16), not in W's order.
  rf::HfWalk walk() {
    return [this](const rf::HfUnits &units) {
      const size_t a = P.m0 / 3 / 16 * 16, b = P.m0 * 2 / 3 / 16 * 16;
      const size_t from[3] = {b, 0, a}, to[3] = {P.m0, a, b};
      for (int x = 0; x < 3; x++)
        if (to[x] > from[x]) units.cells(panel + from[x], pitch, to[x] - from[x], from[x], cudaStreamPerThread);
      CK(cudaDeviceSynchronize());
      walks++;
    };
  }
};

static std::vector<u32> tree(World &w, int s) {
  const HfParams &P = w.P;
  std::vector<u32> heap(kept_digests(P.ln, s) * DW, 0xA5A5A5A5u);
  TablePlan plan;
  plan.lp = std::max(s, std::min(P.ln, 16)), plan.s = s, plan.lines = false;
  plan.key_again = w.key_bytes;
  make_table(P, plan, &w.key, [&](int) { return new RandomCells(P, w.cells, w.stride); }, heap.data());
  return heap;
}
static rf::HfSide side_of(World &w, const HfOpen &mode, const std::vector<u32> &heap, int s, Ext *u_host, int fibres) {
  rf::HfSide side;
  side.P = &w.P, side.mode = &mode, side.Y = &w.Y;
  side.held.seed = &w.key, side.held.key_again = w.key_bytes;
  side.held.nodes = heap.data(), side.held.s = s;
  side.held.u_host = u_host, side.held.u_host_elements = u_host ? rf::hf_side_u_elements(w.P) : 0;
  side.short_lived = [](bool) {};
  side.walk = w.walk();
  side.fibres_a_walk = fibres;
  return side;
}

// The free parameters of a LINK proof, for a toy table. table_data = i0 is the length of a row's
// message. LINK compares the entries before the pads (LinkShape::cut() is pad_entry(0)).
static blk::LinkLines lines_of(const hf::HfTableParams &P) {
  blk::LinkLines L;
  L.table_rows_log = (unsigned)P.b, L.table_log_n = (unsigned)P.ln, L.table_data = P.i0, L.table_random = P.nm;
  const unsigned lu = blk::ceil_log2(P.KD) - 2;
  L.folded.log_ell = lu, L.folded.log_n = lu + 3;
  L.folded.mask_log_ell = 4, L.folded.mask_log_n = lu + 3;
  L.folded.stages = {blk::StageChoice{2, 1, lu + 3, lu + 3}, blk::StageChoice{0, 0, lu + 5, lu + 5}};
  return L;
}

struct Made {
  World &w;
  std::vector<u32> heap;
  const u32 *root() const { return heap.data() + DW; }  // node 1 of the kept heap
  std::vector<uint8_t> commitment() const { return std::vector<uint8_t>((const uint8_t *)root(), (const uint8_t *)root() + 4 * DW); }
  // The table's record: for each row, its answers by blk::record_value(). A row's message is built
  // here: the cells, zeros, then the pads.
  mutable blk::IndexRecord made_record;  // computed at the first call of `record`
  const blk::IndexRecord &record() const {
    if (made_record.y.empty()) made_record = make_record();
    return made_record;
  }
  blk::IndexRecord make_record() const {
    const HfParams &P = w.P;
    const blk::IndexShape shape = hf_index_shape(P);
    const std::vector<blk::SampleSeed> seeds = blk::record_seeds(shape, blk::index_name(hf_digest(root()), shape, hf_digest(P.configuration)));
    blk::IndexRecord rec;
    std::vector<blk::F> message(P.i0);
    for (size_t r = 0; r < P.R; r++) {
      std::fill(message.begin(), message.end(), blk::F::zero());
      for (size_t i = 0; i < P.l; i++) {
        const int cell = w.cells[r * w.stride + i];
        message[i] = cell < 0 ? blk::F::zero() - blk::F{(uint64_t)-cell} : blk::F{(uint64_t)cell};
      }
      for (uint64_t j = 0; j < P.pads(); j++) message[P.pad_entry(j)] = blk::F{hf_mask_ref::pad(w.key_bytes, (uint32_t)r, j, hf_mask_ref::FULL)};
      for (unsigned i = 0; i < shape.answers; i++) rec.y.push_back(blk::record_value(shape, seeds[i], i, message.data()));
    }
    return rec;
  }
};

struct Outcome {
  bool made = false, accepted = false;
  std::string why;
  std::vector<uint8_t> bytes;
  unsigned flags = 0;
  int walks[2] = {0, 0};
  double seconds = 0;
  uint64_t planned = 0;  // the plan's upper bound on the proof's bytes
};

// judged_under: the leaf table whose root the verifier gets: `further`, except in K5.
// an_answer_changed: in the leaf table's record, the same for prover and verifier.
static Outcome run(Made &first, Made &further, const Made &judged_under, int s, bool on_the_card, bool an_answer_changed = false) {
  using namespace blk;
  const HfParams &P = first.w.P;
  Outcome out;
  Setting set = setting_goldilocks();  // the plan's errors are computed over the programs' field
  set.target = (unsigned)P.fhb;
  const LinkLines lines = lines_of(P);
  HfLinkSetup setup;
  const Digest configuration = hf_digest(P.configuration);  // both tables' trees start from it; LINK compares the keepers' and readers' with the chain's
  TableSize table = link_table(set, lines, 0, 0);
  // size_recorded_table() (blocks/numbers/blocks.h) derives the distances, for batches of four
  // requests, as in hf_lin_side_test.cu.
  if (!size_recorded_table(set, table, 4, (unsigned)P.lk)) return out.why = "no distances serve this toy table", out;
  hf_link_setup(P, configuration, set, lines, table, setup);
  if (!setup.refused.empty()) return out.why = "no such parameters: " + setup.refused, out;
  const LinkSized &x = setup.sized;

  HfOpen mode[2];  // only the HfShown part, set below, is read: nothing called here reads `once`
  mode[0].once = mode[1].once = true;
  static_cast<HfShown &>(mode[0]) = setup.shown_first;
  static_cast<HfShown &>(mode[1]) = setup.shown_further;
  rf::HfSide side[2] = {side_of(first.w, mode[0], first.heap, s, nullptr, 0), side_of(further.w, mode[1], further.heap, s, nullptr, 0)};
  HfSeams seams[2] = {rf::hf_side_seams(side[0]), rf::hf_side_seams(side[1])};
  // Without HfCard's `store` and `adopt`, folded rows pass through host vectors:
  // HfKeeper::fold_table() is not tested.
  const HfCard card{[](size_t bytes) { void *p = nullptr; CK((cudaMalloc)(&p, bytes)); return p; }, [](void *p) { CK((cudaFree)(p)); },
                    [](void *host, const void *from, size_t bytes) { CK(cudaMemcpy(host, from, bytes, cudaMemcpyDeviceToHost)); }};
  // `count`, the last argument given, allows any number of opened columns.
  IndexRecord copys_record = further.record(), judged_record = judged_under.record();
  if (an_answer_changed) copys_record.y[0] += E::one(), judged_record.y[0] += E::one();
  HfKeeper keeper(P, mode[0], seams[0], card, first.commitment(), first.record(), [](uint64_t) { return true; });
  HfKeeper copy(P, mode[1], seams[1], card, further.commitment(), copys_record, [](uint64_t) { return true; });
  TestCoins coins(7);
  EngineCpu cpu;
  EngineGpu gpu;
  Engine &engine = on_the_card ? (Engine &)gpu : (Engine &)cpu;
  Writer proof;
  (void)take_the_flags();
  const int before[2] = {first.w.walks, further.w.walks};
  const auto t0 = std::chrono::steady_clock::now();
  if (!hf_make_link(setup, keeper, copy, engine, coins, nullptr, proof)) return out.why = "a keeper does not serve", out;
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.flags = take_the_flags(), out.walks[0] = first.w.walks - before[0], out.walks[1] = further.w.walks - before[1];
  out.made = true, out.bytes = proof.bytes();

  // HfKeeper::open() writes the opened columns as the blocks/ library's opening of a stored tree,
  // which blk::IndexKey reads. The shape is the set-up's: a reader must name the run's distances
  // (IndexShape::distances), or the process ends. hf_link_setup() has checked that it fits P.
  // LINK's error bound holds only for an anchor that has passed its one-time test. No such test
  // runs here: on_a_tests_word() marks the anchor as tested.
  const IndexKey reader(setup.sized.S.table, configuration, hf_digest(first.root()), first.record());
  const IndexKey copy_reader(setup.sized.S.table, configuration, hf_digest(judged_under.root()), judged_record);
  const char *why = "";
  out.accepted = hf_check_link(setup, blk::TableRead::on_a_tests_word(reader, setup.plan->set), copy_reader, out.bytes.data(), out.bytes.size(), &why);
  out.why = why;
  out.planned = setup.plan->size_of(0).bytes;
  if (out.accepted)
    printf("  %s engine: %llu + %llu columns, %llu positions, %zu bytes (planned: at most %llu), each folded row %zu rows of 2^%u, %.3f s (untimed box)\n",
           on_the_card ? "GPU" : "CPU", (unsigned long long)x.S.columns_first, (unsigned long long)x.S.columns_further, (unsigned long long)x.linked.t,
           out.bytes.size(), (unsigned long long)setup.plan->size_of(0).bytes, x.S.rows, setup.codes.base[x.S.code].log_ell(), out.seconds);
  fflush(stdout);
  return out;
}

static void a_shape(const HfParams &P, const std::string &name, bool cpu_too) {
  printf("%s: R = %zu, l = %llu, i0 = %llu, nm = %llu, n = 2^%d, m0 = %zu, k = %zu\n", name.c_str(), P.R, (unsigned long long)P.l, (unsigned long long)P.i0,
         (unsigned long long)P.nm, P.ln, P.m0, P.k);
  const std::mt19937_64 rng(11);
  World w0(P, rng, false, false), w1(P, rng, true, false), w2(P, rng, true, true);
  const int s = P.lcl;
  Made first{w0, tree(w0, s)}, copy{w1, tree(w1, s)}, other{w2, tree(w2, s)};
  expect(first.commitment() != copy.commitment(), name + ": two makings of one W have two roots");

  const Outcome good = run(first, copy, copy, s, true);
  expect(good.made && good.accepted, name + ": K1: made and accepted (" + good.why + ")");
  expect(good.bytes.size() <= good.planned, name + ": K1: a link is no longer than planned (" + std::to_string(good.bytes.size()) + " bytes, planned at most " + std::to_string(good.planned) + ")");
  expect(good.walks[0] == 2 && good.walks[1] == 2 && !good.flags, name + ": K2: two walks for each table and no flag (" + std::to_string(good.walks[0]) + " + " +
                                                                      std::to_string(good.walks[1]) + " walks, flags " + std::to_string(good.flags) + ")");
  if (cpu_too) expect(run(first, copy, copy, s, false).bytes == good.bytes, name + ": K3: both engines write the same bytes");
  const Outcome bad = run(first, other, other, s, true);
  expect(bad.made && !bad.accepted && bad.why == blk::LINK_REFUSED_TEST && !bad.flags, name + ": K4: a copy with one cell changed is refused by the proximity test (" + bad.why + ")");
  const Outcome swapped = run(first, copy, other, s, true);
  expect(swapped.made && !swapped.accepted, name + ": K5: a link is refused under another copy's root");
  const Outcome answer = run(first, copy, copy, s, true, true);
  expect(answer.made && !answer.accepted, name + ": K6: a link is refused if an answer of the copy's record is changed");
}

int main(int argc, char **argv) {
  (void)blk::failures;
  std::vector<int> shapes;
  for (int i = 1; i < argc; i++) shapes.push_back(atoi(argv[i]));
  if (shapes.empty()) shapes = {1, 5, 7};
  for (int which : shapes) a_shape(HfParams::toy(which), "toy " + std::to_string(which), which < 7);
  printf(g_failed ? "hf_link_side_test: %d FAILED\n" : "hf_link_side_test: all passed\n", g_failed);
  return g_failed != 0;
}
