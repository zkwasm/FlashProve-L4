// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test on a GPU (`card` in identifiers) of hf_open_lin(), the part of a batch's opening of the
// weight table T that its requests share. hf_open_lin() runs the shared part of the protocol block
// LIN (blocks/iop/lin.h: one linear claim per request on T). LIN asks a keeper (`blk::IndexHolder`,
// its interface to T) to fold T's rows and to open columns. make_table() (T's tree), `HfKeeper`
// and its callbacks (`seams` in identifiers; prover/hash_commitment/hf_open_side.cuh) are those of
// the programs cp_commit and cp_prove. The verifier, hf_check_lin(), reads the tree through
// `blk::IndexKey`, so that openings of the GPU-built tree must pass check_opening() of the library
// (blocks/, namespace `blk`).
// Limits: random weights; toy shapes (arguments of HfParams::toy(), 0 to 9; another number is not
// rejected and gives shape 9; default 1 5 7, whose error target `fhb` is 16 bits; SUITE.md adds 9,
// at 131 bits); only S4 expects a rejection.
// W: the weights, m0 rows of k cells. M: the same cells arranged as T's R rows, before encoding.
//   S1  accepted, for a batch and as the committer's one-time test (a batch of one);
//   S2  each takes two passes over the weights (`walk`) and raises no failure flag;
//   S3  (at shapes below 7) the CPU engine (`blk::Engine`: the prover's heavy
//       computations, such as encoding and hashing), with the same private randomness (`coins` in
//       identifiers), writes the same bytes as the GPU engine;
//   S5  the same bytes when engine and keeper release GPU memory around the second
//       pass (`leaves`), so that the caller that supplies the weights can use it;
//   S6  (one-time test) the same bytes through cp_commit's `HfOnceTable`
//       (prover/hash_commitment/hf_keeper_once.cuh), which itself finds the trees to release;
//   S7  the library's hiding tests (blocks/tests/hiding/) use the CPU keeper `blk::IndexCpu`. Built
//       by cpu_twin() from M and the masks and salts of `hf_mask_ref` (a second implementation, on
//       the CPU, of the mask and salt generator: prover/hash_commitment/hf_mask_ref.h), it has T's
//       root and gives the same bytes (at shapes below 7). This does not show that masks and
//       salts are uniform or used once;
//   S4  (for a batch) one weight changed after commitment: the recomputed columns fail
//       the callbacks' self-check, which raises the flag OWN_TEST and gives zeros instead, and the
//       verifier rejects ("refused" in the test's failure message) before the proximity tests;
//   S8  T has a record (stored answers to samples of T's rows, masked by pads in each row's
//       message). At shapes below 7 it is that of the CPU keeper from cpu_twin(), with the pads of
//       `hf_mask_ref`; at the others blk::record_value() gives it, on the same pads. `HfKeeper` and
//       `HfOnceTable` are given it. S7's equal root then shows that make_table() puts these pads at
//       the entries where the library reads them, and its equal bytes that the folded row and the
//       opened columns contain them. S7's added negative control: a changed pad gives another root.
// Build with nvcc, from prover/: ./build.sh hc_unit_test tests/hf_lin_side_test
// Run: hf_lin_side_test [shape ..]   Last line "hf_lin_side_test: all passed", exit code 0; 1 if an
// expectation failed. A CUDA error or a failed internal check ends the process at once.

#define BLK_TESTED_TABLE_FOR_A_TEST  // enables `on_a_tests_word` (blocks/iop/foldeq.h): see `run`
#include "engine/engine_cpu.h"
#include "engine/engine_gpu.cuh"
#include "frame/lin_frame.h"
#include "iop/index_cpu.h"
#include "iop/tests/iopp_support.h"

#include "common/setting.h"
#include "weights_commitment/lent_memory.h"
#include "weights_commitment/common.cuh"
#include "hash_commitment/hf_commit_table.cuh"
#include "hash_commitment/hf_open_side.cuh"
#include "hash_commitment/hf_open_lin.h"
#include "hash_commitment/hf_keeper_once.cuh"
#include <chrono>
#include <cstring>
#include <random>
#include <string>
using namespace hf;
typedef uint64_t u64_;
static_assert(sizeof(::Ext) == sizeof(blk::E), "the keeper hands elements of E over as the GPU's: the same three words");

static int g_failed = 0;
static void expect(bool ok, const std::string &what) {
  if (ok) return;
  g_failed++;
  printf("FAILED: %s\n", what.c_str());
  fflush(stdout);
}
// Returns and clears the failure flags (prover/common/slot_flags.h) raised on the CPU and the GPU.
static unsigned take_the_flags() {
  CK(cudaDeviceSynchronize());
  return take_live_flags();
}
// M's rows in GPU memory, for make_table().
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
// The test fixture: random W in three layouts, and a mask key.
struct World {
  const HfParams &P;
  rf::LayW Y;
  std::vector<int8_t> W;     // W[y * k + c]
  std::vector<int8_t> cells; // M, rows at a stride
  size_t stride = 0;
  int8_t *panel = nullptr; // GPU: panel[c * pitch + y], all rows of W
  size_t pitch = 0;
  hf_key32 key{};
  uint8_t key_bytes[32];   // the key's second copy (HfHeld::key_again)
  uint8_t other_bytes[32]; // another key's
  blk::IndexRecord record; // the table's record
  int walks = 0;
  explicit World(const HfParams &P_, std::mt19937_64 &rng) : P(P_) {
    Y.lk = P.lk, Y.k = P.k, Y.m0 = Y.mW = P.m0;
    W.resize(P.m0 * P.k);
    for (auto &b : W) b = (int8_t)(rng() & 255);
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
    for (u32 &w : key.w) w = (u32)rng();
    memcpy(key_bytes, key.w, 32);
    hf_key32 other = key;
    other.w[3] ^= 1;
    memcpy(other_bytes, other.w, 32);
  }
  ~World() {
    CK((cudaFree)(panel));
  }
  // Passes W in up to three units (runs of whole rows, each from a multiple of 16), out of order.
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
// T's tree from make_table(): the digests from level s up.
// Under `World::key`, from which masks, salts and pads are derived.
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

// Private randomness that returns values set by the test: field elements and bytes, each in order.
class GivenCoins : public blk::Coins {
 public:
  std::vector<blk::F> elements;
  std::vector<blk::u8> octets;
  void fill(blk::F *out, size_t n) override {
    BLK_REQUIRE(n <= elements.size() - elements_used_, "more field elements are drawn than were laid down");
    std::copy(elements.begin() + elements_used_, elements.begin() + elements_used_ + n, out);
    elements_used_ += n;
  }
  void bytes(blk::u8 *out, size_t n) override {
    BLK_REQUIRE(n <= octets.size() - octets_used_, "more bytes are drawn than were laid down");
    std::copy(octets.begin() + octets_used_, octets.begin() + octets_used_ + n, out);
    octets_used_ += n;
  }
  bool used_up() const { return elements_used_ == elements.size() && octets_used_ == octets.size(); }

 private:
  size_t elements_used_ = 0, octets_used_ = 0;
};
// `blk::IndexCpu` draws all rows' random tails (T's masks), row by row, then all rows' pads, row by
// row, then a salt per tree leaf, in order. T's pads are the entries that mask the answers of its
// record (stored answers to samples of the table's rows). A row's message is its P.l data cells,
// then zeros (the gap), then its pads: P.i0 entries in all.
// spoilt, for S7's controls, changes 1: a mask; 2: a salt bit; 3: a cell; 4: a pad.
static std::unique_ptr<blk::IndexCpu> cpu_twin(const World &w, int spoilt = 0) {
  const HfParams &P = w.P;
  const blk::IndexShape s = hf_index_shape(P);
  const uint8_t *key = w.key_bytes;
  blk::Table rows(1, s.rows(), s.plain());  // the data cells, then the gap's zeros
  for (size_t r = 0; r < P.R; r++)
    for (size_t i = 0; i < P.l; i++) {
      const int cell = w.cells[r * w.stride + i];
      rows.set(r, i, blk::E::from(cell < 0 ? blk::F::zero() - blk::F{(uint64_t)-cell} : blk::F{(uint64_t)cell}));
    }
  GivenCoins coins;
  for (size_t r = 0; r < P.R; r++)
    for (uint64_t j = 0; j < P.nm; j++) coins.elements.push_back(blk::F{hf_mask_ref::mask(key, (uint32_t)r, j, hf_mask_ref::FULL)});
  for (size_t r = 0; r < P.R; r++)
    for (uint64_t j = 0; j < P.pads(); j++) coins.elements.push_back(blk::F{hf_mask_ref::pad(key, (uint32_t)r, j, hf_mask_ref::FULL)});
  for (uint64_t I = 0; I < ((uint64_t)1 << P.ln); I++)
    for (int sigma = 0; sigma < 5; sigma++) {
      const uint64_t word = hf_mask_ref::salt(key, I, sigma);
      for (int b = 0; b < 8; b++) coins.octets.push_back((blk::u8)(word >> 8 * b));
    }
  if (spoilt == 1) coins.elements[P.R * P.nm / 2] += blk::F::one();
  if (spoilt == 4) coins.elements[P.R * P.nm + P.R * P.pads() / 2] += blk::F::one();
  if (spoilt == 2) coins.octets[coins.octets.size() / 2] ^= 1;
  if (spoilt == 3) rows.set(P.R / 2, P.l / 2, rows.at(P.R / 2, P.l / 2) + blk::E::one());
  std::unique_ptr<blk::IndexCpu> twin(new blk::IndexCpu(s, rows, coins, hf_digest(P.configuration)));
  expect(coins.used_up(), "S7: the CPU keeper has drawn all masks, pads and salts, and nothing else");
  return twin;
}

// T as the library's `blk::TableSize`. blk::size_recorded_table() (blocks/numbers/blocks.h) derives
// the distances, for batches of four requests, as in lines_of().
static blk::TableSize table_of(const hf::HfTableParams &P, const blk::Setting &set) {
  blk::TableSize T;
  T.code.name = "table", T.code.log_n = (unsigned)P.ln, T.code.ell = P.i0, T.code.r = P.nm;
  T.v = (unsigned)P.b, T.answers = set.samples;
  BLK_REQUIRE(blk::size_recorded_table(set, T, 4, (unsigned)P.lk), "no distances serve this toy table");
  return T;
}
// The test's choices for LIN at a toy T: batch size, and the codes and proximity-test stages of
// each group (oracle messages tested together: `folded`, `omega`, `own`).
// Omega: LIN's oracle message with one entry per row of W; a short one gets a longer code.
static blk::LinLines lines_of(const hf::HfTableParams &P, bool once) {
  const unsigned longer = P.m0 < 100 ? 6 : 5;  // a short Omega's random tail outweighs its message
  blk::LinLines L;
  L.k_lo = (unsigned)P.lcl, L.matrix_rows = P.m0;
  L.requests = once ? 1 : 4, L.one_time = once;
  const unsigned lu = blk::ceil_log2(P.KD) - 2, lo = std::max(blk::ceil_log2(std::max<u64_>(P.m0, P.k)) - 1, 3u);
  L.folded.log_ell = lu, L.folded.log_n = lu + 3;
  L.omega.log_ell = L.own.log_ell = lo, L.omega.log_n = L.own.log_n = lo + longer;
  L.folded.mask_log_ell = L.omega.mask_log_ell = L.own.mask_log_ell = 4;
  L.folded.mask_log_n = L.omega.mask_log_n = L.own.mask_log_n = std::min(lu + 3, lo + longer);
  const unsigned last = std::max(lu + 3, lo + longer) + 1;
  L.folded.stages = {blk::StageChoice{2, 1, lu + 2, lu + 2}, blk::StageChoice{0, 0, last, last}};
  L.omega.stages = L.own.stages = {blk::StageChoice{2, 1, lo + longer, lo + longer}, blk::StageChoice{0, 0, last, last}};
  return L;
}


struct Outcome {
  bool made = false, accepted = false;
  std::string why;
  std::vector<uint8_t> bytes;
  unsigned flags = 0;
  int walks = 0;
  double seconds = 0;
};

// leaves: for the second pass the engine's trees leave the GPU and the keeper frees the folded row.
// committers: use `HfOnceTable`'s keeper. instead: if not null, use this keeper of the same table.
// For a batch, blk::TableRead::on_a_tests_word() asserts that T has passed its one-time test, as
// the batch's error bound requires; nothing here checks it. The callbacks passed as `count` to
// `HfKeeper` and `HfOnceTable` return true: the limit on a table's opened columns is not tested.
static Outcome run(World &w, const std::vector<u32> &heap, int s, bool once, bool on_the_card, bool leaves = false, bool committers = false, blk::IndexHolder *instead = nullptr) {
  using namespace blk;
  const HfParams &P = w.P;
  Outcome out;
  Setting set = setting_of_this_build();
  set.target = (unsigned)P.fhb;
  HfLinSetup setup;
  hf_lin_setup(P, set, lines_of(P, once), table_of(P, set), setup);
  if (!setup.refused.empty()) return out.why = "no such parameters: " + setup.refused, out;
  const LinSized &x = setup.sized;
  const Codes &codes = setup.codes;
  const u32 *root = heap.data() + DW;  // node 1 of the kept heap
  const std::vector<uint8_t> commitment((const uint8_t *)root, (const uint8_t *)root + 4 * DW);
  Digest some{};
  some.w[0] = 9;
  const LinBindingRoot binding_root = LinBindingRoot::made_up_without_a_tree(some, setup.sized.lines.requests);  // in place of the batch's binding tree (a Merkle tree over digests of the requests' first parts): one place per request

  // This test's calls read only the `HfShown` part of `mode`: the column count, the height c of the
  // top layer of T's tree (`HfSeams::show` gives the keeper the 2^c digests at depth c once, for
  // all columns) and derived sizes. None reads `once`.
  HfOpen mode;
  mode.once = once;
  static_cast<HfShown &>(mode) = setup.shown;
  rf::HfSide side = side_of(w, mode, heap, s, nullptr, 0);
  HfSeams seams = rf::hf_side_seams(side);
  HfCard card{[](size_t bytes) { void *p = nullptr; CK((cudaMalloc)(&p, bytes)); return p; }, [](void *p) { CK((cudaFree)(p)); },
              [](void *host, const void *from, size_t bytes) { CK(cudaMemcpy(host, from, bytes, cudaMemcpyDeviceToHost)); }, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  EngineCpu cpu;
  EngineGpu gpu;
  if (on_the_card) {  // the engine adopts the folded row in GPU memory
    card.store = [](void *to, const void *host, size_t bytes) { CK(cudaMemcpy(to, host, bytes, cudaMemcpyHostToDevice)); };
    card.adopt = [&gpu](const void *cells, size_t rows, size_t len) { CK(cudaDeviceSynchronize()); return gpu.adopt(cells, Cell::WORD, E_DEGREE, rows, len); };
  }
  HfLinOpened opened;
  int left = 0;
  if (leaves) {
    card.leave = [&] {
      for (CommitId c : opened.context->commit)
        if (c != NO_ID) gpu.leave_card(c), left++;
    };
    card.back = [&] {
      for (CommitId c : opened.context->commit)
        if (c != NO_ID) gpu.back_on_card(c);
    };
    card.moved = [&gpu](TableId t, const void *cells) { CK(cudaDeviceSynchronize()); gpu.moved(t, cells); };
  }
  HfKeeper by_hand(P, mode, seams, card, commitment, w.record, [](uint64_t) { return true; });
  const Digest start = hf_digest(P.configuration);  // a toy configuration digest (`HfParams::toy`)
  std::unique_ptr<HfOnceTable> table;
  if (committers)
    table.reset(new HfOnceTable(P, w.Y, (uint64_t)setup.shown.tcols, w.walk(), side.held, 0, false, commitment.data(), commitment.size(), w.record, [](uint64_t) { return true; }, gpu));
  int over = 0, walks_when_over = -1;
  if (committers) table->when_the_walks_are_over([&] { over++, walks_when_over = w.walks; });
  IndexHolder &keeper = instead ? *instead : committers ? table->holder() : by_hand;
  TestCoins coins(7);
  Engine &engine = on_the_card ? (Engine &)gpu : (Engine &)cpu;
  (void)take_the_flags();
  const int walks_before = w.walks;
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(keeper, x.S, begun)) return out.why = "the keeper does not serve", out;
  hf_open_lin(setup, once ? blk::TableTest::ONE_TIME : blk::TableTest::LATER, keeper, std::move(begun), binding_root, hf_lin_bind_alone(setup, IndexName{keeper.commitment(), keeper.record()}, start), engine, coins, nullptr, opened);
  if (leaves) expect(left > 0, "the folded row's tree has left the card for the second walk");
  hf_open_lin_end(opened);
  if (committers) expect(table->failed() == OWN_GOOD, "the committer's holder: no own test of the seams has failed");
  if (committers) expect(over == 1 && walks_when_over - walks_before == 2, "the committer's holder says once, behind the second walk, that the walks are over");
  table.reset();
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.flags = take_the_flags(), out.walks = w.walks - walks_before;
  out.made = true, out.bytes = opened.bytes.bytes();

  const IndexKey reader(setup.sized.S.fold.index, start, hf_digest(root), w.record);  // the library's reader of a stored tree, with the run's shape and its distances (`IndexShape::distances`); `hf_lin_setup` has checked that it fits T
  HfLinChecked checked;
  if (!hf_check_lin(setup, once ? blk::TableTest::ONE_TIME : blk::TableTest::LATER, once ? blk::TableRead(reader) : blk::TableRead::on_a_tests_word(reader, setup.plan->set), binding_root, hf_lin_bind_alone(setup, IndexName{reader.commitment(), reader.record()}, start), out.bytes.data(), out.bytes.size(), checked)) return out.why = checked.why, out;
  out.accepted = true;
  printf("  %s, %s engine: %llu columns, %zu bytes, G_u %zu rows of 2^%u, %.3f s (untimed box)\n", once ? "one-time" : "a batch",
         on_the_card ? "GPU" : "CPU", (unsigned long long)x.S.fold.columns, out.bytes.size(), x.S.fold.u.main_rows, codes.base[x.S.fold.u.code].log_ell(), out.seconds);
  fflush(stdout);
  return out;
}

// T's record by blk::record_value(), at the seeds derived from T's name: S8 at shapes where
// `blk::IndexCpu` is too large.
static blk::IndexRecord record_of(const World &w, const u32 *root) {
  const HfParams &P = w.P;
  const blk::IndexShape shape = hf_index_shape(P);
  const std::vector<blk::SampleSeed> seeds = blk::record_seeds(shape, blk::index_name(hf_digest(root), shape, hf_digest(P.configuration)));
  blk::IndexRecord record;
  std::vector<blk::F> message(P.i0);
  for (size_t r = 0; r < P.R; r++) {
    std::fill(message.begin(), message.end(), blk::F::zero());
    for (size_t i = 0; i < P.l; i++) {
      const int cell = w.cells[r * w.stride + i];
      message[i] = cell < 0 ? blk::F::zero() - blk::F{(uint64_t)-cell} : blk::F{(uint64_t)cell};
    }
    for (uint64_t j = 0; j < P.pads(); j++) message[P.pad_entry(j)] = blk::F{hf_mask_ref::pad(w.key_bytes, (uint32_t)r, j, hf_mask_ref::FULL)};
    for (unsigned i = 0; i < shape.answers; i++) record.y.push_back(blk::record_value(shape, seeds[i], i, message.data()));
  }
  return record;
}

static void a_shape(const HfParams &P, const std::string &name, bool cpu_too) {
  printf("%s: R = %zu, l = %llu, i0 = %llu, nm = %llu, n = 2^%d, m0 = %zu, k = %zu\n", name.c_str(), P.R, (unsigned long long)P.l, (unsigned long long)P.i0,
         (unsigned long long)P.nm, P.ln, P.m0, P.k);
  std::mt19937_64 rng(11);
  World w(P, rng);
  const int s = P.lcl;
  {
    const std::vector<u32> heap = tree(w, s);
    w.record = cpu_too ? cpu_twin(w)->record() : record_of(w, heap.data() + DW);
    for (int once = 0; once < 2; once++) {
      const std::string what = name + (once ? ", one-time" : ", a batch");
      const Outcome good = run(w, heap, s, once, true);
      expect(good.made && good.accepted, what + ": S1: made and accepted (" + good.why + ")");
      expect(good.walks == 2 && !good.flags, what + ": S2: two walks and no flag (" + std::to_string(good.walks) + " walks, flags " + std::to_string(good.flags) + ")");
      if (cpu_too) expect(run(w, heap, s, once, false).bytes == good.bytes, what + ": S3: both engines write the same bytes");
      expect(run(w, heap, s, once, true, true).bytes == good.bytes, what + ": S5: the same bytes when all GPU memory of the opening is given up around the second walk");
      if (once) expect(run(w, heap, s, once, true, false, true).bytes == good.bytes, what + ": S6: the same bytes through the committer's holder");
      if (cpu_too) {  // `blk::IndexCpu` stores every row's codeword, so only at the smaller shapes
        const std::unique_ptr<blk::IndexCpu> twin = cpu_twin(w);
        expect(twin->root() == hf_digest(heap.data() + DW), what + ": S7: the CPU keeper of the same rows, masks and salts has T's root");
        expect(run(w, heap, s, once, true, false, false, twin.get()).bytes == good.bytes, what + ": S7: the same bytes with the CPU keeper in the real one's place");
        if (!once)  // negative controls: one changed mask, salt bit, cell or pad gives another root
          for (int spoilt = 1; spoilt <= 4; spoilt++) expect(!(cpu_twin(w, spoilt)->root() == hf_digest(heap.data() + DW)), what + ": S7: control " + std::to_string(spoilt) + ": a spoilt twin has another root");
        printf("  S7 played: the CPU keeper of the same rows, masks and salts\n");
      }
    }
    int8_t was, other;  // S4
    CK(cudaMemcpy(&was, w.panel + 3 * w.pitch + 5, 1, cudaMemcpyDeviceToHost));
    other = (int8_t)(was ^ 1);
    CK(cudaMemcpy(w.panel + 3 * w.pitch + 5, &other, 1, cudaMemcpyHostToDevice));
    const Outcome bad = run(w, heap, s, false, true);
    expect(bad.made && !bad.accepted && bad.why.rfind("before the tests", 0) == 0, name + ": S4: a changed cell of W is refused at T's tree (" + bad.why + ")");
    expect((bad.flags & OWN_TEST) != 0, name + ": S4: the side's own test raises its flag");
    CK(cudaMemcpy(w.panel + 3 * w.pitch + 5, &was, 1, cudaMemcpyHostToDevice));
  }
}

int main(int argc, char **argv) {
  (void)blk::failures;
  std::vector<int> shapes;
  for (int i = 1; i < argc; i++) shapes.push_back(atoi(argv[i]));
  if (shapes.empty()) shapes = {1, 5, 7};
  for (int which : shapes) a_shape(HfParams::toy(which), "toy " + std::to_string(which), which < 7);
  printf(g_failed ? "hf_lin_side_test: %d FAILED\n" : "hf_lin_side_test: all passed\n", g_failed);
  return g_failed != 0;
}
