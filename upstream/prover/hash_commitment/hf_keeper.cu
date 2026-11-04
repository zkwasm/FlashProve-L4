// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The table holder (`keeper` in names and messages): implements hf::HfKeeperUnit, through which the
// prover program cp_prove opens a leaf table T (`leaf` in names: the weight matrix W, encoded with
// masks, under a Merkle tree) for a batch. An opening is the protocol block LIN, a batch's test on
// T. Contracts and order of calls: prover/hash_commitment/hf_keeper_unit.h. The opening's thread
// runs hf_open_lin() and blocks in walk() for each pass over W (`walk` in names and messages),
// while the service's thread adds W's units (ranges of its rows) through the add_* calls.
//
// Hidden values: the mask seed, the masks, u (T's rows folded into one), omega (one sum for each
// row of W) and the intermediate results of the opened columns are hidden as W is. This file writes
// them to no file or log. The GPU copies that this file makes, except the seed's (see mask_seed),
// are in memory that the caller lends (see plan()); leaf_end() zeroes it, the seed and u, also
// after a failed opening. LIN also copies omega into host vectors of its own, which neither it nor
// this file zeroes (lin_shared_prove_on_shape() in blocks/iop/lin.h); in cp_prove, operator delete
// zeroes every block that it frees (prover/common/zeroing_heap.h). LIN puts omega into the caller's
// engine (blk::EngineGpu) and has it commit to the folded row (u). For both, the engine uses its
// GPU memory pool, which is not zeroed (prover/engine/device.cuh).
// The tree's stored nodes, which are digests, are not zeroed (see `heap`).
// The pads of a table are hidden too. make_u and make_columns
// (prover/hash_commitment/hf_open_side.cuh) each derive them in host memory, make_columns with a
// copy in the lent GPU memory, and erase them before they return.
//
// Static state includes the pools of lent memory, the live failure flags, the tree hash's scratch
// buffers and the global D (the parameter file's line `D`). So hf_keeper_unit() rejects a second
// object while one is alive; hf_keeper_unit.h explains how to link this object file several times.
#include "iop/lin.h"

#include "common/setting.h"
#ifdef TEST_HOOKS
#error "TEST_HOOKS is only for test programs with a main of their own: never build the weight table's keeper with it"
#endif
// (The switch changes code that this file includes: prover/common/hash.h and
// prover/common/host_common.h.)
// A key's byte order: RFC 8439, section 2.3 (cited in prover/hash_commitment/hc_chacha.cuh).
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "a key's 32 bytes are its eight words, the lowest byte first"); // `leaf_begin` copies the mask seed's bytes as words
#include "weights_commitment/lent_memory.h"
#include "weights_commitment/common.cuh"
#include "weights_commitment/arguments.h"
#include "weights_commitment/in_pieces.h"
#include "weights_commitment/unit_folds.cuh"
#include "common/test_words.h"
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hf_seed.h"  // hf_key32
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_own.cuh"  // the self-checks (`own test` in names)
#include "hash_commitment/hf_shape.cuh"
#include "hash_commitment/hf_walk.h"
#include "hash_commitment/hf_commit_api.h"  // hf::HfKept
#include "hash_commitment/hf_tree.cuh"      // hf::kept_rebuild()
#include "hash_commitment/hf_passes.cuh"
#include "hash_commitment/hf_code.cuh"
#include "hash_commitment/hf_open_side.cuh"
#include "hash_commitment/hf_keeper_unit.h"
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>

static_assert(sizeof(::Ext) == sizeof(blk::E), "the keeper hands elements of E over as the GPU's: the same three words");

namespace lent {
// Marks every buffer taken from lent memory (the pool `borrowed`) as returned and zeroes that
// memory, which stays lent. Exits with code 5 if the memory cannot be zeroed.
static void give_all_borrowed() {
  {
    std::lock_guard<std::mutex> lock(guard);
    for (auto it = live.begin(); it != live.end();) it = it->second.first == &borrowed ? live.erase(it) : std::next(it);
  }
  cudaError_t e = cudaDeviceSynchronize();  // first: no earlier launch on any stream writes after the zeroes
  for (const Piece &p : borrowed.pieces)
    if (e == cudaSuccess) e = cudaMemsetAsync(p.base, 0, p.bytes, cudaStreamPerThread);
  if (e == cudaSuccess) e = cudaStreamSynchronize(cudaStreamPerThread);
  if (e != cudaSuccess) {
    fprintf(stderr, "the keeper of the weight table: GPU memory cannot be zeroed (%s); the process ends\n", cudaGetErrorString(e));
    exit(5);
  }
  std::lock_guard<std::mutex> lock(guard);
  borrowed.forget();
}
}  // namespace lent

namespace rf {
static_assert(sizeof(hf_key32) == 32, "the seed's type holds the seed's bytes");
static_assert(hc_seed_is_no<::hz::CoinK>() && sizeof(hf_key32) == 4 * HC_SEED_WORDS &&
                  !std::is_convertible<hf_key32, ::hz::CoinK>::value && !std::is_convertible<::hz::CoinK, hf_key32>::value,
              "the table's key is no coin key, and a coin key no table's key, for the compiler too");
constexpr size_t SEED_BYTES = 32;
struct Abandoned {};
static bool a_unit_lives = false;

// Returns zeroed, pinned host memory, or null after a line on stderr. madvise() asks that it stay
// out of core dumps and forked children; its result is not checked.
// The memory starts at a multiple of 2 MiB, a huge page's size; no huge pages are asked for.
static void *pinned_room(size_t bytes, const char *what) {
  const size_t huge = (size_t)2 << 20, room = (bytes + huge - 1) / huge * huge;
  char *p = (char *)aligned_alloc(huge, room);
  if (p) {
    madvise(p, room, MADV_DONTDUMP);
    madvise(p, room, MADV_DONTFORK);
    zero_in_pieces(p, room);
  }
  if (p && cudaHostRegister(p, room, cudaHostRegisterDefault) == cudaSuccess) return p;
  fprintf(stderr, "the keeper of the weight table: %.3f GB of host memory for %s cannot be had or pinned\n", room / 1e9, what);
  free(p);
  return nullptr;
}
static void pinned_back(void *p) {
  if (!p) return;
  cudaHostUnregister(p);
  free(p);
}

class Unit : public ::hf::HfKeeperUnit {
 public:
  ~Unit() override {
    if (!forgotten) forget();
    pinned_back(mask_seed), pinned_back(heap), pinned_back(u_host);
    a_unit_lives = false;
  }
  int setup(const ::hf::HfKeeperSetup &s);

  const ::hf::HfParams &parameters() const override { return W.P; }
  int walks() const override { return walks_of_an_opening; }
  std::vector<HfKeeperUnit::Unit> units() const override {
    std::vector<HfKeeperUnit::Unit> list;
    for (const UnitW &u : W.Y.U) list.push_back({u.X, u.l, u.rows});
    return list;
  }

  void lend(void *gpu_memory, size_t bytes) override { lent::lend(gpu_memory, bytes); }
  void take_back() override {
    leave_the_walk_room(true);
    lent::take_back();
  }
  size_t slab_bytes() const override { return slab_bytes_; }
  size_t walk_room_bytes() const override { return walk_room_bytes_; }
  int lend_slab(void *memory, size_t bytes) override;
  size_t lent_peak_bytes() const override { return lent_peak_bytes_; }
  size_t lent_longest_bytes() const override { return lent_longest_bytes_; }

  int plan(const ::hf::HfShown &shown, size_t folded_row_bytes) override;
  int leaf_begin(const uint8_t *table_root, size_t root_bytes, const blk::IndexRecord &record, const uint8_t *seed, const char *nodes, ::hf::HfKeeperHooks hooks) override;
  blk::IndexHolder &holder() override {
    BLK_REQUIRE(keeper != nullptr, "the keeper of the weight table: no leaf is at hand");
    return *keeper;
  }
  void leaf_end() override;

  bool next_walk() override;
  int add_blocks(int X, int layer, const void *blocks, int type, const float *gains, float two_e, cudaStream_t stream) override;
  int add_planes(int X, int layer, int sources, const int8_t *const *low, const int8_t *const *high, const size_t *pitch, cudaStream_t stream) override;
  int add_embedding(const int8_t *bytes, cudaStream_t stream) override { return add_emb_head(12, bytes, nullptr, stream); }
  int add_head(const short *weights, cudaStream_t stream) override { return add_emb_head(13, nullptr, weights, stream); }
  int add_rest(cudaStream_t stream) override;
  void go_on() override;
  void opening_is_over() override;
  void abandon() override;

  const char *keeps_fault() const override {
    return nodes_end != NODES_GOOD ? HF_STOP_NODES_ROOT : side && side->failed == ::hf::OWN_NODE ? ::hf::own_stop(::hf::OWN_NODE) : nullptr;
  }
  const char *own_test_fault() const override {
    if (!side) return nullptr;
    if (side->fold_failed != ::hf::OWN_GOOD) return side->fold_failed == ::hf::OWN_KEY_FILE ? HF_OWN_TEST_KEY_COPY : HF_OWN_TEST_FOLD;
    if (side->failed == ::hf::OWN_GOOD || side->failed == ::hf::OWN_NODE) return nullptr; // OWN_NODE depends on the weights: `keeps_fault` reports it
    return side->failed == ::hf::OWN_KEY_FILE ? HF_OWN_TEST_KEY_COPY : side->failed == ::hf::OWN_SALTS ? HF_OWN_TEST_SALT : side->failed == ::hf::OWN_PADS ? HF_OWN_TEST_PADS : HF_OWN_TEST_MASKS; // (nothing sets `failed` to another code, so the last name is never returned)
  }
  void loader_seconds(double *read, double *rebuild, double *waited) const override {
    if (read) *read = loader_read;
    if (rebuild) *rebuild = loader_rebuild;
    if (waited) *waited = loader_waited;
  }
  int forget() override;
  unsigned take_flags() override { return take_live_flags() | (nodes_end != NODES_GOOD ? (unsigned)OWN_TEST : 0u); }

 private:
  PolC policy;
  HfShape W;  // the shape, the layout of W and the commitment's parameters (W.P)
  bool ready = false, planned = false, generator = false, test_run = false, forgotten = false;

  // ---- The batch's leaf table: set by leaf_begin(); setup() allocates the pinned buffers.
  // leaf_end() zeroes both copies of the seed, the root, u, the slab and the lent memory (see
  // plan()). Nothing zeroes the stored nodes, forget() included; the destructor frees them as they
  // are. They are digests of T's Merkle tree, which hashes each column with a salt. In cp_prove
  // they come from a file in the secret key directory (`isk`), and an opening's Merkle paths
  // publish some of them (docs/limits-all-rows.tsv, row H11.6.1.1).
  // Kernels get the seed as a launch argument (hf::key_for_a_launch()), which nothing here zeroes.
  // Neither forget() nor the destructor joins `loader`: after leaf_begin(), an object destroyed
  // before the second walk or leaf_end() has joined it ends the process (std::terminate).
  u32 root[DW] = {0};  // the table's Merkle root (RT_F), which the stored nodes must hash to
  hf_key32 *mask_seed = nullptr;   // pinned host memory; yields the table's masks, salts and pads
  uint8_t seed_again[32] = {0};    // a second copy: self-checks recompute all three from it
  u32 *heap = nullptr;             // pinned: the tree's stored nodes, from level kept.s up
  ::hf::HfKept kept;
  std::thread loader;              // runs `load_the_nodes`; joined by `join_the_loader`
  std::string nodes_file;
  enum NodesEnd { NODES_GOOD, NODES_FILE, NODES_ROOT };
  NodesEnd nodes_end = NODES_GOOD;
  bool nodes_end_told = false;
  double loader_read = 0, loader_rebuild = 0, loader_waited = 0;
  ::Ext *u_host = nullptr;  // buffer for u, in pinned host memory
  size_t u_elements = 0;
  bool u_written = false;
  ::hf::HfOpen mode;  // the opening's shape as `side` reads it; set by `plan`
  // `seams`: callbacks that compute on W, with their state in `side`. `keeper`: T as the protocol's
  // blk::IndexHolder, built on them.
  std::unique_ptr<HfSide> side;
  std::unique_ptr<HfSeams> seams;
  std::unique_ptr<::hf::HfKeeper> keeper;

  // ---- Memory
  i8 *scratch = nullptr;  // in lent memory: one unit's cells, or those of a range of its rows
  size_t scratch_bytes = 0, slab_bytes_ = 0, walk_room_bytes_ = 0, lent_peak_bytes_ = 0, lent_longest_bytes_ = 0;
  bool in_the_walk_room = false;
  cublasHandle_t blas_of_the_service = nullptr;  // made in setup()

  // ---- A pass over the weights. FOLDING: walk() waits while the service's thread adds units.
  // AFTER_FOLD: go_on() waits until walk() has left this state, so that the next next_walk() sees
  // what follows. OVER: opening_is_over() was called.
  enum State { IDLE, TOWARDS_A_WALK, FOLDING, AFTER_FOLD, OVER };
  std::mutex mutex;
  std::condition_variable changed;
  State state = IDLE;
  const HfUnits *to = nullptr;  // while state is FOLDING: where a unit goes (hf_walk.h)
  vector<char> added;           // per unit: it has been added in this walk
  bool rest_done = false, abandoned = false;
  int walks_of_an_opening = 0, walks_made = 0;

  void load_the_nodes();
  void join_the_loader();
  void enter_the_walk_room();
  void leave_the_walk_room(bool zeroed_by_the_caller);
  void walk(const HfUnits &units);
  bool folding() const { return state == FOLDING && !forgotten; }
  int unit(int X, int layer) const;
  int refuse(const char *call, int X, int layer, int i) const;
  int add_emb_head(int X, const i8 *emb, const short *head, cudaStream_t stream);
  void zero_u(bool plainly);
};

// The loader thread's body: reads the stored nodes (level kept.s: 2^(W.P.ln - kept.s) digests) into
// the heap's second half and recomputes the levels above. The root must then equal `root`
// [hc-nodes-root] (square brackets: the name under which a failure is reported). An opening
// compares the nodes that it recomputes above its opened columns with the stored ones, so this
// check ties what the weights and the mask seed in use give there to the committed root. It makes
// no CUDA call and prints nothing: join_the_loader() prints a failure.
void Unit::load_the_nodes() {
  typedef std::chrono::steady_clock Clock;
  auto seconds = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
  const size_t half = kept.bytes(W.P) / 2;
  const Clock::time_point begun = Clock::now();
  // O_NONBLOCK: a named pipe that no one writes to opens at once and is rejected below, instead of
  // blocking the call.
  const int file = ::open(nodes_file.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOCTTY);
  struct stat is;
  const bool whole = file >= 0 && !fstat(file, &is) && S_ISREG(is.st_mode) && (size_t)is.st_size == half && read_in_pieces(file, 0, half, (char *)heap + half) &&
                     ends_at(file, half);
  if (file >= 0) ::close(file);
  const Clock::time_point read = Clock::now();
  if (whole) ::hf::kept_rebuild(heap, W.P.ln, kept.s, ::hf::TreeTags(W.P.configuration, HF_NUM_F, (u32)W.P.R));
  nodes_end = !whole ? NODES_FILE : memcmp(kept.root(), root, 4 * DW) ? NODES_ROOT : NODES_GOOD;
  loader_read = seconds(begun, read), loader_rebuild = seconds(read, Clock::now());
}
// Joins the loader's thread and prints a failed load once. A failure stops nothing here:
// take_flags() then includes OWN_TEST (a failure flag, prover/common/slot_flags.h), under which no
// message may leave the process.
void Unit::join_the_loader() {
  if (loader.joinable()) {
    const auto begun = std::chrono::steady_clock::now();
    loader.join();
    loader_waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
  }
  if (nodes_end == NODES_GOOD || nodes_end_told) return;
  nodes_end_told = true;
  fprintf(stderr, "%s: the keeper of the weight table: %s %s; this batch ends without a proof\n", HF_STOP_NODES_ROOT, nodes_file.c_str(),
          nodes_end == NODES_FILE ? "is absent, is no plain file, does not have the length of the stored level, or a read of it failed"
                                  : "does not hold the nodes under the table's root");
}

// Parses the parameter file's bytes (`policy` in identifiers) and allocates what lasts as long as
// this object. It opens no file. Returns 0, or 2 after a line on stderr.
int Unit::setup(const ::hf::HfKeeperSetup &s) {
  auto no = [](const char *why) { return fprintf(stderr, "the keeper of the weight table: %s\n", why), 2; };
  if (!s.configuration) return no("set-up requires the configuration's digest, from which the hashes of the table's tree start");
  generator = s.generator, test_run = s.test_run;
  if (!policy.take(s.policy)) return no("the parameter file is not admitted: the line above says why");
  const PolC &PO = policy;
  if (const std::string why = s8::set(PO.file.frame); !why.empty()) return no(("the parameter file is not admitted: " + why).c_str());
  if ((PO["D"] != 2048 && PO["D"] != 32) || PO.m.count("lKP") || PO.m.count("rv")) return no("the parameter file must have D 2048 or 32, and no line lKP or rv");
  if (int rc = W.from_policy(PO)) return rc;
  // The hashes of T's tree start from the configuration digest (docs/format/FORMAT.md, R4.9).
  memcpy(W.P.configuration, s.configuration, 4 * DW);
  const ::hf::HfParams &P = W.P;
  if (generator) W.model.seed = PO["seed_lo"] | ((u64)PO["seed_hi"] << 32);
  if (SERVING_OR(test_run, false)) return no("refused (serving-build): a serving build takes no test's run");
  if (!test_run)
    if (const char *word = test_word_in_the_environment())
      return fprintf(stderr, "the keeper of the weight table: the environment holds %s, a test variable; test variables are accepted only in a test's run\n", word), 2;
  // Self-check of the mask generator on known answers, before a key is read [own-test-generator].
  // In cp_prove a failure ends the program under that name, with a failed self-check's exit code
  // (prover/prover/ends.h): hf_keeper_unit() has installed the program's hook.
  BLK_REQUIRE_NAMED(::hf::generator_own_test(cudaStreamPerThread) == ::hf::OWN_GOOD, "own-test-generator", "own-test-generator: the masks' generator does not give its known answers on this machine");
  // The stored level (the tree level that a file of nodes contains) is the same for all leaf
  // tables; plan() checks an opening's top layer against it.
  if (s.keep_level < 0 || s.keep_level > P.ln) return fprintf(stderr, "%s: the stored level is %d\n", HF_STOP_KEEP_LEVEL, s.keep_level), 2;
  kept.s = s.keep_level;
  mask_seed = (hf_key32 *)pinned_room(SEED_BYTES, "the mask seed");
  heap = (u32 *)pinned_room(kept.bytes(P), "a leaf's stored nodes");
  if (!mask_seed || !heap) return 2;
  kept.heap = heap;
  // The tree hash's scratch buffers, allocated here once at fixed sizes. The idea is public:
  // keeping memory for a later stage: NVIDIA, CUDA Programming Guide,
  // "Stream-Ordered Memory Allocator", "Physical Page Caching Behavior",
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html,
  // stated there for that allocator's pool; here plain cudaMalloc, so cited for the idea only.
  ::treehash_scratch((size_t)48 << 20, (size_t)8 << 20);
  // `scratch` fits the largest unit whose cells (committed small integers) are written to it all
  // at once: a layer's unit given as byte planes. By a unit's kind X: the embedding and the head
  // (12 and 13) pass through it in ranges of rows; the routed experts' units (7, 8, 10) bypass it.
  for (auto &u : W.Y.U)
    if (u.X <= 11 && u.X != 7 && u.X != 8 && u.X != 10) scratch_bytes = std::max(scratch_bytes, (u.rows + 15) / 16 * 16 * W.k);
  scratch_bytes = std::max<size_t>(scratch_bytes, 16);
  u_elements = hf_side_u_elements(P);
  u_host = (::Ext *)pinned_room(u_elements * sizeof(::Ext), "the rows' combination u");
  if (!u_host) return 2;
  // The calling thread's cuBLAS handle is created here, not during an opening: creating it
  // allocates memory from the driver. This assumes (unchecked) that the caller is the service's
  // thread, which will make the add_* calls.
  blas_of_the_service = blas::blas(cudaStreamPerThread, nullptr, 0);
  added.assign(W.Y.U.size(), 0);
  ready = true;
  return 0;
}

// shown.tcols must be at least 1: hf_side_walks() divides by it before anything checks it.
int Unit::plan(const ::hf::HfShown &shown, size_t folded_row_bytes) {
  const ::hf::HfParams &P = W.P;
  if (!ready || planned) return fprintf(stderr, "the keeper of the weight table: plan() comes once, after a set-up that has succeeded\n"), 2;
  // An opening's top layer (all 2^c nodes at depth c) must lie within the stored levels.
  if (kept.s > P.ln - shown.c) return fprintf(stderr, "%s: the stored level %d lies above the top layer of an opening of %d columns\n", HF_STOP_KEEP_LEVEL, kept.s, shown.tcols), 2;
  // One opening by itself must not show more columns of a leaf table than a row has masks
  // [hc-mask-budget]. The caller counts the columns of all openings (HfKeeperHooks::count).
  if (P.nm < (u64)shown.tcols) return fprintf(stderr, "%s: an opening would show more columns of the table than a row has masks\n", HF_STOP_MASK_BUDGET), 2;
  mode = ::hf::HfOpen();  // `side` reads only its HfShown part (tcols, c) and `once`
  static_cast<::hf::HfShown &>(mode) = shown;
  // The GPU memory that the caller lends:
  // - The slab (lend_slab(); pool lent::own): what must last from one add_* call or pass to the
  //   next: a pass's buffers and sums (hf_passes.cuh), the masks' shares, the columns' buffer,
  //   omega.
  // - Lent memory (lend(); pool lent::borrowed), which the caller may take back at each pass.
  //   During a pass: the scratch buffer (`walk_room` in identifiers) and, with `generator`, the
  //   generated cells. Between and after passes: the folded row (u, arranged for the batch's GPU
  //   engine).
  // `spare` allows for the rounding of every buffer up to 512 bytes.
  const size_t spare = (size_t)1 << 20;
  walks_of_an_opening = hf_side_walks(P, mode);  // 2: one pass for u, one for all opened columns
  slab_bytes_ = hcp::up512(P.m0 * sizeof(::Ext)) + hf_side_slab_bytes(P, W.Y, mode, kept.s) + 2 * spare;
  walk_room_bytes_ = hcp::up512(scratch_bytes) + 2 * spare;
  size_t largest = 0;
  for (const UnitW &u : W.Y.U) largest = std::max(largest, (u.rows + 15) / 16 * 16 * W.k);
  const size_t in_a_walk = walk_room_bytes_ + (generator ? hcp::up512(largest) : 0);
  lent_longest_bytes_ = std::max(std::max(walk_room_bytes_, generator ? hcp::up512(largest) : 0), hcp::up512(folded_row_bytes)) + 4096;
  lent_peak_bytes_ = std::max(in_a_walk, hcp::up512(folded_row_bytes)) + spare;
  hf_side_warm_up(W.Y, mode);
  planned = true;
  printf("the keeper of the weight table: W has 2^%d columns, %zu rows, %zu units; the table has %zu rows and 2^%d columns; an opening has %d walks over the weights and shows %d "
         "columns\n",
         W.lk, W.m0, W.Y.U.size(), (size_t)P.R, P.ln, walks_of_an_opening, mode.tcols);
  return 0;
}

int Unit::lend_slab(void *memory, size_t bytes) {
  const char *why = forgotten                        ? "forget() has been called"
                    : !planned                       ? "plan() has not succeeded"
                    : state != IDLE && state != OVER && !(state == TOWARDS_A_WALK && walks_made == 0) ? "an opening's walks have begun"
                    : lent::slab_is_lent             ? "a slab has already been lent"
                    : !memory                        ? "the memory is null"
                    : (uintptr_t)memory % 512        ? "the memory must start at a multiple of 512 bytes"
                    : bytes < slab_bytes_            ? "fewer bytes than slab_bytes() returns"
                                                     : nullptr;
  if (why) return fprintf(stderr, "the keeper of the weight table: lend_slab: %s\n", why), 2;
  lent::zero_what_is_lent(memory, bytes);
  lent::own.add(memory, bytes);
  lent::slab_is_lent = true;
  return 0;
}

// The scratch buffer carries nothing from one add_* call to the next. So the add_* calls take it
// from lent memory here, unless it is already taken, and it is returned when that memory is taken
// back or a pass ends. cudaMalloc and cudaFree are macros here
// (prover/weights_commitment/lent_memory.h): from leaf_begin() to leaf_end() cudaMalloc takes from
// lent memory while lent::borrow is true, else from the slab, and cudaFree returns to either.
void Unit::enter_the_walk_room() {
  if (in_the_walk_room) return;
  const bool before_ = lent::borrow;
  lent::borrow = true;
  CK(cudaMalloc(&scratch, scratch_bytes));
  lent::borrow = before_;
  in_the_walk_room = true;
}
// The buffer contains copies of weights. zeroed_by_the_caller: take_back() zeroes all lent memory
// next. Otherwise the buffer is zeroed here, and the caller must have waited for the streams of the
// add_* calls.
void Unit::leave_the_walk_room(bool zeroed_by_the_caller) {
  if (!in_the_walk_room) return;
  if (!zeroed_by_the_caller) {
    CK(cudaMemsetAsync(scratch, 0, scratch_bytes, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
  }
  cudaFree(scratch);
  scratch = nullptr;
  in_the_walk_room = false;
}

// One pass over the weights, as the opening's callbacks see it (hf_walk.h), on the opening's
// thread. It lets the service's thread add the units of W through `units` and returns after
// add_rest() and go_on(). Throws Abandoned after abandon().
// The rule on memory (stated at HfWalk in hf_walk.h): at this call the opening must have nothing in
// lent memory, because the service's thread may now take all of it back to load the weights.
void Unit::walk(const HfUnits &units) {
  BLK_REQUIRE(walks_made < walks_of_an_opening, "the keeper of the weight table: an opening asks for more walks over the weights than walks() says");
  if (walks_made == 1) join_the_loader();  // what follows the second walk reads the stored nodes
  walks_made++;
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  lent::give_all_borrowed();  // the rule on memory: the service may take all lent memory back now, in order to bring the weights
  {
    std::unique_lock<std::mutex> lock(mutex);
    to = &units;
    std::fill(added.begin(), added.end(), 0);
    rest_done = false;
    state = FOLDING;
    changed.notify_all();
    changed.wait(lock, [&] { return state == AFTER_FOLD; });
    to = nullptr;
    state = TOWARDS_A_WALK;
    changed.notify_all();
  }
  // The service's thread waited for the GPU before it changed the state (go_on()).
  leave_the_walk_room(false);
  if (abandoned) throw Abandoned();
}

bool Unit::next_walk() {
  std::unique_lock<std::mutex> lock(mutex);
  changed.wait(lock, [&] { return state == FOLDING || state == OVER; });
  return state == FOLDING;
}
void Unit::go_on() {
  BLK_REQUIRE(folding() && (rest_done || abandoned), "the keeper of the weight table: go_on() outside a walk, or before add_rest has ended it");
  CK(cudaDeviceSynchronize());
  std::unique_lock<std::mutex> lock(mutex);
  state = AFTER_FOLD;
  changed.notify_all();
  changed.wait(lock, [&] { return state != AFTER_FOLD; });  // see `State`
}
void Unit::opening_is_over() {
  CK(cudaStreamSynchronize(cudaStreamPerThread));  // none of the opening's GPU work may still run when its thread ends
  CK(cudaGetLastError());                          // this thread's launch errors: only this thread can read them
  // Returns the folded row's lent memory, not zeroed: take_back() or leaf_end() zeroes it. The
  // opening now has nothing in lent memory, so take_back() may come. The engine no longer reads the
  // row: LIN released it right after the row's proximity test (lin_shared_tests_prove() in
  // blocks/iop/lin.h). After a failure the caller must not let the engine read the row again.
  if (keeper) keeper->row_back();
  std::unique_lock<std::mutex> lock(mutex);
  state = OVER;
  changed.notify_all();
}
void Unit::abandon() {
  BLK_REQUIRE(folding(), "the keeper of the weight table: abandon() outside a walk");
  abandoned = true;
  go_on();
}

int Unit::unit(int X, int layer) const {
  for (size_t i = 0; i < W.Y.U.size(); i++)
    if (W.Y.U[i].X == X && (W.Y.U[i].l == layer || X >= 12)) return (int)i;
  return -1;
}
int Unit::refuse(const char *call, int X, int layer, int i) const {
  fprintf(stderr, "the keeper of the weight table: %s, kind %d, layer %d: %s\n", call, X, layer,
          forgotten          ? "forget() has been called"
          : state != FOLDING ? "no opening is waiting for its units"
          : i < 0            ? "the layout has no such unit"
          : added[i]         ? "the unit has already been added in this walk"
                             : "the unit or an argument does not have the form that this call requires");
  return 2;
}
// Adds a routed unit from its quantized blocks.
int Unit::add_blocks(int X, int layer, const void *blocks, int type, const float *gains, float two_e, cudaStream_t stream) {
  const int i = unit(X, layer);
  const bool a_type = type == 0 || type == 1 || type == 4 || type == 5 || type == 6; // those of weights_blocks/decode.cuh
  if (i < 0 || !folding() || added[i] || !blocks || (uintptr_t)blocks % 16 || !HfPass::fits(W.Y.U[i]) || !a_type) return refuse("add_blocks", X, layer, i);
  enter_the_walk_room();
  to->blocks(W.Y.U[i], blocks, type, gains, two_e, stream);
  added[i] = 1;
  return 0;
}
// Adds a unit given as low and high byte planes.
int Unit::add_planes(int X, int layer, int sources, const int8_t *const *low, const int8_t *const *high, const size_t *pitch, cudaStream_t stream) {
  const int i = unit(X, layer);
  if (i < 0 || !folding() || added[i] || X == 7 || X == 8 || X == 10 || X > 11 || W.Y.U[i].nlw != 2 || sources != (X == 0 ? 2 : 1) || !low || !high || !pitch || !low[0] ||
      !high[0] || (sources == 2 && (!low[1] || !high[1])))
    return refuse("add_planes", X, layer, i);
  const UnitW &u = W.Y.U[i];
  const size_t ld = (u.rows + 15) / 16 * 16;
  if (ld * W.k > scratch_bytes) return refuse("add_planes", X, layer, i);
  enter_the_walk_room();
  ::g5::PS planes{};
  for (int s = 0; s < sources; s++) planes.w0[s] = low[s], planes.w1[s] = high[s], planes.pitch[s] = pitch[s];
  ::g5::k_cellsP<<<4096, 256, 0, stream>>>(W.um[i], W.model, planes, W.k, ld, scratch);
  to->cells(scratch, ld, u.rows, u.off, stream);
  added[i] = 1;
  return 0;
}
// Adds the embedding (X 12) or the head (X 13).
int Unit::add_emb_head(int X, const i8 *emb, const short *head, cudaStream_t stream) {
  const int i = unit(X, 0);
  const size_t slice = scratch_bytes / W.k / 16 * 16;
  if (i < 0 || !folding() || added[i] || (X == 12 ? !emb : !head) || !slice) return refuse(X == 12 ? "add_embedding" : "add_head", X, 0, i);
  enter_the_walk_room();
  const UnitW &u = W.Y.U[i];
  for (size_t y0 = 0; y0 < u.rows; y0 += slice) {
    const size_t rows = std::min(slice, u.rows - y0), ld = (rows + 15) / 16 * 16;
    ::g5::k_cellsEH<<<4096, 256, 0, stream>>>(W.um[i], (long long)W.sh.V, (long long)W.sh.d, emb, head, W.k, y0, rows, ld, scratch);
    to->cells(scratch, ld, rows, u.off + y0, stream);
  }
  added[i] = 1;
  return 0;
}
// Ends a pass. With `generator` (synthetic weights) it first generates, in lent memory, and adds
// every unit not yet added; otherwise it returns 2 if one is missing. Nothing here compares the
// units with the committed weights. A second pass over other weights is noticed only if it changes
// a node recomputed above an opened column [hc-own-node]; the first pass has no such comparison.
int Unit::add_rest(cudaStream_t stream) {
  if (!folding() || rest_done)
    return fprintf(stderr, "the keeper of the weight table: add_rest: %s\n", !folding() ? "no opening is waiting for its units" : "it has already run in this walk"), 2;
  size_t largest = 0;
  for (size_t i = 0; i < W.Y.U.size(); i++)
    if (!added[i]) {
      if (!generator)
        return fprintf(stderr, "the keeper of the weight table: unit %zu (kind %d, layer %d) has not been added: add it and call again, or abandon()\n", i, W.Y.U[i].X, W.Y.U[i].l), 2;
      largest = std::max(largest, (W.Y.U[i].rows + 15) / 16 * 16);
    }
  rest_done = true;
  if (largest) {
    enter_the_walk_room();
    const bool before = lent::borrow;
    lent::borrow = true;
    i8 *cells;
    CK(cudaMalloc(&cells, largest * W.k));
    for (size_t i = 0; i < W.Y.U.size(); i++)
      if (!added[i]) {
        const UnitW &u = W.Y.U[i];
        const size_t ld = (u.rows + 15) / 16 * 16;
        // Every row is below m0, so the kernel generates each one: RB is null and m is m0.
        ::g5::k_genWm<<<8192, 256, 0, stream>>>(W.umD, (int)W.um.size(), W.model, W.k, u.off, u.rows, ld, nullptr, W.m0, W.m0, 0, cells);
        to->cells(cells, ld, u.rows, u.off, stream);
        added[i] = 1;
      }
    CK(cudaMemsetAsync(cells, 0, largest * W.k, stream));  // weights do not stay in lent memory
    CK(cudaStreamSynchronize(stream));
    cudaFree(cells);
    lent::borrow = before;
  }
  CK(cudaStreamSynchronize(stream));  // the callback zeroes its weight copies once `walk` returns
  return 0;
}

// Takes a batch's leaf table, which is identified by its root only: shape and configuration are
// those that setup() was given. `record`: see prover/hash_commitment/hf_keeper_unit.h.
int Unit::leaf_begin(const uint8_t *table_root, size_t root_bytes, const blk::IndexRecord &record, const uint8_t *seed, const char *nodes, ::hf::HfKeeperHooks hooks) {
  auto no = [](const char *why) { return fprintf(stderr, "the keeper of the weight table: leaf_begin: %s\n", why), 2; };
  if (forgotten || !planned) return no("plan() has not succeeded, or forget() has been called");
  if (keeper) return no("a leaf is at hand: call leaf_end first");
  if (!table_root || !seed || !nodes || !hooks.count) return no("it requires the table's root, its mask seed, the path of its stored nodes and the count of opened columns");
  uint8_t any = 0;
  for (size_t i = 0; i < SEED_BYTES; i++) any |= seed[i];
  if (root_bytes != 4 * DW) return no("it requires the table's root: 32 bytes");
  if (!any) return no("32 zero bytes are no mask seed");
  memcpy(root, table_root, 4 * DW);
  memcpy(mask_seed, seed, SEED_BYTES);
  memcpy(seed_again, seed, SEED_BYTES);
  nodes_file = nodes;
  nodes_end = NODES_GOOD, nodes_end_told = false;
  loader_read = loader_rebuild = loader_waited = 0;
  try {
    loader = std::thread([this] { load_the_nodes(); });
  } catch (const std::exception &) {  // no thread can be started: the work is done here
    load_the_nodes();
  }
  lent::begin_opening();
  walks_made = 0, abandoned = rest_done = false, state = TOWARDS_A_WALK;
  side.reset(new HfSide());
  side->P = &W.P, side->mode = &mode, side->Y = &W.Y;
  side->held.seed = mask_seed, side->held.key_again = seed_again;
  side->held.nodes = heap, side->held.s = kept.s;
  side->held.u_host = u_host, side->held.u_host_elements = u_elements;
  side->short_lived = [](bool yes) { lent::borrow = yes; };
  side->walk = [this](const HfUnits &units) { walk(units); };
  side->fibres_a_walk = 0;  // 0: one pass computes all opened columns
  side->blas = blas_of_the_service;  // the opening's thread uses the service thread's handle; that thread is idle in `next_walk` meanwhile
  seams.reset(new HfSeams(hf_side_seams(*side)));
  u_written = true;
  ::hf::HfCard memory;
  memory.take = [](size_t bytes) {
    void *p = nullptr;
    const bool before = lent::borrow;
    lent::borrow = true;
    CK(cudaMalloc(&p, bytes));
    lent::borrow = before;
    return p;
  };
  memory.take_lasting = [](size_t bytes) {
    void *p = nullptr;
    const bool before = lent::borrow;
    lent::borrow = false;  // from the slab
    CK(cudaMalloc(&p, bytes));
    lent::borrow = before;
    return p;
  };
  memory.give = [](void *p) { cudaFree(p); };
  memory.fetch = [](void *host, const void *card_, size_t bytes) {
    CK(cudaMemcpyAsync(host, card_, bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
  };
  memory.store = [](void *card_, const void *host, size_t bytes) {
    CK(cudaMemcpyAsync(card_, host, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
  };
  if (hooks.adopt)
    memory.adopt = [adopt = hooks.adopt](const void *cells, size_t rows, size_t len) {
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      return adopt(cells, rows, len);
    };
  memory.leave = hooks.leave, memory.back = hooks.back;
  if (hooks.moved)
    memory.moved = [moved = hooks.moved](blk::TableId t, const void *cells) {
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      moved(t, cells);
    };
  // T's commitment in a hash chain's first query (which fixes the statement) is its root.
  keeper.reset(new ::hf::HfKeeper(W.P, mode, *seams, std::move(memory), std::vector<uint8_t>(table_root, table_root + root_bytes), record, std::move(hooks.count)));
  return 0;
}

void Unit::leaf_end() {
  if (forgotten || !keeper) return;
  BLK_REQUIRE(state != FOLDING && state != AFTER_FOLD, "the keeper of the weight table: leaf_end() inside a walk: abandon() first");
  join_the_loader();
  CK(cudaDeviceSynchronize());
  keeper.reset(), seams.reset(), side.reset();
  explicit_bzero(mask_seed, SEED_BYTES);
  explicit_bzero(seed_again, sizeof seed_again);
  memset(root, 0, sizeof root);
  nodes_file.clear();
  zero_u(false);
  scratch = nullptr, in_the_walk_room = false;
  take_live_flags();  // flags that the batch has not read are discarded with it
  lent::end_opening();
  state = IDLE;
}

// Zeroes u's host buffer if an opening may have written to it since it was last zeroed. plainly:
// without helper threads.
void Unit::zero_u(bool plainly) {
  if (!u_host || !u_written) return;
  if (plainly)
    explicit_bzero(u_host, u_elements * sizeof(::Ext));
  else
    zero_in_pieces((char *)u_host, u_elements * sizeof(::Ext));
  u_written = false;
}

int Unit::forget() {
  forgotten = true, ready = false;
  if (mask_seed) explicit_bzero(mask_seed, SEED_BYTES);
  explicit_bzero(seed_again, sizeof seed_again);
  // The wait comes before the zeroing below: add_* calls that the service has queued may still be
  // writing, to u on the host too.
  const bool gpu = cudaDeviceSynchronize() == cudaSuccess;
  zero_u(true);
  if (!gpu) return 1;
  bool zeroed = true;
  for (auto &p : lent::own.pieces) zeroed = cudaMemset(p.base, 0, p.bytes) == cudaSuccess && zeroed;       // the slab: a pass's sums
  for (auto &p : lent::borrowed.pieces) zeroed = cudaMemset(p.base, 0, p.bytes) == cudaSuccess && zeroed;  // lent memory: copies of weights during a pass
  return zeroed && cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}
}  // namespace rf

namespace hf {
std::unique_ptr<HfKeeperUnit> hf_keeper_unit(const HfKeeperSetup &setup) {
  if (rf::a_unit_lives) return fprintf(stderr, "the keeper of the weight table: a unit lives already: a copy of the unit's object file serves one unit at a time\n"), nullptr;
  rf::a_unit_lives = true;
  blk::at_a_failed_guard = setup.at_a_failed_guard;
  if (setup.end_name) g_end_name_at = setup.end_name;
  std::unique_ptr<rf::Unit> unit(new rf::Unit());
  if (unit->setup(setup)) return nullptr;
  return unit;
}
}  // namespace hf
