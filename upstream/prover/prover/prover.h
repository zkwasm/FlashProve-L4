// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Prover: the state of one process of the prover program cp_prove and, as member functions, the
// steps of its set-up and service. run() (service.cuh) shows their order. Those of type int return
// 0 on success, otherwise non-zero.
//
// Terms, with the identifiers' words. A sub-proof (`fork`) covers one part of the forward pass: a
// layer group, its attention or the output head. A batch pass (`wave`) proves up to -in-step
// requests (`member`) on one pass over the weights; they take turns, one layer group at a time, and
// a state buffer (`seat_at_hand`) keeps a request's GPU state between its turns (in_step.cuh). A
// chain is a hash chain, one protocol's Fiat-Shamir transcript. A request's first chain has one
// segment (`link`) for each sub-proof. Every batch is proved against one leaf table (`leaf`), a
// stored weight table with its own masks. The opening (`close`) is the batch's test on that table:
// a shared chain (`fold`), then each request's second chain (batch_opening.h). A keeper unit
// (`library`, `closer`) owns the table's masks and stored nodes; the shared chain needs
// closer->walks() passes over the weights (`walk`). A rider (-overlap) is a batch whose opening is
// made while later batches are proved. The parameter file (`policy`) is policy.txt.
#pragma once
#include "common/made_up_alone.h"
#include "frame/check.h"
#include "frame/files.h"
#include "prover/layer_fork.h"
#include "prover/chain_host.cuh"
#include "hiding/door_chain.cuh"
#include "prover/member_state.cuh"
#include "prover/policy.h"
#include "common/cp_files.h"
#include "prover/slot.h"
#include "prover/spool.h"
#include "prover/stage_clock.h"
#include "prover/weight_stream.cuh"
#include "hash_commitment/hf_keeper_unit.h"
#include "prover/batch_opening.h"
#include <thread>

struct JoinedThread {
  std::thread thread;
  void join() {
    if (thread.joinable()) thread.join();
  }
  ~JoinedThread() { join(); }
};

struct Prover {
  int argc = 0;
  char **argv = nullptr;
  double process_start = 0;
  Arguments args;
  Policy policy;
  // The service: service.cuh. `spool` is the spool directory (-spool): requests in in/, responses in
  // out/.
  Spool spool;
  // batch.number counts the batches released. A slot is a request's
  // place in its batch, given by seat_the_batch(); its failure word holds its flags (slot.h).
  Batch batch; // the batch that is being proved
  // From `isk`, the commitment's secret key directory (prover/prover/isk.cuh): the seed, the number
  // M of leaf tables, the first live one and the stored level of their Merkle trees.
  // read_isk() opens `isk` once, and isk_held keeps the descriptor until the process ends. Whatever
  // cp_prove writes below `isk` it reaches from that descriptor; cp::SecretFolder
  // (prover/common/cp_files.h) gives the reason.
  uint8_t isk_seed[32] = {0};
  cp::SecretFolder isk_held; // the directory -isk, kept open (see above)
  uint32_t isk_leaves = 0, isk_first = 0;
  uint32_t isk_version_word = 0; // isk.txt's line version; read_inputs() compares it with the parameter file's
  int isk_keep = 0;
  std::string isk_ranges; // isk.txt's line ranges, or empty; compared with `parameters_hash_words`
  // Nothing authenticates isk.txt's line "seed test"; only a test build accepts it.
  bool isk_test_seed = false; // isk.txt has that line
  // For the tools in prover/tests/tools that run the set-up steps and prove nothing: they set this
  // before read_inputs(); cp_prove never does. If true, begin_the_batch_on_a_leaf() exits.
  bool makes_no_response = false;
  // -test-coins (test builds) names a file with 32 bytes for each batch, used in place of the
  // system's fresh bytes to derive the batch's private randomness (`coins` in identifiers).
  long long test_coins_batches = -1; // with -test-coins: the number of batches that the file serves
  long long test_coins_used = 0;     // with -test-coins: how many batches have used theirs
  // Null, or the name of a stop that is not due to chance: something in `isk` (isk.txt, the seed, a
  // leaf table's leaf file, stored nodes or column count), the weights or the configuration files
  // do not fit the commitment or forbid serving it, or the file of -keep-for-the-audit
  // cannot be written. If set-up sets it, the process exits before it takes a batch. Later, no
  // further batch is taken, the batch that met the fault is answered "failed", and the service
  // exits. In both cases the name is in the last line.
  const char *isk_fault = nullptr;
  // Null, or the name of the first failed self-check on secret random values, among those whose
  // result travels as a flag and that end nothing where they fail:
  //   - the sub-proofs' mask entries ("slot pads"), compared by value and counted again on the GPU
  //     (finish_request()). The comparison also fails if a segment was compared twice or never, if
  //     the second implementation of the derivation rejects, or if the index-mask entry that a
  //     sub-proof takes, the mask for its claim on the weight table, differs (chain_bind());
  //   - the leaf table's masks, salts and pads in the batch's opening, and their key's second copy
  //     (hf::HfKeeperUnit::own_test_fault()).
  // Only these conditions set it, and only the serving thread does. It is never set from a slot's
  // flag OWN_TEST (prover/common/slot_flags.h), from End::Stopped (prover/prover/ends.h) or from a
  // flag that belongs to no request. Those also carry a check of the lookup argument that depends
  // on weights and prompt, so stopping on them would give a client one bit about a request of its
  // choice and a way to end the service at will. It is never set from ZERO_ELEMENT either, a rare event
  // on challenges, or from NO_ELEMENT, which no draw raises (prover/common/slot_flags.h).
  // Once it is set: no batch whose naming has not begun gets final names in out/ (Spool::shut());
  // no further batch is taken (ServiceWork::must_end()); no call that begins later stages a byte
  // (stage_the_proofs()); the batches in progress run on through their stages; and the process
  // exits at the next end of a batch (end_the_batch()), with this name in its last line and exit
  // code 9 (docs/limits-all-rows.tsv, row C3). That withholds the failing batch, younger ones,
  // and older ones whose openings are still in progress (-overlap). What is staged is overwritten
  // and removed, at the latest by the next start, which answers all these requests "failed".
  const char *self_check_failed = nullptr;
  void a_self_check_failed(const char *name) {
    if (!self_check_failed) self_check_failed = name;
    name_the_end(name); // now, so that the operator learns of the failed self-check even if
                        // something else ends the process before the end of the batch (a file that
                        // cannot be written, a failed CUDA call)
    spool.shut();       // now, and not minutes later when the process ends. Nothing between here
                        // and the end of the batch needs the spool
  }
  // The columns that one leaf table's masks allow to be opened in all
  // (blk::Configured::table_columns()), among them those of its own LINK proof, which ties it to
  // the commitment's first weight table. Each live leaf table's column count (`count.bin`) is
  // checked against it at set-up, before every batch is taken and at every batch's start.
  uint64_t a_leaf_bears = 0;
  // The columns that one batch opens: a batch raises its leaf table's column count by this number.
  uint64_t a_batch_counts = 1;
  int read_the_counts();
  bool a_live_leaf_is_spent();
  void the_counts_refuse(const cp::Counts &c);
  int read_isk();
  int the_ranges_are_guarded();
  int test_the_coins();
  int read_the_seed();
  void wipe_the_seed();
  End begin_the_batch_on_a_leaf();
  End begin_the_batch_in_the_library();
  // Durations, in seconds, of the parts of the last batch's start (begin_the_batch_on_a_leaf()),
  // for -measure's line "measure: head parts".
  struct HeadLaps {
    double system_bytes = 0, engine = 0, leaf_begin = 0, names_and_order = 0;
    cp::CountLaps count;
  } head_laps;
  void end_if_the_keys_differ(uint32_t holds, uint32_t owed, const char *whose);
  // cp_prove never sets it.
  // If bit j is set, slot j's files are never written and its request is answered
  // "failed", whatever its failure word holds.
  uint64_t never_released = 0;
  // The values of a batch's "measure:" line. `start` and `staged` are times since the epoch; the
  // other doubles are durations in seconds.
  struct BatchTimes {
    double start = 0, waves = 0, fold_begin = 0, fold_walk = 0, fold_finish = 0, claims = 0, staged = 0;
    double fold_next = 0;                    // between passes over the weights; not in fold_walk
    double head = 0;                         // the batch's start: `begin_the_batch_on_a_leaf`
    // The keeper unit's loader thread: reading the leaf table's stored nodes; hashing them up to
    // the root; the opening's wait for it.
    double loader_read = 0, loader_rebuild = 0, loader_waited = 0;
    uint32_t leaf = 0;
    double join = 0, stage = 0, end = 0;     // after the opening: waiting for the file writer; staging; erasing
    double file_write = 0, file_sync = 0;    // in the spool code: writing the responses' files; syncing released ones
    double file_make = 0, file_lock = 0;     // in the spool code: creating directories, files; lock
    // cause: a failure of the whole batch, or one of its two alarms (ends.h). member_bytes: the
    // size of the batch's longest proof file.
    End cause = End::None;
    size_t member_bytes = 0;
  };
  // What close_the_books() prints about a batch, kept until the batch's release.
  struct Proved {
    bool there = false;
    int B = 0;
    BatchTimes times;
  };
  void close_the_books(const Proved &b, long long bears);
  int open_the_spool();
  bool name_what_shows();
  void seat_the_batch(std::vector<Spool::Request> &taken);
  int prove_the_batch(BatchTimes &times);
  int the_batch_gets_no_proof(End why, BatchTimes &times);
  // A batch from the end of its batch passes (in prove_the_batch()) until end_the_batch().
  struct Ending {
    Batch *b = nullptr; // its slots: `batch` or, for a rider, its opening's `rode`
    BatchTimes times;
    bool proofs_written = true; // the writer thread's result
    JoinedThread writer;        // stages the proof files' first parts while the opening is made
    End alarm = End::None;      // what the service's loop noted about a rider (ServiceWork::note())
    End shared = End::None;     // a cause of no single batch, noted when an older rider ended
  };
  // What a batch's opening needs, from the batch's start to its end: the keeper unit of its leaf
  // table, the engine on which all its requests and its shared chain run
  // (prover/engine/engine_gpu.cuh: it stores, commits to and opens oracle messages), and the order
  // of its chains. An Opening holds one batch at a time. There is one more than batches can be
  // riders, so that every batch finds a free one at its start: its columns are counted and its
  // engine is made before its first request begins.
  struct Opening {
    std::unique_ptr<hf::HfKeeperUnit> unit;     // for the life of the process
    std::unique_ptr<blk::EngineGpu> engine;     // the batch's: from begin_the_batch_in_the_library() to end_the_batch()
    blk::IndexName table;                       // the batch's: its leaf table's root and record
    blk::u32 entry = 0;                         // .. and that leaf table's number k
    blk::Coins *coins = nullptr;                // the batch's own generator, which the opening's thread holds while it runs
    std::unique_ptr<BatchOpening> order;        // the batch's
    // The sync to disk of the batch's raised column count: see the_count_is_lasting().
    cp::CountSync count;                        // the batch's
    // True from the batch's start, which has written the raised count but not waited for its sync,
    // until the keeper unit's hook `count` asks: the hook says yes once.
    bool counted = false;                       // the batch's
    JoinedThread thread;                        // started by `fold_begin`; runs order->open(),
                                                // which blocks in each `walk` of the keeper unit
    bool abandoned = false;                     // a blocked call of that thread has thrown after abandon()
    std::pair<char *, size_t> slab{nullptr, 0}; // with -overlap: what the unit keeps throughout an opening: a buffer of its
                                                // own, set by hold_the_close_to_its_room()
    Ending ending;
    Batch rode;    // while its batch is a rider: that batch's slots
    int walks = 0; // while its batch is a rider: the completed passes over the weights
  };
  Opening opening[3];
  int openings = 1; // how many of them are set up (setup_opening()): 1, with -overlap 2, with -overlap-two 3
  // Fixed for every batch's shared chain and every request's second chain
  // (set_up_the_shared_chain()).
  hf::HfLinSetup lin_setup;
  int set_up_the_shared_chain();
  // The current opening: calls of a keeper unit and loans of memory act on it. turn_to() exits if
  // memory other than an opening's `slab` is still lent.
  Opening *at_hand = &opening[0];
  void turn_to(Opening &o);
  // The opening of the batch in `batch`, from that batch's start; null once it is a rider.
  Opening *opening_of_the_batch = nullptr;
  End shared_with_the_batch = End::None; // as Ending::shared, for the batch in 'batch', until it becomes a rider or ends
  void stage_the_proofs(Opening &o);
  void join_the_writers();
  int end_the_batch(bool in_a_pass);
  // The riders' openings, the oldest batch first: at most one, with -overlap-two at most two.
  std::vector<Opening *> riders;
  Proved rider_proved;               // a rider's `Proved`, from its end to its release
  std::function<int()> take_ahead;   // takes the next batch's requests, returns their number;
                                     // empty without -overlap
  std::function<void()> release_now; // releases the oldest rider's responses, prints its log lines
  int the_rider_folds(int g, bool the_rounds_only_turn);
  int the_rider_ends(bool in_a_pass = true);
  // Each pass of a rider's opening over the weights takes them from a later batch's batch pass or
  // runs alone (the_riders_walk_alone()).
  bool wave_carries = false;   // the current batch pass supplies the weights to each rider
  int the_riders_walk_ends();
  int the_riders_walk_alone(bool all);
  int serve();
  int named_paths_open();
  int set_up_the_opening_library();
  void forget_the_opening_library();
  // Sizes, set by read_inputs() (prover/prover/inputs.cuh). Lines of the parameter file
  // (explained in verifier/docs/parameter-file.md, "All lines in their order"):
  //   Lr (line L) layers; d: the residual stream's width; f, fe: hidden widths, dense and expert
  //   H query heads and Gk (line G) key and value heads of size dh; NE routed experts
  //   lt, ll, la, nmin, D, lmb, seed (seed_lo, seed_hi); hb_: the line `sec`
  //   lkW: the last number of the line `table`: the weight matrix W has 2^lkW columns
  // Derived: NT = 2^lt tokens; Lam = 2^ll layers in each of NGr layer groups; NI = 2 Lam; Pd, Pf:
  // d, f rounded up to powers of two; lPd, lPf, ldh, lfe, lNI: log2 of d, f, dh, fe, NI, rounded
  // up; dq = H dh, dkv = Gk dh; mo: the odd layers have routed experts; LamM, LamD: a group's
  // layers with and without experts; NC: a layer sub-proof's lookup calls.
  int lt = 0;
  int Lr = 0;
  int ll = 0;
  int d = 0;
  int f = 0;
  int la = 0;
  int nmin = 0;
  int D = 0;
  int lmb = 0;
  u64 seed = 0;
  int Lam = 0;
  int NGr = 0;
  int NT = 0;
  int ls = 0;
  int lPd = 0;
  int Pd = 0;
  int lPf = 0;
  int Pf = 0;
  int NW = 0;
  int NN = 0;
  int NI = 0;
  int xl = 4; // limbs of a residual-stream value: 5 with the parameter file's line "xl 5"
  int lNI = 0;
  int H = 0;
  int Gk = 0;
  int dh = 0;
  int dq = 0;
  int dkv = 0;

  int ldh = 0;
  size_t LA = 0;
  int hb_ = 0;
  int lkW = 0;
  int rkR = 0;    // always 1: a request has no commitment with a rank; the format has no line rkR
  int NC = 0;
  int NC1 = 0; // always equal to NC; never read
  int NE = 0;
  int fe = 0;
  int mo = 0;
  int LamM = 0;
  int LamD = 0;
  int lfe = 0;
  int KLW = 0;
  // KL: the 32-bit words of KL.bin, the quantization scales. A layer's record has KLW = 13 words:
  // kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp (the RoPE flag). After all records come kr kgR kuR for
  // each layer with experts. The layer's first norm computes round(x R / 2^K1) with
  // R = round(2^M1 / sig), sig = isqrt(sum of x^2 + eps); its second norm uses M2 and K2.
  std::vector<u32> KL;
  // For each lookup call c. io: the line iota_c. kap: always 5: a multiplicity has 2^5 bits. kk_:
  // the keys' width in bits. t0_: the padding key. knd: the table's type: 0 R16, 1 R8, 2 FC, 3 FS,
  // 4 any other.
  std::vector<int> io;
  std::vector<int> kap;
  std::vector<int> kk_;
  std::vector<int> t0_;
  std::vector<int> knd;

  // Public tables, built by make_tables(); a final d marks a copy on the GPU.
  // The gate's table Ts: its key step is 2^-gate_key_step, and its values are
  // within +-gate_values_most (the lines eg and smax of the parameter file).
  int gate_key_step = 8, gate_values_most = 4095;
  std::vector<short> Ts;
  short *Tsd = nullptr;
  std::vector<u64> tFC;
  std::vector<u64> tFL;
  u64 *tFCd = nullptr;
  u64 *tFLd = nullptr;
  std::vector<int> Sg;
  std::vector<int> FSw;      // the words of table FS, which build_header() hashes (make_tables())
  bool gate_clamped = false; // the parameter file's line gcl
  int *Sigd = nullptr;
  u64 *tFSd = nullptr;
  std::vector<u64> tFS;
  std::vector<int> Te;
  // iQA, iXG, iXN: indices in Fork::M: the matrix qa, the border matrices (the residual stream's
  // limbs) before and after the layer group.
  int nAF = 0;
  int nAD = 0;
  int iQA = 0;
  int iXG = 0;
  int iXN = 0;
  // The kind of a layer's weight, 0 to 11: queries and keys; values; attention's output; gating and
  // up projections (dense; shared expert; routed); down, dense and shared; routed down; the router.
  // (The embedding has kind 12 and the head kind 13.)
  const char *KN[12] = {"qk", "v", "o", "g", "u", "gs", "us", "gr", "ur", "dn", "dnr", "r"};
  std::vector<int> kinds;
  AOutput attF;
  std::unique_ptr<Weights> weights;
  void *row_pass_scratch = nullptr; // for Weights::row_pass
  // With weights stored as quantized blocks: `weights` as its concrete type, and the source of the
  // bytes.
  wb::BlockWeights *block_weights = nullptr;
  WeightStream weight_stream;
  std::unique_ptr<wb::BlockSource> block_source;
  // FK[g]: layer sub-proof g. XBv[g]: on the GPU, the border matrix before layer group g, kx
  // columns of mx cells. Two buffers alternate: XBv[g] is XBv[g - 2].
  std::vector<Fork> FK;
  std::vector<i8 *> XBv;
  size_t mx = 0;
  size_t kx = 0;

  // On the GPU: MSG[i] is masked value i of the layer or head sub-proof that is running, and
  // MSG[NSM + i] its mask entry.
  T *MSG = nullptr;
  size_t NSM = 0;
  // Each sub-proof in FK keeps SHg's address (prover/head/stack.cuh, setup()).
  gs::Shared SHg;
  // The vocabulary size (the line V).
  int Vv = 0;
  int *tokd = nullptr; // the request's NT + 2 words on the GPU: the tokens, then r and e
  i8 *embC = nullptr;
  size_t npE = 0;
  // One element on the GPU, outside MSG: the mask of the value that a layer sub-proof's merge
  // sends, which is the sub-proof's entry of the request's index-mask message. g_commit() fetches
  // it, and g_equations_and_columns() compares it by value with the entry taken last.
  T *merge_mask = nullptr;
  // A buffer for weight cells (make_weight_cells()), not the sub-proofs' scratch memory, which
  // messages call the slab.
  i8 *slab = nullptr;
  std::function<void(int, const WeightKind &, int)> make_weight_cells;

  // CP: a request's pad tree (crossing-pad commitment); prover/common/pad_tree_rows.h defines its
  // rows. plan_memory() sets it up and allocates rootCP and CPh; no code draws its rows, commits to
  // it or opens it.
  hz::Tree CP;
  u32 *rootCP = nullptr;
  T *CPh = nullptr;
  size_t rscB = 0;
  size_t thW = 0;

  std::vector<void *> arena;
  std::vector<size_t> arenaB;
  size_t ai = 0;
  // xin[g]: on the GPU, the residual stream before layer group g; it alternates as XBv does.
  std::vector<ResidualInt *> xin;

  // AO: one object for each shape of attention sub-proof. AG[g]: group g's object and number in it.
  struct AGr {
    AForkHid *o = nullptr;
    int h = 0;
  };
  std::vector<AGr> AG;
  std::vector<AForkHid *> AO;
  i16 *Aq = nullptr;
  i16 *Ak = nullptr;
  i16 *Av = nullptr;
  i8 *Aqa = nullptr;  // all of the layer sub-proof's cells of qa: copy_qkv_for_attention()
  u32 *Art = nullptr; // ART_WORDS words for each attention sub-proof: allocated, not used
  static constexpr int ART_WORDS = DW * 40;
  u32 *Aef = nullptr;
  // trA: allocated, not used.
  Tr *trA = nullptr;
  HForkHid HH;
  void *rsc = nullptr;
  // Error words on the GPU; finish_request() maps err to WINDOW, errc to UNFIT, errf (the forward
  // passes' flags) to UNFIT or ABOVE.
  u32 *err = nullptr;
  u32 *errc = nullptr;
  u32 *errf = nullptr;
  // Two counts on the GPU, which finish_request() compares with the expected numbers.
  // Cells of the sub-proofs' mask entries that equalled what the batch's key gives.
  unsigned long long *slot_pads_agree = nullptr;
  // Padding cells (`filler`) of the sub-proofs' matrices that were 0 before each sub-proof's first
  // commitment. The residual stream's border matrices are not counted.
  unsigned long long *filler_is_zero = nullptr;
  // begin_request() takes from this count the start value of the two counts above: each holds that
  // start plus what was counted.
  unsigned long long requests_begun = 0;
  u32 *sd = nullptr;
  u32 *hdr = nullptr;
  u32 *ph1 = nullptr;
  // On the GPU: the transcript state (`Tr`, prover/hiding/transcript_shell.cuh) of the request's
  // first chain, which all its sub-proofs continue.
  Tr *tr = nullptr;
  size_t nPh1 = 0;
  u32 *efs = nullptr;

  // hd, hdr: the request's statement as NH 32-bit words, on the host and on the GPU
  // (build_header()).
  std::vector<u32> hd;
  int NHb = 0;
  int atTokens = 0; // position of the tree hash of the request's words (the tokens, r, e) in the statement
  int atY = 0;      // position of y, the output token
  int NH = 0;
  // SC: the sections of the sub-proofs' messages (plan_sections()): an empty one, then secPerFork
  // for each layer sub-proof, the head's (iSecH .. iSecHe - 1), attention's. The sub-proofs'
  // identifiers gid ascend in the same order.
  std::vector<SecD> SC;
  size_t secPerFork = 0;
  size_t iSecH = 0;
  size_t iSecHe = 0;
  std::vector<size_t> oA;
  size_t nw = 0;
  // On the GPU: ch::host.blob copies one layer sub-proof's or the head's sections here, in SC's
  // order. Its callers store the address in a field, `host_words`, that nothing reads.
  u32 *out = nullptr;
  double weights_read_seconds = 0; // block weights: from the first tensor's reading until the last group is in place

  auto is_moe_layer(int l) { return mo && (l & 1); }
  // Layer l's quantization scales, from KL.
  LayerScaleSet layer_constants(int l) {
    const u32 none[3] = {0, 0, 0}, *w = &KL[(size_t)KLW * l];
    const u32 *e = is_moe_layer(l) ? &KL[(size_t)KLW * Lr + 3 * (l / 2)] : none;
    return LayerScaleSet{(int)w[0], (int)w[1], (int)w[2],  (int)w[3], (int)w[4], (int)w[5], (int)w[6],  (int)w[7],
                (int)w[8], (int)w[9], (int)w[10], (int)e[0], (int)e[1], (int)e[2], (int)w[12], (int)w[11]};
  }
  void digit_forms(LayerShape &s) {
    s.hqk = policy["hq"];
    s.hv = policy["hv"];
    s.ho = policy["ho"];
    s.hg = policy["hgD"];
    s.hu = policy["huD"];
    s.hdn = policy["hdD"];
    s.hr = mo ? (int)policy["hr"] : 0;
    s.rcl = policy.m.count("rcl") ? 1 : 0;
    s.gcl = policy.m.count("gcl") ? 1 : 0;
  }
  // Inputs, public tables and shared memory: inputs.cuh.
  int read_inputs();
  int make_tables();

  size_t kind_row_length(int k);

  auto kind_cells(int k) { // cells of one limb plane for weight kind k
    return k == 0                        ? (size_t)Lam * d * dh * 64
           : k == 1                      ? (size_t)Lam * d * dkv
           : k == 2                      ? (size_t)Lam * dq * d
           : k == 3 || k == 4            ? (size_t)LamD * d * f
           : k == 5 || k == 6            ? (size_t)LamM * d * fe
           : k == 7 || k == 8 || k == 10 ? (size_t)LamM * NE * d * fe
           : k == 9                      ? ((size_t)LamD * f + (size_t)LamM * fe) * d
                                         : (size_t)LamM * d * NE;
  }

  size_t parts_of_kind(int kd);

  bool has(const std::vector<std::string> &v, const std::string &s);
  int plan_memory();

  // Set by a layer sub-proof's set-up, below; `nb` and `tscCalls` are also read while proving. nop,
  // nrel: never used.
  LayerShape S;
  size_t nb = 0;
  int NM = 0;
  int muH = 0;
  int muV = 0;
  std::vector<int> tscCalls;
  std::vector<std::string> rdAF;
  std::vector<std::string> rdQA;
  std::vector<std::string> rdXG;
  std::vector<std::string> rdXN;
  std::vector<std::string> rdAD;
  std::vector<std::string> rdAR;
  T *w = nullptr;
  int NM_ = 0;
  size_t s = 0;
  size_t nop = 0;
  size_t nrel = 0;
  int nrows = 0;
  int LS = 0;

  // middle: in a sub-proof's run the caller may use the memory only between
  // fork_memory.middle_begin() and late_begin(); this is not checked.
  void *shared_alloc(int g, size_t bytes, bool middle = false);

  auto alloc_elements(int g, T *&p, size_t n, bool middle = false) { p = (T *)shared_alloc(g, sizeof(T) * n, middle); }

  auto cells_of(Fork &K, const LM &lm) { return K.M[lm.first].m * K.M[lm.first].k * lm.n; }
  // Set-up of layer sub-proof g: layer_setup.cuh; off3(), g_walk_spaces() to g_walk_end() and
  // g_setup_walk(): layer_walk.cuh. In these names, and in g_walk(), `walk` is hw::Walk, a
  // sub-proof's record of its claims.
  int g_setup_layout(int g);
  int g_setup_matrices(int g);
  int g_setup_engines(int g);
  int g_setup_points(int g);
  int g_setup_tables(int g);
  int g_setup_token_ends(int g);

  int spCP = 0;
  std::map<std::string, size_t> OFF;
  std::vector<hw::AF> gClaim;
  hw::Rec gw_RS; // in set-up: the layer sub-proof's batched sumcheck as recorded in its hw::Walk

  size_t off3(const char *U, const std::string &pt);
  int g_walk_spaces(int g);
  int g_walk_relations(int g);
  int g_walk_stack(int g);
  int g_walk_end(int g);
  int g_setup_slots(int g);
  int g_setup_walk(int g);
  int g_setup_equations(int g);
  int g_setup_end(int g);
  int setup_g_forks();
  // These two and setup_buffers(): other_setup.cuh.
  int setup_attention_forks();
  int setup_head_fork();
  int plan_stream_ids(); // stream_plan.cuh
  ChainHost chain; // chain_host.cuh
  // chain_link_plan[link](plan, moment) adds the rounds of segment `link` after its commit point
  // `moment`. frame: everything that follows from the three public files: all sizes, the plans of a
  // proof's three chains, what is in every first query. blk::configure()
  // (blocks/frame/configure.h), which the verifier calls too, makes it once, in chain_make(),
  // before the first batch.
  std::vector<std::function<void(blk::Plan &, unsigned moment)>> chain_link_plan;
  std::unique_ptr<blk::Configured> frame;
  // The three public files' bytes, in the order in which they are hashed into the configuration
  // digest: the parameter file, the layout file, the quantization scales (the bytes of KL.bin).
  // Valid once read_inputs() has returned.
  std::vector<std::string> configuration_files() const { return {policy.bytes, layout_bytes, std::string((const char *)KL.data(), 4 * KL.size())}; }
  blk::Shapes chain_shapes(const std::vector<std::string> &files, const blk::Parameters &par); // chain_frame.cuh
  blk::WfShape chain_ranges(unsigned k_col, const char *&fault);                                // request_member.cuh: from W's layout
  // The engine of the batch in `batch` (Opening::engine).
  blk::EngineGpu &chain_engine();
  blk::Coins &chain_coins();   // the private randomness of the request whose slot is entered
  // h_par: blk::configuration_digest() of the three public files' bytes.
  std::vector<u32> parameters_hash_words; // empty until `read_inputs` has made it
  bool layout_file = false;
  std::string layout_bytes;
  // These four: chain_frame.cuh. link: a segment's number, from 0.
  void chain_first_moment(int link, const std::string &name, u32 gid, HiddenTail &HT, const T *masks, size_t mask_entries);
  int chain_make();
  void chain_begin_request();
  void chain_bind(int link);
  int setup_buffers();

  // The statement and the sections: header.cuh.
  std::vector<u32> hash_words(const std::vector<u32> &v);
  int build_header();
  int plan_sections();
  u64 rep = 0; // the counter of the current request's blinding context

  // Restoring and erasing memory: request.cuh, memory_as_set_up.cuh. AnAttempt: a request's start.
  enum class Before { AnAttempt, ATurn, TheHeadsTurn }; // what wipe_the_working_memory() is called before
  void wipe_the_working_memory(Before before);
  void wipe_behind_the_wave();
  void lay_stale_bytes(const ForkMemory::Buffers &list);
  void count_stale_bytes(const ForkMemory::Buffers &list, const char *when, bool kept);
  ForkMemory::Buffers kept_between_turns() const;
  ForkMemory::Buffers seats_state; // the result of kept_between_turns(), computed once, at set-up
  // Lists of GPU regions, and one range, that no code fills: set_up_the_seats() only gives each of
  // the two vectors one empty list for each state buffer. What the GPU engine allocates for a
  // request's chains (tables, trees, random tails, salts, mask rows) comes from the process's one
  // CUDA memory pool (card_pool()), outside ForkMemory's record, and is not cleared when freed,
  // between requests or at exit (prover/engine/device.cuh lists the exceptions).
  ForkMemory::Buffers chain_shared_regions;
  std::vector<ForkMemory::Buffers> chain_own_regions; // by state buffer
  std::vector<ForkMemory::Buffers> chain_seat_regions;
  std::pair<char *, size_t> chain_dying_stretch{nullptr, 0};
  // True if the prover's recorded GPU memory has risen above its peak at the end of the first batch
  // pass (fork_memory.h, note_what_is_taken()). That is only a warning.
  bool memory_grew() const;
  // Prints one of the three alarms (ends.h: MemoryGrew, CutShort, Unreadable): a constant line,
  // each at most once in a process.
  void say_the_alarm(End alarm);
  bool said_memory_grew = false, said_cut_short = false, said_unreadable = false;

  // The stages of a request's proof: request.cuh.
  void forward_and_counts(int g);
  void begin_request(Slot &s);
  int prove_layer_group(int g);
  int prove_head_fork();
  // After the last proximity test, on the request's first chain: joins the pending claims on the
  // weight table into one, makes the first part of the test on the table and seals the chain
  // (request_member.cuh). It fills s.part.lin_first and s.part.seal.
  void chain_member_and_seal(Slot &s);
  void finish_request(Slot &s);

  // A layer sub-proof's lookup calls, in the order of the parameter file's call lines, and the plan
  // of their lookup forest (the calls' product trees); the same for all layer sub-proofs.
  std::vector<ForestCall> layer_calls;
  ForestPlan layer_forest;
  // The numbers of the lookup calls that contain the lookup segments D and QK.
  int cD = 0;
  int cQK = 0;

  // The steps of prove_layer_group(): layer_prove.cuh.
  int g_commit(int g);
  int g_product_tests(int g);
  int g_weight_tables(int g);
  int g_norm_sums(int g);
  int g_token_sumchecks(int g);
  int g_stack(int g);
  int g_gated_products(int g);
  int g_row_passes(int g);
  int g_fold(int g);
  int g_walk(int g);
  void copy_qkv_for_attention(int g);
  int g_equations_and_columns(int g);
  void compare_attention_cells(AForkHid &attention);
  int prove_attention_fork(int g);

  // Hooks for the streaming of the weights; each may be empty. The first
  // two are called before and after all requests' turns in layer group g.
  std::function<void(int g)> weights_ready, weights_done;
  // Called when the last request of a batch pass has queued its last read of group g's routed
  // experts.
  std::function<void(int g)> experts_done;
  // A round: the requests' turns in one layer group; the last round is the head's. In a batch pass
  // of one request that supplies riders, the_rider_folds() sets this flag, after the turn.
  bool last_of_wave = true; // true in a round's last turn and in the whole last round (wave_order.cuh)
  std::function<void()> weights_forget; // restarts the weight stream (WeightStream::restart())

  // The requests of a batch pass and their state buffers: in_step.cuh. place: a request's index in
  // its batch pass, which is also its state buffer's number.
  MemberState member_state;
  MemberState::Buffers pad_tree_buffers; // the buffers of the request's crossing-pad commitment CP
  int seats = 0;        // the number of state buffers on the GPU: args.in_step
  int seat_at_hand = 0; // the state buffer of the current request
  void plan_member_state(int members);
  int set_up_the_seats();
  void take_the_seat(Slot &s, int place);
  void leave_the_seat(Slot &s);
  void seat_begins(Slot &s, int place);
  void seat_turn(Slot &s, int place, int g);
  void seat_ends(Slot &s, int place);
  // Returns only if the raised column count of the batch in o is on disk, by that batch's own sync
  // (cp::CountSync); otherwise the process exits (isk.cuh). The first step of fold_begin(), which
  // starts the opening's thread, and of stage_the_proofs(), which writes the batch's first bytes to
  // disk.
  void the_count_is_lasting(Opening &o);
  void wave_has_ended();
  void say_of_the_memory() const;
  int waves_ended = 0;
  // The opening: opening_in_process.cuh.
  hf::HfKeeperUnit *closer = nullptr; // the keeper unit of the current opening (`turn_to`)
  int setup_opening();
  int fold_end_for_opening(int which);
  int fold_group_for_opening(int g, const std::vector<int> &unit_kind, const std::vector<int> &unit_layer,
                             int routed_of = -2);
  // GPU memory that is idle at a batch's end and is lent to the keeper unit: close_memory.cuh.
  std::vector<std::pair<char *, size_t>> free_at_the_batchs_close(bool during_the_walk, bool in_a_pass = false);
  void hold_the_close_to_its_room();
  int lend_the_walk_room();
  void lend_for_the_opening(bool in_a_pass);
  void take_back_what_is_lent();
  void back_from_the_close();
  bool walk_room_is_lent = false;
  bool close_has_its_room = false;
  std::vector<std::pair<char *, size_t>> lent_at_the_close; // from lend_the_walk_room() to back_from_the_close()
  // The memory that the keeper unit keeps throughout an opening, and the rest of the sub-proofs'
  // scratch memory. These four: opening_in_process.cuh.
  std::pair<char *, size_t> slab_for_the_library() const;
  std::pair<char *, size_t> slab_without_it() const;
  std::pair<char *, size_t> walk_room() const;
  size_t decoding_needs_at_most();
  // A batch's shared chain: fold_walk.cuh.
  int lend_the_slab();
  int fold_begin();
  std::vector<int> unit_kind, unit_layer; // set by `list_the_units`. Unit i: one layer's weights of
                                          // one kind, or the embedding (kind 12) or the head (13)
  void list_the_units();
  int fold_units_from_the_stream();
  int fold_walk();
  int fold_next();
  int fold_over();
  int fold_walks();
  int fold_rest(bool in_a_pass);
  int batch_end();
  bool opening_library_forgot_all = true;
  int run(int argc, char **argv);
};
