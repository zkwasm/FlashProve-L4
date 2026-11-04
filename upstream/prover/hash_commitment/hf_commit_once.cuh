// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Builds a commitment's key file (`key.bin`) on the GPU: hf::once_make() is phase B of the
// committer program cp_commit. The protocol is not here: blk::make_key() (blocks/frame/check_key.h)
// runs it on the host. This file plans GPU memory and gives make_key() its GPU implementations:
//   hf::HfOnceTable   the anchor (the commitment's first weight table; `the first table` in
//                     messages) as a blk::IndexHolder. For the one-time test (the committer's own
//                     test on the anchor: the block LIN for a batch of one) it folds the table and
//                     opens columns, by passes over the weights (`walk` in names): HfOncePanels.
//   blk::WfGpu        the two lookups of WF, the committer's range proof on the table's entries.
//                     rf::WfWindows generates their keys, one chunk (`piece`) of a part at a time.
//   blk::WfProducts   the products of row weights with the matrix, for the claims that WF leaves.
//   blk::EngineGpu    the engine: a blk::Engine on the GPU, for the run's own oracle messages.
// At once_make(): the order of GPU memory use, secret values, self-checks. The helpers of namespace
// rf also serve prover/hash_commitment/hf_entry_make.cuh, which builds a leaf file (`entry`).
// Include this file after lent_memory.h, common.cuh and arguments.h of prover/weights_commitment/
// and hf_shape.cuh, hf_commit_weights.cuh and hf_commit_making.h of prover/hash_commitment/.
#pragma once
#include "common/blas.h"
#include "common/cp_files.h"
#include "common/element_from_hash.h" // spells_p()
#include "common/made_up_alone.h"
#include "common/slot_flags.h"
#include "engine/engine_gpu.cuh"
#include "engine/wf_heavy.cuh"
#include "frame/check_key.h"
#include "frame/wf_units.h"
#include "hash_commitment/commit_lie.cuh"
#include "hash_commitment/hf_commit_api.h"
#include "hash_commitment/hf_commit_handed.h"
#include "hash_commitment/hf_commit_walk.h"
#include "hash_commitment/hf_open_side.cuh"
#include "hash_commitment/hf_keeper_once.cuh"
#include "hash_commitment/hf_range_plan.h"
#include "hash_commitment/hf_range_windows.cuh"
#include "hash_commitment/hf_wf_keys.cuh"
#include "hash_commitment/hf_commit_coins.h" // `engine_coins_once`
#include <chrono>
#include <cstring>
#include <csignal>
#include <functional>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rf {
// Exit code 4: an internal error, such as a failed self-check.
[[noreturn]] static void hf_once_internal(const char *what) {
  fprintf(stderr, "cp_commit: the key file or an entry: internal error: %s\n", what);
  exit(4);
}

// ---- GPU memory: the slab, two allocations from the driver that become the two pools of
// prover/weights_commitment/lent_memory.h. From take() to give() that header's macro cudaMalloc
// allocates from a pool: lent::borrowed while lent::borrow is set, otherwise lent::own.
// lent::own serves the passes (HfOncePanels' panel and what HfOnceTable allocates for them);
// lent::borrowed holds the folded rows (a folded row, u, is a linear combination of a table's
// rows), as oracle messages, to the end. Each pool is zeroed when it is taken and before it is
// freed.
struct HfOnceSlab {
  // The memory of lent::own and of lent::borrowed.
  void *walks = nullptr, *tables = nullptr;
  size_t bytes = 0; // of both
  // The most that a slab may take: the free memory less 2 GiB (for the driver, the loading of
  // kernels, cuBLAS's allocations and a tool that runs the program) and less `beside` (for the
  // engine); at most cap bytes if cap is not 0 (the option -gpu-memory, with which several small
  // runs under a test seed share a GPU). `beside` comes off the free memory, not off cap: the
  // option limits the slab, not the engine. No byte of an output file depends on the result.
  static size_t most(size_t cap, size_t beside = 0) {
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    const size_t all = free_bytes - std::min(free_bytes, ((size_t)2 << 30) + beside);
    return (cap ? std::min(all, cap) : all) / 512 * 512;
  }
  // for_tables: the bytes of lent::borrowed. for_walks: the least that lent::own must get; it gets
  // all of most(cap) that for_tables leaves. Returns 0, or 2 after a line on stderr.
  int take(size_t for_tables, size_t for_walks, size_t cap) {
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    bytes = most(cap);
    for_tables = (for_tables + 511) / 512 * 512;
    if (bytes < for_tables + for_walks) {
      fprintf(stderr,
              "the key file or an entry: the GPU has too little free memory, or -gpu-memory allows too little: %zu bytes are free, of "
              "which 2 GiB are left alone, and %zu may be taken; the folded rows take %zu bytes, the panel and the walks %zu\n",
              free_bytes, bytes, for_tables, for_walks);
      return 2;
    }
    CK((cudaMalloc)(&walks, bytes - for_tables)); // the runtime's own call: the bare name is a macro (lent_memory.h)
    CK((cudaMalloc)(&tables, for_tables));
    CK(cudaMemset(walks, 0, bytes - for_tables));
    CK(cudaMemset(tables, 0, for_tables));
    ::lent::own.add(walks, bytes - for_tables);
    ::lent::borrowed.add(tables, for_tables);
    ::lent::begin_opening();
    return 0;
  }
  // Call it after the last pass, when every buffer from lent::own has been returned. It zeroes and
  // frees lent::own's memory, for the engine, and frees the driver's buffers whose cudaFree
  // lent_memory.h has deferred since take(). From here on an allocation from lent::own ends the
  // process. If a buffer of lent::own is still in use, nothing is freed and a line is printed: the
  // run may get slower or run out of memory, but its output is not affected.
  void the_walks_are_over() {
    if (!walks) return;
    if (::lent::own.used) {
      printf("the key file or an entry: %zu bytes of the walks' GPU memory are still in use behind the last walk: it stays to the end\n", ::lent::own.used);
      return;
    }
    ::lent::wipe(::lent::own); // waits for the GPU
    ::lent::own.pieces.clear();
    CK((cudaFree)(walks));
    walks = nullptr;
    for (void *p : ::lent::to_free) CK((cudaFree)(p));
    ::lent::to_free.clear();
  }
  // Zeroes (lent::end_opening()) and frees the slab's memory. After it cudaMalloc calls the driver.
  void give() {
    if (!tables) return;
    ::lent::end_opening();
    ::lent::own.pieces.clear();
    ::lent::borrowed.pieces.clear();
    ::lent::own.used = ::lent::borrowed.used = 0;
    if (walks) CK((cudaFree)(walks));
    CK((cudaFree)(tables));
    walks = tables = nullptr;
  }
  ~HfOnceSlab() { give(); }
};

// Declare it before the engine, so that it is destroyed after it. The engine's memory pool
// (blk::card_pool(), prover/engine/device.cuh) keeps for the process whatever was freed into it: at
// full size (Llama 4 Maverick's shape) tens of GB that cudaMemGetInfo() does not report as free.
// The destructor returns them to the driver, not zeroed, for the next step that sizes itself by the
// free memory: the creation of a leaf table (a further table of the same weights), the next slab.
struct HfOncePoolsBack {
  ~HfOncePoolsBack() {
    blk::card_pool_gives_back(); // waits for the GPU
  }
};

// ---- The weights, unit by unit. W is the matrix of all weights' cells (signed bytes); a unit
// (UnitW, prover/common/weight_layout.h) is a range of W's rows. A pass of the walker
// (hf::WeightWalker, prover/hash_commitment/hf_commit_walk.h) delivers each unit's tensors as
// integers in GPU memory, and g5::k_cellsT() (prover/weights_commitment/kernels.cuh) writes them
// into one buffer, the panel, as the unit's cells of W:
//   pan[c * pitch + r] = W[u.off + r][c]
// The pitch is u.rows rounded up to a multiple of 16; k_cellsT() writes 0 into the rows from
// u.rows on. Rows of W that belong to no unit are never visited, so they count as 0.
struct HfOncePanels {
  const ::hf::Committed &c;
  i8 *pan = nullptr;
  size_t most_rows = 0; // the largest pitch
  cudaEvent_t done = nullptr;
  int walks = 0;
  explicit HfOncePanels(const ::hf::Committed &c_) : c(c_) {
    for (const UnitW &u : c.shape.Y.U) most_rows = std::max(most_rows, (u.rows + 15) / 16 * 16);
  }
  size_t bytes() const { return most_rows * c.shape.Y.k; }
  void open() {
    CK(cudaMalloc(&pan, bytes()));
    CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  }
  // One pass (its contract: hf_walk.h): calls units.cells once for every unit. There is one panel,
  // and the callback has one set of buffers, so the event `done` orders each unit's work after the
  // previous unit's, on whatever stream. The host does not wait within a unit, so the walker can
  // copy the next unit meanwhile. Returns with the GPU idle.
  void walk(const HfUnits &units) {
    const LayW &Y = c.shape.Y;
    const long long H = (long long)c.PO["H"], G = (long long)c.PO["G"], dh = (long long)c.shape.sh.dh;
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    c.weights.walk(::hf::WeightWalker::WHOLE_UNITS, [&](const ::hf::Piece &piece, cudaStream_t stream) {
      if (!piece.whole) hf_once_internal("the walk over the weights gave a part of a unit");
      const UnitW &u = *piece.u;
      const size_t pitch = (u.rows + 15) / 16 * 16;
      CK(cudaStreamWaitEvent(stream, done, 0));
      ::g5::k_cellsT<<<8192, 256, 0, stream>>>(c.shape.um[piece.unit], H, G, dh, (const short *)piece.w0,
                                               (const short *)piece.w1, (const int *)piece.w0, Y.k, u.off, u.rows,
                                               pitch, pan);
      CK(cudaGetLastError());
      commit_lie::panel((int8_t *)pan, pitch, (size_t)u.rows, (size_t)u.off, Y.k, stream); // hook for a test's malicious committer; empty in this repository
      units.cells((const int8_t *)pan, pitch, (size_t)u.rows, (size_t)u.off, stream);
      CK(cudaEventRecord(done, stream));
    });
    CK(cudaEventSynchronize(done));
    CK(cudaDeviceSynchronize());
    walks++;
  }
  // Zeroes the panel, frees it, and has the walker zero and free its staging buffers: release()
  // leaves the norms' gains on the GPU (hf_commit_weights.cuh).
  void close() {
    if (!pan) return;
    CK(cudaMemset(pan, 0, bytes()));
    cudaFree(pan);
    pan = nullptr;
    CK(cudaEventDestroy(done));
    c.weights.release();
  }
};

// ---- Host memory for u, the folded row: page-locked, and zeroed before it is unmapped. madvise()
// excludes it from core dumps; its result is not checked.
struct HfOnceHostU {
  Ext *u = nullptr;
  size_t bytes = 0;
  int take(size_t elements) {
    bytes = (elements * sizeof(Ext) + 4095) / 4096 * 4096;
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      fprintf(stderr, "cp_commit: cannot allocate %zu bytes of host memory for the folded rows\n", bytes);
      return 2;
    }
    (void)madvise(p, bytes, MADV_DONTDUMP);
    if (cudaHostRegister(p, bytes, cudaHostRegisterDefault) != cudaSuccess) {
      munmap(p, bytes);
      fprintf(stderr, "cp_commit: cannot page-lock %zu bytes of host memory for the folded rows\n", bytes);
      return 2;
    }
    u = (Ext *)p;
    ::lent::poison_host(u, bytes); // only under a test's variable (lent_memory.h)
    return 0;
  }
  ~HfOnceHostU() {
    if (!u) return;
    explicit_bzero(u, bytes);
    cudaHostUnregister(u);
    munmap(u, bytes);
  }
};

// Waits for the GPU, then reads and clears the failure flags (prover/common/slot_flags.h). Returns
// 0, or 4 after a line on stderr.
static int hf_once_flags(const char *what) {
  CK(cudaGetLastError()); // rejected launches raise no flag: prover/docs/cuda-status.md, "The rule"
  (void)::spells_p(); // first: under the test switch COINS_REFUSED_DRAW its k-th call raises a flag
  unsigned flags = take_live_flags();
  if ((flags & UNFIT) && commit_lie::passes("unfit-key")) flags &= ~(unsigned)UNFIT; // hook for a test's malicious committer, which generates its out-of-range key again after the counting (`passes` is false in this repository)
  if (!flags) return 0;
  fprintf(stderr, "cp_commit: %s failed: failure flags %u (prover/common/slot_flags.h)\n", what, flags);
  return 4;
}

// Allocates from the driver what the process keeps for its lifetime: transform tables, the tree
// hash's scratch, cuBLAS's handle. Call it before a slab takes the free memory.
static void hf_once_ready(const LayW &Y) {
  for (int ln = 11; ln <= ELNO; ln++) ::row_tree::tables(ln);
  ::treehash_scratch((size_t)48 << 20, (size_t)8 << 20);
  ::g_st = cudaStreamPerThread;
  ::hf::HfOpen committer; // `hf_side_warm_up` reads only `once` of it
  committer.once = true;
  hf_side_warm_up(Y, committer);
}

// In bytes. A larger u is kept in page-locked host memory (HfOnceHostU), any other in lent::own.
constexpr size_t HF_ONCE_U_ON_THE_GPU_MOST = (size_t)1 << 30;

// Whether a lookup's parts in the configuration (`fixed`, which the verifier uses too) are those
// for which the committer generates keys (the second argument, from wf_lookup_of()). A part
// (blk::WfPart, blocks/iop/wf.h) has 2^log_places places. Place first_place + p of a run reads, for
// a key's digit q, W's row first_row + p + q * stride; a place in no run is padding. Adjacent runs
// that continue each other in place and row and have the same stride count as one. `name`, `mu`,
// `live` and `first_bound` are not compared.
static bool once_parts_agree(const blk::WfLookup &fixed, const blk::WfLookup &laid) {
  auto joined = [](const std::vector<blk::WfRun> &runs) {
    std::vector<blk::WfRun> j;
    for (const blk::WfRun &r : runs)
      if (!j.empty() && j.back().first_place + j.back().places == r.first_place && j.back().first_row + j.back().places == r.first_row && j.back().stride == r.stride)
        j.back().places += r.places;
      else
        j.push_back(r);
    return j;
  };
  if (fixed.d != laid.d || !(fixed.offset == laid.offset) || !(fixed.padding_key == laid.padding_key) || fixed.digit_weight != laid.digit_weight || fixed.parts.size() != laid.parts.size())
    return false;
  for (size_t p = 0; p < fixed.parts.size(); p++) {
    const std::vector<blk::WfRun> a = joined(fixed.parts[p].runs), b = joined(laid.parts[p].runs);
    if (fixed.parts[p].log_places != laid.parts[p].log_places || a.size() != b.size()) return false;
    for (size_t r = 0; r < a.size(); r++)
      if (a[r].first_place != b[r].first_place || a[r].first_row != b[r].first_row || a[r].places != b[r].places || a[r].stride != b[r].stride) return false;
  }
  return true;
}

// GPU memory from the driver for the packed weights of the current part; zeroed before it is freed.
struct HfOncePool {
  void *at = nullptr;
  size_t bytes = 0;
  void take(size_t n) {
    CK((cudaMalloc)(&at, n)); // the runtime's own call: the bare name is a macro (lent_memory.h)
    bytes = n;
  }
  void give() {
    if (!at) return;
    CK(cudaMemset(at, 0, bytes));
    CK((cudaFree)(at));
    at = nullptr, bytes = 0;
  }
  ~HfOncePool() { give(); }
};

// The least GPU memory that is left to the engine when the slab is taken; fewer columns are
// recomputed in a pass to leave it. No formula gives what the engine needs, and much less than
// this can be too little. The run's last line
// prints the peak of the engine's pool. No byte of an output file depends on this number.
#ifdef FLASHPROVE_L4
constexpr size_t HF_ONCE_ENGINE_ROOM = (size_t)8 << 30;
#else
constexpr size_t HF_ONCE_ENGINE_ROOM = (size_t)24 << 30;
#endif
// Added to the slab's computed size: while the slab exists, other code's small buffers come out of
// it too, since cudaMalloc is lent_memory.h's macro in the whole translation unit.
constexpr size_t HF_ONCE_SLAB_SPARE = (size_t)1 << 30;
// The most bytes that the pool has held from the driver since its high-water mark was last reset;
// with from_here it resets the mark and returns 0. The engine allocates from one pool,
// blk::card_pool() (prover/engine/device.cuh), which keeps what it has once taken from the driver,
// so this peak is what the engine needs beside the slab. (NVIDIA, CUDA Runtime API,
// cudaMemPoolAttr: cudaMemPoolAttrReservedMemHigh, which can be set to 0 only.)
static inline size_t once_pool_peak(cudaMemPool_t pool, bool from_here) {
  unsigned long long high = 0;
  if (from_here)
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high));
  else
    CK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high));
  return (size_t)high;
}
// Kept out of the pool of packed weights, on top of blk::wf_card_needs(): the key
// generator's memory (hf_range_windows.cuh; at full size 10 slots of 256 MiB, 0.6 GB of the
// walker's buffers, 25 MB of tables), with a reserve for the driver, cuBLAS and the engine's small
// trees. The program does not print wf_card_needs()'s value, and no test checks it.
// Part of HF_ONCE_WINDOWS_ROOM goes to the key file's range proof (`group once-ranges` in the
// parameter file, policy.txt), more at a longer code.
constexpr size_t HF_ONCE_WINDOWS_ROOM = (size_t)8 << 30;

#ifndef SERVING_BUILD // a production build has neither this hook's code nor its messages
// Fault-injection hook for tests (`door` in identifiers and messages): the environment variable
// ROWS_COUNT_FAULT damages the anchor's column count, `isk/anchor/count.bin`, right before the
// one-time test's columns are counted. One call of cp_commit creates and lengthens that file, so
// no test can reach it from outside. The variable is read only on synthetic weights under a test
// seed. Its name is a line of prover/tests/serving_build_words.txt, whose lines no production
// program may contain.
//   ROWS_COUNT_FAULT=length:<n>   sets the file's length to n, as if n columns had been opened
//   ROWS_COUNT_FAULT=missing      removes the file
//   ROWS_COUNT_FAULT=folder       puts a directory in its place
//   ROWS_COUNT_FAULT=no-write     while the columns are counted, no file of this process can grow
//                                 beyond the count's present length, so the append fails; SIGXFSZ
//                                 stays ignored afterwards
struct OnceCountDoor {
  struct rlimit before_;
  bool limited_ = false;
  explicit OnceCountDoor(const std::string &isk) {
    const char *word = made_up_under_a_tests_seed() ? TESTENV("ROWS_COUNT_FAULT") : nullptr; // guarded like ROWS_TABLE_FAULT (hf_commit_table.cuh): it can shorten a column count
    if (!word) return;
    const std::string path = ::cp::isk_anchor_count(isk);
    printf("cp_commit: a test's door: the first table's count before the one-time test: %s\n", word);
    fflush(stdout);
    char *end = nullptr;
    bool done = false;
    errno = 0; // stays 0 if the variable's value is malformed
    if (!strncmp(word, "length:", 7)) {
      const unsigned long long n = strtoull(word + 7, &end, 10);
      done = end != word + 7 && !*end && !truncate(path.c_str(), (off_t)n);
    } else if (!strcmp(word, "missing")) {
      done = !unlink(path.c_str());
    } else if (!strcmp(word, "folder")) {
      done = !unlink(path.c_str()) && !mkdir(path.c_str(), 0700);
    } else if (!strcmp(word, "no-write")) {
      // (POSIX, setrlimit(), RLIMIT_FSIZE: a write beyond the limit fails with EFBIG, and SIGXFSZ
      // is sent, which ends a process that does not ignore it.)
      struct stat is;
      struct rlimit limit;
      done = !stat(path.c_str(), &is) && !getrlimit(RLIMIT_FSIZE, &before_) && signal(SIGXFSZ, SIG_IGN) != SIG_ERR &&
             (limit = before_, limit.rlim_cur = (rlim_t)is.st_size, !setrlimit(RLIMIT_FSIZE, &limit));
      limited_ = done;
    }
    if (done) return;
    if (errno)
      fprintf(stderr, "cp_commit: ROWS_COUNT_FAULT=%s: the fault could not be planted: %s\n", word, strerror(errno));
    else
      fprintf(stderr, "cp_commit: ROWS_COUNT_FAULT takes length:<n>, missing, folder or no-write\n");
    exit(2);
  }
  ~OnceCountDoor() { // lifts the limit after the counting: the run's output may go to files too
    if (limited_) setrlimit(RLIMIT_FSIZE, &before_);
  }
};
#endif

} // namespace rf

namespace hf {
// Builds the key file, once phase A (prover/hash_commitment/hf_commit_table.cuh) has built the
// anchor's tree. On return 0 `file`, which must come empty, contains the key file's bytes, which
// the caller writes; after a return of 4 it holds bytes that must not be written. Returns 2
// (arguments that do not fit each other; parts that HfRangePlan::make() rejects; an entry of W out
// of its range; a column count that cannot be raised, or masks that do not cover the one-time
// test's columns; too little host memory) or 4 (a failed self-check), after a line on stderr. Other
// errors exit the process, among them hf_once_internal() (code 4), a CUDA error under CK() (code 5)
// and too little GPU memory in step 2 below (code 2). It works on the calling thread's current GPU.
// configured: made from the run's public files (blk::configure()).
// root: the anchor's commitment as first queries of hash chains (Fiat-Shamir transcripts) contain
// it: its root, 32 bytes.
// record: the anchor's record, from make_the_record() (prover/hash_commitment/hf_commit.cu).
// The caller must have set the key of the key file's private randomness (`coins` in identifiers
// and messages) with RunKeys::take_the_key_files(); it checks that key before and after this call
// (hold_the_key_files()). It must also have created `isk/anchor/count.bin`, the anchor's column
// count, empty: make_key() begins by counting the one-time test's columns there
// (cp::count_first_table_columns()).
//
// GPU memory, in the order of the run. make_key() is one call, so step 2 is a callback, which
// WfProducts::begin() runs at its end. The key file is header | root | stored answers to samples of
// the table's rows (`record`) | one-time chain | place | shared chain | second chain
// (blocks/frame/check_key.h). Steps 1 and 2 belong to the one-time chain, step 3 to the shared one.
//   1  While WF runs there is no slab, and all memory comes from the driver: the current lookup's
//      buffers (at full size: stored levels 25.8 GB, a chunk's keys 8.6 GB, work buffers and digit
//      matrices 3.7 GB), the key generator's slots and the walker's buffers (3.2 GB), and a pool
//      for the packed weights of the current part. The pool is allocated first: the free memory
//      less the others' computed needs and a reserve (HF_ONCE_WINDOWS_ROOM).
//   2  All of that is freed. The parts' vectors (at full size 24 x 2^18 elements) and the engine's
//      small trees stay. The walker reserves its buffers for whole units; then the slab is taken
//      and the panel is allocated in it.
//   3  The table's passes: one for u, the folded row, and omega (one element of the extension field
//      E, `Ext`, for each row of W); then one for each group of `fibres` opened columns. A column's
//      fibre is the 2^s columns under its node of the lowest stored tree level, s = c.kept.s; these
//      passes recompute whole fibres. What leaves the GPU around them:
//      prover/hash_commitment/hf_keeper_once.cuh.
//   4  After the last pass the panel, the walker's buffers and lent::own are freed, for the engine:
//      it allocates from the driver as the proximity tests on u and the second chain need it.
//
// Secret, as derived from the weights or the masks: the lookup keys, the lookups' levels and sums,
// the pool, the panel, the passes' buffers, u, omega and the fibres. Zeroed before they are freed:
// WF's buffers that have held keys, counts, levels or sums and the key generator's slots, by their
// owners (prover/engine/lkp_pieces.cuh, prover/engine/wf_heavy.cuh,
// prover/hash_commitment/hf_range_windows.cuh); pool, panel, slab and the host's u, in this file. A
// buffer inside the slab (omega, the folded row) returns to it unzeroed; the slab is zeroed when it
// is taken and before it is freed. The parts' vectors of step 2 also depend on the weights; their
// owner zeroes them before they are freed (prover/engine/member_heavy.cuh). The engine's other
// buffers are never zeroed (prover/engine/device.cuh). Where the process ends instead of returning,
// no destructor of once_make()'s objects runs, and the buffers that exist then are not zeroed. It
// ends by exit() in hf_once_internal(), CK() and step 2, among others, and by abort() at a failed
// CUDA call in prover/engine/ (ENGINE_CK()) or a failed BLK_REQUIRE() (blocks/base.h).
//
// Self-checks (`own test` in messages). A failure ends the process or makes the return 4:
// once_parts_agree(); a part's tree root is the one that its counts give
// (prover/engine/lkp_pieces.cuh); each round polynomial sums to the running claim
// (blocks/iop/psc.h); the checks of the table's callbacks (`seams` in messages;
// HfOnceTable::failed()); sampled private random values (EngineCoins::hold()); any failure flag
// (hf_once_flags()).
static int once_make(const Committed &c, const blk::Configured &configured, const std::vector<uint8_t> &root, const blk::IndexRecord &record, blk::Writer &file) {
  using namespace rf;
  const HfParams &P = c.shape.P;
  const LayW &Y = c.shape.Y;
  const blk::WfPublic &x = configured.sized().ranges.x;
  const blk::LinShape &S = configured.sized().once.S;
  const blk::WfMatrix m = blk::wf_matrix_of(S);
  const unsigned L = configured.parameters().ranges_part;
  if (!configured.refused().empty() || P.m0 != Y.m0 || P.k != Y.k || !c.key || !c.key_again || !c.kept.heap) {
    fprintf(stderr, "the key file: the configured instance, the parameters, the layout and what phase A has left do not fit each other\n");
    return 2;
  }
  if (m.k_col != (unsigned)c.shape.lk || m.rows != (blk::u64)c.shape.m0 || x.lookups.size() != 2) hf_once_internal("the configured ranges speak of another matrix than the committed one");
  if (live_flags()) hf_once_internal("a failure flag is set that belongs to no part of the key file");

  // The plans from which WfWindows generates keys (hf_wf_keys.cuh). Lookup 0 is on bytes (every
  // cell of W is a signed byte): a part of it is a call B of a plan with lwr = L, that is, of 2^L
  // places. Lookup 1 is on weights (every weight of two digits is in its range): a part is a call W
  // with lwr = L - 1. make() rejects a part of more than 2^HF_RANGE_N_MOST = 2^40 keys. Its last
  // argument, identity_bits = 30, bounds the plan's 32-bit counts of padding keys (`pub`,
  // hf_range_plan.h) by 2^30.
  HfRangePlan bytes, weights;
  for (int two = 0; two < 2; two++) {
    const std::string why = (two ? weights : bytes).make(c.shape.lk, (uint32_t)(L - two), c.shape.m0, range_units(Y), 30);
    if (!why.empty()) {
      fprintf(stderr, "the key file: the ranges' parts: %s\n", why.c_str());
      return 2;
    }
  }
  if (!once_parts_agree(x.lookups[0], wf_lookup_of(bytes, HF_RANGE_B, "bytes", 0, 0, {})) || !once_parts_agree(x.lookups[1], wf_lookup_of(weights, HF_RANGE_W, "weights", 0, 0, {})))
    hf_once_internal("the parts of the public object are not those that the windows lay");

  auto last = std::chrono::steady_clock::now();
  auto lap = [&last](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    printf("the key file: %s: %.1f s\n", stage, std::chrono::duration<double>(now - last).count());
    fflush(stdout);
    last = now;
  };

  c.weights.release();
  hf_once_ready(Y);
  const HfOncePoolsBack pools_go_back_at_every_return;
  HfOnceSlab slab; // taken in step 2. Declared before the engine, so that it is destroyed after it:
                   // what the engine allocates by cudaMalloc while the slab exists is in the slab.
  blk::EngineGpu engine;
  const std::unique_ptr<blk::CardStreamCoins> coins = ::hz::lng::engine_coins_once(::hz::lng::ENGINE_KEY_FILE);
  const blk::NonceSearch search = [&engine](const blk::ChainPrefix &prefix, unsigned bits) { return engine.search_nonce(prefix, bits); };

  // ---- the table's side: sized now, allocated in step 2
  const uint64_t columns = (uint64_t)S.fold.columns;
  HfOpen mode; // only its HfShown part and `once` are read (hf_keeper_once.cuh)
  static_cast<HfShown &>(mode) = hf_shown(P, columns);
  mode.once = true;
  HfOncePanels panels(c);
  // The folded row's memory in lent::borrowed. Its rows and row length are those that FOLDEQ, the
  // block that sends the folded row as an oracle message, passes to the table (blk::FoldedWant).
  const size_t for_tables = HfOnceTable::lent_bytes(S.fold.u.main_rows, configured.sized().key_codes.base[S.fold.u.code].ell);
  // That memory is filled from u, again after the last pass, so u must outlive the passes. Where u
  // is kept: HF_ONCE_U_ON_THE_GPU_MOST.
  const bool large = hf_side_u_elements(P) * sizeof(Ext) > HF_ONCE_U_ON_THE_GPU_MOST;
  HfOnceHostU host;
  if (large)
    if (int rc = host.take(hf_side_u_elements(P))) return rc;
  // What lent::own must hold beside the passes' buffers: omega, the panel and 1 MiB to spare.
  const size_t held_bytes = HfOnceTable::own_bytes(P) + panels.bytes() + (1 << 20);
  int fibres = (int)columns; // fibres to a pass; reduced in step 2, where the free memory is known
  struct ClosePanels { // destroyed before the slab, which holds the panel
    HfOncePanels &panels;
    ~ClosePanels() { panels.close(); }
  } close_panels_at_every_return{panels};
  HfHeld held;
  held.seed = c.key, held.key_again = c.key_again;
  held.nodes = c.kept.heap, held.s = c.kept.s;
  held.u_host = host.u, held.u_host_elements = host.u ? hf_side_u_elements(P) : 0;
  // The limit (`bears`): how many columns of one weight table may be opened (`shown` in names and
  // messages) in all. It is the number of opened columns that a row's masks (`random entries` in
  // some messages) are to hide (blk::Configured::table_columns()).
  const uint64_t bears = configured.table_columns().bears;
  if (bears != blk::hiding_positions(S.fold.index.code) || columns != configured.table_columns().one_time)
    hf_once_internal("the one-time test speaks of a table with other random entries, or draws another number of columns, than the configured instance says");
  const std::string isk = c.O.isk;
  // `isk/anchor`, the directory of the anchor's column count, is opened relative to the descriptor
  // that the run keeps of its secret working directory (making::via(), hf_commit_making.h). So
  // whoever renames the working directory during the run, or puts a link under its name, cannot
  // divert the count to another directory.
  const std::string anchor_folder = ::cp::isk_anchor_dir(isk);
  auto count = [isk, anchor_folder, bears](uint64_t shown) {
    ::cp::Counts k;
    {
#ifndef SERVING_BUILD
      const OnceCountDoor door(isk); // its file-size limit is lifted before a line is printed
#endif
      k = ::cp::count_first_table_columns(making::via(anchor_folder).fd, making::via(anchor_folder).name, shown, bears);
    }
    if (k.say == ::cp::Counts::FAULT)
      fprintf(stderr, "the key file: the first table's count (%s) cannot be raised: %s\n", ::cp::isk_anchor_count(isk).c_str(), k.error ? strerror(k.error) : "it is not the plain file that it must be");
    if (k.say == ::cp::Counts::SPENT)
      fprintf(stderr, "the key file: the first table has shown %llu columns, and its random entries bear %llu in all: %llu more are refused\n", (unsigned long long)k.begun, (unsigned long long)bears,
              (unsigned long long)shown);
    return k.say == ::cp::Counts::ROOM;
  };
  // For commit_lie::walking() (empty in this repository): the first pass is the fold, every later
  // one recomputes fibres.
  HfOnceTable table(P, Y, columns, [&panels](const HfUnits &units) { commit_lie::walking(0, panels.walks ? commit_lie::FIBRES : commit_lie::FOLD), panels.walk(units); }, held, fibres, true, root.data(), root.size(), record, count, engine);

  // ---- the ranges' side
  // WfGpu stores sums at each part's end. WfProducts, given the WfGpu, computes its products from
  // them; without them it generates the part's keys once more.
  const blk::LkpCut cut;
  HfOncePool pool; // declared before `keys`, which is lent it: at an early return `keys` is closed first, then the pool is zeroed and freed
  WfWindows keys(c, bytes, weights);
  {
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    if (c.O.gpu_memory) free_bytes = std::min(free_bytes, (size_t)c.O.gpu_memory);
    size_t others = (size_t)blk::wf_card_needs(x, m, cut) + HF_ONCE_WINDOWS_ROOM;
    if (free_bytes > others + ((size_t)1 << 30)) { // without a pool every pass copies the packed weights again: slower, the same bytes
      pool.take(free_bytes - others);
      keys.lend(pool.at, pool.bytes);
    }
  }
  WfWindows &source = keys;
  blk::WfGpu ranges(x, m, source, cut);
  blk::WfProducts products(x, m, source, cut, &ranges);
  int middle = 0; // the result of `slab.take` in step 2
  products.after_begin([&] {
    const blk::WfGpu::Report r = ranges.report();
    printf("the key file: the ranges: counting walks %.1f s; %zu passes over a part's pieces: bases %.1f, matrix passes %.1f, storing passes %.1f, rounds in memory %.1f, sums by column %.1f\n",
           r.counting, r.passes, r.base, r.matrix, r.store, r.memory, r.sums);
    lap("the ranges and the rows' vectors");
    ranges.done();
    keys.close();
    pool.give();
    c.weights.release();
    engine.release_work_space(); // what WfGpu and the engine have freed is still in the engine's pool, and the encoder keeps its work space: `cudaMemGetInfo` counts neither as free until this call; it waits for the GPU
    c.weights.reserve(::hf::WeightWalker::WHOLE_UNITS); // from the driver: 12.1 GB at full size
    // The slab is sized here, after everything else that allocates from the driver and stays, so
    // that all that is left beside the slab is the engine's.
    const size_t may = HfOnceSlab::most(c.O.gpu_memory, HF_ONCE_ENGINE_ROOM);
    if (!may) { // `take` would read a cap of 0 as no limit
      fprintf(stderr, "the key file: the GPU has too little free memory: nothing is left for the test on the table beside 2 GiB and the %zu bytes that the engine is left, whatever the size of the model\n", HF_ONCE_ENGINE_ROOM);
      exit(2);
    }
    const size_t room = may - std::min(may, for_tables + held_bytes + HF_ONCE_SLAB_SPARE);
    // As many fibres to a pass as fit, with an eighth to spare: halved until they do.
    while (fibres > 1 && hf_side_slab_bytes(P, Y, mode, c.kept.s, fibres, !large) * 9 / 8 > room) fibres = (fibres + 1) / 2;
    if (const char *word = TESTENV("ROWS_ONCE_GROUP")) fibres = std::max(1, std::min(atoi(word), (int)columns)); // a switch for tests; no byte of a file depends on it
    table.fibres_a_walk(fibres);
    // The slab takes what these numbers ask for and no more: the rest is for the engine.
    const size_t for_walks = held_bytes + hf_side_slab_bytes(P, Y, mode, c.kept.s, fibres, !large) * 9 / 8 + HF_ONCE_SLAB_SPARE;
    middle = slab.take(for_tables, for_walks, std::min(may, (for_tables + 511) / 512 * 512 + (for_walks + 511) / 512 * 512));
    if (middle) exit(middle); // a callback cannot make `make_key` return; `take` printed why
    panels.open();
    once_pool_peak(blk::card_pool(), true);
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    printf("the key file: GPU memory in bytes: the slab %zu, of which %zu for the folded row; the panel %zu; %llu columns, %d to a walk; free beside the slab: %zu\n", slab.bytes, for_tables, panels.bytes(),
           (unsigned long long)columns, fibres, free_bytes);
  });

  table.when_the_walks_are_over([&] { // step 4, once the columns are opened
    panels.close(); // and the walker's buffers
    slab.the_walks_are_over();
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    printf("the key file: the walks are over: %zu bytes of GPU memory are free\n", free_bytes);
    fflush(stdout);
  });
  if (!blk::make_key(configured, table.holder(), ranges, products, engine, *coins, search, file)) {
    fprintf(stderr, "the key file: refused: an entry of W is out of its range, or the table's masks do not cover the one-time test's columns\n");
    return 2;
  }
  ::lent::borrow = false;
  lap("the test on the table");
  char pools[160]; // for the last line; made before the flags are read: after that only hold() runs
  snprintf(pools, sizeof pools, "the engine's memory pool held %zu", once_pool_peak(blk::card_pool(), false));
  if (table.failed() != OWN_GOOD && !commit_lie::passes("seams")) {
    fprintf(stderr, "the key file: internal error: an own test of the opening's seams has failed (%d)\n", table.failed());
    return 4;
  }
  if (int rc = hf_once_flags("the key file")) return rc; // read right before the bytes are returned
  // Before bytes leave the process, hold() compares sampled private random values with a second
  // implementation of their derivation (prover/hiding/engine_coins.cuh).
  if (auto *own = dynamic_cast<::hz::lng::EngineCoins *>(coins.get())) {
    if (!own->hold()) hf_once_internal("no random value of the key file was sampled: the coins' own test by value has seen nothing"); // building a key file draws masks, pads and salts: none sampled means that `hold` compared nothing
  } else
    hf_once_internal("the key file's coins are not the program's own generator");
  printf("the key file: %zu bytes; %d walks over the weights for the test on the table; GPU memory at the peak in lent's pools: %zu bytes and %zu; beside the slab, at the most, %s; the least that is left to the engine: %zu\n",
         file.bytes().size(), panels.walks, ::lent::own.peak, ::lent::borrowed.peak, pools, HF_ONCE_ENGINE_ROOM);
  return 0;
}
} // namespace hf
