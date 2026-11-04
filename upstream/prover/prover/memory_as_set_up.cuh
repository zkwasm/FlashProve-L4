// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Restores part of the prover's GPU memory to its content at the end of set-up (`wipe` in
// identifiers), so that a request (`member` in member_state.cuh) finds no other request's data in
// that part. The protocol does not depend on this: a request's proof reads what that request's run
// has written. A restore bounds what a bug could do: a read of stale memory there would return
// zeros or set-up's constants. A sub-proof (`fork` in identifiers) is the argument for one part of
// the forward pass. The record is ForkMemory's list of the buffers that it has allocated; the slab
// is the sub-proofs' shared GPU scratch memory (fork_memory.h). A batch pass (`wave` in
// identifiers) is a set of requests proved on one transfer of the weights to the GPU; a turn is one
// request's work on one layer group, or its start or end.
// Nothing records which memory a run writes and which contains constants, and some buffers are
// divided into many regions. So look() classifies the record's GPU memory by its content at the end
// of set-up, in chunks of 256 KiB (`piece` in identifiers):
//   - A chunk that is all zeros, memory that is allocated later (ForkMemory returns zeroed memory)
//     and the slab are zeroed by a restore.
//   - Adjacent chunks of a buffer that are not all zeros form a range (`stretch`) of constants, so
//     set-up's leftovers must be zeroed before look(). A short range (see SHORT_MOST) has a second
//     copy on the GPU (its `twin`). A restore rewrites it from that copy, changed or not, so a
//     buffer that contains constants and is also written by runs is restored too. A long range
//     (`table` in function names) has a copy in host memory and a digest, and is checked only at
//     the end of a batch pass. Runs are not meant to write there, and nothing restores it between
//     two turns, so if it has changed the whole batch fails (wipe_behind_the_wave(), request.cuh).
//   - The weights and the buffers into which they are streamed (`rooms`) are left alone: they are
//     not meant to contain a request's data. `HfKeeperUnit`, which opens the weight table, zeroes
//     the rooms when it borrows them and before it returns them (close_memory.cuh).
//   - Memory listed in ForkMemory::laid_once is left alone. A buffer into which a run writes
//     constants for later runs must be added to that list, or a restore zeroes it.
// The restore at a request's start and, with -in-step 2 or more, before a turn, wipe(0, ..), spares
// what the requests of a batch pass keep between their turns (the state buffers, and memory
// labelled A_REQUESTS_OWN) and the slab beyond slab_bytes(0). The one at the end of a batch pass,
// wipe(1, ..), spares neither. With -overlap (a batch's opening runs during the next batch),
// weights are decoded into the slab between two turns, and only the next restore removes them;
// wave_has_ended() in in_step.cuh ends the program if they can reach beyond slab_bytes(0).
// Not reached: GPU memory that is neither in the record nor in the slab. It includes __device__ and
// __constant__ variables; what cuBLAS, the CUDA runtime and captured CUDA graphs store, launch
// arguments included; this header's tables; the memory that `HfKeeperUnit` allocates, which is left
// to that code to erase; and the GPU engine's buffers (`blk::Card`, `blk::Pooled`), which hold the
// vectors that the requests' proofs commit to, with their random tails, mask rows, salts and Merkle
// trees. They all come from one CUDA memory pool. Neither class clears what it releases, and
// nothing erases the pool between requests or at exit. Host memory is not reached either.
// Public sources: keeping only non-zero memory regions: Wizer (Bytecode Alliance, README, section
// "How Does it Work?"). Public sources: restoring state after each call: Faasm (Shillaker and
// Pietzuch, arXiv:2002.09344, section 5.2). Here this is done for GPU memory; unlike there, a long
// range is rewritten only if its digest has changed, and some memory is declared by hand.
#pragma once
#include "prover/fork_memory.h"
#include "prover/member_state.cuh"
#include <map>
#include <set>
#include <vector>

struct MemoryStretch {
  char *at;
  size_t bytes;
  unsigned first; // the number of chunks (memory_piece) of the table's earlier entries
};
// A range is processed in chunks of at most this many bytes.
constexpr size_t memory_piece = (size_t)256 << 10;
// Returns the entry of chunk blockIdx.x. Requires blocks of 256 threads; every thread must call it,
// because it contains a barrier. Not checked: n >= 1, table[0].first == 0, and `first` ascends.
// Public sources for the techniques of this function. A barrier inside a loop. Here the test e0 < n does not read
// threadIdx.x, so the block leaves together, and a thread whose index reaches n is just not counted. The loop form is
// public: NVIDIA, Best Practices Guide for version 5.5 of the CUDA Toolkit, "Synchronizing Divergent Threads in a Loop".
// The loop form is public: the same loop condition in all threads at a barrier: NVIDIA, "Parallel
// Thread Execution ISA Version 8.4", section "Parallel Synchronization and Communication
// Instructions: bar, barrier", https://docs.nvidia.com/cuda/parallel-thread-execution/.
static __device__ __forceinline__ MemoryStretch memory_stretch_of(const MemoryStretch *table, int n, int *entry) {
  int below = 0; // the number of entries e with table[e].first <= blockIdx.x
  for (int e0 = 0; e0 < n; e0 += 256)
    below += __syncthreads_count(e0 + (int)threadIdx.x < n && table[e0 + threadIdx.x].first <= blockIdx.x);
  *entry = below - 1;
  return table[below - 1];
}
// Zeroes the table's n ranges. Each must be 16-byte aligned. Needs one 256-thread block per chunk.
static __global__ void __launch_bounds__(256) k_memory_zero(const MemoryStretch *table, int n) {
  int entry;
  const MemoryStretch s = memory_stretch_of(table, n, &entry);
  const size_t from = (size_t)(blockIdx.x - s.first) * memory_piece,
               len = s.bytes - from < memory_piece ? s.bytes - from : memory_piece;
  uint4 *w = (uint4 *)(s.at + from);
#pragma unroll 4
  for (size_t i = threadIdx.x; i < len / 16; i += 256) w[i] = make_uint4(0, 0, 0, 0);
  if (threadIdx.x < (len & 15)) s.at[from + (len & ~(size_t)15) + threadIdx.x] = 0;
}
// For detecting changed memory; not a cryptographic hash. Adds to sums[2 * q] a number computed
// from the range's 8-byte words and their positions, and to sums[2 * q + 1] the count of nonzero
// words. q is the range's index or, with by_piece, the chunk's number. Ranges and launch: as above.
static __global__ void __launch_bounds__(256)
    k_memory_digest(const MemoryStretch *table, int n, bool by_piece, unsigned long long *sums) {
  int entry;
  const MemoryStretch s = memory_stretch_of(table, n, &entry);
  const size_t from = (size_t)(blockIdx.x - s.first) * memory_piece,
               len = s.bytes - from < memory_piece ? s.bytes - from : memory_piece;
  unsigned long long sum = 0, not_zero = 0;
  for (size_t i = threadIdx.x; i < (len + 7) / 8; i += 256) {
    unsigned long long x = 0;
    if (8 * i + 8 <= len)
      x = ((const unsigned long long *)(s.at + from))[i];
    else
      for (size_t j = 8 * i; j < len; j++) x = x << 8 | (unsigned char)s.at[from + j];
    not_zero += x != 0;
    x ^= (from / 8 + i + 1) * 0x9E3779B97F4A7C15ull;
    // The output mix of SplitMix64 (THIRD_PARTY.md).
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    sum += x ^ (x >> 31);
  }
  const size_t q = by_piece ? blockIdx.x : (size_t)entry;
  atomicAdd(&sums[2 * q], sum);
  atomicAdd(&sums[2 * q + 1], not_zero);
}
// For a test: changes one byte.
static __global__ void k_memory_change(char *at) {
  *at ^= 1;
}

struct MemoryAsSetUp {
  using Buffers = ForkMemory::Buffers;
  [[noreturn]] static void failed(const char *what) {
    printf("memory as set-up left it: %s (last CUDA error: %s)\n", what, cudaGetErrorString(cudaGetLastError()));
    exit(1);
  }
  // GPU memory for this header's tables and sums_dev. (cudaMalloc) bypasses fork_memory.h's macro,
  // so it is outside the record. The tables contain only addresses, lengths and chunk counts.
  static void *own_memory(size_t bytes) {
    void *p = nullptr;
    if ((cudaMalloc)(&p, bytes) != cudaSuccess) failed("no GPU memory for a table");
    return p;
  }
  // This header's copies of set-up's constants: on the GPU, labelled TWINS, and in pinned host
  // memory. Tables computed from the weights may be among the constants, so the copies are as
  // secret as the weights. Both are in the record, which ForkMemory::wipe() zeroes at every exit
  // through stop() (ends.h); it returns false if GPU memory could not be zeroed.
  static constexpr const char *TWINS = "GPU copies of the set-up's constants";
  static char *twin_memory(size_t bytes) {
    void *p = nullptr;
    if (fork_memory.device_malloc(&p, bytes) != cudaSuccess) failed("no GPU memory for a copy of constants");
    fork_memory.purpose_of[p] = TWINS;
    return (char *)p;
  }
  static char *copy_memory(size_t bytes) {
    char *p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) failed("no pinned host memory for a copy");
    return p;
  }
  // A launch's ranges, copied to the GPU on the default stream (nvcc --default-stream per-thread).
  struct List {
    MemoryStretch *table = nullptr;
    unsigned stretches = 0, blocks = 0;
    size_t bytes = 0;
    void build(const Buffers &buffers) {
      std::vector<MemoryStretch> t;
      blocks = 0, bytes = 0;
      for (auto &b : buffers) {
        if (!b.second) continue;
        if ((uintptr_t)b.first & 15) failed("an address range does not begin at a multiple of 16 bytes");
        t.push_back({(char *)b.first, b.second, blocks});
        blocks += (unsigned)((b.second + memory_piece - 1) / memory_piece);
        bytes += b.second;
      }
      if (table) (cudaFree)(table);
      table = nullptr;
      stretches = (unsigned)t.size();
      if (t.empty()) return;
      table = (MemoryStretch *)own_memory(t.size() * sizeof t[0]);
      if (cudaMemcpy(table, t.data(), t.size() * sizeof t[0], cudaMemcpyHostToDevice) != cudaSuccess)
        failed("a table could not be copied to the GPU");
    }
  };

  // The label (ForkMemory::purpose) of GPU memory that one request owns from start to end. It
  // outlasts the request's turns, so wipe(0, ..) spares it. fork_free() zeroes it when it is freed.
  static constexpr const char *A_REQUESTS_OWN = "buffers of one request, from its start to its end";
  // A range is short if it has at most SHORT_MOST bytes and all GPU copies stay within TWINS_MOST;
  // look() decides in ascending order of addresses. Every other range is long. Public sources:
  // NVIDIA's CUPTI Checkpoint API (CUPTI documentation, 2.13), made for running a kernel again
  // under a profiler; here the copy restores memory between requests. There a part of memory is
  // rewritten after a check for a difference; copies go to the GPU, then the host. SHORT_MOST and
  // TWINS_MOST are a limit per object under a total cap, as Squid's maximum_object_size_in_memory
  // sets for a web cache. Here only ranges with a host copy are tested, and rewritten if changed.
  static constexpr size_t SHORT_MOST = (size_t)16 << 20, TWINS_MOST = (size_t)1 << 30;

  // Call it once, at the end of set-up, while the GPU is idle and before the first request is read.
  // Public sources: see the top of this file (classifying by content) and SHORT_MOST (the copies).
  void look() {
    const Buffers all = of_the_record(1);
    // Which chunks are not all zeros.
    const std::vector<unsigned long long> pieces = digests(all, true);
    Buffers found;
    size_t piece = 0;
    for (auto &b : all)
      for (size_t from = 0; from < b.second; from += memory_piece, piece++) {
        if (!pieces[2 * piece + 1]) continue;
        char *const at = (char *)b.first + from;
        const size_t len = std::min(memory_piece, b.second - from);
        if (from && !found.empty() && (char *)found.back().first + found.back().second == at)
          found.back().second += len;
        else
          found.push_back({at, len});
      }
    const std::vector<unsigned long long> sums = digests(found, false);
    size_t twins = 0, copies = 0;
    int twin_entries = 0;
    for (size_t i = 0; i < found.size(); i++) {
      Constant c;
      c.at = (char *)found[i].first, c.bytes = found[i].second, c.digest = sums[2 * i];
      if (c.bytes <= SHORT_MOST && twins + c.bytes <= TWINS_MOST) {
        c.twin = twin_memory(c.bytes);
        if (cudaMemcpy(c.twin, c.at, c.bytes, cudaMemcpyDeviceToDevice) != cudaSuccess)
          failed("a copy of constants on the GPU could not be made");
        twin_entries++;
        twins += c.bytes;
      } else {
        c.copy = copy_memory(c.bytes);
        if (cudaMemcpy(c.copy, c.at, c.bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
          failed("a copy of constants in host memory could not be made");
        copies += c.bytes;
      }
      constants.push_back(c);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) failed("the scan of GPU memory at the end of the set-up failed");
    looked = true;
    size_t bytes = 0;
    for (auto &b : all) bytes += b.second;
    printf("# memory as set-up left it: %zu GPU buffers, %.3f GB, looked at: %zu stretches hold constants: %d short ones, "
           "%.3f GB, with twins on the GPU, and %zu long ones, %.3f GB, with copies in host memory; the rest is zeros\n",
           all.size(), bytes / 1e9, constants.size(), twin_entries, twins / 1e9, constants.size() - twin_entries,
           copies / 1e9);
  }

  // A restore, in stream order. `which`: 0 at a request's start and before a turn, 1 at the end of
  // a batch pass. It zeroes what_is_zeroed() and rewrites the short ranges, except their parts in
  // `but` (requests' state may share a range with constants). Not checked: for each `which`, `but`
  // never changes (the table of copies is built once); `stream` is cudaStreamPerThread (see List).
  // It blocks the host only in a call that rebuilds its list (see what_is_zeroed()).
  // Public source: Di Pietro, Lombardi and Villani (CUDA Leaks, arXiv:1305.7383, 2013, section
  // VI.B): zeroing only selected areas of GPU global memory; they would mark the areas to be
  // zeroed, whereas here the areas that are skipped are listed (what_is_zeroed()).
  void wipe(int which, const Buffers &but, cudaStream_t stream) {
    if (!looked) failed("a wipe was requested before the scan at the end of the set-up");
    Wipe &w = wipes[which];
    if (!w.twins_listed) {
      std::vector<StateCopy> lay;
      for (auto &c : constants)
        if (c.twin)
          for (auto &part : minus({{c.at, c.bytes}}, but)) {
            if ((uintptr_t)part.first & 15)
              failed("an address range that a wipe leaves out does not end at a multiple of 16 bytes");
            lay.push_back({(char *)part.first, c.twin + ((char *)part.first - c.at), part.second, w.twin_blocks});
            w.twin_blocks += state_pieces(part.second);
          }
      w.twin_entries = (int)lay.size();
      if (!lay.empty()) {
        w.twin_table = (StateCopy *)own_memory(lay.size() * sizeof lay[0]);
        if (cudaMemcpy(w.twin_table, lay.data(), lay.size() * sizeof lay[0], cudaMemcpyHostToDevice) != cudaSuccess)
          failed("the table of the GPU copies could not be copied to the GPU");
      }
      w.twins_listed = true;
    }
    if (w.allocations != fork_memory.allocations || w.slab != slab_bytes(which)) {
      // A queued launch may still read the table that is in use.
      if (cudaStreamSynchronize(stream) != cudaSuccess) failed("the wait before a list is rebuilt failed");
      w.list.build(what_is_zeroed(which, but));
      w.allocations = fork_memory.allocations, w.slab = slab_bytes(which);
    }
    if (w.list.stretches) k_memory_zero<<<w.list.blocks, 256, 0, stream>>>(w.list.table, (int)w.list.stretches);
    if (w.twin_entries) k_state_copies<<<w.twin_blocks, 256, 0, stream>>>(w.twin_table, w.twin_entries);
    CK(cudaGetLastError()); // after a failed launch the memory would keep what the last request's work left there
  }
  // The slab up to slab_bytes(which) and of_the_record(which), without the constants and `but`. It
  // changes with the record: a sub-proof allocates in its first run. wipe() then rebuilds its list.
  Buffers what_is_zeroed(int which, const Buffers &but) const {
    Buffers from = of_the_record(which);
    if (fork_memory.slab && slab_bytes(which)) from.push_back({fork_memory.slab, slab_bytes(which)});
    Buffers without = but;
    for (auto &c : constants) without.push_back({c.at, c.bytes});
    return minus(from, without);
  }
  size_t bytes_zeroed(int which) const { return wipes[which].list.bytes; }
  // How many bytes of the slab, from its start, a restore zeroes. which == 1: all that is mapped.
  // which == 0: the larger of ForkMemory::peak, beyond which no sub-proof was ever given memory,
  // and ForkMemory::lent_to, the end of the part lent for the life of the process, rounded up to
  // ForkMemory::step. More may be mapped, for code that borrows the slab while no sub-proof runs.
  // Public sources: Erasing memory only as far as a recorded high-water mark is described for the
  // Linux kernel stack (STACKLEAK; J. Edge, Trying to get STACKLEAK into the kernel, LWN.net, 2018)
  // and for GNAT's stack scrubbing with its watermark (GNAT Reference Manual, Security Hardening
  // Features); here the mark is the slab allocator's peak or the lent part's end.
  static size_t slab_bytes(int which) {
    const size_t written = std::max(fork_memory.peak, fork_memory.lent_to);
    return which == 0 ? std::min(fork_memory.mapped, ForkMemory::rounded(written)) : fork_memory.mapped;
  }

  // Call it at the end of a batch pass, after wipe(1, ..). It recomputes the long ranges' digests
  // and rewrites, from its host copy, every range whose digest has changed. It waits for the GPU
  // and returns the number of ranges rewritten. If that is not 0, the caller must fail the batch.
  size_t hold_the_tables() {
    Buffers tables;
    std::vector<const Constant *> which;
    for (auto &c : constants)
      if (!c.twin) tables.push_back({c.at, c.bytes}), which.push_back(&c);
    const std::vector<unsigned long long> sums = digests(tables, false);
    size_t laid = 0;
    for (size_t i = 0; i < tables.size(); i++) {
      if (sums[2 * i] == which[i]->digest) continue;
      if (cudaMemcpy(which[i]->at, which[i]->copy, which[i]->bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        failed("a table of constants could not be restored");
      laid++;
      printf("# memory as set-up left it: a run has changed a long stretch of constants, %zu bytes: it is restored\n",
             which[i]->bytes);
    }
    tables_laid_again += laid;
    return laid;
  }
  size_t tables_laid_again = 0; // over the life of the process
  // For a test (ForkMemory::Test::changed_table): changes the last byte of the first long range, in
  // stream order. If set-up left no long range, the run would show nothing, so the program ends.
  void change_a_table(cudaStream_t stream) {
    for (auto &c : constants) {
      if (c.twin) continue;
      k_memory_change<<<1, 1, 0, stream>>>(c.at + c.bytes - 1);
      printf("test mode: changed-table: the last byte of a long stretch of constants, %zu bytes, is changed\n", c.bytes);
      return;
    }
    printf("test mode: changed-table: the set-up left no long stretch of constants, so the run would show nothing\n");
    exit(2);
  }

  // Returns a without b, as ranges in ascending order. The ranges of a must not overlap each other.
  static Buffers minus(Buffers a, Buffers b) {
    auto by_address = [](const std::pair<void *, size_t> &x, const std::pair<void *, size_t> &y) {
      return (char *)x.first < (char *)y.first;
    };
    std::sort(a.begin(), a.end(), by_address);
    std::sort(b.begin(), b.end(), by_address);
    Buffers out;
    size_t q = 0;
    for (auto &x : a) {
      char *at = (char *)x.first, *const end = at + x.second;
      while (q < b.size() && (char *)b[q].first + b[q].second <= at) q++;
      for (size_t i = q; i < b.size() && (char *)b[i].first < end && at < end; i++) {
        char *const from = (char *)b[i].first, *const to = from + b[i].second;
        if (to <= at) continue;
        if (from > at) out.push_back({at, (size_t)(from - at)});
        at = std::max(at, to);
      }
      if (at < end) out.push_back({at, (size_t)(end - at)});
    }
    return out;
  }

private:
  struct Constant {
    char *at = nullptr;
    size_t bytes = 0;
    unsigned long long digest = 0;
    char *twin = nullptr; // on the GPU, for a short range
    char *copy = nullptr; // in pinned host memory, for a long one
  };
  struct Wipe {
    List list; // what is zeroed
    size_t allocations = ~(size_t)0, slab = ~(size_t)0; // ForkMemory::allocations, slab_bytes() when the list was made
    StateCopy *twin_table = nullptr; // what is written again from the twins
    int twin_entries = 0;
    unsigned twin_blocks = 0;
    bool twins_listed = false;
  };
  std::vector<Constant> constants;
  Wipe wipes[2];
  bool looked = false;
  List digest_list;
  unsigned long long *sums_dev = nullptr;
  size_t sums_room = 0;

  // The record's GPU memory that look() scans and a restore may change: all but the weights, their
  // rooms, the GPU copies (TWINS), laid_once and, for which == 0, memory labelled A_REQUESTS_OWN.
  Buffers of_the_record(int which) const {
    Buffers all;
    for (auto &b : fork_memory.device_buffers) {
      auto purpose = fork_memory.purpose_of.find(b.first);
      const char *const p = purpose == fork_memory.purpose_of.end() ? "" : purpose->second;
      if (!strcmp(p, "rooms for streamed weights") || !strcmp(p, "weights kept on the GPU") || !strcmp(p, TWINS)) continue;
      if (which == 0 && !strcmp(p, A_REQUESTS_OWN)) continue;
      all.push_back({b.first, b.second});
    }
    return minus(all, fork_memory.laid_once);
  }
  // Returns k_memory_digest()'s two words per range or chunk. It waits for the GPU.
  std::vector<unsigned long long> digests(const Buffers &buffers, bool by_piece) {
    if (cudaDeviceSynchronize() != cudaSuccess) failed("the wait before the digests failed");
    digest_list.build(buffers);
    std::vector<unsigned long long> sums(2 * (by_piece ? (size_t)digest_list.blocks : buffers.size()), 0);
    if (sums.empty()) return sums;
    if (digest_list.stretches != buffers.size()) failed("an address range is empty");
    if (sums_room < sums.size()) {
      if (sums_dev) (cudaFree)(sums_dev);
      sums_dev = (unsigned long long *)own_memory(sums.size() * 8);
      sums_room = sums.size();
    }
    if (cudaMemset(sums_dev, 0, sums.size() * 8) != cudaSuccess)
      failed("the buffer for the digests could not be zeroed");
    k_memory_digest<<<digest_list.blocks, 256, 0, cudaStreamPerThread>>>(digest_list.table, (int)digest_list.stretches,
                                                                         by_piece, sums_dev);
    const bool read = cudaMemcpy(sums.data(), sums_dev, sums.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
    // No restore reaches sums_dev, which is outside the record, so it is zeroed here, even after a
    // copy that failed.
    const bool zeroed = cudaMemset(sums_dev, 0, sums_room * 8) == cudaSuccess;
    if (!read) failed("the digests could not be read");
    if (!zeroed) failed("the digests' buffer could not be zeroed");
    return sums;
  }
};
static MemoryAsSetUp memory_as_set_up;
