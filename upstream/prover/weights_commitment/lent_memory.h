// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU memory allocator of cp_commit and of the part of cp_prove that opens the weight commitment;
// its messages say "opening library". Its macros route cudaMalloc and cudaFree through lent::take()
// and lent::give() from the include on, in this translation unit only: include it after
// prover/common/setting.h (for TESTENV, which reads no variable in a production build) and before
// any header that calls them. Each unit has its own pools. take() and give() call the CUDA driver
// only outside an opening (begin_opening() to end_opening()) or under LENT_POISON. In an opening a
// buffer comes from one of two pools: `borrowed`, which the calling code lends from lend() to
// take_back(), or `own` (the slab), for what outlives take_back(). No function here fills `own`:
// see end_opening(). lend() and take_back() zero lent memory; end_opening() zeroes both pools.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
namespace lent { // the allocator, named after the GPU memory that the calling code lends to it
struct Piece {
  char *base;
  size_t bytes;
  std::map<size_t, size_t> free_at; // offset -> length of each free range
};
// Pieces (contiguous GPU memory ranges) to take buffers from, with the bytes in use and their peak.
struct Pool {
  std::vector<Piece> pieces;
  size_t used = 0, peak = 0;
  void add(void *base, size_t bytes) {
    Piece p;
    p.base = (char *)base;
    p.bytes = bytes;
    p.free_at[0] = bytes;
    pieces.push_back(p);
  }
  // Marks every piece as entirely free.
  void forget() {
    for (auto &p : pieces) {
      p.free_at.clear();
      p.free_at[0] = p.bytes;
    }
    used = 0;
  }
};
static Pool borrowed, own; // memory that the calling code lends (`lend`); the slab
// a buffer -> its pool, the index of its piece in the pool, and its rounded length
static std::map<void *, std::pair<Pool *, std::pair<int, size_t>>> live;
static std::mutex guard;          // take and give are called from the opening's host thread and from the calling code's
static bool slab_is_lent = false; // the slab goes back to the calling code at the opening's end
static bool in_opening = false;
static std::vector<void *> to_free; // buffers from the driver that were given back during an opening
// In an opening take() uses `borrowed` if true, else `own`. Including files set it without a lock.
static bool borrow = false;
// Takes n bytes (at least one), rounded up to a multiple of 512, from the shortest free range that
// holds them (the first of several of equal length): a buffer is 512-byte aligned if its piece is.
// Returns nullptr if no free range is long enough: take() then calls exit(4).
static void *best_fit(Pool &pool, size_t n) {
  const size_t want = (std::max<size_t>(n, 1) + 511) / 512 * 512;
  int piece = -1;
  std::map<size_t, size_t>::iterator best;
  for (size_t i = 0; i < pool.pieces.size(); i++)
    for (auto it = pool.pieces[i].free_at.begin(); it != pool.pieces[i].free_at.end(); ++it)
      if (it->second >= want && (piece < 0 || it->second < best->second)) {
        piece = (int)i;
        best = it;
      }
  if (piece < 0) return nullptr;
  const size_t at = best->first, len = best->second;
  pool.pieces[piece].free_at.erase(best);
  if (len > want) pool.pieces[piece].free_at[at + want] = len - want;
  void *p = pool.pieces[piece].base + at;
  live[p] = {&pool, {piece, want}};
  pool.used += want;
  pool.peak = std::max(pool.peak, pool.used);
  return p;
}
// With the test-only environment variable LENT_POISON=<byte, two hex digits>, take() fills every
// buffer with that byte, in an opening up to its rounded length. If the output files change with
// the byte, a buffer was read before it was written (no test in this repository checks it).
// Each GPU fill waits for the device, so such a run's times are meaningless. cp_prove and cp_commit
// pass the byte to the allocator of prover/engine/ (prover/engine/device.cuh, `card_test_byte`).
static int poison_byte() {
  static const int byte = [] {
    const char *test = TESTENV("LENT_POISON");
    return test ? (int)strtoul(test, nullptr, 16) & 255 : -1;
  }();
  return byte;
}
static cudaError_t poison(void *p, size_t n) {
  if (poison_byte() < 0) return cudaSuccess;
  const cudaError_t e = cudaMemset(p, poison_byte(), n);
  return e == cudaSuccess ? cudaDeviceSynchronize() : e;
}
static inline void poison_host(void *p, size_t n) {
  if (poison_byte() >= 0) memset(p, poison_byte(), n);
}
static cudaError_t take(void **p, size_t n) {
  std::lock_guard<std::mutex> lock(guard);
  if (!in_opening) {
    const cudaError_t e = cudaMalloc(p, n);
    return e == cudaSuccess ? poison(*p, n) : e;
  }
  *p = best_fit(borrow ? borrowed : own, n);
  if (*p) return poison(*p, live[*p].second.second);
  if (borrow)
    fprintf(stderr,
            "opening library: no free block of %zu bytes in the GPU memory that the calling program lent; the process "
            "ends. Lend more memory, or one larger piece.\n",
            n);
  else
    fprintf(stderr, "opening library: internal error: no free block of %zu bytes in the slab; the process ends\n", n);
  exit(4);
}
// Returns a buffer to its pool, without zeroing it: a later take() can return its old contents.
static cudaError_t give(void *p) {
  if (!p) return cudaSuccess;
  std::lock_guard<std::mutex> lock(guard);
  auto it = live.find(p);
  // A buffer that is not in `live` is taken to come from the driver. During an opening cudaFree()
  // would be a driver call, so the buffer waits in `to_free` until end_opening().
  if (it == live.end()) {
    if (!in_opening) return cudaFree(p);
    to_free.push_back(p);
    return cudaSuccess;
  }
  Pool &pool = *it->second.first;
  Piece &piece = pool.pieces[it->second.second.first];
  size_t at = (char *)p - piece.base, len = it->second.second.second;
  pool.used -= len;
  live.erase(it);
  auto next = piece.free_at.lower_bound(at);
  if (next != piece.free_at.end() && at + len == next->first) { // merge with the free range after it
    len += next->second;
    next = piece.free_at.erase(next);
  }
  if (next != piece.free_at.begin()) { // and with the one before it
    auto before = std::prev(next);
    if (before->first + before->second == at) {
      before->second += len;
      return cudaSuccess;
    }
  }
  piece.free_at[at] = len;
  return cudaSuccess;
}
// Zeroes every piece of the pool in full and waits for completion, because an opening writes copies
// of the weights, linear combinations of them, and masks and pads (private random values) there.
// The device-wide wait comes first, so that no earlier launch on any stream writes after the zeros.
static void wipe(Pool &pool) {
  if (pool.pieces.empty()) return;
  cudaError_t e = cudaDeviceSynchronize();
  for (auto &p : pool.pieces)
    if (e == cudaSuccess) e = cudaMemsetAsync(p.base, 0, p.bytes, cudaStreamPerThread);
  if (e == cudaSuccess) e = cudaStreamSynchronize(cudaStreamPerThread);
  if (e != cudaSuccess) {
    fprintf(stderr, "opening library: GPU memory cannot be zeroed (%s); the process ends\n", cudaGetErrorString(e));
    exit(5);
  }
}
// Adds to *count the number of the n bytes at p that are not zero. Launched only under LENT_STALE.
// Public sources for the loop and for a count returned through managed memory: see
// prover/prover/fork_memory.h, k_fork_memory_count_not().
__global__ void k_lent_count_not_zero(const unsigned char *p, size_t n, unsigned long long *count) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    c += p[i] != 0;
  atomicAdd(count, c);
}
// Zeroes memory that the calling code (the lender) lends, which still holds its data, before an
// opening takes buffers from it. It waits for the device, because the opening's host thread works
// on a separate stream. With the test-only variable LENT_STALE the memory is first overwritten with
// 0xEE, and a nonzero byte after the zeroing ends the process. LENT_STALE=kept omits the zeroing.
static void zero_what_is_lent(void *base, size_t bytes) {
  const char *test = TESTENV("LENT_STALE");
  const bool kept = test && !strcmp(test, "kept");
  cudaError_t e = cudaDeviceSynchronize(); // the lender's last writes there may still be in flight, on other streams
  if (e == cudaSuccess && test) e = cudaMemset(base, 0xEE, bytes);
  if (e == cudaSuccess && test) e = cudaDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  if (e == cudaSuccess && !kept) e = cudaMemset(base, 0, bytes);
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (e != cudaSuccess) {
    fprintf(stderr,
            "opening library: GPU memory lent by the calling program could not be zeroed (%s); the process ends\n",
            cudaGetErrorString(e));
    exit(5);
  }
  if (!test) return;
  static unsigned long long *left = nullptr;
  if (!left) e = cudaMallocManaged(&left, 8);
  if (e == cudaSuccess) {
    *left = 0;
    k_lent_count_not_zero<<<1024, 256>>>((const unsigned char *)base, bytes, left);
    e = cudaDeviceSynchronize();
  }
  if (e != cudaSuccess) {
    fprintf(stderr, "opening library: test mode: counting the bytes that are not zero failed (%s); the process ends\n",
            cudaGetErrorString(e));
    exit(5);
  }
  printf("opening library: test mode: %zu lent bytes were overwritten with 0xEE", bytes);
  if (kept)
    printf(" and, as this mode asks, NOT zeroed");
  else
    printf(", then zeroed in %.3f ms", ms);
  printf("; bytes that are not zero: %llu%s\n", *left,
         *left ? "; the process ends, because memory that must be zero is not" : "");
  fflush(stdout);
  if (*left) exit(3);
}
static void lend(void *base, size_t bytes) {
  if (!base || (uintptr_t)base % 512) {
    fprintf(stderr, "opening library: lend: the memory is null or does not start at a multiple of 512 bytes; it is not "
                    "accepted\n");
    return;
  }
  zero_what_is_lent(base, bytes);
  borrowed.add(base, bytes);
}
// Zeroes the memory lent with lend() and returns it to the calling code. If a buffer of a running
// opening is still in it, the process ends. Like lend(), begin_opening() and end_opening(), it does
// not lock `guard`: no other thread may be in take() or give() meanwhile.
static void take_back() {
  for (auto &b : live)
    if (b.second.first == &borrowed) {
      fprintf(stderr,
              "opening library: internal error: memory that is being taken back still contains a buffer of a running "
              "opening\n");
      exit(4);
    }
  wipe(borrowed);
  borrowed.pieces.clear();
  borrowed.used = 0;
}
static void begin_opening() {
  in_opening = true;
  borrow = false;
  borrowed.peak = own.peak = 0;
}
// Zeroes both pools and marks them free. A buffer still in use is forgotten: give() would pass it
// to cudaFree(). A lent slab belongs to the calling code again; memory lent with lend() stays lent.
// Other files fill `own`. In cp_prove, Unit::lend_slab() (prover/hash_commitment/hf_keeper.cu) adds
// the slab that the calling code lends and sets `slab_is_lent`. In cp_commit, HfOnceSlab::take()
// (prover/hash_commitment/hf_commit_once.cuh) puts one driver allocation of its own into each pool.
static void end_opening() {
  live.clear();
  wipe(own);
  wipe(borrowed);
  own.forget();
  if (slab_is_lent) own.pieces.clear();
  slab_is_lent = false;
  borrowed.forget();
  in_opening = false;
  for (void *p : to_free) cudaFree(p);
  to_free.clear();
}
} // namespace lent
#define cudaMalloc(p, n) lent::take((void **)(p), (size_t)(n))
#define cudaFree(p) lent::give((void *)(p))
