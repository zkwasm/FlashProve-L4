// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks of CUDA calls, the memory pool of the GPU (`card` in identifiers) and the GPU buffer Card.
// In this directory GPU memory is not cleared when it returns to the pool, an arena or the driver:
// codewords, random tails, salts, mask rows, cells and sums stay until overwritten. That none
// reaches a proof rests on one rule: code that takes memory writes every word that it later reads
// (tests probe it: card_test_byte; a copy can break it: prover/hiding/door_chain.cuh, exception (1)). The
// rule is no defence against another process on the machine or a dump of its memory: the prover's machine is
// trusted. Exceptions: member_heavy.cuh, lkp_pieces.cuh and wf_heavy.cuh say at the top which buffers they
// zero (wf_heavy.cuh lists only those that held keys or counts); products.cuh zeroes cuBLAS's workspace. Host
// copies of private random values: a tree's tails and salts while off the GPU and entries set in an adopted
// table are wiped when freed; samples() (stream_coins.cuh) wipes its notes. Not wiped: what samples() returns
// (its caller must wipe it), vectors local to a call (except set_entries()'s), a tree's mask rows.
#pragma once
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "base.h"
#include "engine/arena.h"

// A failed call ends the process, in cp_prove after a wipe whose GPU part fails on a sticky error.
#define ENGINE_CK(call) \
  do { \
    const cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      ::blk::end_at_a_failed_guard("cuda-call"); \
    } \
  } while (0)
// cudaFuncSetAttribute(), after a check of the calling thread's CUDA status. On CUDA 13.0, the only
// version tried, a successful call resets that status, which would hide an earlier rejected launch
// (prover/docs/cuda-status.md, "A call that resets the status"). This directory has no other call.
#define ENGINE_CK_ATTRIBUTE(...) \
  do { \
    ENGINE_CK(cudaGetLastError()); \
    ENGINE_CK(cudaFuncSetAttribute(__VA_ARGS__)); \
  } while (0)

namespace blk {

// On a failed allocation, prints the bytes requested and the bytes free, then ends the process.
inline void card_memory_taken(cudaError_t status, size_t bytes, const char *file, int line) {
  if (status == cudaSuccess) return;
  size_t free_bytes = 0, total = 0;
  (void)cudaMemGetInfo(&free_bytes, &total);
  std::fprintf(stderr, "%s:%d: %s: %zu bytes were asked for; the driver calls %zu of %zu bytes free (what this process's memory pools keep unused is not among them)\n", file, line,
               cudaGetErrorString(status), bytes, free_bytes, total);
  end_at_a_failed_guard("cuda-call");
}

// The GPU's memory pool. Card allocates from it and frees to it in the order of the default stream,
// so neither call waits for the GPU. The pool is set to keep all freed memory, so a buffer of a
// size used before costs microseconds. Work on a buffer must run on the default stream or on a
// stream that keeps order with it. Under that rule a buffer may be freed, or allocated again, while
// kernels that use it are still queued: the free is queued after those kernels, and so is whatever
// the next owner queues. Memory that the pool keeps is lost to cudaMalloc() elsewhere in the
// process until card_pool_gives_back() is called. A process uses one GPU: the pool belongs to the
// device that is current at the first call. prover/build.sh makes the default stream per-thread.
// (NVIDIA, CUDA Programming Guide, "Stream Ordered Memory Allocator".)
inline cudaMemPool_t card_pool() {
  static const cudaMemPool_t pool = [] {
    int device = 0;
    cudaMemPool_t p;
    unsigned long long all = ~0ull;
    ENGINE_CK(cudaGetDevice(&device));
    ENGINE_CK(cudaDeviceGetMemPool(&p, device));
    ENGINE_CK(cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &all));
    return p;
  }();
  return pool;
}
// Returns the pool's unused memory to the driver. Waits for the GPU.
inline void card_pool_gives_back() {
  ENGINE_CK(cudaDeviceSynchronize());
  ENGINE_CK(cudaMemPoolTrimTo(card_pool(), 0));
}
// Why one pool (pooled.cuh uses it too). An allocation that the pool can serve costs under a
// microsecond of host time; others go to the driver: about 4 ms per GiB on the calling thread. If
// the driver lacks memory for a second pool, it takes back all that a pool keeps unused, whatever
// its release threshold, and the pool fetches it again. Measured, two pools with peaks of 12 GiB in
// turn: 84 ms every turn with 16 GiB left on the GPU; with 30 GiB, nothing after the first turn.
inline size_t card_pool_number(cudaMemPoolAttr which) {
  unsigned long long bytes = 0;
  ENGINE_CK(cudaMemPoolGetAttribute(card_pool(), which, &bytes));
  return (size_t)bytes;
}
// A host call, microseconds: the pool's bytes reserved from the driver now. Shapes alone do not
// fix them: a fold sizes some buffers by which Tensors have equal factors (fold.cuh, key_of()). The
// variation is not measured.
inline size_t card_pool_holds() { return card_pool_number(cudaMemPoolAttrReservedMemCurrent); }

// A test's fill byte, to probe the rule at the top of this file, which no program enforces. While
// it is 0 to 255, every Card or Pooled buffer is filled with it: when allocated, when resize()
// keeps its length, and when its owner calls spent(). A word read but never written then holds the
// byte, so runs with different bytes can differ. Set it while no other thread allocates. Test
// builds of cp_prove and cp_commit set it from the environment variable LENT_POISON; the test that
// compares the runs is not in this repository. A production build (`SERVING_BUILD`) has none of it.
#ifdef SERVING_BUILD
inline void card_fill(void *, size_t, cudaStream_t) {}
#else
inline int card_test_byte = -1;
inline std::atomic<unsigned long long> card_fills{0}, card_bytes_filled{0}; // so that a test can say that it was not empty
inline void card_fill(void *p, size_t bytes, cudaStream_t stream) {
  if (card_test_byte < 0 || !bytes) return;
  ENGINE_CK(cudaMemsetAsync(p, card_test_byte, bytes, stream));
  card_fills++, card_bytes_filled += bytes;
}
#endif

// n elements of T in GPU memory. If the calling thread has set an arena (InArena), the memory comes
// from it and returns to it, under the two rules in arena.h. By its rule of order, the range of a
// Card destroyed while a kernel still uses it may be reused at once: safe only if the next writer
// is queued after that kernel, which two streams do not ensure under --default-stream per-thread.
template <class T> class Card {
public:
  Card() = default;
  explicit Card(size_t n) { resize(n); }
  Card(Card &&o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)), from_(std::exchange(o.from_, nullptr)) {}
  Card &operator=(Card &&o) noexcept {
    if (this != &o) {
      release();
      p_ = std::exchange(o.p_, nullptr), n_ = std::exchange(o.n_, 0), from_ = std::exchange(o.from_, nullptr);
    }
    return *this;
  }
  Card(const Card &) = delete;
  Card &operator=(const Card &) = delete;
  ~Card() { release(); }

  // Discards the contents; the new memory is uninitialized.
  void resize(size_t n) {
    if (n == n_) return spent();
    release();
    if (n && (from_ = card_arena()))
      p_ = (T *)from_->take(n * sizeof(T));
    else if (n)
      card_memory_taken(cudaMallocFromPoolAsync(&p_, n * sizeof(T), card_pool(), 0), n * sizeof(T), __FILE__, __LINE__);
    n_ = n;
    spent();
  }
  // Declares that the old contents will not be read again. A no-op unless card_test_byte is set.
  void spent() { card_fill(p_, n_ * sizeof(T), 0); }
  void release() {
    if (p_ && from_)
      from_->give(p_);
    else if (p_)
      ENGINE_CK(cudaFreeAsync(p_, 0));
    p_ = nullptr, n_ = 0, from_ = nullptr;
  }
  void from_host(const T *host, size_t n) {
    resize(n);
    if (n) ENGINE_CK(cudaMemcpy(p_, host, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void to_host(T *host) const {
    if (n_) ENGINE_CK(cudaMemcpy(host, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
  T *get() const { return p_; }
  size_t size() const { return n_; }

private:
  T *p_ = nullptr;
  size_t n_ = 0;
  CardArena *from_ = nullptr; // null: CUDA's pool
};
// ceil(items / threads). Not checked: that it fits 32 bits. 0 items give 0 blocks, an invalid grid.
inline unsigned blocks_for(u64 items, unsigned threads) { return (unsigned)((items + threads - 1) / threads); }

} // namespace blk
