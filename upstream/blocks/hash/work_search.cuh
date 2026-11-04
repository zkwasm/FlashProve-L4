// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Searches on a GPU for the proof-of-work nonce of one round of a Fiat-Shamir hash chain: the
// smallest nonce below `tries` that passes chain_work_passes() of blocks/hash/hash.h at `bits`
// bits (the low `bits` bits of the first word of chain_call(p, nonce, 0) are zero), or WORK_NONE if
// none does. It is the smallest so that every prover writes the same bytes. The caller chooses
// `tries` and handles WORK_NONE.
// The search's input `p` is the hash state after the round's message. The chain has absorbed Merkle
// roots computed over hidden values, and `p` decides which nonces pass, and so how many pieces (of
// 2^8 nonces) are taken and how often the atomic operations of k_work_search() run. If the proof is
// released, `p` is public (a reader computes the same `p`), and so are the search's outcome and the
// time that it takes: the nonce is written into the proof, so these counts follow a public value.
// Two searches are not public: one that fails, and one for a proof that is made and then withheld,
// whose nonce is written nowhere. What could be learnt from these is a matter of time on the
// prover's machine, and timing is not among what the proofs hide.
#pragma once
#include "hash/hash.h"
#include <cuda_runtime.h>

namespace blk {
constexpr u64 WORK_NONE = ~(u64)0;
// The state of one search, in GPU memory. Before the kernel: `nonce` = WORK_NONE, `next` = 0.
struct WorkCell {
  unsigned long long nonce; // the smallest nonce found so far
  unsigned long long next;  // the number of the next piece that no block has taken
};
// A thread block tries pieces of 2^8 consecutive nonces. Measured on another kernel with the same
// loop, not on k_work_search(): short searches end sooner than with 2^10, and long ones are no
// slower despite four times the atomic adds.
constexpr int WORK_THREADS = 256, WORK_PIECE_BITS = 8;

// Thread blocks take pieces in rising order from the counter `next`. A thread block returns when
// the piece that it has taken begins above a nonce already found, or at or above `tries`. So no
// skipped piece contains the smallest passing nonce, a piece once begun is tried to its end, and
// after the kernel `nonce` is that nonce, or WORK_NONE. No thread block waits for another.
// A block must have WORK_THREADS threads (not checked): with fewer, some nonces are never tried.
// `tries` must be below 2^63, so that `piece << BITS` cannot overflow; start() checks it.
template <class Fn>
__global__ void __launch_bounds__(WORK_THREADS) k_work_search(ChainPrefixT<Fn> p, unsigned bits, u64 tries, WorkCell *cell) {
  constexpr int BITS = WORK_PIECE_BITS;
  __shared__ u64 piece;
  __shared__ bool over;
  for (;;) {
    if (threadIdx.x == 0) {
      piece = atomicAdd(&cell->next, 1ull);
      over = piece << BITS >= tries || piece << BITS > *(volatile unsigned long long *)&cell->nonce;
    }
    __syncthreads();
    const u64 first = piece << BITS;
    const bool end = over;
    __syncthreads();
    if (end) return;
    for (int i = 0; i < (1 << BITS); i += WORK_THREADS) {
      const u64 nonce = first + (u64)(i + (int)threadIdx.x);
      if (nonce < tries && chain_work_passes(p, nonce, bits)) atomicMin(&cell->nonce, (unsigned long long)nonce);
    }
  }
}
// Owns one WorkCell and serves one search at a time: each host thread or stream needs a separate
// searcher (not checked). The first start() allocates the cell on the current GPU; later ones do
// not check the GPU. No CUDA call precedes the first start(), which search() skips if `bits` is 0.
// start() and search() end the process (BLK_REQUIRE of blocks/base.h) if a CUDA call fails,
// start()'s arguments are out of range or search()'s recheck fails.
class WorkSearcher {
public:
  WorkSearcher() = default;
  ~WorkSearcher() {
    if (cell_) (cudaFree)(cell_); // in parentheses for the reason given in `start`
  }
  WorkSearcher(const WorkSearcher &) = delete;
  WorkSearcher &operator=(const WorkSearcher &) = delete;
  // Queues the search on `stream` and returns. Later work there finds the result in cell()->nonce.
  template <class Fn> void start(const ChainPrefixT<Fn> &p, unsigned bits, u64 tries, cudaStream_t stream = cudaStreamPerThread) {
    BLK_REQUIRE(bits <= 64 && tries < (1ull << 63), "a search takes 0 to 64 bits and fewer than 2^63 tries");
    if (!cell_) {
      int device = 0, processors = 0;
      ok(cudaGetDevice(&device), "cudaGetDevice");
      ok(cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute");
      blocks_ = 4u * (unsigned)processors;
      // The parentheses select CUDA's own cudaMalloc and cudaFree even in a program that defines
      // function-like macros of these names to manage scratch memory that it reuses: the cell lives
      // as long as the searcher. It holds nothing hidden.
      ok((cudaMalloc)(&cell_, sizeof(WorkCell)), "cudaMalloc of 16 bytes");
    }
    ok(cudaMemsetAsync(&cell_->nonce, 0xFF, 8, stream), "cudaMemsetAsync"); // WORK_NONE
    ok(cudaMemsetAsync(&cell_->next, 0, 8, stream), "cudaMemsetAsync");
    k_work_search<Fn><<<blocks_, WORK_THREADS, 0, stream>>>(p, bits, tries, cell_);
    ok(cudaGetLastError(), "the kernel's launch");
  }
  const WorkCell *cell() const { return cell_; } // null before the first start()
  // Searches and waits. The host rechecks that the GPU's nonce passes, but not that it is the
  // smallest or that a WORK_NONE is right.
  template <class Fn> u64 search(const ChainPrefixT<Fn> &p, unsigned bits, u64 tries, cudaStream_t stream = cudaStreamPerThread) {
    if (bits == 0) return tries ? 0 : WORK_NONE; // every nonce passes a test of no bits
    start(p, bits, tries, stream);
    unsigned long long nonce = 0;
    ok(cudaMemcpyAsync(&nonce, &cell_->nonce, 8, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
    ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize (it also reports what earlier work of the stream has done)");
    BLK_REQUIRE(nonce == WORK_NONE || (nonce < tries && chain_work_passes(p, nonce, bits)),
                "the search for a nonce has given one that does not pass: a fault of the program or the GPU");
    return nonce;
  }

private:
  static void ok(cudaError_t e, const char *call) {
    if (e != cudaSuccess) std::fprintf(stderr, "the search for a nonce: %s: %s\n", call, cudaGetErrorString(e));
    BLK_REQUIRE(e == cudaSuccess, "a CUDA call of the search for a nonce has failed (the line above says which, and CUDA's words)");
  }
  WorkCell *cell_ = nullptr;
  unsigned blocks_ = 0; // four for each multiprocessor
};
} // namespace blk
