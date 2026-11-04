// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU memory for work buffers that are allocated and released many times a second (Pooled), and
// host-to-GPU copies of short lists that do not wait for the GPU (Uploads).
//
// cudaMalloc() and cudaFree() are slow, and cudaFree() waits for the GPU. CUDA's stream-ordered
// allocator takes memory from a pool and returns it in the order of a stream's work, without
// waiting. The pool keeps what is returned (card_pool(), prover/engine/device.cuh): reuse is cheap.
// Code that takes Pooled memory must write every word that it later reads. Memory is not cleared
// when it is released, so a buffer may hold a former owner's values, hidden ones among them.
// A failed CUDA call or a failed check ends the process (ENGINE_CK, prover/engine/device.cuh;
// BLK_REQUIRE, blocks/base.h).
// Public source: NVIDIA, CUDA Runtime API, "Stream Ordered Memory Allocator"
// (cudaMallocFromPoolAsync, cudaFreeAsync, cudaMemPoolSetAttribute with
// cudaMemPoolAttrReleaseThreshold, cudaMemPoolTrimTo),
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
#pragma once
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "engine/device.cuh"

namespace blk {

// The stream on which the calling thread allocates Pooled buffers: 0, the default stream, at first.
// OnStream sets it for a scope.
inline cudaStream_t &pool_stream() {
  static thread_local cudaStream_t stream = 0;
  return stream;
}
struct OnStream {
  const cudaStream_t before;
  explicit OnStream(cudaStream_t s) : before(pool_stream()) { pool_stream() = s; }
  ~OnStream() { pool_stream() = before; }
  OnStream(const OnStream &) = delete;
  OnStream &operator=(const OnStream &) = delete;
};
// The pool of `Card`, the buffer class for the default stream (device.cuh). This directory has one
// pool, for the reason given there.
inline cudaMemPool_t work_pool() { return card_pool(); }
// Returns the pool's unused memory to the driver, for a program that is short of memory outside the
// pool. Unlike card_pool_gives_back() (device.cuh), it does not synchronize first, so memory
// released since the last synchronization may stay in the pool.
inline void pool_trim() { ENGINE_CK(cudaMemPoolTrimTo(work_pool(), 0)); }

// n elements of T in GPU memory, with the interface of `Card` (device.cuh). Work that is queued on
// the buffer's stream (pool_stream() at the allocation) after the allocation may use the memory.
// The stream must outlive a buffer from the pool: release() queues the free on it.
// The memory comes from the arena (an allocator over GPU memory that the program already holds)
// that the calling thread has set with InArena (arena.h), or else from the pool. The pool reuses
// released memory only after the work that was queued on the stream at the release. An arena may
// reuse it at once, while that work is still queued: all work on an arena's buffers must therefore
// run on one stream, or on streams that keep order with it (CardArena, arena.h).
template <class T> class Pooled {
public:
  Pooled() = default;
  explicit Pooled(size_t n) { resize(n); }
  Pooled(Pooled &&o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)), stream_(o.stream_), from_(std::exchange(o.from_, nullptr)) {}
  Pooled &operator=(Pooled &&o) noexcept {
    if (this != &o) {
      release();
      p_ = std::exchange(o.p_, nullptr), n_ = std::exchange(o.n_, 0), stream_ = o.stream_, from_ = std::exchange(o.from_, nullptr);
    }
    return *this;
  }
  Pooled(const Pooled &) = delete;
  Pooled &operator=(const Pooled &) = delete;
  ~Pooled() { release(); }

  // Discards the contents; the new memory is uninitialized. If n is the current size, memory and
  // stream are kept, but the contents must not be relied on: tests overwrite them.
  void resize(size_t n) {
    if (n == n_) return spent();
    release();
    if (n) {
      stream_ = pool_stream();
      if ((from_ = card_arena()))
        p_ = (T *)from_->take(n * sizeof(T));
      else
        card_memory_taken(cudaMallocFromPoolAsync(&p_, n * sizeof(T), work_pool(), stream_), n * sizeof(T), __FILE__, __LINE__);
    }
    n_ = n;
    spent();
  }
  // Declares that the contents will not be read again. Does nothing unless card_test_byte
  // (device.cuh) is set, as tests do: the buffer is then filled with that byte.
  void spent() { card_fill(p_, n_ * sizeof(T), stream_); }
  void release() {
    if (p_ && from_)
      from_->give(p_);
    else if (p_)
      ENGINE_CK(cudaFreeAsync(p_, stream_));
    p_ = nullptr, n_ = 0, from_ = nullptr;
  }
  // Returns when the copy is done, and so after the stream's earlier work: the caller may then
  // change or free the host's memory. With n = 0 it does not wait.
  void from_host(const T *host, size_t n) {
    resize(n);
    if (!n) return;
    ENGINE_CK(cudaMemcpyAsync(p_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream_));
    ENGINE_CK(cudaStreamSynchronize(stream_));
  }
  // Waits for the stream, unless the buffer is empty.
  void to_host(T *host) const {
    if (!n_) return;
    ENGINE_CK(cudaMemcpyAsync(host, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost, stream_));
    ENGINE_CK(cudaStreamSynchronize(stream_));
  }
  T *get() const { return p_; }
  size_t size() const { return n_; }

private:
  T *p_ = nullptr;
  size_t n_ = 0;
  cudaStream_t stream_ = 0;
  CardArena *from_ = nullptr; // null: CUDA's pool
};

// Host-to-GPU copies of short lists, without a wait. Pooled::from_host() waits for the stream, so
// a function that uploads twenty short lists between its launches runs in step with the GPU.
// Leaving out the wait need not help while the lists are in pageable memory. Handed such a list,
// cudaMemcpyAsync() might hold the caller up much as the wait did; only for a page-locked source do
// we expect it to return with the copy merely queued. Both rest on NVIDIA, CUDA Runtime API,
// "API synchronization behavior", https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html.
// send() copies the list into the calling thread's room (a staging buffer of ROOM bytes of
// page-locked host memory) and only queues the copy from there, so the caller may change or free
// the list when send() returns.
// A list that does not fit is copied directly, with a wait. The room is used as a stack: the
// destructor gives back what its object took, after a wait for the stream unless finished() was
// called after the last send().
// Only for lists that are not secret (descriptors, numbers, coefficients that a verifier knows
// too). The room is not cleared, is kept until the process ends, and passes from a thread that ends
// to the next one that asks. The prover's private random values (masks, salts) do not belong here
// (device.cuh says which host copies of them are wiped and which are not).
// Use on a thread other than the creating one, or out of stack order, ends the process. A thread is
// recognized by the address of its thread_local `Room`, so a later thread that gets the address of
// a finished one is not noticed.
class Uploads {
public:
  static constexpr size_t ROOM = (size_t)32 << 20; // bytes for a thread
  explicit Uploads(cudaStream_t stream) : stream_(stream), mine_(&room()), before_(mine_->used), number_(++mine_->alive) {}
  ~Uploads() {
    BLK_REQUIRE(mine_ == &room(), "uploads: an object ends on another thread than made it");
    BLK_REQUIRE(number_ == mine_->alive, "uploads: a thread's objects end in another order than last made, first ended");
    if (mine_->used != before_ && !finished_) ENGINE_CK(cudaStreamSynchronize(stream_));
    mine_->used = before_, mine_->alive--;
  }
  Uploads(const Uploads &) = delete;
  Uploads &operator=(const Uploads &) = delete;
  // Resizes `to` to n elements and fills it. pool_stream() must be this object's stream, or the
  // process ends. Not checked: a `to` that already has n elements keeps the stream on which it was
  // allocated.
  template <class T> void send(Pooled<T> &to, const T *host, size_t n) {
    BLK_REQUIRE(pool_stream() == stream_, "uploads: the buffer would be taken on another stream than it is filled on");
    to.resize(n);
    send_to(to.get(), host, n);
  }
  // The same, into n elements of GPU memory that the caller owns.
  template <class T> void send_to(T *to, const T *host, size_t n) {
    Room &r = room();
    BLK_REQUIRE(mine_ == &r, "uploads: an object is used on another thread than made it");
    BLK_REQUIRE(number_ == r.alive, "uploads: an object sends while a later one of its thread lives");
    const size_t bytes = n * sizeof(T), at = (r.used + 63) / 64 * 64;
    if (!n) return;
    if (!r.base) {
      ENGINE_CK(cudaHostAlloc(&r.base, ROOM, cudaHostAllocDefault));
      made()++;
    }
    if (at + bytes > ROOM) {
      ENGINE_CK(cudaMemcpyAsync(to, host, bytes, cudaMemcpyHostToDevice, stream_));
      ENGINE_CK(cudaStreamSynchronize(stream_));
      return;
    }
    std::memcpy(r.base + at, host, bytes);
    r.used = at + bytes, finished_ = false;
    ENGINE_CK(cudaMemcpyAsync(to, r.base + at, bytes, cudaMemcpyHostToDevice, stream_));
  }
  // States, unchecked, that the caller has waited for the stream since the last send(). If not, the
  // destructor gives back room that a queued copy still reads, and the GPU gets the lists of the
  // next user of that room, which may be another thread.
  void finished() { finished_ = true; }
  // Rooms allocated so far, of ROOM bytes each and never freed: all of warm()'s, and by send_to()
  // at most the largest number of threads that have had a room at one time (see room()).
  static size_t rooms_made() { return made().load(); }
  // Allocates n rooms for the threads that will ask. Meant for a program's set-up; no program calls
  // it. Without it a thread may allocate at its first send(), in the middle of a
  // batch: that costs time (streams do not run side by side across such a call), and a failure
  // ends the process there and not at set-up.
  static void warm(unsigned n) {
    for (unsigned i = 0; i < n; i++) {
      char *base = nullptr;
      ENGINE_CK(cudaHostAlloc(&base, ROOM, cudaHostAllocDefault));
      made()++;
      GivenBack &list = given_back();
      const std::lock_guard<std::mutex> held(list.lock);
      list.rooms.push_back(base);
    }
  }

private:
  struct Room {
    char *base = nullptr;
    size_t used = 0;
    unsigned alive = 0; // objects of this thread
  };
  // room() is the calling thread's room, held from the thread's first Uploads until the thread
  // ends. Its memory then goes to given_back() for the next thread that asks; otherwise a program
  // that starts a thread for every batch would leave ROOM bytes page-locked for each. Safe if every
  // Uploads is destroyed before its thread ends and finished() was called truthfully: then no
  // queued copy still reads the room.
  // Same pattern: handle_of_this_thread(), prover/common/blas.h.
  struct GivenBack {
    std::mutex lock;
    std::vector<char *> rooms;
  };
  static GivenBack &given_back() {
    static GivenBack *list = new GivenBack; // (never deleted: a thread may end after main())
    return *list;
  }
  static Room &room() {
    struct Held : Room {
      Held() {
        GivenBack &list = given_back();
        const std::lock_guard<std::mutex> held(list.lock);
        if (!list.rooms.empty()) base = list.rooms.back(), list.rooms.pop_back();
      }
      ~Held() {
        if (!base) return;
        GivenBack &list = given_back();
        const std::lock_guard<std::mutex> held(list.lock);
        list.rooms.push_back(base);
      }
    };
    static thread_local Held mine;
    return mine;
  }
  static std::atomic<size_t> &made() {
    static std::atomic<size_t> rooms{0};
    return rooms;
  }
  cudaStream_t stream_;
  Room *mine_;      // the creating thread's
  size_t before_;
  unsigned number_; // among the thread's living objects, from 1
  bool finished_ = true;
};

} // namespace blk
