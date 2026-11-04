// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests of code that takes GPU memory through the engine's buffer classes, `Card`
// (prover/engine/device.cuh) and `Pooled` (prover/engine/pooled.cuh). The loop below runs a case on
// memory in four states, to catch a result that depends on memory that the code never wrote:
//   - an arena (`CardArena`, prover/engine/arena.h: an allocator over memory that it is given)
//     that fills each buffer with 0x00 or with 0xEE as it is taken;
//   - CUDA's memory pool (card_pool()), which both classes use if `InArena` selects no arena:
//     enter() trims it and leaves in it a free range of POOL_BYTES filled with 0xEE or with 0x11.
// Such a result usually changes with the byte, and then cannot match the expected value under both
// bytes. Zeros are not enough: code that forgets to clear passes on them.
// differs() only counts, and the destructor only prints the counts: the test compares, and must
// itself fail on a difference, or if runs_beyond_the_fill() is not 0 (memory in no known state).
// Not tested: reusing arena memory while its former owner's work is queued; `on_take` waits for it.
//   FilledArena memory("my_test");
//   for (memory.first(); memory.more(); memory.next()) {
//     const InArena in(memory.arena()); .. run, compare; on a difference: memory.differs() .. }
#pragma once
#include <cstdio>

#include "engine/arena.h"
#include "engine/device.cuh"

namespace blk {
class FilledArena {
public:
  // The number of memory states.
  static constexpr size_t PASSES = 4;
  static constexpr size_t POOL_BYTES = (size_t)3 << 29; // 1.5 GiB
  // `name` starts each printed line; the pointer is kept, so the string must outlive the object.
  // `bytes` is the arena's size: a run that does not fit in it aborts in CardArena::take().
  explicit FilledArena(const char *name, size_t bytes = (size_t)2 << 30) : name_(name), arena_(name) {
    ENGINE_CK(cudaMalloc(&lent_, bytes));
    arena_.lend(lent_, bytes);
    arena_.on_take = [this](void *p, size_t n) {
      ENGINE_CK(cudaDeviceSynchronize()); // queued work of the former owner may still use `p`
      ENGINE_CK(cudaMemset(p, FILL[pass_], n));
      ENGINE_CK(cudaDeviceSynchronize());
    };
  }
  // Every arena buffer must be freed by now, or CardArena::take_back() aborts. Prints the counts.
  ~FilledArena() {
    ENGINE_CK(cudaDeviceSynchronize());
    for (size_t i = 0; i < PASSES; i++) std::printf("%s: on %s: %d differences\n", name_, NAME[i], differences_[i]);
    std::printf("%s: the arena: %zu buffers taken, at most %.1f MB at a time, %zu bytes in use at the end\n", name_, arena_.takes(), arena_.peak() / 1e6, arena_.used());
    std::printf("%s: the pool: filled %zu times; in %zu of these runs it grew beyond the filled %.1f MB (memory in no known state)\n", name_, pool_fills_, pool_grew_, POOL_BYTES / 1e6);
    arena_.take_back();
    ENGINE_CK(cudaFree(lent_));
  }
  FilledArena(const FilledArena &) = delete;
  FilledArena &operator=(const FilledArena &) = delete;
  // Not thread-safe: threads that share one FilledArena must serialize their calls to it.
  void first() { pass_ = 0, enter(); }
  bool more() {
    if (pass_ < PASSES) return true;
    pass_ = 0; // what runs outside the loop counts as the first state
    return false;
  }
  void next() { leave(), pass_++, enter(); }
  CardArena *arena() { return on_the_pool(pass_) ? nullptr : &arena_; } // for InArena; null: CUDA's pool
  size_t runs_beyond_the_fill() const { return pool_grew_; } // in how many runs on CUDA's pool the pool grew beyond what was filled
  const char *state() const { return NAME[pass_]; }
  int differs() { return differences_[pass_]++; } // the number of earlier differences in this state

private:
  static constexpr bool on_the_pool(size_t pass) { return pass == 0 || pass == 3; }
  // Before a run on the pool. Earlier runs' buffers must be freed: one still in use is not filled.
  // Nor is free memory in a driver allocation that also backs a buffer in use (a test's inputs,
  // say): cudaMemPoolTrimTo() cannot release such an allocation, and a run's buffer may come from
  // it. Otherwise the run's buffers come from the filled range or from what the same run has freed,
  // unless the pool grows: see leave().
  void enter() {
    if (pass_ >= PASSES || !on_the_pool(pass_)) return;
    card_pool_gives_back(); // waits for the GPU
    {
      const InArena none(nullptr); // whatever arena the caller selected: this buffer is the pool's
      Card<unsigned char> all(POOL_BYTES);
      ENGINE_CK(cudaMemset(all.get(), FILL[pass_], POOL_BYTES));
      ENGINE_CK(cudaDeviceSynchronize());
    }
    ENGINE_CK(cudaDeviceSynchronize()); // `all` was freed asynchronously, on the default stream
    pool_fills_++, held_behind_the_fill_ = card_pool_holds(); // the filled range, and what the test holds from before
  }
  // After a run on the pool. Counts the run if the pool has grown: it then took memory from the
  // driver, in no known state, as happens when the run uses more than POOL_BYTES at one time.
  // Not counted: a run that next() does not end (only next() calls leave()), and growth after a
  // trim within the run (card_pool_gives_back(), pool_trim()) up to the size after the fill.
  void leave() {
    if (pass_ < PASSES && on_the_pool(pass_) && card_pool_holds() > held_behind_the_fill_) pool_grew_++;
  }
  // For each state: the fill byte, and the name that state() returns and the destructor prints.
  // If card_test_byte (prover/engine/device.cuh) is 0 or more, `Card` and `Pooled` overwrite each
  // buffer with it after the fill, in every state: leave it at -1.
  static constexpr int FILL[PASSES] = {0xEE, 0x00, 0xEE, 0x11};
  static constexpr const char *NAME[PASSES] = {"CUDA's pool behind a fill with 0xEE", "an arena filled with 0x00", "an arena filled with 0xEE", "CUDA's pool behind a fill with 0x11"};
  const char *name_;
  CardArena arena_;
  void *lent_ = nullptr;
  size_t pass_ = 0;
  int differences_[PASSES] = {0, 0, 0, 0};
  size_t pool_fills_ = 0, pool_grew_ = 0, held_behind_the_fill_ = 0;
};
} // namespace blk
