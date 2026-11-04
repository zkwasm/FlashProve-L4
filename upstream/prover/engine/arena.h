// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A host-side allocator that cuts GPU buffers (`Card`, prover/engine/device.cuh; `Pooled`,
// prover/engine/pooled.cuh) from GPU memory that the program already holds, which CUDA's memory
// pools cannot do. It is for a prover that serves requests: if the prover allocates all its GPU
// memory at set-up and lends it to an arena, no batch (requests proved together) asks the driver
// for memory, so none fails because the driver has none left, and the peak is known before the
// first request. Memory that is idle in one part of a batch can serve another part.
// The programs cp_commit (the committer) and cp_prove (the prover) create no arena; tests in
// prover/engine/tests/ do.
// A caller must keep the two rules at CardArena, on streams and on old contents: the arena checks
// neither.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace blk {

// A list of pieces (ranges of GPU memory passed to lend()), each with its free ranges. The arena
// handles addresses only: it never reads, writes or frees the memory. Misuse that it detects ends
// the process: die() prints a line to stderr and calls std::abort(). Destroying the arena while a
// buffer is live (taken, not given back) is such misuse. A mutex makes calls thread-safe; it orders
// no GPU work.
//
// Order of streams. A range that is given back may be taken again at once, while its former owner's
// kernels are still queued. That is safe only if the new owner's work is queued after them: all
// work on one arena's buffers must run on one stream, or on streams that keep order with each
// other (work queued on one starts only after the work queued earlier on the other is done).
//  - Built with the legacy default stream: a stream created without cudaStreamNonBlocking keeps
//    order with stream 0, but two such streams keep none with each other. An arena can serve
//    stream 0 and at most one other stream. (The GPU engine, `EngineGpu` in
//    prover/engine/engine_gpu.cuh, rejects a cudaStreamNonBlocking stream.)
//  - Built with nvcc's --default-stream per-thread, as prover/build.sh does: stream 0 is the
//    calling thread's own stream and keeps order with no created stream and no other thread's. An
//    engine and its arena must be used by one thread, on that thread's default stream (0 or
//    cudaStreamPerThread; EngineGpu rejects any other). Two threads need two arenas.
//
// Old contents. A buffer is not cleared: it may contain a former owner's values, among them hidden
// ones (values that a proof must not reveal, such as weights). The code that takes a buffer must
// write every byte that it later reads. The code that lends a piece containing hidden values, or
// takes one back, must zero it; take_back() reports how much of each piece was ever used.
class CardArena {
 public:
  // Bytes. Every buffer's length and its offset in its piece are multiples of STEP.
  static constexpr size_t STEP = 512;
  explicit CardArena(std::string name) : name_(std::move(name)) {}
  CardArena(const CardArena &) = delete;
  CardArena &operator=(const CardArena &) = delete;
  ~CardArena() {
    if (!live_.empty()) die("is destroyed while buffers of it live", 0);
  }

  // Adds a piece. Pieces must not overlap. Bytes beyond the last multiple of STEP are not used.
  void lend(void *base, size_t bytes) {
    std::lock_guard<std::mutex> lock(guard_);
    Piece p;
    p.base = (char *)base, p.bytes = bytes / STEP * STEP;
    if (!base || !p.bytes) die("is lent nothing", bytes);
    for (const Piece &q : pieces_)
      if (p.base < q.base + q.bytes && q.base < p.base + p.bytes) die("is lent a piece that overlaps another", bytes);
    p.free_at[0] = p.bytes;
    pieces_.push_back(std::move(p));
    lent_ += bytes / STEP * STEP;
  }
  // Returns all pieces to the caller, each as (address, bytes from its start to the end of the
  // highest buffer ever cut from it). No buffer may be live. peak() and takes() keep their values.
  std::vector<std::pair<void *, size_t>> take_back() {
    std::lock_guard<std::mutex> lock(guard_);
    if (!live_.empty()) die("is taken back while buffers of it live", 0);
    std::vector<std::pair<void *, size_t>> out;
    for (const Piece &p : pieces_) out.emplace_back(p.base, p.touched);
    pieces_.clear();
    lent_ = 0;
    return out;
  }

  // Returns a buffer of `bytes` rounded up to a positive multiple of `STEP`, cut from the shortest
  // free range that holds it (the first of several equally short). If no free range is long enough
  // the process ends. So that a request's data cannot cause this, every length and the order of
  // calls must follow from public shapes (sizes that the verifier also knows): the arena cannot
  // check that.
  void *take(size_t bytes) {
    std::lock_guard<std::mutex> lock(guard_);
    const size_t want = (std::max<size_t>(bytes, 1) + STEP - 1) / STEP * STEP;
    Piece *in = nullptr;
    std::map<size_t, size_t>::iterator best;
    for (Piece &p : pieces_)
      for (auto it = p.free_at.begin(); it != p.free_at.end(); ++it)
        if (it->second >= want && (!in || it->second < best->second)) in = &p, best = it;
    if (!in) die("has no free range for a buffer", want);
    const size_t at = best->first, length = best->second;
    in->free_at.erase(best);
    if (length > want) in->free_at[at + want] = length - want;
    in->touched = std::max(in->touched, at + want);
    void *p = in->base + at;
    live_[p] = want;
    used_ += want, peak_ = std::max(peak_, used_), takes_++;
    if (on_take) on_take(p, want);
    return p;
  }
  // Frees a buffer and joins its range with free neighbours.
  void give(void *p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(guard_);
    const auto it = live_.find(p);
    if (it == live_.end()) die("is given a buffer that is not its own, or twice", 0);
    size_t length = it->second;
    live_.erase(it);
    used_ -= length;
    for (Piece &q : pieces_) {
      if ((char *)p < q.base || (char *)p >= q.base + q.bytes) continue;
      size_t at = (size_t)((char *)p - q.base);
      auto next = q.free_at.lower_bound(at);
      if (next != q.free_at.end() && next->first == at + length) length += next->second, next = q.free_at.erase(next);
      if (next != q.free_at.begin()) {
        auto before = std::prev(next);
        if (before->first + before->second == at) at = before->first, length += before->second, q.free_at.erase(before);
      }
      q.free_at[at] = length;
      return;
    }
    die("has lost the piece of a buffer", 0);
  }

  // Bytes, in multiples of `STEP`; takes(): the number of take() calls.
  size_t lent() const { return locked(lent_); }
  size_t used() const { return locked(used_); }
  size_t peak() const { return locked(peak_); }  // the most bytes in live buffers at one time
  size_t takes() const { return locked(takes_); }
  size_t longest_free() const {
    std::lock_guard<std::mutex> lock(guard_);
    return longest();
  }
  const std::string &name() const { return name_; }

  // For tests: take() calls it with every buffer that it returns, under the arena's lock, so it
  // must not call the arena. FilledArena (prover/engine/tests/filled_arena.cuh) fills each buffer
  // with one byte value in one run and another in the next, to catch a result that depends on
  // memory that was never written. It waits for the GPU before each fill, so it does not test the
  // rule on streams.
  std::function<void(void *, size_t)> on_take;

 private:
  // touched: the byte count that take_back() reports.
  struct Piece {
    char *base = nullptr;
    size_t bytes = 0, touched = 0;
    std::map<size_t, size_t> free_at;  // offset -> length of each free range; no two touch
  };
  size_t locked(const size_t &x) const {
    std::lock_guard<std::mutex> lock(guard_);
    return x;
  }
  size_t longest() const {
    size_t most = 0;
    for (const Piece &p : pieces_)
      for (const auto &f : p.free_at) most = std::max(most, f.second);
    return most;
  }
  [[noreturn]] void die(const char *what, size_t bytes) const {
    std::fprintf(stderr, "GPU memory: the arena \"%s\" %s (%zu bytes): %zu bytes lent in %zu pieces, %zu in %zu buffers, longest free range %zu; the process ends\n",
                 name_.c_str(), what, bytes, lent_, pieces_.size(), used_, live_.size(), longest());
    std::abort();
  }
  const std::string name_;
  mutable std::mutex guard_;
  std::vector<Piece> pieces_;
  std::map<void *, size_t> live_;
  size_t lent_ = 0, used_ = 0, peak_ = 0, takes_ = 0;
};

// The arena from which the calling thread's `Card` and `Pooled` buffers are cut; null: CUDA's pool.
inline CardArena *&card_arena() {
  static thread_local CardArena *arena = nullptr;
  return arena;
}
// Sets the calling thread's arena for this object's lifetime, then restores the former one. A
// buffer goes back to the arena that it was cut from, whichever is set when it is released.
struct InArena {
  CardArena *const before;
  explicit InArena(CardArena *a) : before(card_arena()) { card_arena() = a; }
  ~InArena() { card_arena() = before; }
  InArena(const InArena &) = delete;
  InArena &operator=(const InArena &) = delete;
};

}  // namespace blk
