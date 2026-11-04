// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test helper that finds GPU buffers freed with nonzero content, for tests of classes that promise
// to zero every GPU buffer that has held a hidden value before they free it. For these classes a
// hidden value is one that depends on the model's weights. One such class's check (anything_left()
// in prover/engine/lkp_pieces.cuh) goes through the same list of buffers as its zeroing, wipe():
// neither sees a buffer that the list omits. This helper has no list. The code under test allocates
// from an arena (`CardArena`, prover/engine/arena.h: an allocator over borrowed memory), here one
// region of zeros, so a nonzero 64-bit word in free memory was written and not zeroed. It is
// charged to the allocation (`take` in identifiers and output) that last covered its range of
// CardArena::STEP (512) bytes, even if it was written through another buffer.
// Covered: buffers of two classes for arrays in GPU memory, `Card` (prover/engine/device.cuh) and
// `Pooled` (prover/engine/pooled.cuh), that are allocated while an `InArena` object sets this arena
// for the allocating thread.
// Not covered: memory from elsewhere (cudaMalloc, a library's workspace, CUDA's pool where no arena
// is set), host memory, registers and caches, and a value that its buffer's owner overwrote.
// Every buffer starts as zeros here while `card_test_byte` (prover/engine/device.cuh) is -1, which
// hides a read of memory that was never written: prover/engine/tests/filled_arena.cuh tests for
// that.
#pragma once
#ifndef BLK_LEFT_BEHIND_LOOK_FOR_A_TEST
#error "say #define BLK_LEFT_BEHIND_LOOK_FOR_A_TEST 1 before the first include: else the engine's calls of left_unzeroed() are empty and the look would count every public buffer"
#endif
#include <cxxabi.h>
#include <execinfo.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "engine/arena.h"
#include "engine/device.cuh"
#include "engine/left_unzeroed.h"

namespace blk {
// not_zero[i]: how many of the 64 words of range i (CardArena::STEP bytes) are not zero.
static __global__ void left_behind_k_ranges(const unsigned long long *words, size_t ranges, unsigned char *not_zero) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < ranges; i += (size_t)gridDim.x * blockDim.x) {
    unsigned n = 0;
    for (unsigned k = 0; k < CardArena::STEP / 8; k++) n += words[i * (CardArena::STEP / 8) + k] != 0;
    not_zero[i] = (unsigned char)n;
  }
}
// A range is checked and zeroed each time it is allocated, before CardArena::take() returns;
// left() does the same to all ranges. So what is found in a range was written since its last
// allocation, and what a later owner would have overwritten or zeroed is still found.
// A buffer that held nothing hidden may be freed as it is. The code under test marks such a
// buffer, once it is allocated, with left_unzeroed(buffer, reason)
// (prover/engine/left_unzeroed.h), which reaches this object only on the thread that constructed
// it. Words left by a marked allocation are exempt (`let`) and printed with the reason
// (`ground`); all others count. The reason is the source's claim: nothing here checks it, so a
// hidden value left in a marked buffer is not counted.
// Allocate the test's own buffers (inputs, tables) with no arena set, or what they leave counts:
// const InArena outside(nullptr).
// Use and destroy the object on the thread that constructed it: say(), left() and the marks take
// no lock, and the destructor resets the calling thread's left_unzeroed() listener.
// Example:
//   LeftBehind look("my_test");
//   { const InArena in(look.arena()); look.say("lookups"); .. create, run, destroy the objects .. }
//   const size_t words = look.left();
class LeftBehind {
public:
  // bytes: size of the region; an allocation that does not fit aborts the process.
  // name and each left_unzeroed() reason are kept as pointers: the strings must outlive the object.
  explicit LeftBehind(const char *name, size_t bytes = (size_t)2 << 30) : name_(name), bytes_(bytes / CardArena::STEP * CardArena::STEP), arena_(name), last_(bytes_ / CardArena::STEP, 0) {
    ENGINE_CK(cudaMalloc(&lent_, bytes_));
    ENGINE_CK(cudaMalloc(&on_card_, last_.size()));
    ENGINE_CK(cudaMemset(lent_, 0, bytes_));
    ENGINE_CK(cudaDeviceSynchronize());
    arena_.lend(lent_, bytes_);
    takes_.push_back(Take{0, "nobody"});
    arena_.on_take = [this](void *p, size_t n) {
      const size_t first = (size_t)((char *)p - (char *)lent_) / CardArena::STEP;
      read_and_zero(first, n / CardArena::STEP);
      takes_.push_back(Take{n, said_});
      takes_.back().depth = backtrace(takes_.back().chain, CHAIN);
      for (size_t i = 0; i < n / CardArena::STEP; i++) last_[first + i] = (unsigned)(takes_.size() - 1);
    };
    before_ = left_unzeroed_listener();
    left_unzeroed_listener() = [this](const void *p, size_t, const char *ground) {
      if ((const char *)p < (const char *)lent_ || (const char *)p >= (const char *)lent_ + bytes_) return; // not of this arena
      takes_[last_[(size_t)((const char *)p - (const char *)lent_) / CardArena::STEP]].ground = ground;
    };
  }
  ~LeftBehind() {
    left_unzeroed_listener() = before_;
    ENGINE_CK(cudaDeviceSynchronize());
    arena_.take_back(); // ends the process if a buffer still lives
    ENGINE_CK(cudaMemset(lent_, 0, bytes_));
    ENGINE_CK(cudaFree(on_card_));
    ENGINE_CK(cudaFree(lent_));
  }
  LeftBehind(const LeftBehind &) = delete;
  LeftBehind &operator=(const LeftBehind &) = delete;

  CardArena *arena() { return &arena_; }
  void say(const char *word) { said_ = word; } // label recorded with every later take
  // Call it when every buffer has been freed. Prints one line (60 at most) for each allocation that
  // left nonzero words since the last call: its number (from 1; 0: a range never allocated), its
  // size as the arena rounded it up, its say() label, whether its words count ("IT COUNTS") or are
  // exempt ("LET", with the reason), and its callers. Then it prints one summary line, even if
  // nothing was left. Returns how many of the nonzero words count. The destructor reports nothing:
  // what is left after the last call goes unseen.
  size_t left() {
    BLK_REQUIRE(arena_.used() == 0, "left_behind: a buffer of the arena still lives");
    read_and_zero(0, last_.size());
    size_t count = 0, let = 0, lines = 0;
    for (const auto &c : charged_) {
      const Take &who = takes_[c.first];
      (who.ground ? let : count) += c.second;
      if (lines++ < 60)
        std::printf("%s: LEFT BEHIND: %zu words that are not zero by take %u (%zu bytes; \"%s\"): %s%s; taken by: %s\n", name_, c.second, c.first, who.bytes, who.said.c_str(),
                    who.ground ? "LET: the source says that it holds " : "IT COUNTS", who.ground ? who.ground : "", chain_of(who).c_str());
    }
    std::printf("%s: the look at what is left behind: %zu buffers taken so far, at most %.1f MB at a time; words that are not zero: %zu that count, %zu that are let\n", name_, takes_.size() - 1,
                arena_.peak() / 1e6, count, let);
    charged_.clear();
    let_ = let;
    return count;
  }
  size_t let() const { return let_; }                 // exempt words in the last call of `left`
  size_t taken() const { return takes_.size() - 1; }

private:
  // maximum number of return addresses recorded per allocation
  static constexpr int CHAIN = 14;
  // One allocation. `said`: the say() label at that time; `chain`, `depth`: from backtrace().
  struct Take {
    size_t bytes;
    std::string said;
    const char *ground = nullptr; // reason passed to `left_unzeroed`; null: not marked
    void *chain[CHAIN] = {};
    int depth = 0;
  };
  // Charges the nonzero words of these ranges to their last allocations and zeroes the ranges.
  // Synchronizes the device before, so that the previous owner's queued work has finished, and
  // after, so that the zeros are written before the next owner's work starts.
  void read_and_zero(size_t first, size_t ranges) {
    if (!ranges) return;
    ENGINE_CK(cudaDeviceSynchronize());
    left_behind_k_ranges<<<(unsigned)std::min<size_t>(1024, (ranges + 255) / 256), 256>>>((const unsigned long long *)((const char *)lent_ + first * CardArena::STEP), ranges, on_card_);
    ENGINE_CK(cudaGetLastError());
    host_.resize(ranges);
    ENGINE_CK(cudaMemcpy(host_.data(), on_card_, ranges, cudaMemcpyDeviceToHost));
    bool any = false;
    for (size_t i = 0; i < ranges; i++)
      if (host_[i]) charged_[last_[first + i]] += host_[i], any = true;
    if (any) ENGINE_CK(cudaMemset((char *)lent_ + first * CardArena::STEP, 0, ranges * CardArena::STEP));
    ENGINE_CK(cudaDeviceSynchronize());
  }
  // Up to three callers of the allocation, innermost first: by name where backtrace_symbols() has a
  // C++ name (link the test with -rdynamic; static functions get none), otherwise as the offset
  // that it prints, such as (+0x1234). Skipped, but only where they have names: frames whose name,
  // parameter types included, contains `LeftBehind`, `CardArena`, std::function or
  // std::_Function_handler, or starts with `blk::Card<`.
  static std::string chain_of(const Take &t) {
    std::string out;
    char **names = backtrace_symbols(const_cast<void *const *>(t.chain), t.depth);
    for (int i = 0, shown = 0; names && i < t.depth && shown < 3; i++) {
      std::string s = names[i];
      const size_t open = s.find('('), plus = s.find('+', open == std::string::npos ? 0 : open);
      bool named = false;
      if (open != std::string::npos && plus != std::string::npos && plus > open + 1) {
        int status = 1;
        char *plain = abi::__cxa_demangle(s.substr(open + 1, plus - open - 1).c_str(), nullptr, nullptr, &status);
        if (!status && plain) s = plain, named = true;
        std::free(plain);
      }
      if (s.find("LeftBehind") != std::string::npos || s.find("CardArena") != std::string::npos || s.find("blk::Card<") == 0 || s.find("std::_Function_handler") != std::string::npos || s.find("std::function") != std::string::npos) continue;
      if (named) {
        if (const size_t bracket = s.find('('); bracket != std::string::npos) s.resize(bracket);
      } else if (open != std::string::npos && s.find(')') != std::string::npos)
        s = s.substr(open, s.find(')') - open + 1); // (+0x..): the place in the program
      out += (shown++ ? " < " : "") + s;
    }
    std::free(names);
    return out;
  }
  // lent_: the region. on_card_, host_: the kernel's counts on the GPU and their copy on the host.
  const char *name_;
  size_t bytes_;
  CardArena arena_;
  void *lent_ = nullptr;
  unsigned char *on_card_ = nullptr;
  std::vector<unsigned char> host_;
  std::vector<unsigned> last_; // for each range of STEP bytes: the number of the last take that covered it; 0: none
  std::vector<Take> takes_;
  std::map<unsigned, size_t> charged_; // take -> words that it left, since the last left()
  std::string said_ = "";
  size_t let_ = 0;
  std::function<void(const void *, size_t, const char *)> before_;
};
} // namespace blk
