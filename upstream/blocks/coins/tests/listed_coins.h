// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests of hiding only: a source of private randomness (class `Coins`, blocks/engine/table.h;
// "coins" below) whose values a test sets entry by entry. They are not random: what is masked with
// them is hidden from no one. The file compiles only if BLK_LISTED_COINS_FOR_A_TEST is defined
// before it is included, and never in a production build (SERVING_BUILD).
// With the verifier's messages fixed, everything that a prover sends is a function of its witness
// and coins. A test studies that function (image, rank, what one coin changes) on coins it chooses.
// A first run learns how many values the prover draws; later runs get lists of those lengths:
//   ListedCoins first;                       // learning mode: every value drawn is 0
//   run(first);                              // elements() and bytes() now have the lengths drawn
//   ListedCoins next(my_elements, my_bytes); // a draw beyond a list's end ends the process
//   run(next);  next.all_drawn()             // false if the run drew fewer entries than listed
// Field elements and bytes (such as salts) form two lists, each in the order of its draws; ledger()
// lists the draws from both, in order. all_drawn() compares totals only: it does not notice a run
// that draws the same totals in another order, which ledger() shows.
#pragma once
#ifndef BLK_LISTED_COINS_FOR_A_TEST
#error "listed_coins.h gives coins that a test has chosen: nothing is hidden under them. Only a test of hiding includes it (see its head)."
#endif
#ifdef SERVING_BUILD // the check in blocks/fence.h misses macros defined after it is first included
#error "coins that a test has chosen in the build of a program that serves"
#endif
#include <vector>

#include "engine/table.h"

BLK_MARK_NOT_FOR_SERVING(); // blocks/fence.h: embeds a not-for-serving marker string in the program

namespace blk {

class ListedCoins : public Coins {
public:
  // One call of fill() or bytes(); `Coins::f()` and `Coins::e()` draw through fill().
  struct Draw {
    bool elements; // field elements, or bytes
    size_t first;  // index of the draw's first entry in its list (`elements_` or `bytes_`)
    size_t count;
  };
  ListedCoins() : learning_(true) {}
  ListedCoins(std::vector<F> elements, std::vector<u8> bytes) : learning_(false), elements_(std::move(elements)), bytes_(std::move(bytes)) {}
  ListedCoins(const ListedCoins &) = delete; // a copy would return the same entries a second time
  ListedCoins &operator=(const ListedCoins &) = delete;

  void fill(F *out, size_t n) override {
    room(elements_, elements_at_, n, F::zero());
    ledger_.push_back(Draw{true, elements_at_, n});
    for (size_t i = 0; i < n; i++) out[i] = elements_[elements_at_ + i];
    elements_at_ += n;
  }
  void bytes(u8 *out, size_t n) override {
    room(bytes_, bytes_at_, n, (u8)0);
    ledger_.push_back(Draw{false, bytes_at_, n});
    for (size_t i = 0; i < n; i++) out[i] = bytes_[bytes_at_ + i];
    bytes_at_ += n;
  }
  const std::vector<F> &elements() const { return elements_; }
  const std::vector<u8> &bytes() const { return bytes_; }
  const std::vector<Draw> &ledger() const { return ledger_; }
  bool all_drawn() const { return elements_at_ == elements_.size() && bytes_at_ == bytes_.size(); }

private:
  template <class T> void room(std::vector<T> &list, size_t at, size_t n, T zero) {
    if (learning_) list.resize(at + n, zero);
    BLK_REQUIRE(n <= list.size() - at, "listed coins: the run draws more than the test has listed");
  }
  bool learning_;
  std::vector<F> elements_;
  std::vector<u8> bytes_;
  size_t elements_at_ = 0, bytes_at_ = 0;
  std::vector<Draw> ledger_;
};

} // namespace blk
