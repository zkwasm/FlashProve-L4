// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// An exact test of hiding by linear algebra, for one block of the proof system (blocks/iop/) or for
// a whole proof. With the verifier's messages fixed, what a prover sends and opens (its view) is a
// function of its witness and of the field elements that it draws in private (its coins).
// Exact::compare() asks whether two witnesses of one public statement give the view the same law
// (distribution) under uniform coins.
//
// A pass covers the pair of witnesses and the verifier's messages that were run, no more: anyone
// who relies on a pass must say which pairs were run. The view need not be linear in the witness,
// so a leak can show for special witnesses only. Three faulty provers, tried with random pairs, all
// passed: one leaves zero entries unmasked, one multiplies a mask by the witness, one sends one
// bare bit of the witness (it passes whenever the pair agrees in that bit). Not seen: digests
// (roots, siblings) and salts, since hashing is not linear (that a salted digest hides its leaf is
// another argument); coins that are not uniform; running time; a prover that stops, or draws again,
// for a reason that depends on the witness. This is a test of the programs against the claim, not a
// proof. `World` has the contract of a prover under test, `Exact` the method and its argument.
#pragma once
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
// blocks/coins/tests/listed_coins.h compiles only with this macro: its `ListedCoins` are values
// that a test chooses, not random ones. It does not compile under SERVING_BUILD.
#define BLK_LISTED_COINS_FOR_A_TEST 1
#include "coins/tests/listed_coins.h"
#include "engine/table.h"
#include "field/field.h"
#include "iop/message.h"
#include "toy/toy_coins.h"

namespace blk {

// What a prover sends and opens in one run. F is the base field and E its extension field
// (blocks/field/field.h); an element of E enters `view` as its coordinates over F. Where there is
// no proof, `at` has the elements' numbers instead.
struct Look {
  std::vector<F> view;        // every field element sent or opened, in the order of the proof
  std::vector<size_t> at;     // [i]: the byte of the proof at which the element of view[i] begins
  std::vector<u8> frame;      // bytes that must be the same in all runs; `look_at` says which
  std::vector<Writer::Mark> marks; // labels that the prover set at bytes of the proof, for `where`
  void push(const E &x, size_t place) {
    for (const F &c : x.c) view.push_back(c), at.push_back(place);
  }
};

// The `Look` of a proof whose `Writer` (blocks/iop/message.h) has kept its list of items, which
// Writer::keep_list() turns on. `frame` gets each item's `kind` and size, and the bytes of the
// items that are neither field elements nor digests, such as nonces, listed positions and salts. A
// digest's bytes are left out.
inline Look look_at(const Writer &proof) {
  Look l;
  BLK_REQUIRE(!proof.items().empty(), "exact: the Writer must keep its list (keep_list())");
  for (const Writer::Item &i : proof.items()) {
    const u8 *p = proof.data() + i.at;
    l.frame.push_back((u8)i.kind);
    for (int b = 0; b < 4; b++) l.frame.push_back((u8)(i.bytes >> (8 * b)));
    if (i.kind == 'F' || i.kind == 'E') {
      for (size_t c = 0; c < i.bytes / F_BYTES; c++) {
        F x;
        BLK_REQUIRE(F_from_bytes(p + c * F_BYTES, x), "exact: an element of the proof is not in canonical form");
        l.view.push_back(x), l.at.push_back(i.at);
      }
    } else if (i.kind != 'D') {
      l.frame.insert(l.frame.end(), p, p + i.bytes);
    }
  }
  l.marks = proof.marks();
  return l;
}

// A prover under test: runs on witness number `witness` with these coins and returns its `Look`.
// All witnesses must belong to one public statement (not checked). The view must be a polynomial in
// the coins (not checked): a prover that branches on a coin's size passes with laws far apart. The
// verifier's messages must not depend on what is sent: for a prover that derives them from a hash
// chain (Fiat-Shamir), the macro BLK_TEST_PROVER_MESSAGES_OUTSIDE (blocks/chain/rounds.h) keeps
// what it sends out of the chain. Exact::at() notices a change of these messages only through
// `Look::frame`, the view's length or the number of coins or random bytes drawn, and then ends the
// process.
using World = std::function<Look(unsigned witness, Coins &coins)>;

// Rows over F, brought to echelon form as they come. Only the first `unknowns` columns can hold a
// pivot; the others, if any, are right-hand sides.
class Echelon {
public:
  Echelon(size_t width, size_t unknowns) : width_(width), unknowns_(unknowns) {}
  enum What { NEW, NOTHING, CONTRADICTION };
  // NEW: the row is independent of those before it. NOTHING: it follows from them, in all columns.
  // CONTRADICTION: its unknowns' part follows from them and the rest does not.
  What add(std::vector<F> row) {
    BLK_REQUIRE(row.size() == width_, "echelon: a row of another width");
    for (size_t k = 0; k < rows_.size(); k++) {
      const F f = row[pivot_[k]];
      if (f == F::zero()) continue;
      const std::vector<F> &r = rows_[k];
      for (size_t j = first_[k]; j < width_; j++) row[j] = row[j] - f * r[j];
    }
    size_t p = 0;
    while (p < unknowns_ && row[p] == F::zero()) p++;
    if (p == unknowns_) {
      for (size_t j = unknowns_; j < width_; j++)
        if (row[j] != F::zero()) return CONTRADICTION;
      return NOTHING;
    }
    const F scale = inv(row[p]);
    for (size_t j = p; j < width_; j++) row[j] = row[j] * scale;
    rows_.push_back(std::move(row)), pivot_.push_back(p), first_.push_back(p);
    return NEW;
  }
  size_t rank() const { return rows_.size(); }
  // Clears every pivot's column in all other rows.
  void reduce() {
    for (size_t k = rows_.size(); k-- > 0;)
      for (size_t i = 0; i < k; i++) {
        const F f = rows_[i][pivot_[k]];
        if (f == F::zero()) continue;
        for (size_t j = first_[k]; j < width_; j++) rows_[i][j] = rows_[i][j] - f * rows_[k][j];
      }
  }
  const std::vector<std::vector<F>> &rows() const { return rows_; }
  const std::vector<size_t> &pivots() const { return pivot_; }

private:
  size_t width_, unknowns_;
  std::vector<std::vector<F>> rows_;
  std::vector<size_t> pivot_, first_; // a stored row is zero before its pivot
};

// Matrices. A list of coins has one value for each field element that the prover draws; c0 and c1
// are two lists expanded from `key` and taken as random. A witness's matrix J at a list c (view x
// coins) is measured, not derived: column i is the change of the view when coin i is raised by 1,
// from one run of the honest prover. All algebra is over F: a coin is an element of F, an element
// of E in the view counts as its coordinates, and multiplication by a challenge in E is F-linear.
//
// An affine view. If the view is affine in the coins, its law is uniform on view(c0) + the column
// space of J. Two witnesses then have the same law exactly if (1) some coins give the second
// witness the view that the first has at c0, and (2) the two matrices have the same column space.
//
// The short way. "view(c) = view(c0) + J (c - c0) for all c" is a polynomial identity in c of low
// degree. It is tried at c1, where a false identity fails except with probability about its degree
// over |F|. If it holds for both witnesses with the first one's J, then (2) holds and (1) is one
// linear system: about one run for each coin in all. Otherwise compare() goes the long way.
//
// The long way. PSC (blocks/iop/psc.h) is the sumcheck with a hiding polynomial. When its prover,
// psc_prove(), runs on a product of maps whose end values it sends, it multiplies these maps' mask
// entries (one random element for each map, which masks its end value) with each other: the view is
// not affine. A coin whose column differs between c0 and c1 is called curved, any other coin flat.
// The view is expected to be triangular: a sent end value is affine in its own mask entry, and with
// the curved coins fixed everything else is affine in the flat coins. So (1) is solved in at most
// four steps, each with the second witness's matrix at the coins reached so far: the first step is
// expected to set the curved coins, the second the rest. While a step still moves a curved coin,
// its equations are only a linearization: an entry tied to earlier ones by a relation of degree two
// (a linear claim's value contains the product of two sent values) contradicts them by the
// second-order term. So such a step is solved on its independent equations only, and a
// contradiction counts only in a step that moves no curved coin.
//
// In both ways the prover is then run on the second witness with the coins found, and its view must
// equal the first one's entry by entry. After the long way one such list of coins is not enough: a
// prover that sends a + g a^2 (a: a coin; g: a nonzero constant) for the first witness and a for
// the second has one, yet the first view takes about half of the field's values. So in place of (2)
// shape_fault() requires the triangular shape of both witnesses, at c0 and c1:
//   (a) the same coins are curved;
//   (b) on the entries of the view that are affine in all coins, the curved coins' columns are
//       independent of each other and of the flat columns, so these entries fix the curved coins;
//   (c) the flat columns span one space L for both witnesses.
// Then a witness's law is uniform on the cosets u(N) + L, one for each value N of the curved coins
// (u(N): the view at N with the flat coins 0). By (b) the N' that could give the second witness a
// view of the first is an affine function of that view, so "the second witness reaches the first
// one's view at N" is a polynomial identity in N. It holds at the N of c0, so it holds for all N
// except with probability about its degree over |F|. Both sets of cosets have the same size, so
// they are equal, and so are the laws.
//
// A view outside this shape never passes, even where the laws are equal: a valid chain of masks
// that multiply each other (a_k a_(k-1) + w_k), or curved coins that are sent nowhere, fail with
// the reason in `Verdict::why`. So does a view for which four steps find no coins.
//
// Negative controls: spoil(); Verdict::coin_matters and draw_matters, which are reckoned from an
// honest run's linear system; the faulty fixtures of this directory's
// tests; and blocks/tests/hiding/exact_self_test.cpp, which runs this test on small provers with
// known laws (not the three faulty provers named at the top of this file). This project's builds
// of these tests with a fault in the library itself are not part of this repository (README.md).
class Exact {
public:
  // One call of Coins::fill() (blocks/engine/table.h): coins `first` to `first + count - 1`.
  struct Draw {
    size_t first, count;
  };
  // `key` seeds c0, c1 and the prover's random bytes (such as tree leaves' salts and a hash chain's
  // salt), through `ToyCoins` (blocks/toy/toy_coins.h). After a first run on zeros, which only
  // counts the draws, all runs get the same bytes; another key gives other bytes and so can change
  // the verifier's messages.
  Exact(World world, u64 key) : world_(std::move(world)) {
    ListedCoins learn;
    world_(0, learn);
    coins_ = learn.elements().size();
    for (const ListedCoins::Draw &d : learn.ledger())
      if (d.elements && d.count) draws_.push_back(Draw{d.first, d.count});
    c0_.resize(coins_), c1_.resize(coins_), bytes_.resize(learn.bytes().size());
    ToyCoins(key).fill(c0_.data(), coins_), ToyCoins(key + 1).fill(c1_.data(), coins_), ToyCoins(key + 2).bytes(bytes_.data(), bytes_.size());
    held_.assign(coins_, false);
    begin();
  }
  size_t coins() const { return coins_; }
  size_t entries() const { return first_.view.size(); }
  const std::vector<Draw> &draws() const { return draws_; } // the draws of field elements, in order
  unsigned long long runs() const { return runs_; }

  // Injects a fault into the coins: `wiring` is called on every list of coins before a run. `held`
  // must list the coins that it overwrites (not checked): they count as constants.
  void spoil(std::function<void(std::vector<F> &)> wiring, const std::vector<size_t> &held) {
    wiring_ = std::move(wiring);
    for (size_t i : held) held_[i] = true;
    begin();
  }

  // A typical failure: some combination of entries of the view moves with no coin and differs
  // between the witnesses. Possible causes: a pad (the mask of an out-of-domain sample's answer)
  // that was never drawn; a mask entry used for two scalars; a row opened at one position more than
  // it has random entries; a copy of a mask opened at a list of positions that is not the mask's.
  // Only the reason "no coins give ..." sets `entry`.
  struct Verdict {
    bool same = false;  // the two witnesses have the same law, as far as this test sees
    std::string why;    // if not
    long entry = -1;    // the first entry of the view that the witness fixes, given those before it
    size_t rank = 0;    // of the matrix view x coins; after a failure: as far as it was computed
    size_t curved = 0;  // coins in which the view is not affine
    size_t idle = 0;    // coins that move no entry of the first witness's view
    bool short_way = false; // `follows` held for both witnesses: the short way was taken
    unsigned steps = 0; // in which the second witness's coins were found
    // [i]: every solution of the linear system for (1) at `Exact` moves coin i (some coin of draw
    // i) from its value in c0, so with coin i (all coins of draw i) constant the witnesses would be
    // told apart. After the long way the system is the linearization at the coins found. No further
    // run is made. Valid only if `same`.
    std::vector<bool> coin_matters, draw_matters;
    size_t count(const std::vector<bool> &v) const {
      size_t n = 0;
      for (bool b : v) n += b;
      return n;
    }
  };

  // Compares witness 0 ("the first witness" in these comments) with witness `second`.
  Verdict compare(unsigned second) {
    Verdict v;
    v.curved = v.count(curved_), v.idle = idle_;
    v.coin_matters.assign(coins_, false), v.draw_matters.assign(draws_.size(), false);
    std::vector<F> c = c0_;
    if (straight_) { // the short way
      const Look here = at(second, c0_);
      if (here.view == first_.view) return nothing_tested(v);
      if (follows(here, at(second, c1_))) {
        v.short_way = true;
        Echelon e(coins_ + 1, coins_);
        for (size_t r = 0; r < j1_.size(); r++) {
          std::vector<F> row = j1_[r];
          row.push_back(first_.view[r] - here.view[r]);
          if (e.add(std::move(row)) == Echelon::CONTRADICTION && v.entry < 0) v.entry = (long)r;
        }
        v.rank = e.rank();
        if (v.entry >= 0) {
          v.why = "no coins give the second witness the first one's view";
          return v;
        }
        e.reduce();
        take_away(e, v);
        for (size_t k = 0; k < e.rank(); k++) c[e.pivots()[k]] = c[e.pivots()[k]] + e.rows()[k][coins_];
        v.steps = 1;
        v.same = at(second, c).view == first_.view;
        if (!v.same) v.why = "the prover, run on the coins that were found, gives another view: the view is not what it seemed at two lists of coins";
        return v;
      }
    }
    if (!rank_known_) {
      Echelon one(coins_, coins_);
      for (const std::vector<F> &row : j1_) one.add(row);
      rank_ = one.rank(), rank_known_ = true;
    }
    Matrix j2, j2_at_c0;
    for (v.steps = 0;; v.steps++) {
      const Look here = at(second, c);
      if (v.steps == 0 && here.view == first_.view) return nothing_tested(v);
      j2 = matrix(second, c, here);
      if (v.steps == 0) j2_at_c0 = j2;
      if (here.view == first_.view) break;
      if (v.steps == 4) {
        v.why = "no coins were found in four steps: the view is curved in a way that this test does not follow";
        return v;
      }
      Echelon e(coins_ + 1, coins_);
      long contradiction = -1;
      for (size_t r = 0; r < j2.size(); r++) {
        std::vector<F> row = j2[r];
        row.push_back(first_.view[r] - here.view[r]);
        if (e.add(std::move(row)) == Echelon::CONTRADICTION && contradiction < 0) contradiction = (long)r;
      }
      e.reduce();
      bool curved_moves = false;
      for (size_t k = 0; k < e.rank(); k++) curved_moves = curved_moves || (curved_[e.pivots()[k]] && e.rows()[k][coins_] != F::zero());
      if (contradiction >= 0 && !curved_moves) {
        v.entry = contradiction, v.rank = e.rank();
        v.why = "no coins give the second witness the first one's view";
        return v;
      }
      for (size_t k = 0; k < e.rank(); k++) c[e.pivots()[k]] = c[e.pivots()[k]] + e.rows()[k][coins_];
    }
    v.rank = rank_;
    v.why = shape_fault(j2_at_c0, matrix(second, c1_, at(second, c1_)));
    if (!v.why.empty()) return v;
    { // for `coin_matters`: the system linearized at the coins found, for the whole move from c0
      Echelon e(coins_ + 1, coins_);
      for (size_t r = 0; r < j2.size(); r++) {
        std::vector<F> row = j2[r];
        F sum = F::zero();
        for (size_t i = 0; i < coins_; i++) sum = sum + row[i] * (c[i] - c0_[i]);
        row.push_back(sum);
        e.add(std::move(row));
      }
      e.reduce();
      take_away(e, v);
    }
    v.same = true;
    return v;
  }

  // A witness's view at c0, for a test that asks how the view depends on the witness.
  std::vector<F> view_of(unsigned witness) { return at(witness, c0_).view; }

  // The verdict for equal views at equal coins: the `World` does not pass its witness on to the
  // prover, or nothing that depends on the witness is sent, even masked.
  static Verdict nothing_tested(Verdict v) {
    v.same = false, v.why = "at equal coins the two witnesses give one and the same view: NOTHING WAS TESTED";
    return v;
  }

  // Text for a message: where entry `entry` of the view is in the proof.
  std::string where(long entry) const {
    if (entry < 0) return "";
    const size_t byte = first_.at[(size_t)entry];
    std::string words = "entry " + std::to_string(entry) + " of the view, in the element at byte " + std::to_string(byte);
    const Writer::Mark *last = nullptr;
    for (const Writer::Mark &m : first_.marks)
      if (m.at <= byte) last = &m;
    if (last) words += " (behind the mark: " + last->words + ")";
    return words;
  }

private:
  using Matrix = std::vector<std::vector<F>>; // [entry of the view][coin]

  // The first witness: its view at c0, its matrix there and, unless follows() holds at c1, at c1;
  // then which coins are curved or idle.
  void begin() {
    has_first_ = false;
    first_ = at(0, c0_);
    has_first_ = true;
    j1_ = matrix(0, c0_, first_);
    const Look elsewhere = at(0, c1_);
    straight_ = follows(first_, elsewhere), rank_known_ = false;
    Matrix &other = j1_elsewhere_;
    other.clear();
    if (!straight_) other = matrix(0, c1_, elsewhere);
    curved_.assign(coins_, false), idle_ = 0;
    for (size_t c = 0; c < coins_; c++) {
      bool differs = false, moves = false;
      for (size_t r = 0; r < j1_.size(); r++) differs = differs || (!straight_ && j1_[r][c] != other[r][c]), moves = moves || j1_[r][c] != F::zero();
      curved_[c] = differs, idle_ += !moves && !differs;
    }
  }

  static size_t rank_of(const Matrix &m, const std::vector<size_t> &rows, const std::vector<size_t> &columns) {
    Echelon e(columns.size(), columns.size());
    for (size_t r : rows) {
      std::vector<F> row;
      for (size_t c : columns) row.push_back(m[r][c]);
      e.add(std::move(row));
    }
    return e.rank();
  }
  // Conditions (a), (b), (c) at `Exact`. Returns "" or what is wrong.
  std::string shape_fault(const Matrix &second_at_c0, const Matrix &second_at_c1) const {
    const Matrix *at_c0[2] = {&j1_, &second_at_c0}, *at_c1[2] = {straight_ ? &j1_ : &j1_elsewhere_, &second_at_c1}; // a straight view has one matrix everywhere
    std::vector<size_t> flat[2], every_row, every_coin;
    for (size_t r = 0; r < j1_.size(); r++) every_row.push_back(r);
    for (size_t c = 0; c < coins_; c++) every_coin.push_back(c);
    for (int w = 0; w < 2; w++) {
      const Matrix &a = *at_c0[w], &b = *at_c1[w];
      std::vector<size_t> affine;
      for (size_t r = 0; r < a.size(); r++)
        if (a[r] == b[r]) affine.push_back(r);
      for (size_t c = 0; c < coins_; c++) {
        bool bent = false;
        for (size_t r = 0; r < a.size() && !bent; r++) bent = a[r][c] != b[r][c];
        if (bent != curved_[c]) return "the two witnesses' views are curved in different coins";
        if (!bent) flat[w].push_back(c);
      }
      if (rank_of(a, affine, every_coin) != rank_of(a, affine, flat[w]) + (coins_ - flat[w].size()))
        return "the entries of the view that are affine in all coins do not fix the curved coins: one list of coins that gives the common view says too little";
    }
    Matrix side(j1_.size());
    std::vector<size_t> left, right, both;
    for (size_t r = 0; r < j1_.size(); r++)
      for (int w = 0; w < 2; w++)
        for (size_t c : flat[w]) side[r].push_back((*at_c0[w])[r][c]);
    for (size_t k = 0; k < flat[0].size() + flat[1].size(); k++) (k < flat[0].size() ? left : right).push_back(k), both.push_back(k);
    const size_t one = rank_of(side, every_row, left), two = rank_of(side, every_row, right), together = rank_of(side, every_row, both);
    if (one != two || two != together)
      return "the coins in which the views are affine move the two witnesses' views in different directions (ranks " + std::to_string(one) + ", " + std::to_string(two) + ", together " +
             std::to_string(together) + ")";
    return "";
  }

  // Whether at_second.view = at_first.view + J (c1 - c0), with J the first witness's matrix at c0.
  bool follows(const Look &at_first, const Look &at_second) const {
    for (size_t r = 0; r < j1_.size(); r++) {
      F sum = at_first.view[r];
      for (size_t c = 0; c < coins_; c++) sum = sum + j1_[r][c] * (c1_[c] - c0_[c]);
      if (sum != at_second.view[r]) return false;
    }
    return true;
  }

  // One run of the prover: the witness's `Look` at this list of coins.
  Look at(unsigned witness, std::vector<F> list) {
    if (wiring_) wiring_(list);
    ListedCoins coins(std::move(list), bytes_);
    Look l = world_(witness, coins);
    runs_++;
    BLK_REQUIRE(coins.all_drawn(), "exact: how many coins the prover draws depends on coins or witness");
    BLK_REQUIRE(!has_first_ || l.view.size() == first_.view.size(), "exact: the number of field elements that are shown depends on coins or witness");
    BLK_REQUIRE(!has_first_ || l.frame == first_.frame,
                "exact: nonces, positions, salts or the layout of the proof move with coins or witness: the verifier's messages are not held still");
    return l;
  }

  // The witness's matrix at c; `base` is its `Look` there. One run for each coin that spoil() has
  // not made a constant, shared among child processes: as many as the environment variable
  // BLK_TEST_JOBS says, or 4, and never more than the processors that this process may run on.
  Matrix matrix(unsigned witness, const std::vector<F> &c, const Look &base) {
    const size_t rows = base.view.size(), bytes = std::max<size_t>(1, rows * coins_ * sizeof(F));
    F *shared = (F *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    BLK_REQUIRE(shared != MAP_FAILED, "exact: no memory for a matrix");
    const char *wish = std::getenv("BLK_TEST_JOBS");
    cpu_set_t mine;
    const long most = std::max(1L, sched_getaffinity(0, sizeof mine, &mine) == 0 ? (long)CPU_COUNT(&mine) : sysconf(_SC_NPROCESSORS_ONLN));
    const size_t jobs = (size_t)std::min(wish && std::atol(wish) > 0 ? std::atol(wish) : 4L, most);
    std::fflush(stdout);
    std::vector<pid_t> children;
    for (size_t j = 0; j < jobs; j++) {
      const pid_t child = fork();
      BLK_REQUIRE(child >= 0, "exact: no child process");
      if (child == 0) {
        const rlimit no_core{0, 0}; // no core file if the child aborts
        setrlimit(RLIMIT_CORE, &no_core);
        for (size_t i = j; i < coins_; i += jobs) {
          if (held_[i]) continue; // the memory is zero
          std::vector<F> moved = c;
          moved[i] = moved[i] + F::one();
          const Look l = at(witness, std::move(moved));
          for (size_t r = 0; r < rows; r++) shared[r * coins_ + i] = l.view[r] - base.view[r];
        }
        _exit(0);
      }
      children.push_back(child);
    }
    for (pid_t child : children) {
      int status = 0;
      BLK_REQUIRE(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "exact: a run of the prover has ended its process");
    }
    Matrix m(rows);
    for (size_t r = 0; r < rows; r++) m[r].assign(shared + r * coins_, shared + (r + 1) * coins_);
    munmap(shared, bytes);
    runs_ += coins_;
    return m;
  }

  // Fills `coin_matters` and `draw_matters`. e: the reduced system (matrix | the change of view
  // that moving the coins from c0 must produce).
  void take_away(const Echelon &e, Verdict &v) const {
    std::vector<long> row_of(coins_, -1);
    for (size_t k = 0; k < e.rank(); k++) row_of[e.pivots()[k]] = (long)k;
    std::vector<size_t> free;
    for (size_t i = 0; i < coins_; i++)
      if (row_of[i] < 0) free.push_back(i);
    // "Coin i does not move" as an equation in the free coins' changes.
    auto line = [&](size_t i) {
      std::vector<F> l(free.size() + 1, F::zero());
      if (row_of[i] < 0) {
        for (size_t f = 0; f < free.size(); f++)
          if (free[f] == i) l[f] = F::one();
      } else {
        const std::vector<F> &r = e.rows()[(size_t)row_of[i]];
        for (size_t f = 0; f < free.size(); f++) l[f] = r[free[f]];
        l[free.size()] = r[coins_];
      }
      return l;
    };
    for (size_t i = 0; i < coins_; i++) {
      Echelon alone(free.size() + 1, free.size());
      if (alone.add(line(i)) == Echelon::CONTRADICTION) v.coin_matters[i] = true;
    }
    for (size_t n = 0; n < draws_.size(); n++) {
      Echelon held(free.size() + 1, free.size());
      bool apart = false;
      for (size_t i = draws_[n].first; i < draws_[n].first + draws_[n].count && !apart; i++) apart = held.add(line(i)) == Echelon::CONTRADICTION;
      if (apart) v.draw_matters[n] = true;
    }
  }

  // held_[i]: spoil() has made coin i a constant. straight_: follows() held for witness 0 at c1, so
  // no coin is curved.
  World world_;
  size_t coins_ = 0;
  std::vector<Draw> draws_;
  std::vector<F> c0_, c1_;
  std::vector<u8> bytes_;
  std::function<void(std::vector<F> &)> wiring_;
  std::vector<bool> held_;
  bool has_first_ = false;
  Look first_;
  Matrix j1_, j1_elsewhere_; // witness 0's matrix at c0, and at c1 (empty if `straight_`)
  size_t rank_ = 0, idle_ = 0;
  bool straight_ = false, rank_known_ = false;
  std::vector<bool> curved_;
  unsigned long long runs_ = 0;
};

} // namespace blk
