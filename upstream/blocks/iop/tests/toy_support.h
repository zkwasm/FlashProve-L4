// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Helpers for tests of the sumcheck blocks HP (hiding polynomial), PSC (sumcheck masked with it)
// and LINE (joins two claims on a polynomial's halves): a prime field of any size below 2^63, so
// that small cases can be enumerated; a channel whose challenges (`coins` in identifiers) the test
// chooses; a sumcheck prover that follows the definition literally; stand-in mask rows; a check of
// PSC's output claims (`row` in failing_row(), `row_target`).
#pragma once
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/hp.h"
#include "iop/line.h"
#include "iop/mask.h"
#include "iop/psc.h"
#include "iop/summand.h"

namespace blk {

// In the protocol only SEND, the block that sends oracle messages, makes receipts (SampleReceipt,
// blocks/iop/mask.h: needed to take mask entries). These tests have no oracle messages: their mask
// rows are plain vectors. blocks/iop/send.h also defines SendAccess: include only one of the two.
struct SendAccess {
  static SampleReceipt make(u32 oracle, u32 first, u32 count) { return SampleReceipt(oracle, first, count); }
};

// ---- Z/P for a prime P < 2^63 ----
// Not checked: the bound (a.v + b.v must fit in 64 bits) and primality (for inv()).
template <u64 P>
struct Fp {
  u64 v = 0;
  static constexpr u64 size = P;
  static Fp zero() { return {0}; }
  static Fp one() { return {1 % P}; }
  static Fp from(u64 x) { return {x % P}; }
  friend Fp operator+(Fp a, Fp b) { return {(a.v + b.v) % P}; }
  friend Fp operator-(Fp a, Fp b) { return {(a.v + P - b.v) % P}; }
  friend Fp operator*(Fp a, Fp b) { return {u64(u128(a.v) * b.v % P)}; }
  friend bool operator==(Fp a, Fp b) { return a.v == b.v; }
  friend bool operator!=(Fp a, Fp b) { return a.v != b.v; }
};
template <u64 P>
Fp<P> inv(Fp<P> a) {
  BLK_REQUIRE(a.v != 0, "inv(0)");
  Fp<P> r = Fp<P>::one();
  for (u64 e = P - 2; e; e >>= 1, a = a * a)
    if (e & 1) r = r * a;
  return r;
}

// A small generator for test data (SplitMix64, Steele, Lea, Flood:
// "Fast splittable pseudorandom number generators", OOPSLA 2014). Not for anything secret.
struct TestRng {
  u64 s;
  explicit TestRng(u64 seed) : s(seed) {}
  u64 word() {
    u64 z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  template <class K>
  K element() {  // uniform by rejection
    const u64 limit = ~0ull - (~0ull % K::size + 1) % K::size;
    u64 x;
    do x = word();
    while (x > limit);
    return K::from(x);
  }
  template <class K>
  K nonzero() {
    K x;
    do x = element<K>();
    while (x == K::zero());
    return x;
  }
  template <class K>
  std::vector<K> vector(size_t n) {
    std::vector<K> v(n);
    for (K& x : v) x = element<K>();
    return v;
  }
};

// ---- The channel ----
// The verifier's challenges come from a function that the test supplies (its argument: must the
// challenge be non-zero?). The test must give both sides of a run functions that return the same
// sequence. Each close function ends a prover's message and gives the verifier's, if any. `log`
// records the order of events on this side, unless keep_log is false.
template <class K>
class ToyChannel {
 public:
  using Coins = std::function<K(bool nonzero)>;
  explicit ToyChannel(Coins coins, bool keep_log = true) : coins_(std::move(coins)), keep_log_(keep_log) {}
  struct Draw {
    ToyChannel* ch;
    K draw() {
      ch->note("draw");
      return ch->coins_(false);
    }
    K draw_nonzero() {
      ch->note("draw, not zero");
      return ch->coins_(true);
    }
  };
  K close_epsilon(const std::string& name) {
    BLK_REQUIRE(last_was_hiding_sum_, "close_epsilon: the message must end with a hiding polynomial's sum");
    last_was_hiding_sum_ = false;
    note("eps " + name);
    return coins_(false);
  }
  Draw close_algebraic(const std::string& name) {
    last_was_hiding_sum_ = false;
    note("alg " + name);
    return Draw{this};
  }
  void close_empty(const std::string& name) {
    last_was_hiding_sum_ = false;
    note("empty " + name);
  }
  std::vector<std::string> log;

 protected:
  void note(const std::string& s) {
    if (keep_log_) log.push_back(s);
  }
  Coins coins_;
  bool keep_log_;
  bool last_was_hiding_sum_ = false;
};

template <class K>
class ToyProverChannel : public ToyChannel<K> {
 public:
  using ToyChannel<K>::ToyChannel;
  void send(const K& x) {
    tape.push_back(x);
    this->last_was_hiding_sum_ = false;
    this->note("scalar");
  }
  void send_hiding_sum(const K& x) {
    tape.push_back(x);
    this->last_was_hiding_sum_ = true;
    this->note("hiding sum");
  }
  std::vector<K> tape;  // all that the prover sent
};

// `tape` is kept by reference and must outlive the channel.
template <class K>
class ToyVerifierChannel : public ToyChannel<K> {
 public:
  ToyVerifierChannel(const std::vector<K>& tape, typename ToyChannel<K>::Coins coins)
      : ToyChannel<K>(std::move(coins)), tape_(tape) {}
  bool ok() const { return ok_; }
  bool recv(K& x) {
    if (at_ >= tape_.size()) return ok_ = false;
    x = tape_[at_++];
    this->last_was_hiding_sum_ = false;
    this->note("scalar");
    return true;
  }
  bool recv_hiding_sum(K& x) {
    if (at_ >= tape_.size()) return ok_ = false;
    x = tape_[at_++];
    this->last_was_hiding_sum_ = true;
    this->note("hiding sum");
    return true;
  }
  bool at_end() const { return at_ == tape_.size(); }

 private:
  const std::vector<K>& tape_;
  size_t at_ = 0;
  bool ok_ = true;
};

template <class K>
typename ToyChannel<K>::Coins random_coins(u64 seed) {
  auto rng = std::make_shared<TestRng>(seed);
  return [rng](bool nonzero) { return nonzero ? rng->template nonzero<K>() : rng->template element<K>(); };
}
// Challenges that the test lists. A listed zero is returned even where the protocol asks for a
// non-zero one: tests such as psc_hiding_test.cpp use this to show what that condition is for.
template <class K>
typename ToyChannel<K>::Coins listed_coins(const std::vector<K>& list) {
  auto at = std::make_shared<size_t>(0);
  return [list, at](bool) {
    BLK_REQUIRE(*at < list.size(), "listed_coins: the list is used up");
    return list[(*at)++];
  };
}

// ---- Maps of the witness ----
// A map is a polynomial that belongs to the witness; here, the function that evaluates it.
template <class K>
using ToyMap = std::function<K(const std::vector<K>& point)>;

// The multilinear extension of a table of 2^k entries; bit t of the index belongs to variable t.
template <class K>
K mle_at(std::vector<K> table, const std::vector<K>& point) {
  BLK_REQUIRE(table.size() == size_t(1) << point.size(), "mle_at: 2^k entries");
  for (const K& g : point) {
    for (size_t i = 0; i < table.size() / 2; ++i) table[i] = table[2 * i] + g * (table[2 * i + 1] - table[2 * i]);
    table.resize(table.size() / 2);
  }
  return table[0];
}
template <class K>
ToyMap<K> multilinear(std::vector<K> table) {
  return [table](const std::vector<K>& point) { return mle_at(table, point); };
}
// Variable 0 ranges over the points of D (extended by Lagrange interpolation on D), the others over
// {0,1}. The value at point number i of D and bits b of the other variables is table[i + |D| b].
template <class K>
ToyMap<K> lagrange_then_multilinear(std::vector<K> table, std::vector<K> D) {
  return [table, D](const std::vector<K>& point) {
    std::vector<K> rest(table.size() / D.size(), K::zero());
    for (size_t i = 0; i < D.size(); ++i) {
      K lag = K::one();
      for (size_t j = 0; j < D.size(); ++j)
        if (j != i) lag = lag * (point[0] - D[j]) * inv(D[i] - D[j]);
      for (size_t b = 0; b < rest.size(); ++b) rest[b] = rest[b] + lag * table[i + D.size() * b];
    }
    return mle_at(rest, std::vector<K>(point.begin() + 1, point.end()));
  };
}
template <class K>
K eq_at(const std::vector<K>& a, const std::vector<K>& b) {
  K r = K::one();
  for (size_t t = 0; t < a.size(); ++t) r = r * (a[t] * b[t] + (K::one() - a[t]) * (K::one() - b[t]));
  return r;
}

// ---- The sumcheck without masks, by the definition ----
// Applies each functional (`Weight`, blocks/iop/hp.h: how one variable is summed) to a polynomial
// that it first interpolates from values at 0, 1, 2, ..: slow and plain. The characteristic must be
// at least the largest Weight::d.
template <class K>
class DefinitionSumcheck : public SumcheckProver<K> {
 public:
  // maps: first the shown (end values sent, masked), then the kept (not sent); see `Summand`.
  DefinitionSumcheck(std::vector<Weight<K>> vars, Summand<K> S, std::vector<ToyMap<K>> maps)
      : vars_(std::move(vars)), S_(std::move(S)), maps_(std::move(maps)) {}
  std::vector<K> round() override { return in_variable(point_.size(), point_); }
  void bind(const K& g) override { point_.push_back(g); }
  std::vector<K> end_values() override {
    std::vector<K> v;
    for (const ToyMap<K>& m : maps_) v.push_back(m(point_));
    return v;
  }
  K sum() { return vars_.empty() ? rest(0, {}) : vars_[0].apply(in_variable(0, {})); }

 private:
  // Coefficients in variable j of the summand, with the variables before j set to `prefix` and the
  // functionals of those after j applied.
  std::vector<K> in_variable(size_t j, std::vector<K> prefix) {
    std::vector<K> values;
    prefix.push_back(K::zero());
    for (unsigned n = 0; n < vars_[j].d; ++n) {
      prefix.back() = small_int<K>(n);
      values.push_back(rest(j + 1, prefix));
    }
    return poly_from_values_at_0_1_2(std::move(values));
  }
  K rest(size_t j, const std::vector<K>& prefix) {
    if (j < vars_.size()) return vars_[j].apply(in_variable(j, prefix));
    std::vector<K> y, u;
    for (size_t i = 0; i < maps_.size(); ++i) (i < S_.shown ? y : u).push_back(maps_[i](prefix));
    return S_.at_values(prefix, y, u);
  }
  std::vector<Weight<K>> vars_;
  Summand<K> S_;
  std::vector<ToyMap<K>> maps_;
  std::vector<K> point_;
};

// Replays round polynomials and end values computed earlier (they depend on witness and challenges,
// not on masks): the run must get the same challenges. Set `at` to 0 before each run.
template <class K>
class ReplaySumcheck : public SumcheckProver<K> {
 public:
  std::vector<std::vector<K>> rounds;
  std::vector<K> ends;
  size_t at = 0;
  std::vector<K> round() override { return rounds[at]; }
  void bind(const K&) override { ++at; }
  std::vector<K> end_values() override { return ends; }
};

// ---- Mask rows and the check of PSC's output claims ----
// `row` is captured by reference and must outlive the result.
template <class K>
MaskValues<K> mask_row(const std::vector<K>& row) {  // one oracle message, number 0
  return [&row](u32 oracle, u32 index) {
    BLK_REQUIRE(oracle == 0 && index < row.size(), "mask_row: no such entry");
    return row[index];
  };
}
template <class K>
K mask_form(const std::vector<MaskTerm<K>>& terms, const MaskValues<K>& masks) {
  K r = K::zero();
  for (const MaskTerm<K>& m : terms) r = r + m.coeff * masks(m.slot.oracle(), m.slot.index());
  return r;
}
// In the protocol the blocks after PSC check its output claims (PscOutput, blocks/iop/psc.h)
// against the committed vectors. Here the test has these vectors: `maps`, ordered as for
// DefinitionSumcheck, and `masks`. Returns "" if all claims hold, else the first false one's name.
template <class K>
std::string failing_row(const PscOutput<K>& out, const Summand<K>& S, const std::vector<ToyMap<K>>& maps,
                        const MaskValues<K>& masks) {
  K u = mask_form(out.row_masks, masks);
  for (const KeptTerm<K>& t : out.row_maps) u = u + t.coeff * maps[S.shown + t.map](out.point);
  if (!(u == out.row_target)) return "(U)";
  for (size_t j = 0; j < out.ends.size(); ++j)
    if (!(maps[j](out.ends[j].point) + mask_form(out.ends[j].masks, masks) == out.ends[j].value))
      return "(E" + std::to_string(j) + ")";
  return "";
}

template <class K>
bool same_terms(const std::vector<MaskTerm<K>>& a, const std::vector<MaskTerm<K>>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].slot.oracle() != b[i].slot.oracle() || a[i].slot.index() != b[i].slot.index() || !(a[i].coeff == b[i].coeff))
      return false;
  return true;
}
template <class K>
bool same_output(const PscOutput<K>& a, const PscOutput<K>& b) {
  if (!(a.point == b.point) || !same_terms(a.row_masks, b.row_masks) || !(a.row_target == b.row_target) ||
      a.row_maps.size() != b.row_maps.size() || a.ends.size() != b.ends.size())
    return false;
  for (size_t i = 0; i < a.row_maps.size(); ++i)
    if (a.row_maps[i].map != b.row_maps[i].map || !(a.row_maps[i].coeff == b.row_maps[i].coeff)) return false;
  for (size_t i = 0; i < a.ends.size(); ++i)
    if (!(a.ends[i].point == b.ends[i].point) || !same_terms(a.ends[i].masks, b.ends[i].masks) ||
        !(a.ends[i].value == b.ends[i].value))
      return false;
  return true;
}

}  // namespace blk
