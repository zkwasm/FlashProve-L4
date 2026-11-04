// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Reference CPU implementation of SumcheckProver (blocks/iop/summand.h), straight from the
// definition, for maps that are multilinear extensions of tables. It adds no masks and hides
// nothing. Every variable must be summed over {0,1}, plainly or under equality weights: its
// `Weight` (blocks/iop/hp.h) sums g(X) as (`moment[0]` - `moment[1]`) g(0) + `moment[1]` g(1).
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "iop/summand.h"

namespace blk {

template <class K>
class SumcheckCpu : public SumcheckProver<K> {
public:
  // maps: Summand::shown maps, then Summand::kept ones; 2^k values each, k = vars.size();
  // index bit t is vars[t].
  SumcheckCpu(const Summand<K> &summand, const std::vector<Weight<K>> &vars, std::vector<std::vector<K>> maps)
      : s_(summand), vars_(vars), maps_(std::move(maps)) {
    BLK_REQUIRE(maps_.size() == size_t(s_.shown) + s_.kept, "sum-check: one table for each map");
    for (const auto &m : maps_) BLK_REQUIRE(m.size() == size_t(1) << vars_.size(), "sum-check: a table of 2^k values for each map");
    for (const Weight<K> &w : vars_)
      for (unsigned t = 2; t < w.d; t++) BLK_REQUIRE(w.moment[t] == w.moment[1], "sum-check: a variable is not summed over {0,1}");
  }
  // Evaluates the summand vars_[j].d * 2^(k-j-1) times. Not checked: the round polynomial has
  // degree below vars_[j].d; if not, the result is wrong.
  std::vector<K> round() override {
    const size_t j = point_.size(), k = vars_.size();
    BLK_REQUIRE(j < k, "sum-check: no variable is left");
    const size_t rest = size_t(1) << (k - j - 1);
    std::vector<K> values(vars_[j].d, K::zero()), point(point_), y(s_.shown), u(s_.kept);
    point.resize(k, K::zero());
    for (unsigned node = 0; node < vars_[j].d; node++) {
      const K x = small_int<K>(node);
      point[j] = x;
      for (size_t b = 0; b < rest; b++) {
        K weight = K::one();
        for (size_t t = j + 1; t < k; t++) {
          const bool one = (b >> (t - j - 1)) & 1;
          point[t] = one ? K::one() : K::zero();
          weight = weight * (one ? vars_[t].moment[1] : vars_[t].moment[0] - vars_[t].moment[1]);
        }
        for (size_t i = 0; i < maps_.size(); i++) {
          const K lo = maps_[i][2 * b], hi = maps_[i][2 * b + 1], v = lo + x * (hi - lo);
          (i < s_.shown ? y[i] : u[i - s_.shown]) = v;
        }
        values[node] = values[node] + weight * s_.at_values(point, y, u);
      }
    }
    return poly_from_values_at_0_1_2<K>(std::move(values));
  }
  void bind(const K &g) override {
    BLK_REQUIRE(point_.size() < vars_.size(), "sum-check: no variable is left");
    for (auto &m : maps_) {
      for (size_t b = 0; b < m.size() / 2; b++) m[b] = m[2 * b] + g * (m[2 * b + 1] - m[2 * b]);
      m.resize(m.size() / 2);
    }
    point_.push_back(g);
  }
  std::vector<K> end_values() override {
    BLK_REQUIRE(point_.size() == vars_.size(), "sum-check: variables are left");
    std::vector<K> v;
    for (const auto &m : maps_) v.push_back(m[0]);
    return v;
  }

private:
  Summand<K> s_;
  std::vector<Weight<K>> vars_;
  std::vector<std::vector<K>> maps_;
  std::vector<K> point_;
};

} // namespace blk
