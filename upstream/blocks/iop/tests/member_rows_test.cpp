// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests member_inner::stretch_in_the_cube() (blocks/iop/member.h), by which
// member_weights_on_rows() writes row weights (values[i] on row first + i, 0 elsewhere) as a sum of
// Tensors (product-form vectors, blocks/iop/tensor.h). For every stretch of rows in every cube of
// k <= 6 variables, it checks each entry, the multilinear extension at a pseudo-random point, at
// most 2k + 1 Tensors, and that dense factors are slices of `values` itself. For one long,
// unaligned stretch (2^20 - 4321 rows, k = 22): only the extension and at most 2k Tensors.
#include <cstdio>

#include "iop/member.h"
#include "iop/tests/iopp_support.h"

using namespace blk;

int main() {
  TestCoins data(5);
  size_t stretches = 0;
  for (unsigned k = 0; k <= 6; k++)
    for (u64 first = 0; first <= (u64(1) << k); first++)
      for (u64 len = 0; first + len <= (u64(1) << k); len++) {
        auto values = std::make_shared<std::vector<E>>(len);
        for (E &x : *values) x = data.e();
        const std::vector<Tensor> g = member_inner::stretch_in_the_cube(k, first, values);
        stretches++;
        if (g.size() > 2 * size_t(k) + 1) return std::printf("FAILED: %zu Tensors for a stretch in a cube of %u variables\n", g.size(), k), 1;
        std::vector<E> point(k);
        for (E &x : point) x = data.e();
        E extension = E::zero(), direct = E::zero();
        for (const Tensor &t : g) {
          if (t.vars() != k || (t.has_dense() && t.values.get() != values.get())) return std::printf("FAILED: a Tensor of another height, or with a vector of its own\n"), 1;
          extension += t.mle(point.data());
        }
        for (u64 j = 0; j < (u64(1) << k); j++) {
          E sum = E::zero();
          for (const Tensor &t : g) sum += t.at(j);
          const E want = j >= first && j < first + len ? (*values)[j - first] : E::zero();
          if (sum != want) return std::printf("FAILED: entry %llu of the stretch (%llu, %llu) in a cube of %u variables\n", (unsigned long long)j, (unsigned long long)first, (unsigned long long)len, k), 1;
          direct += eq_index(j, point.data(), k) * want;
        }
        if (extension != direct) return std::printf("FAILED: the extension at a point differs\n"), 1;
      }
  const unsigned k = 22;
  const u64 first = 1234567, len = (u64(1) << 20) - 4321;
  auto values = std::make_shared<std::vector<E>>(len);
  for (E &x : *values) x = data.e();
  const std::vector<Tensor> g = member_inner::stretch_in_the_cube(k, first, values);
  std::vector<E> point(k);
  for (E &x : point) x = data.e();
  E extension = E::zero(), direct = E::zero();
  for (const Tensor &t : g) extension += t.mle(point.data());
  for (u64 i = 0; i < len; i++) direct += eq_index(first + i, point.data(), k) * (*values)[i];
  if (g.size() > 2 * k || extension != direct) return std::printf("FAILED: the long stretch (%zu Tensors)\n", g.size()), 1;
  std::printf("ok: %zu stretches, and one of %llu rows in %zu Tensors\n", stretches, (unsigned long long)len, g.size());
  return 0;
}
