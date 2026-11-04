// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests `Binder` (blocks/chain/public.h), which writes named public inputs (items) into the first
// query (hash input) of a hash chain, a Fiat-Shamir transcript. On a few examples: the bytes differ
// if the items differ, and include the plan (the chain's list of rounds). Items out of order, or
// added after a channel (`ProverRounds`) has started, abort the process.
// More tests of `Public`, an item's handle: blocks/chain/tests/rounds_test.cpp.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets a test build a `Plan` by hand (blocks/numbers/plan.h)
#include <cstdio>
#include <string>
#include <vector>

#include "chain/public.h"
#include "chain/rounds.h"
#include "chain/tests/death.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      failures++; \
    } \
  } while (0)

static Plan small_plan(u64 rows) {
  Plan plan(setting_goldilocks());
  plan.send_scalars("x", 1);
  PlanRound &round = plan.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
  round.draw_E = 1, round.rows = rows;
  return plan;
}

int main() {
  const Plan plan = small_plan(2);
  Digest made_up{}; // a configuration's digest
  made_up.w[0] = 77;
  using List = std::vector<u64>;

  // Where one item ends and the next begins is part of the bytes.
  auto two = [&](const char *name_a, const List &a, const char *name_b, const List &b) {
    Binder binder(7, plan);
    binder.parameter(name_a, a), binder.parameter(name_b, b);
    return binder.parameters_digest();
  };
  const Digest base = two("ab", {1, 2}, "c", {3});
  CHECK(two("ab", {1, 2}, "c", {3}) == base);
  CHECK(two("a", {1, 2}, "bc", {3}) != base);
  CHECK(two("ab", {1}, "c", {2, 3}) != base);
  CHECK(two("ab", {1, 2, 3}, "c", {}) != base);
  CHECK(two("c", {3}, "ab", {1, 2}) != base);
  {
    Binder nested(7, plan), flat(7, plan);
    nested.parameter("lists", std::vector<List>{{1}, {2, 3}});
    flat.parameter("lists", std::vector<List>{{1, 2}, {3}});
    CHECK(nested.parameters_digest() != flat.parameters_digest());
  }
  {
    // Two plans that differ in nothing but the number of rows that a round's error counts.
    const Plan other = small_plan(5);
    Binder a(7, plan), b(7, other);
    CHECK(a.parameters_digest() != b.parameters_digest());
  }

  // Order: parameters, then the statement, then the chain.
  // ends_with() (blocks/chain/tests/death.h): true if the code, run in a child process, aborts
  // with a message containing the text.
  CHECK(ends_with("the parameters are closed", [&] {
    Binder binder(7, plan);
    binder.statement("target", (u64)1);
    binder.parameter("table", (u64)2);
  }));
  CHECK(ends_with("the parameters are closed", [&] {
    Binder binder(7, plan);
    binder.parameters_digest(); // closes them too: the run's value may already be hashed from it
    binder.parameter("table", (u64)2);
  }));
  CHECK(ends_with("its first query takes nothing more", [&] {
    Binder binder(7, plan);
    binder.configuration(made_up);
    Writer proof;
    ProverRounds ch(binder, Digest{}, proof);
    binder.statement("target", (u64)1);
  }));
  CHECK(ends_with("do not stand in this first query", [&] {
    Binder binder(7, plan), other(7, plan);
    binder.random_bytes_are(other.statement("a root", Digest{}));
  }));

  // Every chain names its configuration (the public files' digest), once, as the first item after
  // the plan.
  CHECK(ends_with("every chain names its configuration", [&] {
    Binder binder(7, plan);
    Writer proof;
    ProverRounds ch(binder, Digest{}, proof);
  }));
  CHECK(ends_with("the configuration is the first item behind the plan", [&] {
    Binder binder(7, plan);
    binder.parameter("table", (u64)2);
    binder.configuration(made_up);
  }));
  CHECK(ends_with("names its configuration once", [&] {
    Binder binder(7, plan);
    binder.configuration(made_up), binder.configuration(made_up);
  }));

  {  // `Binder` keeps a read-only copy (`Planned`, blocks/numbers/plan.h): a round added to the
    // `Plan` later reaches neither the first query nor `plan()`, which a channel reads.
    Plan growing = small_plan(2);
    Binder binder(7, growing), control(7, small_plan(2));
    growing.close("a round entered late", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    CHECK(binder.plan()->rounds_.size() + 1 == growing.rounds_.size() && binder.parameter_bytes() == control.parameter_bytes());
  }

  // Derived values: a channel reads them like the `Public` values they come from.
  {
    Binder binder(7, plan);
    binder.configuration(made_up);
    const Public<List> list = binder.parameter("list", List{4, 5, 6});
    const Public<u64> factor = binder.parameter("factor", (u64)10);
    const Public<u64> sum = list.derive([](const List &v) { return v[0] + v[1] + v[2]; });
    const Public<u64> product = derive(sum, factor, [](u64 s, u64 f) { return s * f; });
    Writer proof;
    ProverRounds ch(binder, Digest{}, proof);
    CHECK(ch.get(sum) == 15 && ch.get(product) == 150 && ch.get(element(list, 1)) == 5);
  }
  CHECK(ends_with("two values of two first queries", [&] {
    Binder binder(7, plan), other(7, plan);
    derive(binder.parameter("a", (u64)1), other.parameter("b", (u64)2), [](u64 a, u64 b) { return a + b; });
  }));
  CHECK(ends_with("no such entry", [&] {
    Binder binder(7, plan);
    element(binder.parameter("list", List{4}), 1);
  }));

  if (failures) return std::printf("public_test: %d FAILED\n", failures), 1;
  std::printf("public inputs: ok\n");
  return 0;
}
