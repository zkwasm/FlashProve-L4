// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests that a planner reads each of its sizes (structs such as `GroupSize`) once
   (blocks/numbers/plan.h, "READ ONCE") and that a round enters the plan complete. A planner, such
   as `plan_fold`, adds the rounds and messages of one protocol step to a `Plan` (a proof's rounds
   with their error bounds). It takes its sizes by reference and allocates memory, so a replaced
   global operator new, or a second thread, can run code during the call. If a planner read a field
   once for a round's error bound and again for what the plan states about the round, that code
   could make the two differ. This
   file replaces operator new but defines no test macro of the library, which it uses as any program
   does. Not tested: the functions that run a proof, planners that `main` does not call (such as
   `plan_iopp`), fields that no case changes, groups with mask rows (`GroupSize::mask`), scopes
   other than `Scope::ONCE`; each field is changed in one direction only. With one of the planners'
   22 copies of an argument (blocks/numbers/blocks.h) replaced by a reference, the test must fail
   under g++ and clang++ (checked by a test that is not part of this repository), except for four that
   it cannot detect: `tree` of `plan_lin_shared`, `trees` of `plan_loc`, `plan_cs`, `plan_base`. */
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>

#include "numbers/plan.h"

// The replaced operator new calls a hook (`turn`) at a chosen allocation (see `play`, `whole`). The
// test detects a second read only while the compiler lets a replaced operator new write globals
// (g++ 13.3 and clang++ do). A compiler that assumes otherwise may compile a header that reads
// twice as if it read once: the test then succeeds and detects nothing, and nothing in this
// repository would notice or report it.
static long calls = 0, turn_at = -1;  // allocations counted; which one fires the hook (-1: none)
static void (*turn)() = nullptr;
[[gnu::noinline]] void *operator new(std::size_t n) {
  if (calls++ == turn_at && turn) turn();
  if (void *p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void *p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void *p, std::size_t) noexcept { std::free(p); }

using namespace blk;

static int failures = 0;
#define CHECK(x, what) \
  do { \
    if (!(x)) std::printf("FAILED, line %d: %s\n", __LINE__, what), failures++; \
  } while (0)

// operator== skips `said`, `rule` and `m` of `group` and `said` of `table`: compared separately.
static bool same(const PlanRound &a, const PlanRound &b) {
  return a.name == b.name && a.where == b.where && a.kind == b.kind && a.why == b.why && a.scope == b.scope && Real::cmp(a.error, b.error) == 0 && a.rests == b.rests && a.work_bits == b.work_bits &&
         a.count == b.count && a.part == b.part && a.message_bytes == b.message_bytes && a.roots == b.roots && a.answers == b.answers && a.draw_E == b.draw_E && a.draw_nonzero == b.draw_nonzero &&
         a.draw_positions == b.draw_positions && a.log_n == b.log_n && a.draw_trits == b.draw_trits && a.rows == b.rows && a.first_list == b.first_list && a.group == b.group &&
         a.group.said == b.group.said && a.group.rule == b.group.rule && a.group.m == b.group.m && a.table == b.table && a.table.said == b.table.said && a.combinations == b.combinations &&
         a.sharing == b.sharing && a.priced == b.priced;
}
static bool same(const Planned &a, const Planned &b) {
  if (a->refused() != b->refused() || a->rounds_.size() != b->rounds_.size() || a->messages_.size() != b->messages_.size()) return false;
  for (size_t i = 0; i < a->rounds_.size(); ++i)
    if (!same(a->rounds_[i], b->rounds_[i])) return false;
  for (size_t i = 0; i < a->messages_.size(); ++i) {
    const PlanMessage &x = a->messages_[i], &y = b->messages_[i];
    if (x.name != y.name || x.where != y.where || x.kind != y.kind || x.bytes != y.bytes || x.expected != y.expected || x.count != y.count || x.part != y.part || x.in_chain != y.in_chain || x.hash_calls != y.hash_calls) return false;
  }
  return true;
}

// The sizes that the hook changes: globals, so that a plain function can reach them.
static GroupSize g, to;
static TableSize T;
static LinSize *x = nullptr;
static SampleSize *u = nullptr;
static PscSize psc;
static LookupSize lookup;
static std::vector<SampleSize> *sent = nullptr;
static std::vector<TreeSize> trees;
static TreeSize tree;

template <class X, class A, class B>
static void between(X &x, A a, B b) { x = x == (X)a ? (X)b : (X)a; }  // toggles x between a and b

static const Setting s = setting_of_this_build();
static unsigned long tried = 0;

// One case. `lay` calls one planner on the globals. `turned` changes one field of a size (for
// plan_chain(), its handle; a few cases change more at once); a second call changes the field back.
// For every k, the hook calls `turned` at allocation k within `lay`; the plan must then equal, in
// every round and message, one of the two honest plans: the one made with the field unchanged, or
// the one made with it changed from the start. The two honest plans must differ and neither may be
// rejected (`refused`): otherwise the case fails.
static void play(const char *what, const std::function<void(Plan &)> &lay, void (*turned)()) {
  auto planned = [&](long at) {
    Plan p(s);
    p.begin_scope(Scope::ONCE);
    calls = 0, turn_at = at, turn = turned;
    lay(p);
    const long made = calls;
    const bool fired = at >= 0 && at < made;
    turn_at = -1, turn = nullptr;
    if (fired) turned();  // back
    return std::make_pair(p.finish(), made);
  };
  const auto as_it_was = planned(-1);
  turned();
  const auto as_turned = planned(-1);
  turned();
  char line[200];
  std::snprintf(line, sizeof line, "%s: an honest plan is refused, or the two honest plans are alike: the case shows nothing", what);
  CHECK(!as_it_was.first->refused() && !as_turned.first->refused() && !same(as_it_was.first, as_turned.first), line);
  long parted = 0;
  for (long k = 0; k < as_it_was.second; ++k) {  // runs agree up to the hook: no larger k can fire
    const Planned q = planned(k).first;
    if (!same(q, as_it_was.first) && !same(q, as_turned.first)) ++parted;
    ++tried;
  }
  std::snprintf(line, sizeof line, "%s: at %ld of %ld allocations the plan is neither of the two honest ones", what, parted, as_it_was.second);
  CHECK(!parted, line);
}

// Checks that a round enters the plan complete. For every k, look() copies the unfinished plan at
// allocation k within `lay`; the copy's rounds must equal the first rounds of the finished plan.
static Plan *in_the_making = nullptr;
static Planned found;
static void look() { found = in_the_making->finish(); }
static void whole(const char *what, const std::function<void(Plan &)> &lay) {
  long made = 1, half = 0, with_a_round = 0;
  for (long k = 0; k < made; ++k) {
    Plan p(s);
    p.begin_scope(Scope::ONCE);
    in_the_making = &p, found = Planned();
    calls = 0, turn_at = k, turn = look;
    lay(p);
    if (!k) made = calls;  // includes the allocations of `look`: at the last few k no hook fires
    turn_at = -1, turn = nullptr, in_the_making = nullptr;
    const Planned q = p.finish();
    bool a_beginning = found->rounds_.size() <= q->rounds_.size();
    for (size_t i = 0; a_beginning && i < found->rounds_.size(); ++i) a_beginning = same(found->rounds_[i], q->rounds_[i]);
    if (!a_beginning) ++half;
    if (!found->rounds_.empty()) ++with_a_round;
    ++tried;
  }
  char line[200];
  std::snprintf(line, sizeof line, "%s: at %ld of %ld allocations a round lies in the plan that is not yet what it will be", what, half, made);
  CHECK(!half, line);
  CHECK(with_a_round, "the hook never found a round in the plan: the case shows nothing");
}

int main() {
  CodeSize c;
  c.log_n = 20, c.ell = 1 << 17, c.r = 101;
  g.main = c, g.k = 4, g = group_of_m(g, 50);
  to = g, to.main.log_n = 19, to.main.ell = 1 << 13, to.k = 2, to = group_of_m(to, 40);
  T.code.log_n = 33, T.code.ell = 402653184, T.code.r = 44040191, T.v = 11, T.distances.radius = T.distances.later_radius = T.code.n() / 4;
  const SampleSize sample{24, {g}};
  LinSize lin{sample, sample, sample};
  lin.instances = 64, lin.k_lo = 18, lin.rows_log = 22;
  x = &lin;
  SampleSize one = sample;
  u = &one;
  std::vector<SampleSize> two = {sample, SampleSize{20, {to}}};
  sent = &two;
  tree.name = "a tree with a name of more than fifteen bytes", tree.log_n = 33, tree.leaf_bytes = 16384;
  TreeSize small = tree;
  small.log_n = 20;
  trees = {small};
  psc.degree = {3, 3, 4};
  lookup = LookupSize(10, 8, 4, true);
  const std::string where = "a label of more than fifteen bytes, so that its copy allocates";

  play("LIN, which test", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); }, [] { x->one_time = !x->one_time; });
  play("LIN, the table's later radius", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); }, [] { between(T.distances.later_radius, T.code.n() / 4, T.code.n() / 8); });
  play("LIN, the requests that share the point", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); }, [] { between(x->instances, 64, 1); });
  play("LIN, the table's rows", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); }, [] { between(T.v, 11, 10); });
  play("LINK, the table's later radius", [&](Plan &p) { u64 a = 0, b = 0; plan_link(p, where, T, *u, tree, a, b); }, [] { between(T.distances.later_radius, T.code.n() / 4, T.code.n() / 8); });
  play("LINK, the sample's cube", [&](Plan &p) { u64 a = 0, b = 0; plan_link(p, where, T, *u, tree, a, b); }, [] { between(u->cube_bits, 24, 20); });
  static u64 first, further;  // `plan_link`'s outputs, by reference too: written last, never read
  play("LINK, the caller's two numbers", [&](Plan &p) { plan_link(p, where, T, *u, tree, first, further); }, [] { between(T.v, 11, 10), first += 7, further += 7; });
  play("LINK, the tables' tree", [&](Plan &p) { u64 a = 0, b = 0; plan_link(p, where, T, *u, tree, a, b); }, [] { between(tree.leaf_bytes, 16384, 64); });  // opened twice
  play("FOLD, the choice m", [&](Plan &p) { plan_fold(p, where, g, 1); }, [] { between(g.m, 50, 30); });
  play("FOLD, the rows", [&](Plan &p) { plan_fold(p, where, g, 1); }, [] { between(g.k, 4, 3); });
  play("FOLD's rounds, the choice m", [&](Plan &p) { plan_fold_rounds(p, where, g); }, [] { between(g.m, 50, 30); });
  play("LOC, the choice m", [&](Plan &p) { plan_loc(p, where, g, trees); }, [] { between(g.m, 50, 30); });
  play("LOC, the code's length", [&](Plan &p) { plan_loc(p, where, g, trees); }, [] { between(g.main.log_n, 20, 21); });
  play("LOC, a tree's leaves", [&](Plan &p) { plan_loc(p, where, g, trees); }, [] { between(trees[0].leaf_bytes, 16384, 64); });
  play("CS, the stage that is left", [&](Plan &p) { plan_cs(p, where, g, to, 15, 3, 2, trees, 1); }, [] { between(g.m, 50, 30); });
  play("CS, the stage that follows", [&](Plan &p) { plan_cs(p, where, g, to, 15, 3, 2, trees, 1); }, [] { between(to.m, 40, 20); });
  play("BASE, the choice m", [&](Plan &p) { plan_base(p, where, g, trees, 1); }, [] { between(g.m, 50, 30); });
  play("BASE, the message", [&](Plan &p) { plan_base(p, where, g, trees, 1); }, [] { between(g.main.r, 101, 77); });
  play("SEND, a cube", [&](Plan &p) { plan_send(p, where, *sent, 1); }, [] { between((*sent)[0].cube_bits, 24, 30); });
  play("BAT, the rows' groups", [&](Plan &p) { plan_bat(p, where, u->groups, 5); }, [] { static const TestedIn a{g}, b{to}; static bool other = false; other = !other, u->groups = other ? b : a; });
  play("PSC, a degree", [&](Plan &p) { plan_psc(p, where, psc); }, [] { between(psc.degree[1], 3, 5); });
  play("PSC, the degree in Y", [&](Plan &p) { plan_psc(p, where, psc); }, [] { between(psc.degree_Y, 2, 3); });
  play("the look-up, its keys", [&](Plan &p) { plan_lookup(p, where, lookup); }, [] { between(lookup.v, 10, 9); });
  play("the look-up, its table", [&](Plan &p) { plan_lookup(p, where, lookup); }, [] { between(lookup.d, 8, 7); });
  whole("LIN", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); });
  whole("LINK", [&](Plan &p) { plan_link(p, where, T, *u, tree, first, further); });
  whole("FOLD", [&](Plan &p) { plan_fold(p, where, g, 1); });
  whole("CS", [&](Plan &p) { plan_cs(p, where, g, to, 15, 3, 2, trees, 1); });
  whole("BASE", [&](Plan &p) { plan_base(p, where, g, trees, 1); });
  whole("the look-up", [&](Plan &p) { plan_lookup(p, where, lookup); });
  T.answers = 2, T.distances.radius = T.distances.later_radius = 0;
  CHECK(size_recorded_table(s, T, 64, T.v + 18), "no distances for the table");
  play("the record, its answers", [&](Plan &p) { plan_record(p, where, T); }, [] { between(T.answers, 2, 3); });
  play("an opening, the tree's leaves", [&](Plan &p) { plan_open(p, where, tree, 100); }, [] { between(tree.leaf_bytes, 16384, 64); });
  play("an opening, the tree's depth", [&](Plan &p) { plan_open(p, where, tree, 100); }, [] { between(tree.log_n, 33, 30); });
  play("LOC, a tree's depth", [&](Plan &p) { plan_loc(p, where, g, trees); }, [] { between(trees[0].log_n, 20, 18); });
  play("LIN, the table's tree", [&](Plan &p) { plan_lin_shared(p, where, T, *x, tree); }, [] { between(tree.log_n, 33, 30); });
  // A planner that iterates over an argument's entries reads each entry once, so changing one entry
  // detects nothing. The next two cases change both groups of a `TestedIn` at once: a planner that
  // iterated over the caller's own object would, for some k, read the first group unchanged and the
  // second changed. list_of() takes the largest list size (`Analysis::list`) among the groups, and
  // a larger `m` gives a larger one. Unchanged, `m` is 20 and 20; changed, 50 and 30; the mixture,
  // 20 and 30, gives the list size of neither.
  static const TestedIn shortest{group_of_m(g, 20), group_of_m(g, 20)}, longest_first{g, group_of_m(g, 30)};
  const auto both_groups = [] { static bool other = false; other = !other, u->groups = other ? longest_first : shortest; };
  u->groups = shortest;
  play("BAT, two groups", [&](Plan &p) { plan_bat(p, where, u->groups, 5); }, both_groups);
  play("LINK, the sample's two groups", [&](Plan &p) { u64 a = 0, b = 0; plan_link(p, where, T, *u, tree, a, b); }, both_groups);
  // plan_chain() reads a finished plan through a handle (`Planned`): first the rounds, then the
  // messages.
  static Planned chains[2], handle;
  for (int i = 0; i < 2; ++i) {
    Plan p(s);
    p.begin_scope(Scope::ONCE);
    i ? (void)plan_base(p, where, g, trees, 1) : plan_fold(p, where, g, 1);
    chains[i] = p.finish();
  }
  handle = chains[0];
  play("a chain, its handle", [&](Plan &p) { plan_chain(p, handle); }, [] { static bool other = false; other = !other, handle = chains[other]; });
  std::printf(failures ? "FAILED: %d checks (%lu plans)\n" : "ok%.0d: %lu plans, each laid with a hook at one allocation: with a field turned there, one of the two honest plans; looked at there, a beginning of the finished plan\n", failures, tried);
  return failures != 0;
}
