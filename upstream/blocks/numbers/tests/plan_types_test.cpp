// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests the two plan types of blocks/numbers/plan.h. A plan lists the rounds and messages of a
   protocol run. The members of `Plan` that add a round or a message are private, open only to the
   planners: the plan_... functions that class `MayLay` (blocks/numbers/blocks.h) lists as friends.
   A `Planned`, which `finish` makes from a `Plan`, is a read-only handle.
   The static_asserts check this with type traits, which cannot tell a private member from a
   misspelled one. As the control, blocks/numbers/tests/plan_types_by_hand_test.cpp includes this
   file with BLK_PLANS_BY_HAND_FOR_A_TEST defined, which makes the members public: each expression
   compared with BY_HAND must then be well formed. Sources that try to bypass the two types,
   and must not compile, are in blocks/tests/must_not_build/. The types guard only against
   mistakes: a cast or a macro, for example, gets past them.
   `main` has the run-time checks. There a chain is a finished plan appended by `plan_chain`, and
   a repeat (`push_repeat`) stands for n copies of its rounds and messages. */
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>

#include "numbers/plan.h"  // it includes blocks/numbers/blocks.h at its end

using namespace blk;

#ifdef BLK_PLANS_BY_HAND_FOR_A_TEST
constexpr bool BY_HAND = true;
#else
constexpr bool BY_HAND = false;
#endif
static int failures = 0;
#define CHECK(x, what) \
  do { \
    if (!(x)) std::printf("FAILED, line %d: %s\n", __LINE__, what), failures++; \
  } while (0)

// CAN(name, expression): name<T>::value tells whether the expression, in which X is an lvalue of
// type T, is well formed in this file. `start_under` and `rounds_for_a_test` are members that
// `Plan` must not have.
#define CAN(name, expression) \
  template <class T, class = void> struct name : std::false_type {}; \
  template <class T> struct name<T, std::void_t<decltype(expression)>> : std::true_type {}
#define X std::declval<T &>()
const Bound &a_bound();  // never called
CAN(can_close, X.close(std::string(), RoundKind::SEAL, ErrorKind::ZERO, a_bound(), Priced::NOTHING));
CAN(can_close_whole, X.close(std::string(), RoundKind::SEAL, ErrorKind::ZERO, a_bound(), Priced::NOTHING, PlanRound()));
CAN(can_close_none, X.close_none(std::string()));
CAN(can_enter, X.enter(PlanRound()));
CAN(can_append, X.append(Planned()));
CAN(can_send_scalars, X.send_scalars(std::string(), 1));
CAN(can_send_answers, X.send_answers(std::string(), 1));
CAN(can_send_root, X.send_root(std::string()));
CAN(can_file, X.file(std::string(), MessageKind::HEAD, 1, 1.0, 0));
CAN(can_label, X.last_message_at(std::string()));
CAN(sees_rounds, X.rounds_);
CAN(sees_messages, X.messages_);
CAN(sees_opened, X.one_list_opened_);
CAN(can_start_under, X.start_under(Setting()));
CAN(has_the_door, X.rounds_for_a_test());
CAN(can_refuse, X.refuse(std::string()));  // the control: a public call is seen
CAN(helper_close, T::close(std::declval<Plan &>(), "", std::string(), RoundKind::SEAL, ErrorKind::ZERO, a_bound(), Priced::NOTHING));
CAN(helper_enter, T::enter(std::declval<Plan &>(), PlanRound()));
CAN(helper_chain, T::chain(std::declval<Plan &>(), Planned()));
static_assert(can_refuse<Plan>::value, "the looking glass is blind");
static_assert(can_close<Plan>::value == BY_HAND && can_close_whole<Plan>::value == BY_HAND && can_close_none<Plan>::value == BY_HAND && can_enter<Plan>::value == BY_HAND && can_append<Plan>::value == BY_HAND, "a round can be laid from outside, or not under the word");
static_assert(can_send_scalars<Plan>::value == BY_HAND && can_send_answers<Plan>::value == BY_HAND && can_send_root<Plan>::value == BY_HAND && can_file<Plan>::value == BY_HAND && can_label<Plan>::value == BY_HAND, "a message can be laid from outside, or not under the word");
static_assert(sees_rounds<Plan>::value == BY_HAND && sees_messages<Plan>::value == BY_HAND && sees_opened<Plan>::value == BY_HAND, "what is being written can be reached from outside, or not under the word");
static_assert(!can_start_under<Plan>::value && !has_the_door<Plan>::value, "`Plan` has a member that it must not have");
static_assert(!helper_close<MayLay>::value && !helper_enter<MayLay>::value && !helper_chain<MayLay>::value, "the planners' helpers can be called from outside");  // (control for `close`: blocks/tests/must_not_build/the_planners_helper.cpp, rejected as private; none for the other two)

// A `Planned` with rounds comes only from finish(); callers cannot assign to or destroy its data.
static_assert(std::is_convertible<Plan, Planned>::value == BY_HAND && std::is_constructible<Planned, Plan>::value == BY_HAND && std::is_constructible<Plan, Planned>::value == BY_HAND, "the tests' roads are open without their word, or shut under it");
static_assert(!std::is_constructible<Planned, const Planned::Data *>::value && !std::is_constructible<Planned, std::shared_ptr<const Planned::Data>>::value, "a Planned can be made of data of the caller's own");
static_assert(!std::is_destructible<Planned::Data>::value, "a plan's data can be ended from outside");  // (a type that cannot be destroyed fails every is_constructible trait, so none is used: its private constructor and deleted copy constructor are not tested, only read in plan.h)
static_assert(!std::is_assignable<Plan &, Plan>::value && !std::is_assignable<Plan &, const Plan &>::value, "a Plan's setting can be replaced by an assignment");
#define NOT_WRITABLE(expression, type) static_assert(!std::is_assignable<decltype((expression)), type>::value, "a number that a plan holds can be written from outside: " #expression)
#define Q std::declval<Planned &>()
NOT_WRITABLE(Q->rounds_[0].draw_positions, u64);
NOT_WRITABLE(Q->rounds_[0].combinations, u64);
NOT_WRITABLE(Q->rounds_[0].table.test, TableTest);
NOT_WRITABLE(Q->rounds_[0].error, Real);
NOT_WRITABLE(Q->rounds_[0].priced, Priced);
NOT_WRITABLE(Q->rounds_.back(), PlanRound);
NOT_WRITABLE(Q->rounds_, std::vector<PlanRound>);
NOT_WRITABLE(Q->messages_[0].bytes, u64);
NOT_WRITABLE(Q->messages_, std::vector<PlanMessage>);
NOT_WRITABLE(Q->parts, std::vector<std::string>);
NOT_WRITABLE(Q->set, Setting);
NOT_WRITABLE(Q->set.target, unsigned);
NOT_WRITABLE(*Q, const Planned::Data &);
NOT_WRITABLE(std::declval<Plan &>().set, Setting);
NOT_WRITABLE(std::declval<Plan &>().set.target, unsigned);
static_assert(std::is_assignable<decltype((std::declval<PlanRound &>().draw_positions)), u64>::value && std::is_assignable<Planned &, Planned>::value, "the looking glass is blind");

/* Every friend declaration in class `MayLay` must name a function that blocks/numbers/blocks.h
   defines: if one did not, any source could define that function and so become a friend. This
   array is a copy of the list, kept by hand. Each entry takes an address by the declaration's full
   type, which builds only if that header defines such a function. tests/new/run_new.py, stage
   `fences`, checks that the list has as many lines as this array has entries, so a declaration that
   has outlived its function is detected. */
static const void *const listed[] = {
    (const void *)(void (*)(Plan &, const std::string &, const std::vector<SampleSize> &, unsigned, bool))plan_send,
    (const void *)(void (*)(Plan &, const std::string &, const TestedIn &, u64))plan_bat,
    (const void *)(void (*)(Plan &, const std::string &, u64))plan_bat_sigma,
    (const void *)(void (*)(Plan &, const std::string &, const PscSize &))plan_psc,
    (const void *)(void (*)(Plan &, const std::string &, unsigned))plan_zero,
    (const void *)(void (*)(Plan &, const std::string &))plan_line,
    (const void *)(void (*)(Plan &, const std::string &, u64))plan_shift,
    (const void *)(void (*)(Plan &, const std::string &, unsigned))plan_join,
    (const void *)(void (*)(Plan &, const std::string &, unsigned))plan_lkp,
    (const void *)(void (*)(Plan &, const std::string &, const LookupSize &))plan_lookup,
    (const void *)(void (*)(Plan &, const std::string &, u64))plan_rng,
    (const void *)(void (*)(Plan &, const std::string &, const GroupSize &, unsigned))plan_fold,
    (const void *)(void (*)(Plan &, const std::string &, const GroupSize &))plan_fold_rounds,
    (const void *)(u64 (*)(Plan &, const std::string &, const GroupSize &, const std::vector<TreeSize> &))plan_loc,
    (const void *)(u64 (*)(Plan &, const std::string &, const GroupSize &, const GroupSize &, unsigned, u64, unsigned, const std::vector<TreeSize> &, unsigned))plan_cs,
    (const void *)(u64 (*)(Plan &, const std::string &, const GroupSize &, const std::vector<TreeSize> &, unsigned))plan_base,
    (const void *)(void (*)(Plan &, const std::string &, const TableSize &))plan_record,
    (const void *)(void (*)(Plan &, const std::string &))plan_seal,
    (const void *)(u64 (*)(Plan &, const std::string &, const TableSize &, const LinSize &, const TreeSize &))plan_lin_shared,
    (const void *)(void (*)(Plan &, const std::string &, const TableSize &, const SampleSize &, const TreeSize &, u64 &, u64 &))plan_link,
    (const void *)(void (*)(Plan &, const char *, const std::string &))plan_none,
    (const void *)(void (*)(Plan &, const char *, const std::string &, u64))plan_scalars,
    (const void *)(void (*)(Plan &, const std::string &, unsigned))plan_roots,
    (const void *)(void (*)(Plan &, const std::string &, const TreeSize &, u64))plan_open,
    (const void *)(void (*)(Plan &, const std::string &, bool))plan_chain_start,
    (const void *)(void (*)(Plan &, const char *, const std::string &, MessageKind, u64, double, u64))plan_file_item,
    (const void *)(void (*)(Plan &, const Planned &))plan_chain,
};

int main() {
  const Setting s = setting_of_this_build();
  for (const void *routine : listed) CHECK(routine, "a listed planner has no address");
  {  // no handle is empty
    Planned none;
    CHECK(!none->holds() && none->refused() && none->why_not() == "nothing was planned" && none->rounds_.empty(), "a Planned that nobody planned holds, or names nothing");
    Plan p(s);
    plan_line(p, "a line");
    Planned a = p.finish();
    CHECK(a->holds() && a->cause() == Cause::NONE && a->why_not().empty() && a->rounds_.size() == 1, "a plan of one round does not hold");
    Planned b = std::move(a);
    CHECK(a->holds() && a->rounds_.size() == 1 && &*a == &*b, "a handle that was moved from is empty, or names other data");
  }
  {  // a `Cursor` keeps the plan it was made on when the caller's variable is assigned another
    Plan one(s), other(s);
    plan_line(one, "the first plan"), plan_zero(other, "the second plan", 3);
    Planned q = one.finish();
    Cursor walk(q);
    q = other.finish();
    CHECK(walk.next_round(R_LINE).where == "the first plan" && walk.at_end(), "a Cursor walks another plan than it was made on");
    plan_line(one, "laid behind finish()");  // a finished `Plan` can be added to and finished again
    Planned again = one.finish();
    CHECK(again->rounds_.size() == 2 && q->rounds_.size() == 1, "a Plan that is written on is not finished anew");
  }
  {  // `finish` rejects (`refused`) a plan with an open message or repeat but keeps its messages
    Plan open_message(s), open_repeat(s);
    plan_scalars(open_message, "scalars", "nowhere", 2);
    open_repeat.push_repeat(3), plan_line(open_repeat, "inside");
    const Planned a = open_message.finish(), b = open_repeat.finish();
    CHECK(a->refused() && a->why_not().find("in the middle of a prover's message") != std::string::npos && a->messages_.size() == 1, "a plan that ends in the middle of a prover's message is not refused by name");
    CHECK(b->refused() && b->why_not().find("inside a repeat") != std::string::npos, "a plan that ends inside a repeat is not refused by name");
  }
  {  // nested repeats do not wrap: unchecked, a * b = 1 mod 2^64 gives count 1 inside them, 0 after
    const u64 a = 3, b = 0xAAAAAAAAAAAAAAABull;
    Plan p(s);
    p.push_repeat(a), p.push_repeat(b), p.pop_repeat(), p.pop_repeat();
    CHECK(a * b == 1 && p.times() == 1 && p.refused(), "repeats of 2^64 times or more are taken");
    Plan chain(s), whole(s), control(s);  // nor when a chain with a repeat is appended in a repeat
    chain.push_repeat((u64)1 << 32), plan_line(chain, "a line"), chain.pop_repeat();
    whole.push_repeat((u64)1 << 32), plan_chain(whole, chain.finish()), whole.pop_repeat();
    control.push_repeat((u64)1 << 31), plan_chain(control, chain.finish()), control.pop_repeat();
    CHECK(whole.finish()->refused() && control.finish()->holds() && control.finish()->rounds() == (u64)1 << 63, "a chain's counts wrap under a repeat");
    Plan items(s), many(s);  // likewise a chain that has a message and no round
    items.push_repeat((u64)1 << 32), plan_file_item(items, "item", "", MessageKind::HEAD, 1, 1.0, 0), items.pop_repeat();
    many.push_repeat((u64)1 << 32), plan_chain(many, items.finish()), many.pop_repeat();
    CHECK(items.finish()->holds() && items.finish()->rounds_.empty() && many.finish()->refused(), "a chain's messages' counts wrap under a repeat");
  }
  {  // a chain whose errors were computed under one `Setting` is rejected in a plan under another
    Setting weaker = s;
    weaker.target = s.target / 2;
    Plan chain(weaker), whole(s), same(weaker);
    plan_line(chain, "priced at half the target");
    plan_chain(whole, chain.finish()), plan_chain(same, chain.finish());
    CHECK(whole.finish()->refused() && whole.finish()->why_not().find("another setting") != std::string::npos, "a chain under another setting is taken");
    CHECK(same.finish()->holds(), "control: a chain under the same setting is refused");
    Plan refused(s), carries(s);
    refused.refuse("made to fail");
    plan_chain(carries, refused.finish());
    CHECK(carries.finish()->why_not() == "made to fail", "a chain's refusal is dropped");
  }
  {  // `why_not` names the failing round by number and label
    Setting strict = s;
    strict.target = 400;  // more bits than 1 / |E| gives (E: the challenges' field)
    Plan p(strict);
    plan_seal(p, "a round that holds"), plan_line(p, "too weak");
    const Planned q = p.finish();
    CHECK(q->cause() == Cause::ABOVE_THE_TARGET && q->why_not().find("round 2, ") == 0 && q->why_not().find("too weak") != std::string::npos, "the first cause does not name the round");
  }
  {  // `refuse` with an empty reason rejects the plan too
    Plan p(s);
    p.refuse("");
    CHECK(p.refused() && p.finish()->cause() == Cause::REFUSED, "refuse(\"\") refuses nothing");
  }
  {  // a `TestedIn` (a list of groups) that was moved from is not empty: a move is a copy
    GroupSize g;
    g.main.log_n = 20, g.main.ell = 1 << 17, g.main.r = 101, g.k = 4, g = group_of_m(g, 50);
    TestedIn a(g);
    const TestedIn b(std::move(a));
    CHECK(a.begin() != a.end() && b.begin() != b.end(), "a TestedIn that was moved from is empty: its rounds would be priced without their list");
  }
  {  // a drawn matrix whose number of entries does not fit in 64 bits is rejected
    Plan p(s), q(s);
    plan_rng(p, "too many", ~(u64)0 / 2), plan_rng(q, "few", 3);
    CHECK(p.refused() && p.why().find("does not fit 64 bits") != std::string::npos && !q.refused(), "the number of drawn entries wraps, or a small matrix is refused");
  }
  {  // each of the four numbers of a table's distances is compared
    const TableDistances d{1, 2, 3, 4};
    CHECK((d == TableDistances{1, 2, 3, 4} && !(d == TableDistances{9, 2, 3, 4}) && !(d == TableDistances{1, 9, 3, 4}) && !(d == TableDistances{1, 2, 9, 4}) && !(d == TableDistances{1, 2, 3, 9})), "a number of a table's distances is not compared");
  }
  std::printf(failures ? "FAILED\n" : "ok\n");
  return failures ? 1 : 0;
}
