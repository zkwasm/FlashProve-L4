// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the BLK_REQUIRE checks (blocks/base.h; `stop` in identifiers and output) of six headers in
// blocks/iop/: mask.h, psc.h, hp.h, line.h, summand.h, psc_rows.h. A failed check prints its
// message and aborts: a caller broke a rule that no type enforces. Some checks are all that stands
// between a caller's mistake and a loss of hiding that no verifier or honest run can see. They
// stop the straightforward breaches of three rules, not every breach: (a) one `MaskLedger` for an
// oracle message's mask entries (random values, each added to one sent scalar); (b) no entry given
// out twice, beyond the last, or after a proximity test has begun (MaskLedger::close()); (c) as
// many fresh entries as PSC, the sumcheck with a hiding polynomial (psc.h), sends scalars. No
// other test triggers the checks for (a) or (c): without them every other test would still pass.
// Each check but three is triggered in a child process, which must abort with that check's
// message; a control, the same call made correctly, must run to its end. This shows that the
// checks exist and fire on the case tried, not that the rules suffice. Not tested: the check in
// MaskSlot::oracle(); the two in Weight::checked(), which no public constructor can make fail.
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "field/field.h"
#include "iop/psc_rows.h"
#include "toy_support.h"

using namespace blk;
using K = Fp<101>;

static int failures = 0, stops_seen = 0;
// Runs `f` in a child process. Returns "" if it aborted with `why` (part of a message; null: any
// text) on stderr, else how it ended.
static std::string how_it_ends(const char* why, const std::function<void()>& f) {
  std::fflush(nullptr);
  int fd[2];
  BLK_REQUIRE(pipe(fd) == 0, "test: pipe failed");
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "test: fork failed");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    ::close(fd[0]);
    dup2(fd[1], 2);
    f();
    _exit(0);
  }
  ::close(fd[1]);
  std::string said;
  char buffer[256];
  for (ssize_t got; (got = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, size_t(got));
  ::close(fd[0]);
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "test: waitpid failed");
  if (WIFEXITED(status)) return WEXITSTATUS(status) == 0 ? "it ran to its end" : "it left with a status other than 0";
  if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT) return "it ended in another way than by abort()";
  return !why || said.find(why) != std::string::npos ? "" : "it was stopped with other words: " + said;
}
// `wrong` must abort with the message `why`; `right`, the control, must run to its end.
static void stop(const char* why, const std::function<void()>& wrong, const std::function<void()>& right) {
  const std::string a = how_it_ends(why, wrong), b = how_it_ends(nullptr, right);
  if (!a.empty()) ++failures, std::printf("FAIL: \"%s\": %s\n", why, a.c_str());
  if (b != "it ran to its end") ++failures, std::printf("FAIL: the control beside \"%s\" does not run to its end\n", why);
  ++stops_seen;
}

// One small sumcheck: two variables under equality weights, a product of two `shown` maps (their
// end values are sent, masked; a `kept` map's is not: summand.h). SendAccess::make(0, 0, n)
// (toy_support.h) makes a `SampleReceipt` for n mask entries: this test sends no oracle message.
struct Small {
  Summand<K> S{2, 0, 2, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[1]}; }};
  std::vector<Weight<K>> vars = {Weight<K>::equality(K::from(5), 3), Weight<K>::equality(K::from(9), 3)};
  std::vector<ToyMap<K>> maps = {multilinear(std::vector<K>{K::from(1), K::from(2), K::from(3), K::from(4)}),
                                 multilinear(std::vector<K>{K::from(7), K::from(6), K::from(5), K::from(8)})};
  std::vector<K> row = std::vector<K>(16, K::from(3));
  SumClaim<K> claim(K shift = K::zero()) const {
    SumClaim<K> c;
    c.vars = vars;
    c.target = DefinitionSumcheck<K>(vars, S, maps).sum() + shift;
    return c;
  }
  size_t fresh() const { return psc_masks(claim(), S); }
  // `more`: mask entries passed in excess of what psc_prove() requires (negative: too few).
  std::vector<K> prove(const Summand<K>& summand, SumcheckProver<K>& inner, int more = 0, K shift = K::zero()) const {
    ToyProverChannel<K> ch(random_coins<K>(1));
    MaskLedger ledger(SendAccess::make(0, 0, 16));
    (void)psc_prove(ch, claim(shift), summand, inner, ledger.take(size_t(int(fresh()) + more)), mask_row(row));
    return ch.tape;
  }
  std::vector<K> prove() const {
    DefinitionSumcheck<K> inner(vars, S, maps);
    return prove(S, inner);
  }
  void verify(const std::vector<K>& tape, const Summand<K>& summand, int more = 0) const {
    ToyVerifierChannel<K> ch(tape, random_coins<K>(1));
    MaskLedger ledger(SendAccess::make(0, 0, 16));
    PscOutput<K> out;
    (void)psc_verify(ch, claim(), summand, ledger.take(size_t(int(fresh()) + more)), out);
  }
};
// Wraps `inner`. long_round: round() returns 4 coefficients; few_ends: drops the last end value.
struct Bent : SumcheckProver<K> {
  SumcheckProver<K>& inner;
  bool long_round, few_ends;
  Bent(SumcheckProver<K>& i, bool l, bool f) : inner(i), long_round(l), few_ends(f) {}
  std::vector<K> round() override {
    std::vector<K> s = inner.round();
    if (long_round) s.resize(4, K::one());
    return s;
  }
  void bind(const K& c) override { inner.bind(c); }
  std::vector<K> end_values() override {
    std::vector<K> v = inner.end_values();
    if (few_ends) v.pop_back();
    return v;
  }
};

int main() {
  const Small w;
  // ---- mask.h: what hiding rests on ----
  stop("this receipt has been moved away or has made its ledger",
       [] {
         SampleReceipt r = SendAccess::make(0, 0, 4);
         MaskLedger first(std::move(r));
         MaskLedger second(std::move(r));  // would hand the same entries out again
       },
       [] {
         SampleReceipt r = SendAccess::make(0, 0, 4);
         MaskLedger first(std::move(r));
       });
  stop("the mask rows have no free entry left", [] { MaskLedger l(SendAccess::make(0, 0, 2)); (void)l.take(3); },
       [] { MaskLedger l(SendAccess::make(0, 0, 2)); (void)l.take(2); });
  stop("a proximity test on this oracle message has begun", [] { MaskLedger l(SendAccess::make(0, 0, 2)); l.close(); (void)l.take(); },
       [] { MaskLedger l(SendAccess::make(0, 0, 2)); (void)l.take(); l.close(); });
  stop("this slot has been moved away",
       [] {
         MaskLedger l(SendAccess::make(0, 0, 2));
         MaskSlot a = l.take();
         [[maybe_unused]] MaskSlot b = std::move(a);
         (void)a.index();  // a second use of the entry, through the moved-from `a`
       },
       [] {
         MaskLedger l(SendAccess::make(0, 0, 2));
         MaskSlot a = l.take();
         MaskSlot b = std::move(a);
         (void)b.index();
       });
  // ---- psc.h ----
  const std::vector<K> tape = w.prove();
  for (int more : {-1, 1}) {
    stop("psc: wrong number of fresh mask entries",
         [&] {
           DefinitionSumcheck<K> inner(w.vars, w.S, w.maps);
           (void)w.prove(w.S, inner, more);
         },
         [&] { (void)w.prove(); });
    stop("psc: wrong number of fresh mask entries", [&] { w.verify(tape, w.S, more); }, [&] { w.verify(tape, w.S); });
  }
  {
    Summand<K> no_degree = w.S, nothing{0, 1, 0, [](const std::vector<K>&, const std::vector<K>&) { return std::vector<K>{K::zero(), K::one()}; }};
    no_degree.degree = 0;
    stop("psc: degree > 0 exactly if a map is shown", [&] { (void)psc_vars(w.claim(), no_degree); }, [&] { (void)psc_vars(w.claim(), w.S); });
    stop("psc: nothing to sum over and nothing to show", [&] { (void)psc_vars(SumClaim<K>{}, nothing); }, [&] { (void)psc_vars(w.claim(), nothing); });
    Summand<K> long_answer = w.S;
    long_answer.at = [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[1], K::one()}; };
    stop("psc: Summand::at gives one coefficient for each kept map", [&] { w.verify(tape, long_answer); }, [&] { w.verify(tape, w.S); });
  }
  stop("psc: one end value for each map",
       [&] {
         DefinitionSumcheck<K> inner(w.vars, w.S, w.maps);
         Bent bent(inner, false, true);
         (void)w.prove(w.S, bent);
       },
       [&] {
         DefinitionSumcheck<K> inner(w.vars, w.S, w.maps);
         Bent bent(inner, false, false);
         (void)w.prove(w.S, bent);
       });
  stop("psc: the summand's round polynomial has too many coefficients",
       [&] {
         DefinitionSumcheck<K> inner(w.vars, w.S, w.maps);
         Bent bent(inner, true, false);
         (void)w.prove(w.S, bent);
       },
       [&] { (void)w.prove(); });
  stop("psc: the round polynomial does not sum to the running claim",
       [&] {
         DefinitionSumcheck<K> inner(w.vars, w.S, w.maps);
         (void)w.prove(w.S, inner, 0, K::one());  // an honest prover on a false target
       },
       [&] { (void)w.prove(); });
  // ---- line.h, summand.h ----
  {
    auto at = [](u64 x) { return PointClaim<K>{{K::from(x)}, {}, K::one()}; };
    stop("line: the two claims must be at one point",
         [&] { ToyProverChannel<K> ch(random_coins<K>(1)); (void)line(ch, at(2), at(3)); },
         [&] { ToyProverChannel<K> ch(random_coins<K>(1)); (void)line(ch, at(2), at(2)); });
    stop("SumClaim::at_point: one degree bound for each variable", [&] { (void)SumClaim<K>::at_point(at(2), {3, 3}); },
         [&] { (void)SumClaim<K>::at_point(at(2), {3}); });
    stop("Summand: one coefficient for each kept map", [&] { (void)w.S.at_values({}, {K::one(), K::one()}, {K::one()}); },
         [&] { (void)w.S.at_values({}, {K::one(), K::one()}, {}); });
  }
  // ---- hp.h ----
  {
    const Weight<K> eq = Weight<K>::equality(K::from(5), 3);
    const std::vector<Weight<K>> two = {eq, eq};
    stop("Weight: at least two coefficients", [] { (void)Weight<K>::plain(1); }, [] { (void)Weight<K>::plain(2); });
    stop("Weight: the weights of the points sum to zero",
         [] {
           std::vector<Fp<5>> all;
           for (u64 x = 0; x < 5; ++x) all.push_back(Fp<5>::from(x));
           (void)Weight<Fp<5>>::domain(all, 2);  // five points in the field of five elements
         },
         [] { (void)Weight<Fp<5>>::domain({Fp<5>::from(0), Fp<5>::from(1)}, 2); });
    stop("Weight::apply: the polynomial has too many coefficients", [&] { (void)eq.apply(std::vector<K>(4, K::one())); },
         [&] { (void)eq.apply(std::vector<K>(3, K::one())); });
    stop("Weight::complete: the polynomial must have d coefficients", [&] { std::vector<K> g(2); eq.complete(g, K::one()); },
         [&] { std::vector<K> g(3); eq.complete(g, K::one()); });
    stop("hp_entries: a variable that no constructor of Weight has made", [] { (void)hp_entries(std::vector<Weight<K>>(1)); }, [&] { (void)hp_entries(two); });
    stop("hp_place: no such coefficient", [&] { (void)hp_place(two, 0, 0); }, [&] { (void)hp_place(two, 1, 0); });  // only the last variable has a constant
    stop("hp_form: one challenge for each variable", [&] { (void)hp_form(two, {K::one()}); }, [&] { (void)hp_form(two, {K::one(), K::one()}); });
    const std::vector<K> entries(hp_entries(two), K::from(2));
    stop("HidingPolynomial: wrong number of coefficients", [&] { HidingPolynomial<K> u(two, std::vector<K>(entries.size() + 1)); },
         [&] { HidingPolynomial<K> u(two, entries); });
    auto bound = [&](int binds) {
      HidingPolynomial<K> u(two, entries);
      for (int i = 0; i < binds; ++i) u.bind(K::one());
      return u;
    };
    stop("HidingPolynomial::round: all variables are bound", [&] { (void)bound(2).round(); }, [&] { (void)bound(1).round(); });
    stop("HidingPolynomial::bind: all variables are bound", [&] { (void)bound(3); }, [&] { (void)bound(2); });
    stop("HidingPolynomial::value: bind all variables first", [&] { (void)bound(1).value(); }, [&] { (void)bound(2).value(); });
  }
  // ---- psc_rows.h (over the build's challenge field, `E`) ----
  {
    auto output = [] {
      PscOutput<E> out;
      MaskLedger l(SendAccess::make(0, 0, 1));
      out.point = {E::one()};
      out.row_masks.push_back({l.take(), E::one()});
      out.row_maps.push_back({0, E::one()});
      out.row_target = E::zero();
      return out;
    };
    const MapAt map = row_of_table(1, Part::MAIN, 2, 1);
    stop("psc_row_u: one MapAt for each kept map", [&] { PscOutput<E> o = output(); (void)psc_row_u(o, {}, "U"); },
         [&] { PscOutput<E> o = output(); (void)psc_row_u(o, {map}, "U"); });
    stop("psc_row_u: the row has been made already", [&] { PscOutput<E> o = output(); (void)psc_row_u(o, {map}, "U"); (void)psc_row_u(o, {map}, "U"); },
         [&] { PscOutput<E> o = output(); (void)psc_row_u(o, {map}, "U"); });
    stop("row_of_table: no such row", [] { (void)row_of_table(1, Part::MAIN, 2, 4); }, [] { (void)row_of_table(1, Part::MAIN, 2, 3); });
    stop("block_of_table: the point has more coordinates than the table has index bits",
         [] { Row r; block_of_table(1, Part::MAIN, 1, 0)(r, {E::one(), E::one()}, E::one()); },
         [] { Row r; block_of_table(1, Part::MAIN, 2, 0)(r, {E::one(), E::one()}, E::one()); });
  }
  if (failures)
    std::printf("psc_stops_test: FAILED (%d checks failed, see above)\n", failures);
  else
    std::printf("psc_stops_test: ok (%d stops, each ended its child with its own words, and each control ran to its end)\n", stops_seen);
  return failures ? 1 : 0;
}
