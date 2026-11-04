// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the hash chain (`Chain`: the Fiat-Shamir transcript, one hash query per round) and the
// typed channels over it (`ProverRounds`, `VerifierRounds`), which follow a plan (`Plan`: the list
// of all rounds). It tests the queries' format, misuse that must abort, and a toy protocol whose
// altered proofs must be rejected (`refuse`, `refused` in identifiers and output). The toy has
// every kind of chain round except a seal (a digest of a whole chain, on which a following chain
// builds), and opens a tree. It proves nothing, but every byte of its proof is checked by the toy
// verifier or hashed into the chain before a challenge that enters the last message, which the
// verifier compares.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets this test build plans by hand (blocks/numbers/plan.h)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "chain/closed.h"
#include "chain/rounds.h"
#include "chain/tests/death.h"
#include "numbers/round_names.h"
#include "tree/opening.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      failures++; \
    } \
  } while (0)

// ---- what the types do not let one write ----
// These lines fail to compile if a draw handle (`SeedDraw`, ..: returned by close_seed(), ..)
// gains another kind's call, which would let an out-of-domain sample's seed (a challenge that the
// prover must answer next) share a verifier message with positions, or with elements that could
// serve as a fold point.
template <class T, class = void> struct has_positions : std::false_type {};
template <class T> struct has_positions<T, std::void_t<decltype(std::declval<T &>().positions(0u))>> : std::true_type {};
template <class T, class = void> struct has_seed : std::false_type {};
template <class T> struct has_seed<T, std::void_t<decltype(std::declval<T &>().seed(1u, 1u))>> : std::true_type {};
template <class T, class = void> struct has_draw : std::false_type {};
template <class T> struct has_draw<T, std::void_t<decltype(std::declval<T &>().draw())>> : std::true_type {};
template <class T, class = void> struct has_word : std::false_type {};
template <class T> struct has_word<T, std::void_t<decltype(std::declval<T &>().word())>> : std::true_type {};
static_assert(has_seed<SeedDraw>::value && !has_positions<SeedDraw>::value && !has_draw<SeedDraw>::value && !has_word<SeedDraw>::value,
              "a round of seeds draws seeds only");
static_assert(has_positions<PositionsDraw>::value && !has_seed<PositionsDraw>::value && !has_draw<PositionsDraw>::value &&
                  !has_word<PositionsDraw>::value,
              "a round of positions draws no seed");
static_assert(has_draw<AlgebraicDraw>::value && !has_seed<AlgebraicDraw>::value && !has_positions<AlgebraicDraw>::value,
              "an algebraic round draws elements only");
static_assert(!has_word<BitsDraw>::value && !has_draw<BitsDraw>::value && !has_seed<BitsDraw>::value && !has_positions<BitsDraw>::value,
              "a round of bits draws matrix entries only: no raw words, from which anything could be made");
static_assert(!std::is_copy_constructible<ProverRounds>::value && !std::is_copy_constructible<VerifierRounds>::value,
              "a channel cannot be forked");
static_assert(!std::is_copy_constructible<SeedDraw>::value && !std::is_copy_constructible<PositionsDraw>::value &&
                  !std::is_copy_constructible<AlgebraicDraw>::value && !std::is_copy_constructible<BitsDraw>::value,
              "a handle cannot be copied");
static_assert(!std::is_move_constructible<SeedDraw>::value && !std::is_move_constructible<PositionsDraw>::value &&
                  !std::is_move_constructible<AlgebraicDraw>::value && !std::is_move_constructible<BitsDraw>::value,
              "a handle cannot be moved: the one moved from would stay usable");

struct TestCoins : Coins { // splitmix64 (Steele, Lea, Flood, 2014); for tests only
  u64 s;
  explicit TestCoins(u64 seed) : s(seed) {}
  u64 word() {
    u64 z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  void fill(F *out, size_t n) override {
    for (size_t i = 0; i < n; i++)
      while (!F_from_word(word(), out[i])) {
      }
  }
  void bytes(u8 *out, size_t n) override {
    for (size_t i = 0; i < n; i++) out[i] = (u8)word();
  }
};

// ---- the toy protocol ----
// What a round draws from the extension field E: `elements` in all, `nonzero` of them not 0.
static void set_draws(PlanRound &r, u64 elements, u64 nonzero) { r.draw_E = elements, r.draw_nonzero = nonzero; }

// The toy's public inputs that its verifier reads. They must all be `Public` handles. Nothing
// enforces this: a plain copy would reach the verifier without being hashed into the chain's first
// query (the hash input that fixes the public inputs).
struct ToyBound {
  Public<u64> log_n, log_n_zk;
  Public<std::vector<u64>> main_symbols, mask_symbols; // of the tree's parts: blocks/tree/opening.h
};

// The toy's public inputs. work_..: bits of proof of work in the rounds "positions", "bits" and
// "seed". Main rows have 2^log_n symbols, mask rows 2^log_n_zk. `label`: which protocol the chain
// runs.
struct Toy {
  unsigned work_positions = 9, work_bits_round = 5, work_seed = 0;
  unsigned log_n = 6, log_n_zk = 3;
  u64 draws = 7;
  u64 label = 1;
  std::vector<u8> statement = {1, 2, 3}, record = {9, 9};
  unsigned parameter = 0; // stands for any parameter: it enters the parameters' digest

  OracleTreeShape shape() const {
    OracleTreeShape s;
    s.log_n = log_n, s.log_n_zk = log_n_zk;
    s.main_symbols = {2, 1}, s.mask_symbols = {3, 0};
    s.salt_bytes = SALT_FLOOR_BYTES; // must equal the plan's `Setting::salt_bytes`
    return s;
  }
  Plan plan() const {
    Setting set = setting_goldilocks();
    set.set_work(ErrorKind::QUERY, Scope::REQUEST, work_positions);
    set.set_work(ErrorKind::SMALL, Scope::REQUEST, work_bits_round);
    set.set_work(ErrorKind::SAMPLE, Scope::REQUEST, work_seed);
    Plan p(set);
    p.send_root("tree");
    set_draws(p.close("seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 4, 1);
    p.send_answers("answers", 2);
    p.send_scalars("sum", 1);
    p.close("epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    p.send_scalars("a", 1);
    p.close_none("a silent verifier");
    p.send_scalars("b", 1);
    set_draws(p.close("algebraic", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING), 2, 2);
    p.close("bits", RoundKind::BITS, ErrorKind::SMALL, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_trits = 4100; // an empty prover message
    p.send_scalars("c", 1);
    PlanRound &q = p.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    q.draw_positions = draws, q.log_n = log_n, q.draw_E = 1;
    p.send_scalars("sum 2", 1);
    p.close("second epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    p.send_scalars("the last message", 1);
    p.close_none(R_LAST);
    return p;
  }
  // Writes the toy's public inputs into a chain's first query. The `Binder`
  // (blocks/chain/public.h), which collects the first query, has already written the plan, with
  // its bits of proof of work.
  ToyBound bind(Binder &b) const {
    Digest files{}; // arbitrary but for `parameter`: the toy has no public files
    files.w[0] = 0xC0F1, files.w[7] = (u32)parameter;
    b.configuration(files);
    const OracleTreeShape s = shape();
    ToyBound in;
    in.log_n = b.parameter("log_n", (u64)log_n), in.log_n_zk = b.parameter("log_n_zk", (u64)log_n_zk);
    in.main_symbols = b.parameter("main symbols", std::vector<u64>(s.main_symbols.begin(), s.main_symbols.end()));
    in.mask_symbols = b.parameter("mask symbols", std::vector<u64>(s.mask_symbols.begin(), s.mask_symbols.end()));
    b.parameter("any parameter", (u64)parameter);
    b.statement("statement", statement), b.statement("record", record);
    return in;
  }
};
static ToyBound toy_bind(Binder &b, const Toy &toy) { return toy.bind(b); }

// Accumulates every challenge (the matrix entries, `trits` in identifiers, by way of `factor`) and
// one value from each opened column into `sum`, the prover's last message.
struct Seen {
  E sum = E::zero(), factor = E::one();
  void add(const E &x) {
    factor = factor * (E::theta(1) + E::one());
    sum = sum + factor * x;
  }
  void add_u64(u64 x) { add(E::from(from_i64((i64)(x >> 8)))); }
};

// A `Binder` with an arbitrary configuration digest, the same in every chain of a test.
struct WorldBinder : Binder {
  WorldBinder(u64 label, const Plan &plan) : Binder(label, plan) {
    Digest files{};
    files.w[0] = 0xC0F1;
    configuration(files);
  }
};
struct ToyBinder : Binder {
  ToyBinder(const Toy &toy, const Plan &plan) : Binder(toy.label, plan) { toy.bind(*this); }
};
// A chain's salt (its "random bytes"), for tests where any value will do.
static Digest some_bytes() {
  Digest d{};
  d.w[0] = 0x5A17;
  return d;
}
// The first query of a `Binder` without a statement, built by hand.
[[maybe_unused]] static FirstQuery by_hand(u64 label, const Binder &b, const Digest &bytes) {
  FirstQuery q;
  q.label = label, q.parameters = b.parameters_digest(), q.salt = bytes;
  return q;
}

static std::vector<u8> toy_prove(const Toy &toy, u64 seed) {
  const Plan plan = toy.plan();
  TestCoins coins(seed);
  OracleTreeShape s = toy.shape();
  std::vector<F> main(s.main_width() << s.log_n), mask(s.mask_width() << s.log_n_zk);
  coins.fill(main.data(), main.size());
  coins.fill(mask.data(), mask.size());
  auto fetch = [&](u64 x, std::vector<F> &col) { // the main rows' symbols at x, then the mask rows' at its short position
    const u64 y = short_position(s.log_n, s.log_n_zk, x);
    col.assign(main.begin() + x * s.main_width(), main.begin() + (x + 1) * s.main_width());
    col.insert(col.end(), mask.begin() + y * s.mask_width(), mask.begin() + (y + 1) * s.mask_width());
  };
  ToyBinder binder(toy, plan);
  u8 salt[DIGEST_BYTES];
  coins.bytes(salt, DIGEST_BYTES);
  Digest bytes;
  digest_from_bytes(salt, bytes);
  // The run's value, which every leaf's hash starts from, needs no channel: the tree can be built
  // before one exists.
  s.run = run_value(toy.label, binder.parameters_digest(), bytes);
  s.parameters = binder.configuration_digest();
  OracleTree tree(s, coins);
  std::vector<F> col;
  for (u64 x = 0; x < ((u64)1 << s.log_n); x++) fetch(x, col), tree.set_leaf(x, col.data(), col.size());

  Writer proof;
  ProverRounds ch(binder, bytes, proof);
  CHECK(ch.run() == s.run && ch.configuration() == s.parameters);
  Seen seen;
  ch.send_root(tree.finish());
  const SampleSeed sample = ch.close_seed("seed").seed(3, 2);
  for (const E &z : sample.z) seen.add(z);
  seen.add(sample.beta);
  ch.send_answer(coins.e());
  ch.send_answer(coins.e());
  ch.send_hiding_sum(coins.e());
  seen.add(ch.close_epsilon("epsilon"));
  ch.send(coins.e());
  ch.close_empty("a silent verifier");
  ch.send(coins.e());
  {
    AlgebraicDraw d = ch.close_algebraic("algebraic");
    seen.add(d.draw_nonzero());
    seen.add(d.draw_nonzero());
  }
  {
    BitsDraw d = ch.close_bits("bits");
    int count[3] = {0, 0, 0};
    for (signed char t : d.trits(4100)) count[t + 1]++, seen.factor = seen.factor + E::from(from_i64(t));
    // A matrix entry is 0 with probability 1/2, 1 or -1 with 1/4 each. The bounds are more than
    // six standard deviations from the means.
    CHECK(count[1] > 1850 && count[1] < 2250 && count[0] > 850 && count[0] < 1200 && count[2] > 850 && count[2] < 1200);
  }
  ch.send(coins.e());
  PositionsDraw d = ch.close_positions("positions");
  const PositionList list = d.positions(s.log_n);
  for (u64 x : list.draws()) seen.add_u64((x + 1) << 8 | list.index_of(x) << 20);
  seen.add(d.batching());
  Opening o;
  tree.open(list, o, fetch);
  for (size_t i = 0; i < list.size(); i++) seen.add(E::from(o.main_column(i)[0] + o.mask_column(i)[0]));
  write_opening(ch.file_only(), s, list, o);
  ch.send_hiding_sum(coins.e());
  seen.add(ch.close_epsilon("second epsilon"));
  ch.send(seen.sum);
  ch.close_empty(R_LAST);
  ch.finish();
  return proof.bytes();
}

// Three ways to write a verifier function. Each must reject whatever is not an honest proof, and
// none may abort because of a proof's bytes.
enum class Style {
  AT_ONCE, // returns at the first failure
  RUN_ON,  // ignores failures and runs to its end
  HALF     // leaves `middle` at its first failure, as a sub-protocol would; the caller then runs on
};
// The toy verifier, called through run_verifier() (blocks/chain/closed.h), which gives it the
// channel and the `ToyBound`, nothing else of the `Toy`.
template <Style style> static bool toy_decide(VerifierRounds &ch, const ToyBound &in, Notes *) {
  OracleTreeShape s;
  s.log_n = (unsigned)ch.get(in.log_n), s.log_n_zk = (unsigned)ch.get(in.log_n_zk);
  s.main_symbols.assign(ch.get(in.main_symbols).begin(), ch.get(in.main_symbols).end());
  s.mask_symbols.assign(ch.get(in.mask_symbols).begin(), ch.get(in.mask_symbols).end());
  s.salt_bytes = ch.plan()->set.salt_bytes;
  s.run = ch.run(), s.parameters = ch.configuration(); // from the verifier's own channel
  Seen seen;
  SentRoot root;
  E x;
  auto done = [&] { // a verifier that returns early may call `finish` itself: that must do no harm
    (void)ch.finish();
    return false;
  };
#define STEP(call) \
  do { \
    if (!(call) && style == Style::AT_ONCE) return done(); \
  } while (0)
  STEP(ch.recv_root(root));
  const SampleSeed sample = ch.close_seed("seed").seed(3, 2);
  for (const E &z : sample.z) seen.add(z);
  seen.add(sample.beta);
  auto middle = [&]() -> bool {
#define PART(call) \
  do { \
    if (!(call) && style != Style::RUN_ON) return false; \
  } while (0)
    PART(ch.recv_answer(x));
    PART(ch.recv_answer(x));
    PART(ch.recv_hiding_sum(x));
    seen.add(ch.close_epsilon("epsilon"));
    PART(ch.recv(x));
    ch.close_empty("a silent verifier");
    PART(ch.recv(x));
    {
      AlgebraicDraw d = ch.close_algebraic("algebraic");
      seen.add(d.draw_nonzero());
      seen.add(d.draw_nonzero());
    }
    PART(ch.ok());
    BitsDraw d = ch.close_bits("bits");
    PART(ch.ok());
    for (signed char t : d.trits(4100)) seen.factor = seen.factor + E::from(from_i64(t));
    return true;
#undef PART
  };
  STEP(middle());
  STEP(ch.recv(x));
  PositionsDraw d = ch.close_positions("positions");
  const PositionList list = d.positions(s.log_n);
  for (u64 p : list.draws()) seen.add_u64((p + 1) << 8 | list.index_of(p) << 20);
  seen.add(d.batching());
  Opening o;
  STEP(receive_opening(ch, root, s, list, o));
  // A verifier that runs on still reads `o`, which always has its full size, with zeros where
  // nothing was read.
  for (size_t i = 0; i < list.size(); i++) seen.add(E::from(o.main_column(i)[0] + o.mask_column(i)[0]));
  STEP(ch.recv_hiding_sum(x));
  seen.add(ch.close_epsilon("second epsilon"));
  if (ch.recv(x) && x != seen.sum) ch.refuse("the last message is not what the verifier's messages ask for");
  ch.close_empty(R_LAST);
  return true; // `run_verifier` takes the verdict from its `VerifierSession`
#undef STEP
}
static bool toy_verify(const Toy &toy, const std::vector<u8> &bytes, const char **why = nullptr, Style style = Style::AT_ONCE) {
  const Plan plan = toy.plan();
  return run_verifier(toy.label, plan, toy, toy_bind,
                      style == Style::AT_ONCE  ? toy_decide<Style::AT_ONCE>
                      : style == Style::RUN_ON ? toy_decide<Style::RUN_ON>
                                               : toy_decide<Style::HALF>,
                      bytes.data(), bytes.size(), why);
}
// Rejected in all three styles.
static bool refused(const Toy &toy, const std::vector<u8> &bytes) {
  return !toy_verify(toy, bytes, nullptr, Style::AT_ONCE) && !toy_verify(toy, bytes, nullptr, Style::RUN_ON) &&
         !toy_verify(toy, bytes, nullptr, Style::HALF);
}

// ---- programming errors: the process must abort ----
// ends_with(text, body) (blocks/chain/tests/death.h): true if body, run in a child process,
// aborts and its stderr contains text.
[[maybe_unused]] static void test_orders() {
  const Toy toy;
  const Plan plan = toy.plan();
  const Digest bytes = some_bytes();
  const Digest root{};
  const E e = E::one();
  auto upto_seed = [&](ProverRounds &ch) {
    ch.send_root(root);
    return ch.close_seed("seed").seed(3, 2);
  };
  CHECK(runs_through([&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e), ch.send_answer(e), ch.send_hiding_sum(e);
    ch.close_epsilon("epsilon");
  }));
  CHECK(ends_with("begins with the answers", [&] { // something else before the answers
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send(e);
  }));
  CHECK(ends_with("begins with the answers", [&] { // a root before the answers
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e), ch.send_root(root);
  }));
  CHECK(ends_with("answers to a sample are owed", [&] { // a verifier message between a seed and its answers
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e);
    ch.close_algebraic("epsilon");
  }));
  CHECK(ends_with("no seed has asked for", [&] { // an answer that no seed asked for
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_answer(e);
  }));
  CHECK(ends_with("must be the last thing sent", [&] { // epsilon, but no hiding sum was sent
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e), ch.send_answer(e), ch.send(e);
    ch.close_epsilon("epsilon");
  }));
  CHECK(ends_with("must be the last thing sent", [&] { // the sum is not the last thing sent
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e), ch.send_answer(e), ch.send_hiding_sum(e), ch.send(e);
    ch.close_epsilon("epsilon");
  }));
  CHECK(ends_with("another kind of verifier message", [&] { // a round under another kind than planned
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root);
    ch.close_positions("seed");
  }));
  CHECK(ends_with("not the next planned one", [&] { // a round out of turn
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root);
    ch.close_algebraic("algebraic");
  }));
  CHECK(ends_with("has not the planned bytes", [&] { // a message of another size than planned
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root), ch.send(e);
    ch.close_seed("seed");
  }));
  CHECK(ends_with("this verifier message is over", [&] { // a draw after the prover's next send
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root);
    SeedDraw d = ch.close_seed("seed");
    d.seed(3, 1);
    ch.send_answer(e);
    d.seed(3, 1);
  }));
  auto upto_epsilon = [&](ProverRounds &ch) {
    upto_seed(ch);
    ch.send_answer(e), ch.send_answer(e), ch.send_hiding_sum(e);
    ch.close_epsilon("epsilon");
    ch.send(e), ch.close_empty("a silent verifier"), ch.send(e);
  };
  CHECK(ends_with("must be the last thing sent", [&] { // a sum from before the last close does not count
    Plan two(plan.set);
    two.send_scalars("sum", 1);
    two.close("first", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    two.close("second", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    Writer w;
    WorldBinder ch_binder(1, two);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_hiding_sum(e);
    ch.close_epsilon("first");
    ch.close_epsilon("second");
  }));
  CHECK(ends_with("more elements are drawn", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    AlgebraicDraw d = ch.close_algebraic("algebraic");
    d.draw(), d.draw(), d.draw();
  }));
  CHECK(ends_with("less was drawn", [&] { // an element that the plan counts on is not drawn
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    ch.close_algebraic("algebraic").draw();
    ch.close_bits("bits");
  }));
  CHECK(ends_with("have all the planned number of coordinates", [&] { // a seed with more coordinates than planned: the surplus could serve as challenges
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root);
    ch.close_seed("seed").seed(5, 2);
  }));
  CHECK(ends_with("this verifier message is over", [&] { // an old handle after the next close
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    AlgebraicDraw d = ch.close_algebraic("algebraic");
    d.draw_nonzero(), d.draw_nonzero();
    ch.close_bits("bits");
    d.draw();
  }));
  auto upto_positions = [&](ProverRounds &ch) {
    upto_epsilon(ch);
    {
      AlgebraicDraw d = ch.close_algebraic("algebraic");
      d.draw_nonzero(), d.draw_nonzero();
    }
    ch.close_bits("bits").trits(4100);
    ch.send(e);
  };
  CHECK(ends_with("are drawn once", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_positions(ch);
    PositionsDraw d = ch.close_positions("positions");
    d.positions(toy.log_n), d.positions(toy.log_n);
  }));
  CHECK(ends_with("comes after the positions", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_positions(ch);
    ch.close_positions("positions").batching();
  }));
  CHECK(ends_with("less was drawn", [&] { // a round of positions in which none is drawn
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_positions(ch);
    ch.close_positions("positions");
    ch.send_hiding_sum(e);
  }));
  CHECK(ends_with("has random bytes in its first query", [&] { // a tree's leaves start from the run's value
    ToyBinder bare(toy, plan);
    bare.no_random_bytes();
    Writer w;
    ProverRounds ch(bare, bytes, w);
    ch.send_root(root);
  }));
  CHECK(ends_with("has no run's value", [&] {
    ToyBinder bare(toy, plan);
    bare.no_random_bytes();
    Writer w;
    ProverRounds ch(bare, bytes, w);
    ch.run();
  }));
  {
    // A salt that both sides supply, by random_bytes_are(): the proof does not contain it, and the
    // two chains agree only if both supply the same value.
    Plan one(plan.set);
    one.send_root("tree");
    one.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    auto handed_in = [](Binder &b, u32 word) {
      Digest d{};
      d.w[3] = word;
      b.random_bytes_are(b.statement("a root that is fresh in every run", d));
    };
    WorldBinder handed(1, one);
    handed_in(handed, 77);
    Writer w;
    ProverRounds p(handed, bytes, w);
    p.send_root(root);
    const E there = p.close_algebraic("round").draw();
    p.finish();
    CHECK(w.size() == DIGEST_BYTES); // the root alone
    for (u32 other : {77u, 78u}) {
      WorldBinder mine(1, one);
      handed_in(mine, other);
      Reader r(w.data(), w.size());
      VerifierSession v_session;
      VerifierRounds v(mine, r, v_session);
      SentRoot d;
      CHECK(v.recv_root(d));
      CHECK((v.run() == p.run()) == (other == 77));
      CHECK((v.close_algebraic("round").draw() == there) == (other == 77));
      CHECK(v.finish());
    }
  }
  // A verifier message (matrix entries, positions, elements) must be drawn in full;
  // finish() checks this too.
  {
    Plan bits(plan.set), places(plan.set), last(plan.set);
    bits.close("bits", RoundKind::BITS, ErrorKind::SMALL, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_trits = 10;
    bits.send_scalars("x", 1), bits.close_none("end");
    {
      PlanRound &r = places.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
      r.draw_positions = 3, r.log_n = 4; // and no batching element
    }
    places.send_scalars("x", 1), places.close_none("end");
    last.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    CHECK(ends_with("less was drawn", [&] {
      Writer w;
      WorldBinder ch_binder(1, bits);
      ProverRounds ch(ch_binder, bytes, w);
      ch.close_bits("bits").trits(9);
      ch.send(e);
    }));
    CHECK(ends_with("less was drawn", [&] {
      Writer w;
      WorldBinder ch_binder(1, places);
      ProverRounds ch(ch_binder, bytes, w);
      ch.close_positions("positions");
      ch.send(e);
    }));
    CHECK(ends_with("less was drawn", [&] {
      Writer w;
      WorldBinder ch_binder(1, last);
      ProverRounds ch(ch_binder, bytes, w);
      ch.close_algebraic("round");
      ch.finish();
    }));
  }
  {
    // A round of kind EPSILON draws exactly one element: the challenge that follows the sum of a
    // hiding polynomial (the random polynomial added to a sumcheck's summand).
    Plan two(plan.set);
    two.send_scalars("sum", 1);
    two.close("epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 2;
    CHECK(ends_with("not planned with this element alone", [&] {
      Writer w;
      WorldBinder ch_binder(1, two);
      ProverRounds ch(ch_binder, bytes, w);
      ch.send_hiding_sum(e);
      ch.close_epsilon("epsilon");
    }));
  }
  CHECK(ends_with("has not the planned bytes, roots", [&] { // the right bytes, but an answer where the plan has a plain scalar
    Plan plain(plan.set);
    set_draws(plain.close("seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 2, 1);
    plain.send_scalars("no answer", 1);
    plain.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    Writer w;
    WorldBinder ch_binder(1, plain);
    ProverRounds ch(ch_binder, bytes, w);
    ch.close_seed("seed").seed(1, 1);
    ch.send_answer(e);
    ch.close_algebraic("round");
  }));
  CHECK(ends_with("a seed without an answer", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send_root(root);
    ch.close_seed("seed").seed(3, 0);
  }));
  CHECK(ends_with("a protocol's label is below 2^32", [&] { // the labels above belong to the records' seeds
    WorldBinder high(LABEL_RECORD, plan);
    Writer w;
    ProverRounds ch(high, bytes, w);
  }));
  CHECK(ends_with("a proof lists no position", [&] {
    Setting listing = plan.set;
    listing.position_bytes = 1;
    Plan listed(listing);
    Writer w;
    WorldBinder ch_binder(1, listed);
    ProverRounds ch(ch_binder, bytes, w);
  }));
  CHECK(ends_with("positions from another range", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_positions(ch);
    ch.close_positions("positions").positions(toy.log_n + 1);
  }));
  CHECK(ends_with("a nonce has 1 to 8 bytes", [&] {
    Setting odd = plan.set;
    odd.nonce_bytes = 9;
    Plan p9(odd);
    Writer w;
    WorldBinder ch_binder(1, p9);
    ProverRounds ch(ch_binder, bytes, w);
  }));
  CHECK(ends_with("a draw before the first round", [&] { Chain(FirstQuery()).draw_E(); }));
  CHECK(ends_with("a round without work has no nonce", [&] {
    Chain c{FirstQuery()};
    ChainAbsorb m = c.message();
    c.step(Chain::prefix(m, ChainKind::ALGEBRAIC, 0), 1, 0);
  }));
  CHECK(ends_with("made from no first query", [&] { Chain().message(); }));
  CHECK(ends_with("another number of rows", [&] {
    Plan rows(plan.set);
    rows.close("batch", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    rows.rounds_.back().rows = 5;
    Writer w;
    WorldBinder ch_binder(1, rows);
    ProverRounds ch(ch_binder, bytes, w);
    AlgebraicDraw d = ch.close_algebraic("batch");
    d.require_rows(5);
    d.require_rows(4);
  }));
  {
    // A plan entry (`PlanRound`) can state numbers from which its round's error was computed
    // (listed in `Kind`). The code that runs the round must ask about each with its require_..()
    // call, which aborts if the plan states another number or none. (No rows means 0 rows:
    // require_rows(0) passes.) If no call is made, the process aborts at the next send, close or
    // finish.
    PricedGroup group;
    group.main = PricedCode{10, 100, 20}, group.mask = PricedCode{8, 30, 20}, group.k = 3, group.h = 1, group.slots = 2, group.said = true;
    PricedTable table;
    table.code = PricedCode{12, 1000, 24}, table.v = 5, table.answers = 2, table.test = TableTest::ONE_TIME, table.said = true;
    enum Kind { FIRST_LIST, GROUP, CODES, TABLE, COMBINATIONS, ROWS, SHARING, KINDS };
    enum Way { ASKS, ASKS_ANOTHER, PLAN_SAYS_NOTHING, NEVER_ASKS, WAYS };
    const char *const stop[KINDS] = {"first list has another number of draws", "begins a stage on another group", "draws positions on rows of other codes", "draws columns of another table",
                                     "draws a matrix of another number of combinations", "batches another number of rows",
                                     "draws a point for another number of requests"};
    for (int kind = 0; kind < KINDS; kind++)
      for (int way = 0; way < WAYS; way++) {
        auto play = [&] {
          Plan said(plan.set);
          said.close("asked", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
          said.close("next", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
          PlanRound &r = said.rounds_[said.rounds_.size() - 2];
          if (way != PLAN_SAYS_NOTHING) {
            if (kind == FIRST_LIST) r.first_list = 7;
            if (kind == GROUP || kind == CODES) r.group = group;
            if (kind == TABLE) r.table = table;
            if (kind == COMBINATIONS) r.combinations = 33;
            if (kind == ROWS) r.rows = 5;
            if (kind == SHARING) r.sharing = 8;
          }
          Writer w;
          WorldBinder ch_binder(1, said);
          ProverRounds ch(ch_binder, bytes, w);
          AlgebraicDraw d = ch.close_algebraic("asked");
          (void)d.draw();
          if (way != NEVER_ASKS) {
            const bool other = way == ASKS_ANOTHER;
            PricedGroup g = group;
            PricedTable t = table;
            if (other) g.mask.r++, t.test = TableTest::LATER;
            if (kind == FIRST_LIST) ch.require_first_list(other ? 6 : 7);
            if (kind == GROUP) ch.require_group(g);
            if (kind == CODES) ch.require_codes(g);
            if (kind == TABLE) ch.require_table(t);
            if (kind == COMBINATIONS) ch.require_combinations(other ? 1 : 33);
            if (kind == ROWS) ch.require_rows(other ? 4 : 5);
            if (kind == SHARING) ch.require_sharing(other ? 4 : 8);
          }
          (void)ch.close_algebraic("next").draw();
        };
        if (way == ASKS) CHECK(runs_through(play));
        if (way == ASKS_ANOTHER || way == PLAN_SAYS_NOTHING) CHECK(ends_with(stop[kind], play));
        if (way == NEVER_ASKS) CHECK(ends_with("no block has asked for it", play));
      }
    CHECK(ends_with("at the round asked: said by the plan and asked by nobody: the table (require_table)", [&] {
      Plan said(plan.set);
      said.close("asked", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
      said.close("next", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
      said.rounds_[said.rounds_.size() - 2].table = table;
      Writer w;
      WorldBinder ch_binder(1, said);
      ProverRounds ch(ch_binder, bytes, w);
      (void)ch.close_algebraic("asked").draw();
      (void)ch.close_algebraic("next").draw();
    }));
    // A line printed before the abort message names the round and both numbers.
    CHECK(ends_with("at the round asked: the plan says 33 combinations; the run has 1 combinations", [&] {
      Plan said(plan.set);
      said.close("asked", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
      said.rounds_.back().combinations = 33;
      Writer w;
      WorldBinder ch_binder(1, said);
      ProverRounds ch(ch_binder, bytes, w);
      (void)ch.close_algebraic("asked").draw();
      ch.require_combinations(1);
    }));
    // A require_..() call counts for one round only: every close clears it, or the first round's
    // call would pass for the second's too. For each `Kind`, two rounds state the same thing: code
    // that asks only at the first aborts on leaving the second, in both channels.
    for (int kind = 0; kind < KINDS; kind++) {
      Plan twice(plan.set);
      for (const char *name : {"first", "second", "next"}) twice.close(name, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
      for (size_t i = 0; i < 2; i++) {
        PlanRound &r = twice.rounds_[i];
        if (kind == FIRST_LIST) r.first_list = 7;
        if (kind == GROUP || kind == CODES) r.group = group;
        if (kind == TABLE) r.table = table;
        if (kind == COMBINATIONS) r.combinations = 33;
        if (kind == ROWS) r.rows = 5;
        if (kind == SHARING) r.sharing = 8;
      }
      auto ask = [&](Rounds &ch) {
        if (kind == FIRST_LIST) ch.require_first_list(7);
        if (kind == GROUP) ch.require_group(group);
        if (kind == CODES) ch.require_codes(group);
        if (kind == TABLE) ch.require_table(table);
        if (kind == COMBINATIONS) ch.require_combinations(33);
        if (kind == ROWS) ch.require_rows(5);
        if (kind == SHARING) ch.require_sharing(8);
      };
      auto walk = [&](Rounds &ch, bool at_the_second_too) {
        (void)ch.close_algebraic("first").draw();
        ask(ch);
        (void)ch.close_algebraic("second").draw();
        if (at_the_second_too) ask(ch);
        (void)ch.close_algebraic("next").draw();
      };
      Writer honest;
      {
        WorldBinder b(1, twice);
        ProverRounds ch(b, bytes, honest);
        walk(ch, true);
        ch.finish();
      }
      for (bool at_the_second_too : {true, false}) {
        auto prover = [&] {
          Writer w;
          WorldBinder b(1, twice);
          ProverRounds ch(b, bytes, w);
          walk(ch, at_the_second_too);
          ch.finish();
        };
        auto verifier = [&] {
          Reader r(honest.data(), honest.size());
          WorldBinder b(1, twice);
          VerifierSession session;
          VerifierRounds ch(b, r, session);
          walk(ch, at_the_second_too);
          if (!ch.finish()) std::abort(); // honest proof: a rejection is not the fault under test
        };
        if (at_the_second_too) {
          CHECK(runs_through(prover));
          CHECK(runs_through(verifier));
        } else {
          CHECK(ends_with("at the round second: said by the plan and asked by nobody", prover));
          CHECK(ends_with("at the round second: said by the plan and asked by nobody", verifier));
        }
      }
    }
  }
  {
    // Every prover message must be closed by its planned round, even one whose verifier message is
    // empty; such a round accepts no root.
    CHECK(ends_with("is not the next planned one", [&] { // the planned empty round is not closed
      Writer w;
      ToyBinder ch_binder(toy, plan);
      ProverRounds ch(ch_binder, bytes, w);
      upto_seed(ch);
      ch.send_answer(e), ch.send_answer(e), ch.send_hiding_sum(e);
      ch.close_epsilon("epsilon");
      ch.send(e), ch.send(e);
      ch.close_algebraic("algebraic");
    }));
    Setting asks = plan.set;
    Plan rooted(asks), open_end(asks);
    rooted.send_scalars("x", 1);
    rooted.close_none("silent");
    rooted.send_root("tree");
    rooted.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    open_end.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    CHECK(ends_with("a root stands in the query of the next", [&] {
      Writer w;
      WorldBinder b(1, rooted);
      ProverRounds ch(b, bytes, w);
      ch.send_root(root);
      ch.close_empty("silent");
    }));
    CHECK(ends_with("still open at the chain's end", [&] {
      Writer w;
      WorldBinder b(1, open_end);
      ProverRounds ch(b, bytes, w);
      ch.close_algebraic("round").draw();
      ch.send(e);
      ch.finish();
    }));
  }
  CHECK(ends_with("all drawn with 0 or, in a round of elements alone, all without", [&] { // a plan that does not say which
    Plan mixed(plan.set);
    set_draws(mixed.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING), 2, 1);
    Writer w;
    WorldBinder b(1, mixed);
    ProverRounds ch(b, bytes, w);
    ch.close_algebraic("round");
  }));
  CHECK(ends_with("have all the planned number of coordinates", [&] { // two seeds of 1 and 3 coordinates where two of 2 are planned
    Plan two(plan.set);
    set_draws(two.close("seeds", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 6, 2);
    Writer w;
    WorldBinder b(1, two);
    ProverRounds ch(b, bytes, w);
    ch.close_seed("seeds").seed(1, 1);
  }));
  {
    // Two chains in one file: the first ends inside it.
    Plan one(plan.set);
    one.send_scalars("x", 1);
    one.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    Writer w;
    for (u64 label : {1, 2}) {
      WorldBinder b(label, one);
      ProverRounds ch(b, bytes, w);
      ch.send(e);
      ch.close_algebraic("round").draw();
      if (label == 1) ch.finish_inside_a_file();
      else ch.finish();
    }
    Reader r(w.data(), w.size());
    E x;
    WorldBinder b1(1, one), b2(2, one);
    {
      VerifierSession ch_session;
      VerifierRounds ch(b1, r, ch_session);
      CHECK(ch.recv(x));
      ch.close_algebraic("round").draw();
      CHECK(ch.finish_inside_a_file());
    }
    {
      VerifierSession ch_session;
      VerifierRounds ch(b2, r, ch_session);
      CHECK(ch.recv(x));
      ch.close_algebraic("round").draw();
      CHECK(ch.finish());
    }
    Reader again(w.data(), w.size());
    WorldBinder b3(1, one);
    VerifierSession whole_session;
    VerifierRounds whole(b3, again, whole_session);
    CHECK(whole.recv(x));
    whole.close_algebraic("round").draw();
    CHECK(!whole.finish()); // `finish` also requires that every byte was read
    CHECK(ends_with("before the last planned round", [&] { // all planned rounds are still required
      Reader r2(w.data(), w.size());
      WorldBinder b4(1, one);
      VerifierSession ch_session;
      VerifierRounds ch(b4, r2, ch_session);
      (void)ch.finish_inside_a_file();
    }));
  }
  CHECK(ends_with("says nothing has no work", [&] { // a plan that was not made by close_none()
    Plan worked(plan.set);
    PlanRound silent;
    silent.name = "silent", silent.kind = RoundKind::EMPTY, silent.message_bytes = E_BYTES, silent.work_bits = 3;
    worked.enter(silent);
    Writer w;
    WorldBinder b(1, worked);
    ProverRounds ch(b, bytes, w);
    ch.send(e);
    ch.close_empty("silent");
  }));
  CHECK(ends_with("a label is below 2^48", [&] {
    FirstQuery high;
    high.label = (u64)1 << 48;
    Chain c(high);
  }));
  CHECK(ends_with("not of a seal", [&] {
    Plan sealed(plan.set);
    sealed.send_root("tree");
    sealed.close("seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
    Writer w;
    WorldBinder b(1, sealed);
    ProverRounds ch(b, bytes, w);
    ch.send_root(root);
    Sealed out;
    (void)ch.seal("seal", out);
  }));
  CHECK(ends_with("drawn without 0 than the round is planned with", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    {
      AlgebraicDraw d = ch.close_algebraic("algebraic");
      d.draw(), d.draw();
    }
    ch.close_bits("bits");
  }));
  CHECK(ends_with("are drawn in one call", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    {
      AlgebraicDraw d = ch.close_algebraic("algebraic");
      d.draw_nonzero(), d.draw_nonzero();
    }
    BitsDraw b = ch.close_bits("bits");
    b.trits(10), b.trits(10);
  }));
  CHECK(ends_with("more entries are drawn", [&] {
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_epsilon(ch);
    {
      AlgebraicDraw d = ch.close_algebraic("algebraic");
      d.draw_nonzero(), d.draw_nonzero();
    }
    ch.close_bits("bits").trits(4101);
  }));
  CHECK(ends_with("has not the planned bytes, roots", [&] { // the right number of bytes, but no root among them
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    ch.send(e), ch.send(F::one());
    ch.close_seed("seed");
  }));
  CHECK(ends_with("while answers to a sample are owed", [&] { // the protocol ends between a seed and its answers
    Plan only(plan.set);
    only.send_root("tree");
    set_draws(only.close("seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 4, 1);
    Writer w;
    WorldBinder ch_binder(1, only);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e);
    ch.finish();
  }));
  // The verifier's channel has the same rules until it has rejected the proof.
  {
    const std::vector<u8> honest = toy_prove(toy, 1);
    CHECK(ends_with("begins with the answers", [&] {
      Reader r(honest.data(), honest.size());
      ToyBinder ch_binder(toy, plan);
      VerifierSession ch_session;
      VerifierRounds ch(ch_binder, r, ch_session);
      SentRoot d;
      E x;
      (void)ch.recv_root(d);
      ch.close_seed("seed").seed(3, 2);
      (void)ch.recv(x);
    }));
    CHECK(ends_with("before the last planned round", [&] {
      Reader r(honest.data(), honest.size());
      ToyBinder ch_binder(toy, plan);
      VerifierSession ch_session;
      VerifierRounds ch(ch_binder, r, ch_session);
      SentRoot d;
      (void)ch.recv_root(d);
      (void)ch.finish();
    }));
    // After a rejection the channel enforces neither order nor plan: calls return fixed values.
    CHECK(runs_through([&] {
      Reader r(honest.data(), 40); // the salt and a piece of the root
      ToyBinder ch_binder(toy, plan);
      VerifierSession ch_session;
      VerifierRounds ch(ch_binder, r, ch_session);
      SentRoot d;
      E x;
      if (ch.recv_root(d)) _exit(1);
      if (ch.recv(x) || ch.recv_answer(x)) _exit(1);
      const PositionList fixed = ch.close_positions("no such round").positions(7);
      if (fixed.log_n() != 7 || fixed.draws() != std::vector<u64>{0}) _exit(1);
      if (ch.close_epsilon("epsilon") != E::zero()) _exit(1);
      if (ch.close_algebraic("algebraic").draw_nonzero() != E::one()) _exit(1);
      Sealed sealed;
      if (ch.seal("seal", sealed) || sealed.digest() != Digest{}) _exit(1);
      if (ch.recv(x)) _exit(1);
      if (ch.finish()) _exit(1);
    }));
  }
  CHECK(ends_with("before the last planned round", [&] { // the protocol ends early
    Writer w;
    ToyBinder ch_binder(toy, plan);
    ProverRounds ch(ch_binder, bytes, w);
    upto_seed(ch);
    ch.send_answer(e), ch.send_answer(e);
    ch.finish();
  }));
}

// The hash chain without a channel.
[[maybe_unused]] static void test_chain() {
  FirstQuery q;
  q.statement = {1, 2}, q.record = {3};
  auto first_word = [](const FirstQuery &f) {
    Chain c(f);
    ChainAbsorb empty = c.message();
    c.close(empty, ChainKind::ALGEBRAIC);
    return c.next_word();
  };
  const u64 w0 = first_word(q);
  CHECK(w0 == first_word(q));
  // Each change below alters the first word drawn, even moving one byte from statement to record.
  FirstQuery o = q;
  o.label = 1;
  CHECK(first_word(o) != w0);
  o = q, o.parameters.w[0] = 1;
  CHECK(first_word(o) != w0);
  o = q, o.statement = {1}, o.record = {2, 3};
  CHECK(first_word(o) != w0);
  o = q, o.record.clear();
  CHECK(first_word(o) != w0);
  o = q, o.salt.w[7] = 1;
  CHECK(first_word(o) != w0);
  o = q, o.salted = false;
  CHECK(first_word(o) != w0);

  // Proof of work: search_nonce() finds the smallest valid nonce; a failed step() changes nothing.
  Chain c(q);
  const u8 message[16] = {5};
  // `ChainPrefix` (blocks/hash/hash.h): a round's query without its nonce.
  auto prefix = [&](const Chain &chain, size_t bytes, ChainKind kind, unsigned work_bits) {
    ChainAbsorb a = chain.message();
    a.push(message, bytes);
    return Chain::prefix(a, kind, work_bits);
  };
  const unsigned bits = 8;
  const ChainPrefix p = prefix(c, 16, ChainKind::POSITIONS, bits);
  const u64 nonce = search_nonce(p, bits);
  for (u64 n = 0; n < nonce; n++) CHECK(!c.step(p, n, bits));
  CHECK(c.rounds() == 0);
  CHECK(c.step(p, nonce, bits) && c.rounds() == 1);
  // The nonce is part of the query: another valid nonce gives another verifier message.
  Chain c2(q);
  const u64 nonce2 = search_nonce(p, bits, nonce + 1);
  CHECK(c2.step(p, nonce2, bits));
  CHECK(c.next_word() != c2.next_word());
  // Draws are in range; equal queries give equal draws.
  Chain a(q), b(q);
  CHECK(a.step(prefix(a, 8, ChainKind::ALGEBRAIC, 0), 0, 0) && b.step(prefix(b, 8, ChainKind::ALGEBRAIC, 0), 0, 0));
  for (int i = 0; i < 1000; i++) {
    const E x = a.draw_E(), y = b.draw_E();
    CHECK(x == y);
    for (const F &f : x.c) CHECK(f.v < P);
    CHECK(a.draw_position(5) == b.draw_position(5));
    CHECK(a.draw_position(0) == 0 && b.draw_position(0) == 0);
  }
  // A round's kind and its bits of proof of work are part of the query.
  Chain k1(q), k2(q), k3(q);
  CHECK(k1.step(prefix(k1, 8, ChainKind::ALGEBRAIC, 0), 0, 0) && k2.step(prefix(k2, 8, ChainKind::SEED, 0), 0, 0));
  CHECK(k1.value() != k2.value());
  CHECK(k3.step(prefix(k3, 8, ChainKind::ALGEBRAIC, 1), search_nonce(prefix(k3, 8, ChainKind::ALGEBRAIC, 1), 1), 1));
  CHECK(k1.value() != k3.value());
  // The answer to a round's query is 32-byte blocks: the test block (only with proof of work: the
  // low `work` bits of its first word are 0), the chain's new value, then the verifier message. No
  // 64-bit word repeats among its first 32.
  for (unsigned work : {0u, 3u}) {
    Chain s(q);
    const ChainPrefix p = prefix(s, 8, ChainKind::BITS, work);
    const u64 nonce = search_nonce(p, work);
    CHECK(s.step(p, nonce, work));
    std::vector<u64> words, all;
    for (unsigned i = 0; i < 20; i++) words.push_back(s.next_word());
    for (u64 j = 0; j < 4; j++)
      for (u64 w : chain_call(p, nonce, j).w) all.push_back(w);
    const size_t value_at = work ? 4 : 0;
    for (int i = 0; i < 4; i++) CHECK(all[value_at + i] == (s.value().w[2 * i] | (u64)s.value().w[2 * i + 1] << 32));
    for (size_t i = 0; i < words.size(); i++) CHECK(words[i] == all[value_at + 4 + i]);
    for (size_t i = 0; i < all.size(); i++)
      for (size_t j = i + 1; j < all.size(); j++) CHECK(all[i] != all[j]);
  }
#if BLK_COUNT_HASH_CALLS
  // A round that sends two or four elements of E and draws one costs one compression function call
  // (more if two of the first four words drawn are rejected as field elements).
  Chain one(q);
  const u64 before = hash_calls_counted();
  ChainAbsorb m = one.message();
  m.push(E::one()), m.push(E::theta(1));
  one.close(m, ChainKind::ALGEBRAIC);
  one.draw_E();
  CHECK(hash_calls_counted() - before == 1);
  std::printf("a round (2 E sent, 1 E drawn): %llu calls of the compression function; ", (unsigned long long)(hash_calls_counted() - before));
  {
    const u64 then = hash_calls_counted();
    ChainAbsorb four = one.message();
    for (int i = 0; i < 4; i++) four.push(E::theta(1));
    one.close(four, ChainKind::ALGEBRAIC);
    one.draw_E();
    CHECK(hash_calls_counted() - then == 1);
  }
#endif
}

// Rebuilds a chain's queries with the functions of blocks/hash/hash.h and compares with `Chain`.
[[maybe_unused]] static void test_format() {
  FirstQuery q;
  q.label = 77, q.parameters.w[2] = 5, q.statement = {1, 2, 3}, q.record = {4, 5}, q.salt.w[0] = 9;
  Hasher h(Tag(Domain::CHAIN_START), 77);
  h.push(q.parameters);
  h.push_u64(3), h.push(q.statement.data(), 3);
  h.push_u64(2), h.push(q.record.data(), 2);
  h.push_u64(1), h.push(node(Tag(Domain::RUN), 77, q.parameters, q.salt)); // the run's value, not the salt itself
  Digest rho = h.finish();
  Chain c(q);
  CHECK(c.value() == rho);
  {
    // Whether the chain's salt is written in the proof changes no query. (A chain with `salted`
    // false has no run's value and can send no tree.)
    FirstQuery handed = q;
    handed.salt_in_proof = false;
    CHECK(Chain(handed).value() == rho);
  }
  const E sent[2] = {E::theta(1), E::theta(2) + E::one()};
  // One round of each kind. A query's last word before the nonce is number + 256 * work.
  const struct {
    ChainKind kind;
    u64 number;
    unsigned work;
  } rounds[] = {{ChainKind::ALGEBRAIC, 1, 0}, {ChainKind::EPSILON, 2, 0}, {ChainKind::SEED, 3, 0}, {ChainKind::POSITIONS, 4, 6},
                {ChainKind::BITS, 5, 3},      {ChainKind::SEAL, 6, 0},    {ChainKind::EMPTY, 7, 0}};
  u64 round = 0;
  for (const auto &r : rounds) {
    round++;
    ChainAbsorb by_hand(rho, round, 77); // the chain's label is in every query
    for (const E &x : sent)
      for (const F &f : x.c) by_hand.push_u64(f.v);
    by_hand.push_u64(r.number + 256 * r.work);
    const ChainPrefix p = by_hand.prefix();
    const u64 nonce = search_nonce(p, r.work);
    // Answer blocks: halves of hash calls 0, 1, 2; with proof of work the test block comes first.
    std::vector<u64> row;
    for (u64 j = 0; j < 3; j++)
      for (u64 w : chain_call(p, nonce, j).w) row.push_back(w);
    if (r.work) {
      CHECK((row[0] & (((u64)1 << r.work) - 1)) == 0);
      row.erase(row.begin(), row.begin() + 4);
    }
    Digest next;
    for (int i = 0; i < 4; i++) next.w[2 * i] = (u32)row[i], next.w[2 * i + 1] = (u32)(row[i] >> 32);
    const Chain before = c;
    ChainAbsorb m = c.message();
    m.push(sent[0]), m.push(sent[1]);
    CHECK(c.step(Chain::prefix(m, r.kind, r.work), nonce, r.work));
    CHECK(c.value() == next && c.rounds() == round);
    if (r.work == 0) { // `Chain::round` (no proof of work; the message as an array) gives the same
      Chain fixed = before;
      const u64 words[6] = {sent[0].c[0].v, sent[0].c[1].v, sent[0].c[2].v, sent[1].c[0].v, sent[1].c[1].v, sent[1].c[2].v};
      fixed.round(words, r.kind);
      CHECK(fixed.value() == c.value() && fixed.rounds() == c.rounds() && fixed.draw_E() == Chain(c).draw_E());
    }
    for (size_t i = 4; i < 16; i++) CHECK(c.next_word() == row[i]);
    rho = next;
  }
  // record_seeds() (blocks/chain/rounds.h): the sample seeds of a committed table's record (its
  // stored sample answers) depend only on the bytes that name the table and on the two counts.
  {
    Digest r1{}, r2{};
    r2.w[0] = 1;
    auto record_seed = [](const Digest &name, unsigned variables) { // a name of 32 bytes, one seed
      u8 n[DIGEST_BYTES];
      digest_to_bytes(name, n);
      return record_seeds(n, DIGEST_BYTES, variables, 1)[0];
    };
    const SampleSeed a = record_seed(r1, 4), b = record_seed(r1, 4), other = record_seed(r2, 4);
    CHECK(a.z == b.z && a.beta == b.beta && a.z.size() == 4 && a.beta != E::zero());
    CHECK(a.z != other.z);
    const SampleSeed shorter = record_seed(r1, 2);
    CHECK(shorter.z[0] == a.z[0] && shorter.z[1] == a.z[1]);
    u8 bytes[DIGEST_BYTES];
    digest_to_bytes(r1, bytes);
    const std::vector<SampleSeed> two = record_seeds(bytes, DIGEST_BYTES, 4, 2);
    CHECK(two.size() == 2 && two[0].z == a.z && two[0].beta == a.beta && two[1].z != a.z);
    CHECK(record_seeds(bytes, DIGEST_BYTES - 1, 4, 1)[0].z != a.z);
  }
  // A channel draws what a bare `Chain` draws; its proof is the salt, then the sent items in order.
  Plan plan(setting_goldilocks());
  plan.send_root("root");
  plan.send_scalars("x", 1);
  plan.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  Writer w;
  WorldBinder binder(77, plan);
  ProverRounds ch(binder, q.salt, w);
  Digest root{};
  root.w[5] = 6;
  ch.send_root(root), ch.send(sent[1]);
  const E drawn = ch.close_algebraic("round").draw();
  ch.finish();
  Chain again(by_hand(77, binder, q.salt));
  ChainAbsorb m = again.message();
  m.push(root), m.push(sent[1]);
  again.close(m, ChainKind::ALGEBRAIC);
  CHECK(again.draw_E() == drawn);
  Writer expected;
  expected.put(q.salt), expected.put(root), expected.put(sent[1]);
  CHECK(w.bytes() == expected.bytes());
}

// Repeats a channel's SEED, EPSILON, BITS, POSITIONS and SEAL rounds on a bare `Chain`, to check
// the kind in each query and how each verifier message is cut from the chain's words.
// Prover and verifier share the channel's code, so they would agree on a wrong format too.
[[maybe_unused]] static void test_channel_format() {
  Setting set = setting_goldilocks();
  Plan plan(set);
  plan.send_scalars("x", 1);
  set_draws(plan.close("seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 3, 1);
  plan.send_answers("y", 1);
  plan.send_scalars("sum", 1);
  plan.close("epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  plan.close("bits", RoundKind::BITS, ErrorKind::SMALL, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_trits = 40;
  plan.send_scalars("x", 1);
  {
    PlanRound &r = plan.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    r.draw_positions = 5, r.log_n = 9, r.draw_E = 1;
  }
  plan.send_scalars("x", 1);
  plan.close("seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
  WorldBinder binder(5, plan);
  const FirstQuery q = by_hand(5, binder, some_bytes());
  const E x = E::theta(1), y = E::theta(2), sum = E::one() + E::theta(1);
  Writer w;
  ProverRounds ch(binder, q.salt, w);
  Chain c(q);
  auto round = [&](std::initializer_list<E> sent, ChainKind kind) {
    ChainAbsorb m = c.message();
    for (const E &e : sent) m.push(e);
    c.close(m, kind);
  };

  ch.send(x);
  const SampleSeed seed = ch.close_seed("seed").seed(2, 1);
  round({x}, ChainKind::SEED);
  CHECK(seed.z.size() == 2 && seed.z[0] == c.draw_E() && seed.z[1] == c.draw_E()); // the point first,
  CHECK(seed.beta == c.draw_E_nonzero());                                          // then the factor

  ch.send_answer(y), ch.send_hiding_sum(sum);
  const E eps = ch.close_epsilon("epsilon");
  round({y, sum}, ChainKind::EPSILON);
  CHECK(eps == c.draw_E());

  const std::vector<signed char> entries = ch.close_bits("bits").trits(40);
  round({}, ChainKind::BITS);
  for (int i = 0; i < 40;) {
    const u64 word = c.next_word(); // 32 entries a word, two bits each, the low bits first
    for (int k = 0; k < 32 && i < 40; k++, i++) {
      const unsigned two = (unsigned)(word >> (2 * k)) & 3; // bit 0: not 0; bit 1: the sign
      CHECK(entries[i] == (two == 1 ? 1 : two == 3 ? -1 : 0));
    }
  }

  ch.send(x);
  {
    PositionsDraw d = ch.close_positions("positions");
    const PositionList list = d.positions(9);
    round({x}, ChainKind::POSITIONS);
    CHECK(list.draws().size() == 5);
    for (u64 p : list.draws()) CHECK(p == (c.next_word() & 511)); // a word a position, its low bits
    CHECK(d.batching() == c.draw_E());                            // and no word between
  }

  ch.send(x);
  Sealed the_end;
  CHECK(ch.seal("seal", the_end));
  const Digest sealed = the_end.digest();
  round({x}, ChainKind::SEAL);
  for (int i = 0; i < 4; i++) {
    const u64 word = c.next_word();
    CHECK(sealed.w[2 * i] == (u32)word && sealed.w[2 * i + 1] == (u32)(word >> 32));
  }
  ch.finish();

  // record_seeds() by hand: an unsalted chain of its own on the table's name, with one SEED round.
  const u8 name[3] = {7, 8, 9};
  FirstQuery own;
  own.label = LABEL_RECORD, own.salted = false, own.statement = {7, 8, 9};
  Chain r(own);
  r.round(ChainKind::SEED);
  const std::vector<SampleSeed> seeds = record_seeds(name, 3, 1, 2);
  CHECK(seeds.size() == 2 && seeds[0].z[0] == r.draw_E() && seeds[0].beta == r.draw_E_nonzero());
  CHECK(seeds[1].z[0] == r.draw_E() && seeds[1].beta == r.draw_E_nonzero());
}

// A channel started from a `Binder`: the public inputs are in the chain's first query, and only a
// channel of that chain, or of a chain that follows it, can read them.
[[maybe_unused]] static void test_binder() {
  Plan plan(setting_goldilocks());
  plan.send_scalars("x", 1);
  plan.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  plan.close("seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
  Digest bytes{};
  bytes.w[0] = 11;
  struct Run {
    E drawn;
    std::vector<u8> proof;
  };
  auto prove = [&](u64 table, u64 target) {
    WorldBinder b(3, plan);
    const Public<u64> t = b.parameter("table", table), y = b.statement("target", target);
    Writer w;
    ProverRounds ch(b, bytes, w);
    CHECK(ch.get(t) == table && ch.get(y) == target);
    ch.send(E::one());
    Run r;
    r.drawn = ch.close_algebraic("round").draw();
    Sealed end;
    CHECK(ch.seal("seal", end));
    ch.finish();
    r.proof = w.bytes();
    return r;
  };
  const Run honest = prove(5, 6);
  CHECK(prove(5, 6).drawn == honest.drawn);
  CHECK(prove(4, 6).drawn != honest.drawn && prove(5, 7).drawn != honest.drawn); // either input changes the challenge
  for (u64 table : {5, 4}) {
    WorldBinder b(3, plan);
    const Public<u64> t = b.parameter("table", table), y = b.statement("target", (u64)6);
    Reader r(honest.proof.data(), honest.proof.size());
    VerifierSession ch_session;
    VerifierRounds ch(b, r, ch_session);
    E x;
    CHECK(ch.get(t) == table && ch.get(y) == 6 && ch.recv(x));
    CHECK((ch.close_algebraic("round").draw() == honest.drawn) == (table == 5));
  }
  CHECK(ends_with("does not stand in this chain's first query", [&] { // an input of another chain
    WorldBinder b(3, plan), other(3, plan);
    const Public<u64> elsewhere = other.parameter("table", (u64)5);
    Writer w;
    ProverRounds ch(b, bytes, w);
    ch.get(elsewhere);
  }));
  CHECK(ends_with("does not stand in this chain's first query", [&] { // an input that no Binder has made
    WorldBinder b(3, plan);
    Writer w;
    ProverRounds ch(b, bytes, w);
    ch.get(Public<u64>());
  }));
  CHECK(ends_with("takes nothing more", [&] { // an input that comes after the chain has begun
    WorldBinder b(3, plan);
    Writer w;
    ProverRounds ch(b, bytes, w);
    b.statement("late", (u64)1);
  }));
  CHECK(runs_through([&] { // a chain that follows another, by its seal, reads that chain's inputs
    WorldBinder first(3, plan);
    const Public<u64> t = first.parameter("table", (u64)5);
    Writer w;
    ProverRounds one(first, bytes, w);
    one.send(E::one());
    one.close_algebraic("round").draw();
    Sealed end;
    if (!one.seal("seal", end)) _exit(1);
    WorldBinder second(4, plan);
    second.follows(end);
    ProverRounds two(second, bytes, w);
    if (two.get(t) != 5) _exit(1);
  }));
  {
    // Two chains of one session (`VerifierSession`: one verdict on all its chains), in two files.
    // With `spoiled`, a cheating prover truncates the first file and builds the second chain on the
    // all-zero seal that a verifier gets after a rejection, which the prover knows beforehand. The
    // second verifier channel then starts rejected, whatever the caller did with the first chain's
    // result.
    WorldBinder p1(3, plan);
    Writer w1, w2;
    ProverRounds one(p1, bytes, w1);
    one.send(E::one());
    one.close_algebraic("round").draw();
    Sealed honest_end;
    CHECK(one.seal("seal", honest_end));
    one.finish();
    for (bool spoiled : {false, true}) {
      std::vector<u8> first = w1.bytes();
      if (spoiled) first.resize(first.size() - 1);
      WorldBinder v1(3, plan);
      Reader r1(first.data(), first.size());
      VerifierSession a_session;
      VerifierRounds a(v1, r1, a_session);
      E x;
      (void)a.recv(x);
      a.close_algebraic("round").draw();
      Sealed end;
      CHECK(a.seal("seal", end) == !spoiled);
      CHECK(a.finish() == !spoiled); // to accept, call `finish` on every chain, earlier ones first
      CHECK((end.digest() == Digest{}) == spoiled && (end.digest() == honest_end.digest()) == !spoiled);
      WorldBinder p2(4, plan);
      p2.follows(end);
      Writer second;
      ProverRounds two(p2, bytes, second);
      two.send(E::one());
      two.close_algebraic("round").draw();
      Sealed unused;
      CHECK(two.seal("seal", unused));
      two.finish();
      WorldBinder v2(4, plan);
      v2.follows(end);
      Reader r2(second.data(), second.size());
      VerifierRounds b(v2, r2, a_session); // it follows a: one session
      CHECK(b.stands_on(a) && !a.stands_on(b) && !b.stands_on(b));
      CHECK(b.ok() == !spoiled);
      (void)b.recv(x);
      b.close_algebraic("round").draw();
      (void)b.seal("seal", unused);
      CHECK(b.finish() == !spoiled);
      if (spoiled) CHECK(std::string(b.why()).find("an earlier chain") != std::string::npos);
    }
  }
}

// receive_opening() (blocks/tree/opening.h) checks its caller's shape, root and list against the
// channel that reads the opening, and aborts on a mismatch: that is the caller's bug. Also tests
// the session's records.
[[maybe_unused]] static void test_opening_held() {
  Plan plan(setting_goldilocks());
  plan.send_root("tree");
  {
    PlanRound &r = plan.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    r.draw_positions = 4, r.log_n = 3;
  }
  TestCoins coins(11);
  Writer w;
  WorldBinder prover_binder(1, plan);
  ProverRounds p(prover_binder, some_bytes(), w);
  OracleTreeShape s;
  s.log_n = 3, s.log_n_zk = 1, s.main_symbols = {2}, s.mask_symbols = {1}, s.salt_bytes = plan.set.salt_bytes;
  s.run = p.run(), s.parameters = p.configuration();
  auto column = [&](u64 x, std::vector<F> &out) { out = {F{100 + x}, F{200 + x}, F{300 + short_position(3, 1, x)}}; };
  OracleTree tree(s, coins);
  std::vector<F> col;
  for (u64 x = 0; x < 8; x++) column(x, col), tree.set_leaf(x, col.data(), col.size());
  p.send_root(tree.finish());
  const PositionList provers = p.close_positions("positions").positions(3);
  Opening sent;
  tree.open(provers, sent, column);
  write_opening(p.file_only(), s, provers, sent);
  p.finish();
  // drew() goes by a list's origin, not its values: an equal list made by from_draws() fails.
  CHECK(p.drew(provers) && p.drew(provers.part(1, 2)) && p.drew(provers.short_list(1)));
  CHECK(!p.drew(PositionList::from_draws(provers.draws(), 3)) && !p.drew(PositionList()));
  CHECK(provers.part(1, 2).draws() == std::vector<u64>(provers.draws().begin() + 1, provers.draws().begin() + 3));
  CHECK(provers.short_list(1).all() == provers.short_positions(1) && provers.short_list(1).log_n() == 1);
  // what: 0 honest; 1 to 4 one mismatch each; 5 the whole list taken by part()
  auto verify = [&](int what) {
    Reader r(w.data(), w.size());
    WorldBinder verifier_binder(1, plan);
    VerifierSession v_session;
    VerifierRounds v(verifier_binder, r, v_session);
    SentRoot root;
    bool good = v.recv_root(root);
    const PositionList drawn = v.close_positions("positions").positions(3);
    OracleTreeShape own = s;
    own.run = v.run(), own.parameters = v.configuration();
    if (what == 1) own.salt_bytes += 8;
    if (what == 2) own.parameters.w[0] ^= 1;
    const PositionList list = what == 3   ? PositionList::from_draws(drawn.draws(), 3) // the same positions, drawn by no channel
                              : what == 4 ? provers                                    // the same positions, drawn by another channel
                              : what == 5 ? drawn.part(0, drawn.draws().size())
                                          : drawn;
    Opening o;
    good = good && receive_opening(v, root, own, list, o) && v.finish();
    return good;
  };
  CHECK(verify(0) && verify(5));
  {
    // A session keeps one record of vertex values (`TreeValues`, blocks/tree/tree.h) for each root.
    // Two chains of one session read the same tree from the same file. With `planted`, a wrong
    // value is put at both vertices of depth 1 before the second reading, as if by an earlier
    // chain, and the second chain must reject: this catches a verifier that forgets the values
    // between chains.
    auto two_chains = [&](bool planted, bool two_sessions, bool second_ended) {
      VerifierSession session, other;
      bool good = true;
      SentRoot root;
      for (int k = 0; k < 2; k++) {
        Reader r(w.data(), w.size());
        WorldBinder b(1, plan);
        VerifierRounds v(b, r, k && two_sessions ? other : session);
        good = v.recv_root(root) && good;
        const PositionList drawn = v.close_positions("positions").positions(3);
        OracleTreeShape own = s;
        own.run = v.run(), own.parameters = v.configuration();
        if (k && planted) {
          Digest bent{};
          bent.w[0] = 1;
          session.record(root.digest()).put(1, 0, bent), session.record(root.digest()).put(1, 1, bent); // one of the two lies on every path
        }
        Opening o;
        const bool read = receive_opening(v, root, own, drawn, o);
        if (k && planted && !two_sessions) CHECK(!read && std::string(v.why()).find("two values under one root") != std::string::npos);
        good = read && good;
        if (k == 0 || second_ended) good = v.finish() && good;
      }
      CHECK(session.roots() == 1 && session.chains() == (two_sessions ? 1u : 2u));
      const bool verdict = session.accepted() && (!two_sessions || other.accepted()); // of both chains, in one session or in two
      CHECK(verdict == good);
      return verdict;
    };
    CHECK(two_chains(false, false, true));
    CHECK(!two_chains(true, false, true));
    CHECK(two_chains(true, true, true)); // control: the other session has no injected value
    CHECK(ends_with("has not been brought to its end", [&] { (void)two_chains(false, false, false); }));
    CHECK(ends_with("a verdict on no chain", [&] { (void)VerifierSession().accepted(); }));
    CHECK(runs_through([&] {
      // After a rejection accepted() returns false even if a chain is unfinished: a verifier may
      // stop at the first rejection. way 0: the channel has rejected. way 1: a read through
      // file_only(), the proof's `Reader` itself, has failed without telling the channel. way 2: as
      // way 1, but the verdict is asked only after the channel is destroyed.
      for (int way = 0; way < 3; way++) {
        VerifierSession session;
        std::vector<u8> cut(w.bytes().begin(), w.bytes().begin() + (way == 0 ? 40 : 64)); // the random bytes, and a part or all of the root
        Reader r(cut.data(), cut.size());
        WorldBinder b(1, plan);
        {
          VerifierRounds v(b, r, session);
          SentRoot root;
          if (v.recv_root(root) != (way != 0)) _exit(1);
          if (way != 0 && v.file_only().take(1)) _exit(1); // nothing is left: the reader fails, the channel is not told
          if (way != 2 && session.accepted()) _exit(1);
        }
        if (session.accepted()) _exit(1);
      }
    }));
    {
      // A rejected opening leaves nothing in the session's record: after an altered file, the
      // honest file's opening still passes in the same session (whose verdict stays no).
      VerifierSession session;
      bool answers[2];
      for (int k = 0; k < 2; k++) {
        std::vector<u8> file = w.bytes();
        if (k == 0) file[file.size() - 1] ^= 1; // the last given digest
        Reader r(file.data(), file.size());
        WorldBinder b(1, plan);
        VerifierRounds v(b, r, session);
        SentRoot root;
        (void)v.recv_root(root);
        const PositionList drawn = v.close_positions("positions").positions(3);
        OracleTreeShape own = s;
        own.run = v.run(), own.parameters = v.configuration();
        Opening o;
        answers[k] = receive_opening(v, root, own, drawn, o);
        // Rejected at the root, after the path check has entered its values; none of them remains.
        if (k == 0) CHECK(v.why() && std::string(v.why()).find("does not lead to its root") != std::string::npos && session.record(root.digest()).entered() > 0 && session.record(root.digest()).vertices() == 0);
        (void)v.finish();
      }
      CHECK(!answers[0] && answers[1] && !session.accepted()); // one of its chains was rejected
    }
  }
  CHECK(ends_with("neither this chain nor one that it follows has received", [&] { // a root that no chain has received
    Reader r(w.data(), w.size());
    WorldBinder b(1, plan);
    VerifierSession session;
    VerifierRounds v(b, r, session);
    SentRoot got;
    (void)v.recv_root(got);
    const PositionList drawn = v.close_positions("positions").positions(3);
    OracleTreeShape own = s;
    own.run = v.run(), own.parameters = v.configuration();
    Opening o;
    (void)receive_opening(v, SentRoot(), own, drawn, o);
  }));
  CHECK(ends_with("neither this chain nor one that it follows has received", [&] { // a root that another chain has received, with the same bytes
    VerifierSession session;
    Reader r1(w.data(), w.size()), r2(w.data(), w.size());
    WorldBinder b1(1, plan), b2(1, plan);
    VerifierRounds other(b1, r1, session), v(b2, r2, session);
    SentRoot theirs, mine;
    (void)other.recv_root(theirs), (void)v.recv_root(mine);
    const PositionList drawn = v.close_positions("positions").positions(3);
    OracleTreeShape own = s;
    own.run = v.run(), own.parameters = v.configuration();
    Opening o;
    (void)receive_opening(v, theirs, own, drawn, o);
  }));
  CHECK(ends_with("do not start from the values of the chain that received the root", [&] { // leaves hashed from another run's value
    Reader r(w.data(), w.size());
    WorldBinder b(1, plan);
    VerifierSession session;
    VerifierRounds v(b, r, session);
    SentRoot got;
    (void)v.recv_root(got);
    const PositionList drawn = v.close_positions("positions").positions(3);
    OracleTreeShape own = s;
    own.run = v.run(), own.parameters = v.configuration();
    own.run.w[0] ^= 1;
    Opening o;
    (void)receive_opening(v, got, own, drawn, o);
  }));
  CHECK(ends_with("has not the length that the chain's plan says", [&] { (void)verify(1); }));
  CHECK(ends_with("do not start from the chain's configuration", [&] { (void)verify(2); }));
  CHECK(ends_with("neither this chain nor one that it follows has drawn", [&] { (void)verify(3); }));
  CHECK(ends_with("neither this chain nor one that it follows has drawn", [&] { (void)verify(4); }));
  CHECK(ends_with("a part beyond the list's draws", [&] { provers.part(3, 2); }));
  {
    // After a rejection the channel returns a fixed list that no chain has drawn: receive_opening()
    // returns false without aborting.
    std::vector<u8> cut(w.bytes().begin(), w.bytes().begin() + 40);
    Reader r(cut.data(), cut.size());
    WorldBinder verifier_binder(1, plan);
    VerifierSession v_session;
    VerifierRounds v(verifier_binder, r, v_session);
    SentRoot root;
    CHECK(!v.recv_root(root));
    const PositionList fixed = v.close_positions("positions").positions(3);
    OracleTreeShape own = s;
    own.run = v.run(), own.parameters = v.configuration();
    Opening o;
    CHECK(!receive_opening(v, root, own, fixed, o) && !v.finish());
  }
}

// ProverRounds::lend() copies the chain out to a borrower (a GPU kernel), which runs a stretch of
// rounds itself. take_back() repeats them from the bytes that the borrower sent, and aborts unless
// the drawn elements and the chain's last value agree.
[[maybe_unused]] static void test_lent() {
  Plan plan(setting_goldilocks());
  plan.send_scalars("left open by the block before", 1);
  plan.send_scalars("sum", 1);
  plan.close("epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  for (int j = 0; j < 3; j++) {
    plan.send_scalars("coefficients", 2);
    plan.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  }
  plan.close("point", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 3; // nothing sent; three coordinates
  plan.send_scalars("last", 1);
  set_draws(plan.close("nonzero", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING), 1, 1);
  plan.send_scalars("end values", 2);
  plan.close_none("silent"); // the verifier message is empty; the borrower makes the query too
  plan.send_scalars("the next block's", 2);
  plan.send_root("the next block's tree");
  set_draws(plan.close("after", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 3, 1);
  plan.send_answers("answer", 1);
  plan.close_none(R_LAST);
  const Digest bytes = some_bytes();
  TestCoins coins(5);
  // Fields: name, kind, drawn with draw_E_nonzero(), number of elements (default 1).
  const std::vector<ProverRounds::LentRound> stretch = {{"epsilon", RoundKind::EPSILON, false}, {"round", RoundKind::ALGEBRAIC, false},
                                                        {"round", RoundKind::ALGEBRAIC, false}, {"round", RoundKind::ALGEBRAIC, false},
                                                        {"point", RoundKind::ALGEBRAIC, false, 3}, {"nonzero", RoundKind::ALGEBRAIC, true},
                                                        {"silent", RoundKind::EMPTY, false, 0}};
  // Acts as the borrower, by direct calls of `Chain`. spoil: its chain gets a u64 that is not in
  // `sent`.
  auto elsewhere = [&](ProverRounds::Lent l, std::vector<u8> &sent, std::vector<E> &drawn, bool spoil) {
    auto send = [&](ChainAbsorb &m, const E &x) {
      u8 b[E_BYTES];
      E_to_bytes(x, b);
      sent.insert(sent.end(), b, b + E_BYTES);
      m.push(x);
    };
    send(l.message, coins.e());
    l.chain.close(l.message, ChainKind::EPSILON);
    drawn.push_back(l.chain.draw_E());
    for (int j = 0; j < 3; j++) {
      ChainAbsorb m = l.chain.message();
      send(m, coins.e() * drawn.back());
      send(m, coins.e());
      if (spoil && j == 1) m.push_u64(0);
      l.chain.close(m, ChainKind::ALGEBRAIC);
      drawn.push_back(l.chain.draw_E());
    }
    l.chain.round(ChainKind::ALGEBRAIC);
    for (int i = 0; i < 3; i++) drawn.push_back(l.chain.draw_E());
    ChainAbsorb m = l.chain.message();
    send(m, coins.e());
    l.chain.close(m, ChainKind::ALGEBRAIC);
    drawn.push_back(l.chain.draw_E_nonzero());
    ChainAbsorb values = l.chain.message();
    send(values, coins.e());
    send(values, coins.e());
    l.chain.close(values, ChainKind::EMPTY);
    ChainAbsorb open = l.chain.message();
    send(open, coins.e());
    return l.chain;
  };
  Writer w;
  WorldBinder ch_binder(1, plan);
  ProverRounds ch(ch_binder, bytes, w);
  ch.send(coins.e());
  std::vector<u8> sent;
  std::vector<E> there;
  const Chain end = elsewhere(ch.lend(), sent, there, false);
  const std::vector<E> here = ch.take_back(stretch, sent.data(), sent.size(), there.data(), end);
  CHECK(here == there);
  // What the borrower sent after its last round is an open message: the channel's next close takes
  // it in, whatever that round's kind.
  ch.send(coins.e());
  ch.send_root(Digest{});
  const SampleSeed last = ch.close_seed("after").seed(2, 1);
  ch.send_answer(coins.e());
  ch.close_empty(R_LAST);
  ch.finish();

  // The verifier makes only its ordinary calls.
  Reader r(w.data(), w.size());
  WorldBinder v_binder(1, plan);
  VerifierSession v_session;
  VerifierRounds v(v_binder, r, v_session);
  E x;
  std::vector<E> seen;
  CHECK(v.recv(x) && v.recv_hiding_sum(x));
  seen.push_back(v.close_epsilon("epsilon"));
  for (int j = 0; j < 3; j++) {
    CHECK(v.recv(x) && v.recv(x));
    seen.push_back(v.close_algebraic("round").draw());
  }
  {
    AlgebraicDraw point = v.close_algebraic("point");
    for (int i = 0; i < 3; i++) seen.push_back(point.draw());
  }
  CHECK(v.recv(x));
  seen.push_back(v.close_algebraic("nonzero").draw_nonzero());
  SentRoot root;
  CHECK(v.recv(x) && v.recv(x));
  v.close_empty("silent");
  CHECK(v.recv(x) && v.recv(x) && v.recv_root(root));
  const SampleSeed same = v.close_seed("after").seed(2, 1);
  CHECK(same.z == last.z && same.beta == last.beta);
  CHECK(v.recv_answer(x));
  v.close_empty(R_LAST);
  CHECK(v.finish());
  {
    // A changed byte in the message "end values" changes the seed of the round "after".
    std::vector<u8> b = w.bytes();
    b[b.size() - E_BYTES - DIGEST_BYTES - 3 * E_BYTES] ^= 1;
    Reader r2(b.data(), b.size());
    WorldBinder v2_binder(1, plan);
    VerifierSession v2_session;
    VerifierRounds v2(v2_binder, r2, v2_session);
    CHECK(v2.recv(x) && v2.recv_hiding_sum(x));
    v2.close_epsilon("epsilon");
    for (int j = 0; j < 3; j++) {
      CHECK(v2.recv(x) && v2.recv(x));
      v2.close_algebraic("round").draw();
    }
    {
      AlgebraicDraw point = v2.close_algebraic("point");
      point.draw(), point.draw(), point.draw();
    }
    CHECK(v2.recv(x));
    v2.close_algebraic("nonzero").draw_nonzero();
    CHECK(v2.recv(x) && v2.recv(x));
    v2.close_empty("silent");
    CHECK(v2.recv(x) && v2.recv(x) && v2.recv_root(root));
    CHECK(v2.close_seed("after").seed(2, 1).z != last.z);
  }
  CHECK(seen == there);

  // Mistakes of a borrower or of its caller. `Again`: a prover's channel just before lend().
  struct Again {
    Writer w;
    WorldBinder b;
    ProverRounds p;
    std::vector<u8> sent;
    std::vector<E> drawn;
    Again(const Plan &plan, const Digest &bytes) : b(1, plan), p(b, bytes, w) { p.send(E::one()); }
  };
  CHECK(ends_with("other elements were drawn", [&] { // the borrower's chain got a u64 not in `sent`
    Again a(plan, bytes);
    const Chain e2 = elsewhere(a.p.lend(), a.sent, a.drawn, true);
    a.p.take_back(stretch, a.sent.data(), a.sent.size(), a.drawn.data(), e2);
  }));
  CHECK(ends_with("other elements were drawn", [&] { // it uses another element than the chain gave
    Again a(plan, bytes);
    const Chain e2 = elsewhere(a.p.lend(), a.sent, a.drawn, false);
    a.drawn.back() = a.drawn.back() + E::one();
    a.p.take_back(stretch, a.sent.data(), a.sent.size(), a.drawn.data(), e2);
  }));
  CHECK(ends_with("has not the value that its messages give", [&] { // it ran a round more than it says
    Again a(plan, bytes);
    Chain e2 = elsewhere(a.p.lend(), a.sent, a.drawn, false);
    e2.round(ChainKind::ALGEBRAIC);
    a.p.take_back(stretch, a.sent.data(), a.sent.size(), a.drawn.data(), e2);
  }));
  CHECK(ends_with("draws again after its next round", [&] { // it goes on drawing from the verifier message before the stretch
    Plan two(plan.set);
    two.close("first", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    two.close("second", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    Writer w2;
    WorldBinder p_binder(1, two);
    ProverRounds p(p_binder, bytes, w2);
    p.close_algebraic("first").draw();
    p.lend().chain.draw_E();
  }));
  CHECK(ends_with("not field elements in canonical form", [&] {
    Again a(plan, bytes);
    const Chain e2 = elsewhere(a.p.lend(), a.sent, a.drawn, false);
    for (int i = 0; i < 8; i++) a.sent[i] = 0xFF;
    a.p.take_back(stretch, a.sent.data(), a.sent.size(), a.drawn.data(), e2);
  }));
  CHECK(ends_with("the chain was not lent out", [&] {
    Again a(plan, bytes);
    a.p.take_back({}, nullptr, 0, nullptr, Chain(FirstQuery()));
  }));
  for (int what = 0; what < 5; what++)
    CHECK(ends_with("the chain is lent out", [&] { // until it is back, these five calls abort
      Again a(plan, bytes);
      a.p.lend();
      if (what == 0) a.p.close_algebraic("round");
      if (what == 1) a.p.finish();
      if (what == 2) a.p.file_only();
      if (what == 3) a.p.lend();
      Sealed d;
      if (what == 4) (void)a.p.seal("seal", d);
    }));
  CHECK(ends_with("this verifier message is over", [&] { // a handle from before the lending
    Plan two(plan.set);
    two.close("first", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 2;
    Writer w2;
    WorldBinder p_binder(1, two);
    ProverRounds p(p_binder, bytes, w2);
    AlgebraicDraw d = p.close_algebraic("first");
    d.draw(), d.draw();
    p.lend();
    d.draw();
  }));
  CHECK(ends_with("less was drawn", [&] { // lent with one of the round's two elements not drawn
    Plan two(plan.set);
    two.close("first", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 2;
    Writer w2;
    WorldBinder p_binder(1, two);
    ProverRounds p(p_binder, bytes, w2);
    p.close_algebraic("first").draw();
    p.lend();
  }));
  CHECK(ends_with("while answers to a sample are owed", [&] {
    Plan two(plan.set);
    set_draws(two.close("seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 2, 1);
    Writer w2;
    WorldBinder p_binder(1, two);
    ProverRounds p(p_binder, bytes, w2);
    p.close_seed("seed").seed(1, 1);
    p.lend();
  }));
  {
    // Cannot be lent: a round with proof of work, a round of seeds, a round planned with another
    // number of elements than the caller states.
    Setting with_work = plan.set;
    with_work.set_work(ErrorKind::FIELD, Scope::REQUEST, 4);
    Plan hard(with_work), seeds(plan.set), twice(plan.set);
    hard.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    set_draws(seeds.close("round", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING), 1, 1);
    twice.close("round", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 2;
    const E none[2] = {E::zero(), E::zero()};
    auto back = [&](const Plan &of, RoundKind kind) {
      Writer w2;
      WorldBinder p_binder(1, of);
      ProverRounds p(p_binder, bytes, w2);
      const Chain c = p.lend().chain;
      p.take_back({{"round", kind, false, 1}}, nullptr, 0, none, c);
    };
    CHECK(ends_with("a round with work cannot be lent out", [&] { back(hard, RoundKind::ALGEBRAIC); }));
    CHECK(ends_with("only rounds that draw elements of E, or nothing", [&] { back(seeds, RoundKind::SEED); }));
    CHECK(ends_with("draws the planned number of elements", [&] { back(twice, RoundKind::ALGEBRAIC); }));
  }
  {
    // A hiding polynomial's sum must be sent last before close_epsilon(), across take_back() too.
    // `tail_is_hiding_sum` (take_back()'s last argument) is taken on the caller's word: only a
    // length is checked.
    Plan sums(plan.set);
    sums.send_scalars("sum and one more", 2);
    sums.close("epsilon", RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    u8 one[E_BYTES];
    E_to_bytes(E::one(), one);
    CHECK(ends_with("must be the last thing sent", [&] { // bytes come back after the sum
      Writer w2;
      WorldBinder p_binder(1, sums);
      ProverRounds p(p_binder, bytes, w2);
      p.send_hiding_sum(E::one());
      const Chain c = p.lend().chain;
      p.take_back({}, one, E_BYTES, nullptr, c);
      p.close_epsilon("epsilon");
    }));
    CHECK(runs_through([&] { // a sum that the borrower sent last, on the caller's word
      Writer w2;
      WorldBinder p_binder(1, sums);
      ProverRounds p(p_binder, bytes, w2);
      p.send(E::one());
      const Chain c = p.lend().chain;
      p.take_back({}, one, E_BYTES, nullptr, c, true);
      p.close_epsilon("epsilon");
      p.finish();
    }));
    CHECK(ends_with("no bytes are left open", [&] {
      Writer w2;
      WorldBinder p_binder(1, sums);
      ProverRounds p(p_binder, bytes, w2);
      const Chain c = p.lend().chain;
      p.take_back({}, nullptr, 0, nullptr, c, true);
    }));
  }
  CHECK(ends_with("the chain is lent out", [&] {
    Writer w2;
    WorldBinder p_binder(1, plan);
    ProverRounds p(p_binder, bytes, w2);
    p.send(E::one());
    p.lend();
    p.send(E::one());
  }));
  CHECK(ends_with("fewer bytes come back", [&] {
    Again a(plan, bytes);
    const Chain e2 = elsewhere(a.p.lend(), a.sent, a.drawn, false);
    a.p.take_back(stretch, a.sent.data(), 5 * E_BYTES, a.drawn.data(), e2);
  }));
}

// A chain that ends in a seal, and a chain with that seal in its first query (Binder::follows()).
[[maybe_unused]] static void test_seal() {
  Plan plan(setting_goldilocks());
  plan.send_scalars("a", 1);
  plan.close("algebraic", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  plan.send_scalars("b", 1);
  plan.close("seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
  const Digest bytes{}; // not read: these chains have no random bytes
  auto prove = [&](const E &b, Writer &w) {
    WorldBinder ch_binder(1, plan);
    ch_binder.no_random_bytes();
    ProverRounds ch(ch_binder, bytes, w);
    ch.send(E::one());
    ch.close_algebraic("algebraic").draw();
    ch.send(b);
    Sealed d;
    CHECK(ch.seal("seal", d));
    ch.finish();
    return d.digest();
  };
  Writer w, w2;
  const Digest end = prove(E::theta(1), w);
  CHECK(w.size() == 2 * E_BYTES); // no salt, and nothing of the seal
  CHECK(end != prove(E::theta(2), w2)); // the open message is under the seal
  Reader r(w.data(), w.size());
  WorldBinder ch_binder(1, plan);
  ch_binder.no_random_bytes();
  VerifierSession ch_session;
  VerifierRounds ch(ch_binder, r, ch_session);
  E x;
  CHECK(ch.recv(x));
  ch.close_algebraic("algebraic").draw();
  CHECK(ch.recv(x));
  Sealed seen_end;
  CHECK(ch.seal("seal", seen_end) && seen_end.digest() == end);
  CHECK(ch.finish());
  {
    // A chain that fails after its seal makes the chain that follows it fail too. spoiled: the
    // first chain has a byte too many, which its finish() finds. end_forgotten: that finish() is
    // not called.
    Plan next(plan.set);
    next.send_scalars("c", 1);
    next.close("algebraic", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    auto session = [&](bool spoiled, bool end_forgotten) {
      std::vector<u8> first = w.bytes();
      if (spoiled) first.push_back(0);
      Reader r1(first.data(), first.size());
      WorldBinder b1(1, plan);
      b1.no_random_bytes();
      VerifierSession one_session;
      VerifierRounds one(b1, r1, one_session);
      E y;
      CHECK(one.recv(y));
      one.close_algebraic("algebraic").draw();
      CHECK(one.recv(y));
      Sealed s;
      CHECK(one.seal("seal", s)); // nothing is wrong up to here
      Writer w4;
      {
        WorldBinder pb(1, plan);
        pb.no_random_bytes();
        Writer again;
        ProverRounds p1(pb, bytes, again);
        p1.send(E::one()), p1.close_algebraic("algebraic").draw(), p1.send(E::theta(1));
        Sealed ps;
        CHECK(p1.seal("seal", ps));
        p1.finish();
        WorldBinder pb2(2, next);
        pb2.no_random_bytes(), pb2.follows(ps);
        ProverRounds p2(pb2, bytes, w4);
        p2.send(E::one()), p2.close_algebraic("algebraic").draw();
        p2.finish();
      }
      Reader r2(w4.data(), w4.size());
      WorldBinder b2(2, next);
      b2.no_random_bytes(), b2.follows(s);
      VerifierRounds two(b2, r2, one_session); // `two` follows `one`: they must share a session
      CHECK(two.recv(y));
      two.close_algebraic("algebraic").draw();
      const bool first_good = end_forgotten || one.finish();
      CHECK(end_forgotten || first_good == !spoiled);
      const bool good = two.finish();
      if (spoiled) CHECK(two.why() && std::string(two.why()).find("an earlier chain") != std::string::npos);
      return good;
    };
    CHECK(session(false, false));
    CHECK(!session(true, false));
    CHECK(ends_with("follows a chain of another session", [&] {
      Reader r1(w.data(), w.size());
      WorldBinder b1(1, plan);
      b1.no_random_bytes();
      VerifierSession first_session, second_session;
      VerifierRounds one(b1, r1, first_session);
      E y;
      (void)one.recv(y);
      one.close_algebraic("algebraic").draw();
      (void)one.recv(y);
      Sealed s;
      (void)one.seal("seal", s);
      (void)one.finish();
      WorldBinder b2(2, next);
      b2.no_random_bytes(), b2.follows(s);
      Reader r2(w.data(), w.size());
      VerifierRounds two(b2, r2, second_session);
    }));
    CHECK(ends_with("has not been brought to its end", [&] { (void)session(false, true); }));
    CHECK(ends_with("has not been brought to its end", [&] { // the same order on the prover's side
      WorldBinder pb(1, plan);
      pb.no_random_bytes();
      Writer w5, w6;
      ProverRounds p1(pb, bytes, w5);
      p1.send(E::one()), p1.close_algebraic("algebraic").draw(), p1.send(E::one());
      Sealed ps;
      (void)p1.seal("seal", ps);
      WorldBinder pb2(2, next);
      pb2.no_random_bytes(), pb2.follows(ps);
      ProverRounds p2(pb2, bytes, w6);
      p2.send(E::one()), p2.close_algebraic("algebraic").draw();
      p2.finish();
    }));
    CHECK(ends_with("name two configurations", [&] { // not the followed chain's configuration
      WorldBinder pb(1, plan);
      pb.no_random_bytes();
      Writer w5, w6;
      ProverRounds p1(pb, bytes, w5);
      p1.send(E::one()), p1.close_algebraic("algebraic").draw(), p1.send(E::one());
      Sealed ps;
      (void)p1.seal("seal", ps);
      p1.finish();
      Binder other(2, next);
      Digest files{};
      files.w[0] = 0xBAD;
      other.configuration(files);
      other.no_random_bytes(), other.follows(ps);
      ProverRounds p2(other, bytes, w6);
    }));
  }
  auto sealed = [&](ProverRounds &p) {
    p.send(E::one());
    p.close_algebraic("algebraic").draw();
    p.send(E::one());
    Sealed d;
    (void)p.seal("seal", d);
  };
  CHECK(ends_with("has been sealed", [&] {
    Writer w3;
    WorldBinder p_binder(1, plan);
    p_binder.no_random_bytes();
    ProverRounds p(p_binder, bytes, w3);
    sealed(p);
    p.send(E::one());
  }));
  CHECK(ends_with("has been sealed", [&] {
    Writer w3;
    WorldBinder p_binder(1, plan);
    p_binder.no_random_bytes();
    ProverRounds p(p_binder, bytes, w3);
    sealed(p);
    p.close_algebraic("algebraic");
  }));
  CHECK(ends_with("this verifier message is over", [&] { // a handle from before the seal
    Plan only(plan.set);
    only.close("algebraic", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
    only.close("seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING);
    Writer w3;
    WorldBinder p_binder(1, only);
    p_binder.no_random_bytes();
    ProverRounds p(p_binder, bytes, w3);
    AlgebraicDraw d = p.close_algebraic("algebraic");
    d.draw();
    Sealed end_of_it;
    (void)p.seal("seal", end_of_it);
    d.draw();
  }));
}

[[maybe_unused]] static bool meets(const char *why, const char *expected) { return why && std::string(why).find(expected) != std::string::npos; }

int main() {
  const Toy toy;
  const std::vector<u8> proof = toy_prove(toy, 1);
  const char *why = nullptr;
  const bool accepted = toy_verify(toy, proof, &why);
  CHECK(accepted && why == nullptr);
  CHECK(toy_verify(toy, proof, nullptr, Style::RUN_ON) && toy_verify(toy, proof, nullptr, Style::HALF));
  CHECK(toy_prove(toy, 1) == proof);
  CHECK(toy_prove(toy, 2) != proof);
  test_chain();
  test_format();
  test_channel_format();
  test_binder();
  test_orders();
  test_seal();
  test_opening_held();
  test_lent();

  // Flipping the low or the high bit of any byte gets a proof rejected: as it is, and with proof of
  // work in the first round.
  size_t flips = 0;
  Toy early = toy;
  early.work_seed = 6;
  const std::vector<u8> ground = toy_prove(early, 1);
  CHECK(toy_verify(early, ground));
  const std::pair<const Toy *, const std::vector<u8> *> runs[] = {{&toy, &proof}, {&early, &ground}};
  for (const auto &run : runs) {
    const std::vector<u8> &honest = *run.second;
    for (size_t i = 0; i < honest.size(); i++)
      for (u8 flip : {(u8)1, (u8)0x80}) {
        std::vector<u8> b = honest;
        b[i] ^= flip;
        CHECK(refused(*run.first, b));
        flips++;
      }
    // Cut short at every length; a byte too many.
    for (size_t n = 0; n < honest.size(); n++) CHECK(refused(*run.first, std::vector<u8>(honest.begin(), honest.begin() + n)));
    std::vector<u8> b = honest;
    b.push_back(0);
    CHECK(refused(*run.first, b));
  }
  // The reasons reported for truncated proofs.
  CHECK(!toy_verify(toy, {}, &why) && meets(why, "salt is missing"));
  CHECK(!toy_verify(early, std::vector<u8>(ground.begin(), ground.begin() + 2 * DIGEST_BYTES + 3), &why) && meets(why, "a nonce is missing"));
  CHECK(!toy_verify(toy, std::vector<u8>(proof.begin(), proof.begin() + DIGEST_BYTES + 5), &why) && meets(why, "a value that the prover sends is missing"));
  {
    // Not in canonical form: the modulus p as the first coordinate of the first answer.
    std::vector<u8> b = proof;
    const u64 p = P;
    for (int i = 0; i < 8; i++) b[2 * DIGEST_BYTES + i] = (u8)(p >> (8 * i));
    CHECK(!toy_verify(toy, b, &why) && meets(why, "not in canonical form"));
    CHECK(refused(toy, b));
  }
  // Rejected: a proof made for one statement, record, label or parameter, checked under another.
  Toy other = toy;
  other.statement[0] ^= 1;
  CHECK(refused(other, proof));
  other = toy, other.record[1] ^= 1;
  CHECK(refused(other, proof));
  other = toy, other.record.clear();
  CHECK(refused(other, proof));
  other = toy, other.label = 2;
  CHECK(refused(other, proof));
  other = toy, other.parameter = 1;
  CHECK(refused(other, proof));
  // Rejected: honest proofs for less or more proof of work than the verifier's plan asks for.
  for (int delta : {-1, 1, -9}) {
    other = toy, other.work_positions = (unsigned)((int)toy.work_positions + delta);
    const std::vector<u8> p2 = toy_prove(other, 1);
    CHECK(toy_verify(other, p2));
    CHECK(refused(toy, p2));
  }
  if (failures)
    std::printf("rounds_test: %d FAILURES\n", failures);
  else
    std::printf("rounds_test: all passed (a proof of %zu bytes, %zu flips, each shown to three styles of verifier)\n", proof.size(), flips);
  return failures ? 1 : 0;
}
