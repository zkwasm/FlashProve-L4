// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests FOLD (blocks/iop/fold.h: the sumcheck that folds a group's rows into three single rows)
// followed by BASE (blocks/iop/base_case.h: the last step of a proximity test): prover against
// verifier at toy size, with the real hash chain, Merkle trees and CPU engine (`EngineCpu`: it
// stores a prover's large tables; code outside it is "the host"). FOLD's output, a few single rows
// with one linear claim on them, is called a complex (blocks/iop/single_rows.h).
//
// The test checks, for the cases listed in main():
//   a true claim (`GroupRow`; `row` in identifiers) is accepted, and FOLD's output claim holds for
//   the prover's folded rows;
//   where BASE follows, a claim with a false value (`target`) is rejected (`refused` in identifiers
//   and output), and so is a proof with any one byte changed, the last dropped or one added; a
//   verifier program that misuses a mask oracle aborts;
//   with kept members (oracle messages whose main rows are not folded) BASE cannot follow: both
//   sides reach the same output claim, and the complex, read at a list of positions, gives the
//   symbols of the encodings of the prover's rows.
// It also tests plans (lists of a run's rounds) that record a wrong group or none
// (other_group_test()), BASE alone, two folds in a row, host reads of coefficients that only the
// engine stores, which must abort, and the prover's self-checks of its masks against their draws
// (own-test-base-mask, own-test-fold-mask). Cheating provers are not part of this repository.

#define BLK_PLANS_BY_HAND_FOR_A_TEST  // this test builds plans by hand (blocks/numbers/plan.h)
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "chain/closed.h"
#include "engine/engine_cpu.h"
#include "iop/base_case.h"
#include "iop/fold.h"

using namespace blk;

// A limit of this test: no run here depends on BASE's comparison of the last mask row's symbols
// (blocks/iop/base_case.h): with BASE's loop over the rows cut to the main row, the whole test
// passed. Only that comparison sees a prover that changes that row's word (not in this repository).

// A mask (an oracle message made from private randomness alone) serves one FOLD or one BASE;
// asserted here: it cannot be copied. A caller cannot default-construct or copy a `FoldStart`,
// without which fold_epsilon() draws no eps.
static_assert(!std::is_copy_constructible<FoldMask>::value && !std::is_copy_assignable<FoldMask>::value, "a FoldMask can be copied");
static_assert(!std::is_copy_constructible<BaseMask>::value && !std::is_copy_assignable<BaseMask>::value, "a BaseMask can be copied");
static_assert(!std::is_default_constructible<FoldStart>::value && !std::is_copy_constructible<FoldStart>::value,
              "a FoldStart can be made without FOLD's first message");

// The string is part of a rejection message of BASE (blocks/iop/base_case.h), and no line here
// reads it. FOLD's verifier tests no equation itself, so a cheat in FOLD leaves an output claim
// that is false for the committed rows, which BASE rejects.
static const char *const EXPECTED_REASON = "differs from the opened symbols";

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// Test data and a stand-in for private randomness (`coins` in identifiers).
// SplitMix64 (Steele, Lea, Flood, OOPSLA 2014). Not for anything secret.
class TestCoins : public Coins {
 public:
  explicit TestCoins(u64 seed) : s_(seed) {}
  u64 word() {
    u64 z = (s_ += 0x9E3779B97F4A7C15ull);
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

 private:
  u64 s_;
};
// [own-test-base-mask, own-test-fold-mask] TestCoins that record each draw of field elements, and
// so can answer last_draw() and not_the_coins() (`Coins`, blocks/engine/table.h). not_the_coins()
// returns 0 or 1, not a count, and ignores `left_out`, which is empty in this test. fill() adds 1 to
// the first element of draw number `loses` after recording it: a value lost right after its draw,
// before any copy of it is taken.
struct Recalling : TestCoins {
  using TestCoins::TestCoins;
  std::vector<std::vector<F>> drawn;
  long loses = -1;      // the number of that draw; -1: none
  bool numbers = true;  // false: last_draw() gives no number, so the self-checks compare nothing
  void fill(F *out, size_t n) override {
    TestCoins::fill(out, n);
    drawn.emplace_back(out, out + n);
    if (n && (long)drawn.size() - 1 == loses) out[0] = out[0] + F::one();
  }
  u32 last_draw() const override { return numbers ? (u32)drawn.size() - 1 : NO_DRAW; }
  u64 not_the_coins(u32 draw, const F *values, u64 n, const std::vector<u64> &) const override {
    u64 differ = drawn.at(draw).size() != n;
    for (u64 i = 0; i < n && !differ; i++) differ += !(values[i] == drawn[draw][i]);
    return differ;
  }
};

// One oracle message of the group. `main_deg`: 1 for main rows over F, 3 over E. `kept`: see
// GroupLayout (blocks/iop/relation.h).
struct Member {
  size_t main_rows;
  unsigned main_deg;
  size_t mask_rows;
  bool kept;
};
struct Case {
  const char *name;
  std::vector<Member> members;  // by decreasing main and mask row counts: GroupLayout's block order
  bool early;                   // FOLD's and BASE's masks go out in the first prover message
  bool batched;                 // FOLD's claim is a + x b, with x drawn in a round before FOLD
  unsigned work = 4;            // bits of proof of work on every round
  bool kept() const {           // if true, no BASE: FOLD's output is read at a list of positions
    for (const Member &m : members)
      if (m.kept) return true;
    return false;
  }
};

constexpr unsigned DRAWS = 8;

// Code{log2 of the length, message entries, random tail entries}.
static Codes toy_codes() {
  const size_t r = tail_for_positions(DRAWS);
  Codes c;
  c.base.push_back(Code{8, 32, r});
  c.mask.push_back(Code{7, 16, r});
  return c;
}

// The group has 2^k main rows and 2^h mask rows (GroupLayout).
static FoldShape toy_shape(const Case &t) {
  FoldShape S;
  S.code = 0, S.mask_code = 0;
  GroupLayout &L = S.layout;
  u64 main = 0, mask = 0;
  for (size_t i = 0; i < t.members.size(); i++) {
    L.members.push_back((OracleId)i);
    const bool folded = t.members[i].main_rows && !t.members[i].kept;
    L.kept.push_back(t.members[i].kept);
    L.main_first.push_back(folded ? main : GroupLayout::NONE);
    L.mask_first.push_back(t.members[i].mask_rows ? mask : GroupLayout::NONE);
    main += folded ? u64(1) << ceil_log2(t.members[i].main_rows) : 0;
    mask += t.members[i].mask_rows ? u64(1) << ceil_log2(t.members[i].mask_rows) : 0;
  }
  L.h = ceil_log2(mask);
  L.k = std::max(std::max(ceil_log2(main), L.h), 1u);
  L.log_ell = ceil_log2(toy_codes().base[0].ell), L.log_ell_zk = ceil_log2(toy_codes().mask[0].ell);
  return S;
}

// The members' shapes. They have no out-of-domain samples: neither FOLD nor BASE reads samples.
static std::vector<OracleNotice> toy_notices(const Case &t) {
  std::vector<OracleNotice> n;
  for (const Member &m : t.members) {
    OracleNotice x;
    x.shape.name = "member", x.shape.code = 0, x.shape.main_rows = m.main_rows, x.shape.main_deg = m.main_deg;
    x.shape.mask_rows = m.mask_rows, x.shape.mask_code = 0, x.shape.answers = 0;
    n.push_back(x);
  }
  return n;
}

static Setting toy_setting(unsigned work) {
  Setting s = setting_goldilocks();
  for (ErrorKind kind : {ErrorKind::FIELD, ErrorKind::CLOSENESS, ErrorKind::QUERY}) s.set_work(kind, Scope::REQUEST, work);
  return s;
}
// The group that a plan records beside a round (`PricedGroup`, blocks/numbers/plan.h): the codes
// and row counts under which FOLD, or BASE, is to run. Filled in field by field, not by
// group_as_run(), which FOLD and BASE call.
static PricedGroup said_group(bool with_mask, unsigned k, unsigned h, u64 slots) {
  const Codes c = toy_codes();
  PricedGroup g;
  g.main.log_n = c.base[0].log_n, g.main.ell = c.base[0].ell, g.main.r = c.base[0].r;
  if (with_mask) g.mask.log_n = c.mask[0].log_n, g.mask.ell = c.mask[0].ell, g.mask.r = c.mask[0].r;
  g.k = k, g.h = h, g.slots = slots, g.said = true;
  return g;
}
static PricedGroup fold_group(const Case &t, const FoldShape &S) {
  u64 kept_rows = 0, slots = 1;
  for (const Member &m : t.members) kept_rows += m.kept ? m.main_rows : 0;
  while (slots < 1 + kept_rows) slots *= 2;  // a main row and the kept rows side by side
  return said_group(true, S.layout.k, S.layout.h, slots);
}
static PricedGroup base_group(unsigned masks) { return said_group(masks != 0, 0, 0, 1); }
// For other_group_test(): `bent_number` alters one group that toy_plan() records. Its tens digit
// picks the round: 0 FOLD's first (eps), 1 BASE's first (lambda), 2 the one in which BASE draws
// positions, 3 FOLD's further rounds. Its units digit picks the number that changes; 9 marks the
// group as not recorded (`said` false). -1: nothing is altered.
static int bent_number = -1;
static void bend(PricedGroup &fold, PricedGroup &base, PricedGroup &positions, PricedGroup &rounds) {
  PricedGroup &g = bent_number < 10 ? fold : bent_number < 20 ? base : bent_number < 30 ? positions : rounds;
  switch (bent_number % 10) {
  case 0: g.main.log_n++; break;
  case 1: g.main.ell *= 2; break;
  case 2: g.main.r++; break;
  case 3: g.mask.log_n++; break;
  case 4: g.mask.ell *= 2; break;
  case 5: g.mask.r++; break;
  case 6: g.k++; break;
  case 7: g.h++; break;
  case 8: g.slots *= 2; break;
  default: g.said = false;
  }
}
// The rounds of one case's run. All its oracle messages are read at one list of positions, so they
// have one leaf class, and those of one prover message share a Merkle tree. Each round's error is
// entered as 0: the test is not about error accounting.
static Plan toy_plan(const Case &t, const FoldShape &S) {
  Plan p(toy_setting(t.work));
  const Codes codes = toy_codes();
  PricedGroup of_fold = fold_group(t, S), of_base = base_group(2), of_positions = base_group(2), of_rounds = fold_group(t, S);
  if (bent_number >= 0) bend(of_fold, of_base, of_positions, of_rounds);
  p.send_root("members");
  if (t.batched) p.close(R_BAT_SIGMA, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  if (!t.early) p.send_root("fold mask");
  p.send_scalars("hiding sum", 1);
  PlanRound &eps = p.close(R_FOLD_EPSILON, RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
  eps.draw_E = 1, eps.group = of_fold;
  for (unsigned j = 0; j < S.layout.k; j++) {
    p.send_scalars("round polynomial", 2);
    PlanRound &round = p.close(R_FOLD_ROUND, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
    round.draw_E = 1, round.group = of_rounds;
  }
  if (t.kept()) {
    PlanRound &r = p.close(R_LOC, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    r.draw_positions = DRAWS, r.log_n = codes.base[0].log_n;
    return p;
  }
  if (!t.early) p.send_root("base mask");
  p.send_scalars("masked target", 1);
  PlanRound &lambda = p.close(R_BASE_LAMBDA, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
  lambda.draw_E = 1, lambda.group = of_base;
  p.send_scalars("masked word", codes.base[0].ell + codes.base[0].r + 2 * (codes.mask[0].ell + codes.mask[0].r));
  PlanRound &r = p.close(R_BASE, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  r.draw_positions = DRAWS, r.log_n = codes.base[0].log_n, r.group = of_positions;
  return p;
}

// A claim's public coefficients, without its target. On the main table: three tensors (structured
// vectors: blocks/iop/tensor.h), one with a dense (written-out) factor on a row variable and two
// variables of the index within a row, and three single entries, two at one index. On the mask
// table: a written-out vector. On each kept member's main table: a tensor and a single entry.
static GroupRow toy_row(const Case &t, const FoldShape &S, u64 seed) {
  TestCoins data(seed);
  const GroupLayout &L = S.layout;
  GroupRow row;
  std::vector<E> z(L.k + L.log_ell);
  for (E &x : z) x = data.e();
  row.c.add(Tensor::eq(z.data(), (unsigned)z.size(), data.e()));
  std::vector<F> f(L.k + L.log_ell);
  data.fill(f.data(), f.size());
  row.c.add(Tensor::product(f.data(), (unsigned)f.size(), data.e()));
  Tensor dense = Tensor::eq(z.data(), (unsigned)z.size(), data.e());
  dense.dense_vars = {L.k + 1, L.k + 2, 0};
  for (unsigned t : dense.dense_vars) dense.set_pair(t, E::one(), E::one());
  std::vector<E> values(11);
  for (E &x : values) x = data.e();
  dense.set_values(std::make_shared<const std::vector<E>>(values), 3);
  row.c.add(dense);
  const u64 main = u64(1) << (L.k + L.log_ell);
  const u64 twice = data.word() % main;
  row.c.add_entry(twice, data.e()), row.c.add_entry(data.word() % main, data.e()), row.c.add_entry(twice, data.e());
  row.d.resize(size_t(1) << (L.h + L.log_ell_zk));
  for (E &x : row.d) x = data.e();
  for (size_t i = 0; i < t.members.size(); i++) {
    if (!t.members[i].kept) continue;
    std::vector<E> y(ceil_log2(t.members[i].main_rows) + L.log_ell);
    for (E &x : y) x = data.e();
    KeptForm f;
    f.oracle = (OracleId)i;
    f.c.add(Tensor::eq(y.data(), (unsigned)y.size(), data.e()));
    f.c.add_entry(data.word() % (u64(1) << y.size()), data.e());
    row.kept.push_back(f);
  }
  return row;
}
static std::vector<E> main_form(const FoldShape &S, const GroupRow &row) {
  std::vector<E> c(size_t(1) << (S.layout.k + S.layout.log_ell), E::zero());
  row.c.add_to(c.data(), S.layout.k + S.layout.log_ell);
  return c;
}
// a + x b.
static GroupRow combine(GroupRow a, const GroupRow &b, const E &x) {
  a.c.add(b.c, x);
  for (size_t i = 0; i < a.kept.size(); i++) a.kept[i].c.add(b.kept[i].c, x);
  for (size_t i = 0; i < a.d.size(); i++) a.d[i] += x * b.d[i];
  a.target += x * b.target;
  return a;
}

// Every chain must name a configuration digest in its first query (the hash input that fixes the
// public inputs before the prover speaks); the test makes one up.
static Digest made_up_configuration() {
  Digest d;
  for (u32 i = 0; i < 8; i++) d.w[i] = 0x5EED0000u + i;
  return d;
}
// Everything public about a run goes into the chain's first query: as parameters the codes, the
// oracle messages' shapes, the run's switches and the seed of the claims' coefficients; as the
// statement the claims' targets. run_verifier() (blocks/chain/closed.h) calls each verifier, a
// plain function, with the channel (`VerifierRounds`) and `Handles`, not `Inputs`: of a run's
// inputs it sees only what the first query contains. Only the verifier programs that are wrong on
// purpose, and must abort, build their own channel.
struct Inputs {
  std::vector<OracleNotice> sent;
  std::vector<u64> numbers;
  std::vector<E> targets;
};
struct Handles {
  Public<Codes> codes;
  Public<std::vector<OracleNotice>> sent;
  Public<std::vector<u64>> numbers;
  Public<std::vector<E>> targets;
};
static Handles bind(Binder &b, const Inputs &in) {
  b.configuration(made_up_configuration());
  Handles h;
  h.codes = b.parameter("codes", toy_codes());
  h.sent = b.parameter("oracle messages", in.sent);
  h.numbers = b.parameter("switches, seed of the coefficients", in.numbers);
  h.targets = b.statement("targets", in.targets);
  return h;
}
static std::vector<u64> numbers_of(const Case &t, u64 seed) {
  std::vector<u64> n = {(u64)t.early, (u64)t.batched, (u64)t.work, seed};
  for (const Member &m : t.members) n.push_back(m.kept);
  return n;
}
// For a verifier: the inverse of toy_notices() and numbers_of().
static Case case_of(const std::vector<OracleNotice> &sent, const std::vector<u64> &numbers) {
  BLK_REQUIRE(numbers.size() == 4 + sent.size(), "fold_base_test: the bound numbers do not fit the bound oracle messages");
  Case t{"as bound", {}, numbers[0] != 0, numbers[1] != 0, (unsigned)numbers[2]};
  for (size_t i = 0; i < sent.size(); i++) t.members.push_back(Member{sent[i].shape.main_rows, sent[i].shape.main_deg, sent[i].shape.mask_rows, numbers[4 + i] != 0});
  return t;
}
// Pseudo-random bytes for a chain's salt.
static Digest random_bytes(u64 seed) {
  TestCoins c(seed);
  u8 bytes[DIGEST_BYTES];
  c.bytes(bytes, sizeof bytes);
  Digest d;
  digest_from_bytes(bytes, d);
  return d;
}

struct Proved {
  std::vector<u8> proof;
  E target[2];
  bool output_holds;
  E claim;                          // the target of FOLD's output claim
  size_t factors = 0;               // written-out factors moved into engine tables: `hold_factors`
  std::vector<E> symbols, encoded;  // with kept members: the complex as opened; as re-encoded
  size_t draws = 0;                 // prove(): the draws of field elements from the prover's generator
};
// [own-test-fold-mask] A fault injected into prove(). FOLD's self-check must catch it
// (fold_mask_as_drawn(), blocks/iop/fold.h), unless `no_numbers` leaves it nothing to compare with.
// For cases with `batched`: there prove() calls FOLD's three steps one by one.
struct Fault {
  long loses = -1;  // `Recalling::loses`
  // A stray write: coefficient number `zeroed` of the fresh mask's hiding polynomial becomes 0; with
  // `after_start` only the last of its three elements of F. If `zeroed` is the number of
  // coefficients (only with `after_start`): the first entry of the mask rows' random tail, as the
  // context keeps it. -1: none.
  long zeroed = -1;
  bool after_start = false;      // the write follows fold_start_prove(); if false, fold_mask_make()
  bool exit_after_start = false; // for a child process: exits with 0 right after fold_start_prove()
  bool no_numbers = false;       // the generator does not number its draws (`Recalling::numbers`)
  bool not_recorded = false;     // the mask is made by fold_mask_make(S, coins), which records no number
};
// Runs the prover. The first claim's public target is the true value of its left side plus
// `off_target`; the second claim's target is always its true value (a second claim is used only
// where two claims are batched). The generator numbers its draws, so the self-checks compare, in
// every run but those with `Fault::no_numbers`.
static Proved prove(const Case &t, u64 seed, const E &off_target, const Fault &fault = Fault()) {
  const FoldShape S = toy_shape(t);
  const Plan plan = toy_plan(t, S);
  const Codes codes = toy_codes();
  Recalling coins(seed);
  coins.loses = fault.loses, coins.numbers = !fault.no_numbers;
  TestCoins data(seed + 1);
  EngineCpu engine;
  Writer proof;

  GroupRow row[2] = {toy_row(t, S, seed + 3), toy_row(t, S, seed + 4)};  // the second is used where two claims are batched
  const std::vector<E> c[2] = {main_form(S, row[0]), main_form(S, row[1])};

  std::vector<OracleDraft> drafts;
  const std::vector<OracleNotice> notices = toy_notices(t);
  E value[2] = {E::zero(), E::zero()};
  for (size_t i = 0; i < notices.size(); i++) {
    OracleDraft d;
    d.shape = notices[i].shape, d.leaf_class = notices[i].leaf_class;
    if (d.shape.main_rows) {
      Table m(d.shape.main_deg, d.shape.main_rows, codes.base[0].ell);
      data.fill(m.a.data(), m.a.size());
      for (int a = 0; a < 2; a++)
        for (const KeptForm &f : row[a].kept) {
          if (f.oracle != i) continue;
          std::vector<E> form(m.entries(), E::zero());
          f.c.add_to(form.data(), ceil_log2(m.entries()));
          for (u64 j = 0; j < m.entries(); j++) value[a] += form[j] * m.entry(j);
        }
      for (size_t b = 0; b < m.rows && !t.members[i].kept; b++)
        for (size_t j = 0; j < m.len; j++)
          for (int a = 0; a < 2; a++) value[a] += c[a][(S.layout.main_first[i] + b) + (j << S.layout.k)] * m.at(b, j);
      d.main = engine.put(m);
    }
    if (d.shape.mask_rows) {
      d.mask = Table(E_DEGREE, d.shape.mask_rows, codes.mask[0].ell);
      data.fill(d.mask.a.data(), d.mask.a.size());
      for (size_t b = 0; b < d.mask.rows; b++)
        for (size_t j = 0; j < d.mask.len; j++)
          for (int a = 0; a < 2; a++) value[a] += row[a].d[(S.layout.mask_first[i] + b) + (j << S.layout.h)] * d.mask.at(b, j);
    }
    drafts.push_back(std::move(d));
  }
  row[0].target = value[0] + off_target, row[1].target = value[1];

  Binder binder(1, plan);
  const Public<Codes> bound_codes = bind(binder, Inputs{notices, numbers_of(t, seed), {row[0].target, row[1].target}}).codes;
  ProverRounds ch(binder, random_bytes(seed + 2), proof);
  ProverContext P(bound_codes, ch, engine, coins);

  FoldMask fold_mask = fault.not_recorded ? fold_mask_make(S, coins) : fold_mask_make(S, P);
  // Where the stray write lands: a move of the FoldMask passes the vector's buffer on.
  E *const coefficients = const_cast<E *>(fold_mask.coefficients().data());
  const long tail = (long)fold_mask.coefficients().size();
  if (fault.zeroed >= 0 && fault.zeroed < tail && !fault.after_start) coefficients[fault.zeroed] = E::zero();
  BaseMask base_mask = base_mask_make(codes, 0, 0, 2, engine, coins);
  const size_t members = drafts.size();
  if (t.early) {  // kept members: no BASE; its mask, if sent here, would be opened with this tree
    drafts.push_back(fold_mask_draft(codes, S, fold_mask));
    if (!t.kept()) drafts.push_back(base_mask_draft(0, 0, 2, 0, base_mask));
  }
  const std::vector<OracleId> ids = send_oracles(P, std::move(drafts));
  if (t.early) fold_mask.sent_as(ids[members]);
  if (t.early && !t.kept()) base_mask.sent_as(ids[members + 1]);

  SingleRows rows;
  SingleRowsWitness witness;
  if (t.batched) {  // FOLD by its three steps here, not by `fold_prove`
    const E x = ch.close_algebraic(R_BAT_SIGMA).draw();
    FoldStart start = fold_start_prove(P, S, std::move(fold_mask));
    if (fault.exit_after_start) _exit(0);
    if (fault.zeroed >= 0 && fault.zeroed < tail && fault.after_start) coefficients[fault.zeroed].c[E_DEGREE - 1] = F::zero();
    if (fault.zeroed == tail && fault.after_start) (t.early ? P.behind[ids[members]] : P.behind.back()).mask_tail.set(0, 0, E::zero());
    fold_epsilon(ch, start);
    fold_rounds_prove(P, S, combine(row[0], row[1], x), std::move(start), rows, witness);
  } else {
    fold_prove(P, S, row[0], std::move(fold_mask), rows, witness);
  }
  CHECK(P.fold_draws.empty(), "FOLD left the numbers of its mask's draws in the context");
  Proved out;
  out.output_holds = single_rows_hold(P, rows, witness);
  out.claim = rows.target;
  if (t.kept()) {
    const PositionList at = ch.close_positions(R_LOC).positions(codes.base[0].log_n);
    out.symbols = single_rows_open(P, rows, at);
    // The same symbols, computed from the prover's own messages and random tails.
    const std::vector<E> main = table_row(engine.get(witness.main), 0), main_tail = table_row(witness.main_tail, 0);
    for (size_t i = 0; i < at.size(); i++) {
      out.encoded.push_back(symbol_at(codes.base[0], at[i], main.data(), main_tail.data()));
      for (size_t w = 0; w < t.members.size(); w++) {
        if (!t.members[w].kept) continue;
        const TreeBook::Place place = P.book.place(ids[w]);
        const Table m = engine.get(P.behind[ids[w]].main), tail = engine.tail(P.commit[place.tree], place.part);
        for (size_t b = 0; b < m.rows; b++)
          out.encoded.push_back(symbol_at(codes.base[0], at[i], table_row(m, b).data(), table_row(tail, b).data()));
      }
      for (unsigned j = 0; j < 2; j++)
        out.encoded.push_back(symbol_at(codes.mask[0], mask_position(codes.base[0], codes.mask[0], at[i]), witness.mask[j].data(), witness.mask_tail[j].data()));
    }
  } else {
    base_prove(P, rows, witness, std::move(base_mask), 0);
  }
  ch.finish();
  out.proof = proof.bytes();
  out.draws = coins.drawn.size();
  out.target[0] = row[0].target, out.target[1] = row[1].target;
  return out;
}

// What a verifier function passes out for the test to read after the verdict.
struct Left {
  E claim = E::zero();     // the target of the last FOLD's output claim
  std::vector<E> symbols;  // with kept members: the complex as read at the list
};
static const char *last_reason;
static E last_claim;
static std::vector<E> last_symbols;
static bool decide(VerifierRounds &ch, const Handles &in, Left &left, Notes *) {
  const Case t = case_of(ch.get(in.sent), ch.get(in.numbers));
  const u64 seed = ch.get(in.numbers)[3];
  const FoldShape S = toy_shape(t);
  VerifierContext V(in.codes, ch);
  GroupRow row[2] = {toy_row(t, S, seed + 3), toy_row(t, S, seed + 4)};
  row[0].target = ch.get(in.targets)[0], row[1].target = ch.get(in.targets)[1];

  std::vector<OracleNotice> notices = toy_notices(t);
  const size_t members = notices.size();
  if (t.early) {
    notices.push_back(fold_mask_notice(S));
    if (!t.kept()) notices.push_back(base_mask_notice(0, 0, 2, 0));
  }
  std::vector<OracleId> ids;
  if (!receive_oracles(V, notices, ids)) return false;
  FoldMask fold_mask;
  BaseMask base_mask;
  if (t.early) fold_mask.sent_as(ids[members]);
  if (t.early && !t.kept()) base_mask.sent_as(ids[members + 1]);

  SingleRows rows;
  if (t.batched) {
    const E x = ch.close_algebraic(R_BAT_SIGMA).draw();
    std::unique_ptr<FoldStart> start;
    if (!fold_start_verify(V, S, std::move(fold_mask), start)) return false;
    fold_epsilon(ch, *start);
    if (!fold_rounds_verify(V, S, combine(row[0], row[1], x), std::move(*start), rows)) return false;
  } else if (!fold_verify(V, S, row[0], std::move(fold_mask), rows)) {
    return false;
  }
  left.claim = rows.target;
  if (t.kept()) {
    const PositionList at = ch.close_positions(R_LOC).positions(V.st.codes().base[0].log_n);
    if (!single_rows_read(V, rows, at, left.symbols)) return false;
  } else if (!base_verify(V, rows, std::move(base_mask), 0)) {
    return false;
  }
  return true;
}
static bool verify(const Case &t, u64 seed, const E *target, const std::vector<u8> &proof) {
  Left left;
  const char *why = nullptr;
  const bool yes = run_verifier(1, toy_plan(t, toy_shape(t)), Inputs{toy_notices(t), numbers_of(t, seed), {target[0], target[1]}}, bind, decide, proof.data(), proof.size(), left, &why);
  last_reason = why ? why : "", last_claim = left.claim, last_symbols = left.symbols;
  return yes;
}

// ---- two folds in a row ----
// `link` in the names below means one of the two folds. The second group has three fresh rows over
// F with a mask row, and an oracle message of two rows over E that repeats what the first fold
// left: its folded main row, and in the second row its two folded mask rows one after the other.
// Nothing here ties this oracle message to the first group's oracles (that is the work of the code
// switch, blocks/iop/cs.h): the test is about the claim.
static const Case FIRST = {"first link", {{3, 1, 1, false}, {1, 3, 1, false}}, false, false};
static const Case SECOND = {"second link", {{3, 1, 1, false}, {2, 3, 0, false}}, false, false};
static FoldShape second_shape() {
  FoldShape S = toy_shape(SECOND);
  S.layout.members = {3, 4};  // after the first link's two members and its fresh mask
  return S;
}
static Plan two_links_plan() {
  Plan p(toy_setting(4));
  const Codes codes = toy_codes();
  bool second = false;
  for (const FoldShape &S : {toy_shape(FIRST), second_shape()}) {
    p.send_root("members");
    if (second) p.send_scalars("the second group's own target", 1);  // depends on the first fold's challenges: cannot be in the first query
    p.send_root("fold mask");
    p.send_scalars("hiding sum", 1);
    PlanRound &eps = p.close(R_FOLD_EPSILON, RoundKind::EPSILON, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
    const PricedGroup group = fold_group(second ? SECOND : FIRST, S);
    eps.draw_E = 1, eps.group = group;
    second = true;
    for (unsigned j = 0; j < S.layout.k; j++) {
      p.send_scalars("round polynomial", 2);
      PlanRound &round = p.close(R_FOLD_ROUND, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
      round.draw_E = 1, round.group = group;
    }
  }
  p.send_root("base mask");
  p.send_scalars("masked target", 1);
  PlanRound &lambda = p.close(R_BASE_LAMBDA, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
  lambda.draw_E = 1, lambda.group = base_group(2);
  p.send_scalars("masked word", codes.base[0].ell + codes.base[0].r + 2 * (codes.mask[0].ell + codes.mask[0].r));
  PlanRound &r = p.close(R_BASE, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  r.draw_positions = DRAWS, r.log_n = codes.base[0].log_n, r.group = base_group(2);
  return p;
}
// The second group's claim: a claim of its own plus x times the first fold's output claim, moved
// onto the rows that repeat the first fold's rows (rows 4 and 5 of the group).
static GroupRow second_row(const FoldShape &S, u64 seed, const E &own_target, const SingleRows &first, const E &x) {
  GroupRow row = toy_row(SECOND, S, seed + 5);
  row.target = own_target + x * first.target;
  Form carried = first.c;
  carried.insert_choice(0, S.layout.k, 4);
  for (unsigned j = 0; j < 2; j++)
    for (size_t i = 0; i < first.d[j].size(); i++) carried.add_entry(5 + ((j * first.d[j].size() + i) << S.layout.k), first.d[j][i]);
  row.c.add(carried, x);
  return row;
}
static std::vector<OracleNotice> two_links_notices() {
  std::vector<OracleNotice> all = toy_notices(FIRST);
  for (const OracleNotice &n : toy_notices(SECOND)) all.push_back(n);
  return all;
}
// A prover's coefficient vector written out, including terms that only the engine stores.
static std::vector<E> written_with(EngineCpu &engine, const Form &c, unsigned vars) {
  std::vector<E> out(size_t(1) << vars, E::zero());
  c.add_written_to(out.data(), vars);
  for (const EngineTerm &t : c.engine_terms) {
    const Table table = engine.get(t.vector);
    std::vector<E> w(table.entries());
    for (u64 j = 0; j < table.entries(); j++) w[j] = table.entry(j);
    for (u64 j = 0; j < out.size(); j++) out[j] += engine_term_at(t, w, j, vars);
  }
  return out;
}
// How the two folds keep their folded coefficient vectors in the engine. HINT(S): the output claim
// stays structured, with a hint that names the kept vector; EngineCpu aborts if a hint that a fold
// gives it differs from the structured terms. ALONE: the claim is the kept vector only. The second
// fold's is kept only to be compared: BASE writes its claim's vector out on the host, so a
// BOTH_ALONE run stops before BASE.
enum class Kept { FIRST_HINT, FIRST_ALONE, BOTH_HINTS, BOTH_ALONE };
static std::vector<E> two_links_c[2];  // of the last run: the two output claims' coefficient vectors, written out
// Moves every written-out factor of c's tensors into an engine table, which the tensor then names
// (Tensor::held). Returns the tables, for the caller to drop.
static std::vector<TableId> hold_factors(Form &c, Engine &engine) {
  std::vector<TableId> made;
  for (Tensor &v : c.tensors) {
    if (!v.values) continue;
    Table t(E_DEGREE, 1, v.values->size());
    for (size_t i = 0; i < t.len; i++) t.set(0, i, (*v.values)[i]);
    made.push_back(engine.put(t));
    v.values.reset();
    v.set_held(made.back(), v.first, t.len);
  }
  return made;
}
// targets_unknown: both claims get an arbitrary target, both shapes set target_from_the_rounds.
// factors_held: the claims' written-out factors are engine tables, dropped after their fold.
static Proved prove_two_links(u64 seed, Kept kept = Kept::FIRST_HINT, bool targets_unknown = false, bool factors_held = false) {
  const bool alone = kept == Kept::FIRST_ALONE || kept == Kept::BOTH_ALONE, both = kept == Kept::BOTH_HINTS || kept == Kept::BOTH_ALONE;
  FoldShape S1 = toy_shape(FIRST);
  S1.keep_folded_coefficients = true, S1.folded_coefficients_alone = alone, S1.target_from_the_rounds = targets_unknown;
  FoldShape S2 = second_shape();
  S2.keep_folded_coefficients = both, S2.folded_coefficients_alone = both && alone, S2.target_from_the_rounds = targets_unknown;
  const Plan plan = two_links_plan();
  const Codes codes = toy_codes();
  TestCoins coins(seed), data(seed + 1);
  EngineCpu engine;
  Writer proof;
  Proved out;

  // A group's oracle messages from tables, and the value of a claim's left side on them.
  auto make = [&](const Case &t, const FoldShape &S, std::vector<Table> main, const GroupRow &row, E &value) {
    const std::vector<E> c = main_form(S, row);
    const std::vector<OracleNotice> notices = toy_notices(t);
    std::vector<OracleDraft> drafts;
    for (size_t i = 0; i < notices.size(); i++) {
      OracleDraft d;
      d.shape = notices[i].shape, d.leaf_class = notices[i].leaf_class;
      for (size_t b = 0; b < main[i].rows; b++)
        for (size_t j = 0; j < main[i].len; j++) value += c[(S.layout.main_first[i] + b) + (j << S.layout.k)] * main[i].at(b, j);
      d.main = engine.put(main[i]);
      if (d.shape.mask_rows) {
        d.mask = Table(E_DEGREE, d.shape.mask_rows, codes.mask[0].ell);
        data.fill(d.mask.a.data(), d.mask.a.size());
        for (size_t b = 0; b < d.mask.rows; b++)
          for (size_t j = 0; j < d.mask.len; j++) value += row.d[(S.layout.mask_first[i] + b) + (j << S.layout.h)] * d.mask.at(b, j);
      }
      drafts.push_back(std::move(d));
    }
    return drafts;
  };
  auto random_table = [&](unsigned deg, size_t rows) {
    Table m(deg, rows, codes.base[0].ell);
    data.fill(m.a.data(), m.a.size());
    return m;
  };

  GroupRow row1 = toy_row(FIRST, S1, seed + 3);
  std::vector<OracleDraft> first = make(FIRST, S1, {random_table(1, 3), random_table(3, 1)}, row1, row1.target);
  Binder binder(1, plan);
  const Public<Codes> bound_codes = bind(binder, Inputs{two_links_notices(), {seed}, {row1.target}}).codes;
  ProverRounds ch(binder, random_bytes(seed + 2), proof);
  ProverContext P(bound_codes, ch, engine, coins);
  send_oracles(P, std::move(first));
  SingleRows rows1, rows2;
  SingleRowsWitness w1, w2;
  out.target[0] = row1.target;
  if (targets_unknown) row1.target = E::from(from_u64(12345));  // the first query above has the true target, like the verifier's
  std::vector<TableId> held_tables;
  if (factors_held) held_tables = hold_factors(row1.c, engine);
  out.factors = held_tables.size();
  fold_prove(P, S1, row1, fold_mask_make(S1, coins), rows1, w1);
  for (TableId t : held_tables) engine.drop(t);

  Table again(E_DEGREE, 2, codes.base[0].ell);
  const Table folded = engine.get(w1.main);
  for (size_t i = 0; i < again.len; i++) again.set(0, i, folded.at(0, i));
  for (unsigned j = 0; j < 2; j++)
    for (size_t i = 0; i < w1.mask[j].size(); i++) again.set(1, j * w1.mask[j].size() + i, w1.mask[j][i]);
  E own = E::zero();
  send_oracles(P, make(SECOND, S2, {random_table(1, 3), again}, toy_row(SECOND, S2, seed + 5), own));
  ch.send(own);
  out.target[1] = E::zero();  // not used: the verifier reads the second target from the proof
  GroupRow row2 = second_row(S2, seed, own, rows1, data.e());
  if (targets_unknown) row2.target = E::from(from_u64(67890));
  if (factors_held) held_tables = hold_factors(row2.c, engine);
  out.factors += factors_held ? held_tables.size() : 0;
  fold_prove(P, S2, std::move(row2), fold_mask_make(S2, coins), rows2, w2);
  if (factors_held)
    for (TableId t : held_tables) engine.drop(t);
  out.output_holds = single_rows_hold(P, rows1, w1) && single_rows_hold(P, rows2, w2) && rows1.c.held.size() == (alone ? 0u : 1u) &&
                     rows1.c.engine_terms.size() == (alone ? 1u : 0u) && rows1.c.tensors.empty() == alone && rows2.c.held.size() == (both && !alone ? 1u : 0u) &&
                     rows2.c.engine_terms.size() == (both && alone ? 1u : 0u);
  two_links_c[0] = written_with(engine, rows1.c, codes.base[0].log_ell()), two_links_c[1] = written_with(engine, rows2.c, codes.base[0].log_ell());
  engine.drop(fold_kept_coefficients(rows1.c));
  out.claim = rows2.target;
  if (both) engine.drop(fold_kept_coefficients(rows2.c));
  if (both && alone) {  // BOTH_ALONE: no BASE (see `Kept`); the proof so far
    out.proof = proof.bytes();
    return out;
  }
  base_prove(P, rows2, w2, base_mask_make(codes, 0, 0, 2, engine, coins), 0);
  ch.finish();
  out.proof = proof.bytes();
  return out;
}
static bool decide_two_links(VerifierRounds &ch, const Handles &in, Left &left, Notes *) {
  const FoldShape S1 = toy_shape(FIRST), S2 = second_shape();
  const u64 seed = ch.get(in.numbers)[0];
  VerifierContext V(in.codes, ch);
  // x is test data, not a challenge: as in prove_two_links(), the next value after all the
  // prover's tables.
  TestCoins data(seed + 1);
  const Codes codes = toy_codes();
  std::vector<F> skip((3 + 3 + 3) * codes.base[0].ell + 3 * E_DEGREE * codes.mask[0].ell);
  data.fill(skip.data(), skip.size());
  const E x = data.e();

  std::vector<OracleId> ids;
  SingleRows rows1, rows2;
  GroupRow row1 = toy_row(FIRST, S1, seed + 3);
  row1.target = ch.get(in.targets)[0];
  if (!receive_oracles(V, toy_notices(FIRST), ids) || !fold_verify(V, S1, row1, FoldMask(), rows1)) return false;
  E own;
  if (!receive_oracles(V, toy_notices(SECOND), ids) || !ch.recv(own) || !fold_verify(V, S2, second_row(S2, seed, own, rows1, x), FoldMask(), rows2))
    return false;
  left.claim = rows2.target;
  return base_verify(V, rows2, BaseMask(), 0);
}
static bool verify_two_links(u64 seed, const E *target, const std::vector<u8> &proof, E &claim) {
  Left left;
  const bool yes = run_verifier(1, two_links_plan(), Inputs{two_links_notices(), {seed}, {target[0]}}, bind, decide_two_links, proof.data(), proof.size(), left);
  claim = left.claim;
  return yes;
}

// Runs f in a child process. True if the child died of SIGABRT and its stderr contains `text`.
template <class Fn>
static bool aborts_saying(const char *text, Fn f) {
  std::fflush(stdout);
  int words[2];
  if (pipe(words) != 0) return false;
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    dup2(words[1], 2);
    f();
    _exit(0);
  }
  close(words[1]);
  std::string said;
  char buffer[256];
  for (ssize_t n; (n = read(words[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(words[0]);
  int status = 0;
  return waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT &&
         said.find(text) != std::string::npos;
}

// A verifier that starts FOLD `folds` times, each time with the same early-sent oracle as its mask.
static void one_mask_for_folds(int folds, const Case &t, u64 seed, const Proved &p) {
  const FoldShape S = toy_shape(t);
  const Plan plan = toy_plan(t, S);
  Reader reader(p.proof.data(), p.proof.size());
  Binder binder(1, plan);
  const Public<Codes> bound_codes = bind(binder, Inputs{toy_notices(t), numbers_of(t, seed), {p.target[0], p.target[1]}}).codes;
  VerifierSession session;  // one verdict, one chain
  VerifierRounds ch(binder, reader, session);
  VerifierContext V(bound_codes, ch);
  std::vector<OracleNotice> notices = toy_notices(t);
  notices.push_back(fold_mask_notice(S));
  notices.push_back(base_mask_notice(0, 0, 2, 0));
  std::vector<OracleId> ids;
  receive_oracles(V, notices, ids);
  for (int i = 0; i < folds; i++) {
    FoldMask mask;
    mask.sent_as(ids[ids.size() - 2]);
    std::unique_ptr<FoldStart> start;
    fold_start_verify(V, S, std::move(mask), start);
  }
  std::abort();  // reached with one fold only
}

// A faulty verifier on an honest proof. `which` = 0: names a group member as FOLD's fresh mask.
// 1: names FOLD's fresh mask as BASE's masking oracle. 2: calls BASE after a FOLD with kept rows.
static void wrong_program(int which, const Case &t, u64 seed, const Proved &p) {
  const FoldShape S = toy_shape(t);
  const Plan plan = toy_plan(t, S);
  Reader reader(p.proof.data(), p.proof.size());
  Binder binder(1, plan);
  const Public<Codes> bound_codes = bind(binder, Inputs{toy_notices(t), numbers_of(t, seed), {p.target[0], p.target[1]}}).codes;
  VerifierSession session;  // one verdict, one chain
  VerifierRounds ch(binder, reader, session);
  VerifierContext V(bound_codes, ch);
  GroupRow row = toy_row(t, S, seed + 3);
  row.target = p.target[0];
  std::vector<OracleId> ids;
  if (!receive_oracles(V, toy_notices(t), ids)) return;
  FoldMask fold_mask;
  BaseMask base_mask;
  SingleRows rows;
  if (which == 0) fold_mask.sent_as(ids[0]);
  if (!fold_verify(V, S, row, std::move(fold_mask), rows)) return;
  if (which == 1) base_mask.sent_as(rows.fresh);
  (void)base_verify(V, rows, std::move(base_mask), 0);
}

// ---- BASE alone ----
// On the plain rows of one oracle message: one main row over E and at most one mask row.
static SingleRows plain_rows(unsigned masks, u64 seed) {
  TestCoins data(seed + 3);
  const Codes codes = toy_codes();
  SingleRows S;
  S.code = 0, S.mask_code = 0, S.masks = masks;
  S.layout.members = {0}, S.layout.kept = {false}, S.layout.main_first = {0};
  S.layout.mask_first = {masks ? 0 : GroupLayout::NONE};
  S.layout.log_ell = ceil_log2(codes.base[0].ell), S.layout.log_ell_zk = ceil_log2(codes.mask[0].ell);
  std::vector<E> z(S.layout.log_ell);
  for (E &x : z) x = data.e();
  S.c.add(Tensor::eq(z.data(), (unsigned)z.size(), data.e()));
  S.c.add_entry(3, data.e());
  if (masks) {
    S.d.emplace_back(codes.mask[0].ell);
    for (E &x : S.d[0]) x = data.e();
  }
  return S;
}
static Plan base_alone_plan(unsigned masks) {
  Plan p(toy_setting(4));
  const Codes codes = toy_codes();
  p.send_root("rows"), p.send_root("base mask");
  p.send_scalars("masked target", 1);
  PlanRound &lambda = p.close(R_BASE_LAMBDA, RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound((u64)0), Priced::NOTHING);
  lambda.draw_E = 1, lambda.group = base_group(masks);
  p.send_scalars("masked word", codes.base[0].ell + codes.base[0].r + masks * (codes.mask[0].ell + codes.mask[0].r));
  PlanRound &r = p.close(R_BASE, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  r.draw_positions = DRAWS, r.log_n = codes.base[0].log_n, r.group = base_group(masks);
  return p;
}
static OracleNotice plain_notice(unsigned masks) {
  OracleNotice n;
  n.shape.name = "rows", n.shape.code = 0, n.shape.main_rows = 1, n.shape.main_deg = E_DEGREE;
  n.shape.mask_rows = masks, n.shape.mask_code = 0, n.shape.answers = 0;
  return n;
}
static bool decide_base_alone(VerifierRounds &ch, const Handles &in, Notes *) {
  VerifierContext V(in.codes, ch);
  SingleRows T = plain_rows((unsigned)ch.get(in.numbers)[0], ch.get(in.numbers)[1]);
  T.target = ch.get(in.targets)[0];
  std::vector<OracleId> ids;
  return receive_oracles(V, ch.get(in.sent), ids) && base_verify(V, T, BaseMask(), 0);
}
// An engine that loses the masking word M* (the main row's message in BASE's masking oracle) before
// the oracle is committed. All that is sent then still agrees. This was run on a copy of the
// library without the comparison named next: the verifier accepted, and the prover had sent
// lambda M, its main row's message times the challenge, unmasked. So base_prove() compares M* as
// the engine returns it with M* as drawn, and aborts if they differ. No test of this repository
// takes the comparison out to show that this test then fails.
// lose_put: the put() call with this number, counted from 0, stores zeros instead of its table.
struct LosingEngine : EngineCpu {
  int lose_put = -1, puts = 0;
  TableId put(const Table &t) override { return EngineCpu::put(puts++ == lose_put ? Table(t.deg, t.rows, t.len) : t); }
};
// lost: the engine loses the masking word; base_prove() must abort.
// loses: `Recalling::loses` for the prover's `coins`. Returns how many draws of field elements
// they made.
static size_t base_alone(unsigned masks, u64 seed, bool lost = false, long loses = -1) {
  const Plan plan = base_alone_plan(masks);
  const Codes codes = toy_codes();
  Recalling coins(seed);
  coins.loses = loses;
  TestCoins data(seed + 1);
  LosingEngine engine;
  if (lost) engine.lose_put = 1;  // table 0 is the data's, 1 the masking word's: checked below
  Writer proof;
  SingleRows S = plain_rows(masks, seed);
  OracleDraft d;
  d.shape = plain_notice(masks).shape;
  Table main(E_DEGREE, 1, codes.base[0].ell);
  data.fill(main.a.data(), main.a.size());
  d.main = engine.put(main);
  if (masks) {
    d.mask = Table(E_DEGREE, 1, codes.mask[0].ell);
    data.fill(d.mask.a.data(), d.mask.a.size());
  }
  S.target = engine.evaluate(d.main, S.c) + (masks ? inner(S.d[0], table_row(d.mask, 0)) : E::zero());
  Binder binder(1, plan);
  const Public<Codes> bound_codes = bind(binder, Inputs{{plain_notice(masks)}, {masks, seed}, {S.target}}).codes;
  ProverRounds pch(binder, random_bytes(seed + 2), proof);
  ProverContext P(bound_codes, pch, engine, coins);
  const OracleId o = send_oracles(P, {d})[0];
  SingleRowsWitness W;
  W.main = P.behind[o].main, W.main_tail = engine.tail(P.commit[P.book.place(o).tree], P.book.place(o).part);
  if (masks) W.mask = {table_row(P.behind[o].mask, 0)}, W.mask_tail = {table_row(P.behind[o].mask_tail, 0)};
  CHECK(engine.puts == 1, "BASE alone: the masking word's table is not the second that the engine gets: the test with a lost word would show nothing");
  base_prove(P, S, W, base_mask_make(codes, 0, 0, masks, engine, coins), 0);
  pch.finish();
  for (int wrong = 0; wrong < 2; wrong++) {
    const bool yes = run_verifier(1, plan, Inputs{{plain_notice(masks)}, {masks, seed}, {wrong ? S.target + E::one() : S.target}}, bind, decide_base_alone, proof.data(), proof.size());
    CHECK(yes == !wrong, wrong ? "BASE alone: a false claim is accepted" : "BASE alone: an honest proof is refused");
  }
  return coins.drawn.size();
}

// Runs f in a child process. 0: the child died of SIGABRT and its stderr contains `text`.
// 1: f returned true. 2: anything else (another signal or message; f returned false).
template <class Fn>
static int stopped_or_through(const char *text, Fn f) {
  std::fflush(stdout);
  int words[2];
  if (pipe(words) != 0) return 2;
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    dup2(words[1], 2);
    _exit(f() ? 0 : 3);
  }
  close(words[1]);
  std::string said;
  char buffer[256];
  for (ssize_t n; (n = read(words[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(words[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child) return 2;
  if (WIFSIGNALED(status)) return WTERMSIG(status) == SIGABRT && said.find(text) != std::string::npos ? 0 : 2;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 1 : 2;
}
// Here the plan is wrong, not the prover. Beside each round listed at `bent_number` a plan records
// the group for which that round's error was computed (`PricedGroup`). FOLD and BASE compare it
// with the group on which they run, by require_group() or, where positions are drawn,
// require_codes() (blocks/chain/rounds.h). bend() makes one of ten changes to the recorded group at
// one of these four places, and both prover and verifier must then abort in that comparison. The
// recorded group enters no query, so an honest proof takes the verifier up to the altered round.
// This repository has no negative control of this test: one would replace one of the two
// comparisons by `true`, and the runs altered at its rounds must then go through.
// Taking out a block's call instead would abort honest runs too: the
// channel demands the call wherever the plan records a group (tested in
// blocks/chain/tests/rounds_test.cpp).
static void other_group_test(const std::vector<Case> &with, size_t &runs, size_t &refused) {
  for (const Case &t : with) {
    const Proved p = prove(t, 100, E::zero());
    for (bent_number = 0; bent_number < 40; bent_number++) {
      if (t.kept() && bent_number >= 10 && bent_number < 30) continue;  // kept members: no BASE, so none of its rounds
      const char *const words = bent_number >= 20 && bent_number < 30 ? "draws positions on rows of other codes" : "begins a stage on another group";
      for (bool prover : {true, false}) {
        const int how = stopped_or_through(words, [&] { return prover ? !prove(t, 100, E::zero()).proof.empty() : verify(t, 100, p.target, p.proof); });
        runs++, refused += how == 0;
        if (how == 1) std::printf("ACCEPTED a plan that is priced for another group than runs, or names none: the %s's side ran through, bend %d: %s\n", prover ? "prover" : "verifier", bent_number, t.name);
        CHECK(how != 2, "a run under a bent plan neither ended at the block's question nor ran through: another stop, or a refusal");
      }
    }
    bent_number = -1;
    CHECK(verify(t, 100, p.target, p.proof), "the control: under the plan as it is the honest proof is refused");
  }
}

// A case: {name, {{main_rows, main_deg, mask_rows, kept}, ...}, early, batched, work}.
// `runs` counts the runs with a changed proof byte or an altered plan, `refused` those that were
// caught. Returns 0 only if no CHECK failed; the last asks that the counts are equal.
int main() {
  const std::vector<Case> cases = {
      {"k=1 h=0, no work", {{2, 1, 1, false}}, false, false, 0},
      {"k=1 h=1", {{1, 3, 2, false}}, false, false},
      {"k=3 h=1, two members", {{3, 1, 1, false}, {1, 3, 1, false}}, false, false},
      {"k=3 h=1, two members, early masks", {{3, 1, 1, false}, {1, 3, 1, false}}, true, false},
      {"k=3 h=1, two members, two claims batched", {{3, 1, 1, false}, {1, 3, 1, false}}, false, true},
      {"k=2 h=2", {{4, 1, 3, false}, {0, 1, 1, false}}, false, false},
      {"k=2 h=0, no mask rows sent, early masks, two claims batched", {{4, 1, 0, false}}, true, true},
      {"k=2 h=0, one kept row", {{3, 1, 1, false}, {1, 1, 0, true}}, false, false},
      {"k=1 h=1, three kept rows in two members", {{2, 1, 1, false}, {2, 1, 1, true}, {1, 1, 0, true}}, false, false},
      {"k=2 h=0, one kept row, early masks, two claims batched", {{3, 1, 1, false}, {1, 1, 0, true}}, true, true},
  };
  size_t runs = 0, refused = 0;
  for (const Case &t : cases)
    for (u64 seed = 100; seed < 104; seed++) {
      const Proved p = prove(t, seed, E::zero());
      CHECK(p.output_holds, "FOLD's output claim does not hold for the folded rows");
      CHECK(verify(t, seed, p.target, p.proof), "an honest proof is refused");
      CHECK(last_claim == p.claim, "prover and verifier leave FOLD with different claims");
      if (t.kept()) {
        CHECK(p.symbols == p.encoded && !p.symbols.empty(), "the complex read at a position is not the encoding of the prover's rows");
        CHECK(last_symbols == p.symbols, "prover and verifier read different symbols");
        const Proved g = prove(t, seed, E::one());
        CHECK(!g.output_holds, "a false target gives a true output claim");
        // No BASE follows, so nothing checks FOLD's output claim: a changed proof must be rejected
        // or lead to another claim or other symbols.
        for (size_t i = 0; seed == 100 && i < p.proof.size(); i++) {
          std::vector<u8> changed = p.proof;
          changed[i] ^= (u8)(1u << (i % 8));
          runs++;
          if (!verify(t, seed, p.target, changed) || last_claim != p.claim || last_symbols != p.symbols) refused++;
          else std::printf("a proof with byte %zu changed went unnoticed: %s\n", i, t.name);
        }
        if (seed == 100 && !t.early && !t.batched) CHECK(aborts_saying("the complex has kept rows", [&] { wrong_program(2, t, seed, p); }), "BASE took a complex with kept rows");
        continue;
      }
      const E other[2] = {p.target[0] + E::one(), p.target[1]};
      CHECK(!verify(t, seed, other, p.proof), "an honest proof is accepted for another target");
      const Proved f = prove(t, seed, E::one());
      CHECK(!f.output_holds, "a false target gives a true output claim");
      CHECK(!verify(t, seed, f.target, f.proof), "a false target is accepted");
      if (seed == 100) {
        for (size_t i = 0; i < p.proof.size(); i++) {
          std::vector<u8> changed = p.proof;
          changed[i] ^= (u8)(1u << (i % 8));
          runs++;
          if (!verify(t, seed, p.target, changed)) refused++;
          else std::printf("ACCEPTED a proof with byte %zu changed: %s\n", i, t.name);
        }
        std::vector<u8> shorter(p.proof.begin(), p.proof.end() - 1), longer = p.proof;
        longer.push_back(0);
        CHECK(!verify(t, seed, p.target, shorter), "a proof cut short is accepted");
        CHECK(!verify(t, seed, p.target, longer), "a proof with a byte added is accepted");
        if (!t.early && !t.batched) {
          CHECK(aborts_saying("named as the fresh mask", [&] { wrong_program(0, t, seed, p); }), "a member of the group served as its fresh mask");
          CHECK(aborts_saying("named as the masking oracle", [&] { wrong_program(1, t, seed, p); }), "FOLD's fresh mask served as BASE's masking oracle");
        }
        if (t.early) {
          CHECK(aborts_saying("serves one block", [&] { one_mask_for_folds(2, t, seed, p); }), "one oracle served as the fresh mask of two folds");
          CHECK(!aborts_saying("serves one block", [&] { one_mask_for_folds(1, t, seed, p); }), "the test of a mask used twice stops for another reason");
        }
      }
    }
  other_group_test({cases[2], cases[8]}, runs, refused);
  for (u64 seed = 300; seed < 304; seed++) base_alone(0, seed), base_alone(1, seed);
  for (unsigned masks : {0u, 1u})  // `base_prove` compares the masking word where it uses it
    CHECK(aborts_saying("the masking word that the engine gives back is not the one that was drawn", [&] { base_alone(masks, 300, true); }), "BASE sent its word under a masking message that was lost");
  // base_mask_make() copies the masking word after its draw, so a value lost in between is the same
  // in word and copy, and the comparison tested above passes. base_prove() therefore also compares
  // what it uses with the draws themselves (not_the_coins()). Counted: the draws whose lost value
  // makes the run abort with "own-test-base-mask": exactly 1 without a mask row (the masking
  // word's), 3 with one (also those of the masking oracle's mask row and of its random tail).
  for (unsigned masks : {0u, 1u}) {  // each draw in turn loses a value (`Recalling`)
    const size_t draws = base_alone(masks, 300);
    unsigned caught = 0;
    for (size_t d = 0; d < draws; d++) caught += aborts_saying("own-test-base-mask", [&] { base_alone(masks, 300, false, (long)d); });
    std::printf("BASE alone, %u mask rows: of %zu draws of elements %u end by own-test-base-mask when they lose a value\n", masks, draws, caught);
    CHECK(caught == (masks ? 3u : 1u), "BASE sent its word under a masking word that was lost right behind its draw, or the look cries at another draw");
  }
  // FOLD's self-check of its fresh mask (own-test-fold-mask), where FOLD sends the mask itself
  // (cases[4]) and where it was sent ahead (cases[6]). The honest runs above have passed it.
  for (const Case &t : {cases[4], cases[6]}) {
    const long coefficients = 2 * (long)toy_shape(t).layout.k + 1;
    auto caught = [&](const Fault &fault) { return aborts_saying("own-test-fold-mask", [&] { (void)prove(t, 100, E::zero(), fault); }); };
    // Each draw in turn loses a value (`Recalling`). The child exits after fold_start_prove(), so
    // only the check there can end it: it must, for each coefficient's draw (the run's first draws)
    // and for the draw of the mask rows' random tail, and for no other draw.
    const Proved honest = prove(t, 100, E::zero());
    long ended = 0, ended_at_a_coefficient = 0;
    for (long d = 0; d < (long)honest.draws; d++)
      if (caught(Fault{d, -1, false, true})) ended++, ended_at_a_coefficient += d < coefficients;
    std::printf("FOLD, %s: of %zu draws of elements %ld end by own-test-fold-mask in FOLD's first step when they lose a value\n", t.name, honest.draws, ended);
    CHECK(ended == coefficients + 1 && ended_at_a_coefficient == coefficients, "FOLD sent its sum with a coefficient or a tail value that was lost right after its draw, or the self-check fails at another draw");
    for (long i : {0L, coefficients / 2, coefficients - 1}) {  // stray writes
      CHECK(caught(Fault{-1, i, false, true}), "FOLD sent its sum with a coefficient that was overwritten after fold_mask_make()");
      CHECK(caught(Fault{-1, i, true, false}), "FOLD ran its rounds with a coefficient that was overwritten after fold_start_prove()");
      // Why the check is there: without numbers nothing is compared, and the verifier accepts the
      // proof that is made with the coefficient zeroed before the table.
      const Proved lost = prove(t, 100, E::zero(), Fault{-1, i, false, false, true});
      CHECK(lost.proof != honest.proof && verify(t, 100, lost.target, lost.proof), "the control: a coefficient zeroed before the table is made should give another proof, which verifies");
    }
    CHECK(caught(Fault{-1, coefficients, true, false}), "FOLD ran its rounds with a tail value that was overwritten after fold_start_prove()");
    CHECK(caught(Fault{-1, -1, false, true, false, true}), "FOLD sent its sum for a mask whose draws' numbers were not recorded, though the generator numbers its draws");
  }
  for (u64 seed = 200; seed < 204; seed++) {
    const Proved p = prove_two_links(seed);
    E claim;
    CHECK(p.output_holds, "two links: an output claim does not hold for the folded rows");
    CHECK(verify_two_links(seed, p.target, p.proof, claim), "two links: an honest proof is refused");
    CHECK(claim == p.claim, "two links: prover and verifier end with different claims");
    const E other[2] = {p.target[0] + E::one(), p.target[1]};
    CHECK(!verify_two_links(seed, other, p.proof, claim), "two links: a false first claim is accepted");
    // However the prover stores its folded coefficients (`Kept`), both output claims have the same
    // coefficient vectors, entry by entry, and the proof has the same bytes (BOTH_ALONE: a prefix).
    const std::vector<E> c1 = two_links_c[0], c2 = two_links_c[1];
    for (Kept kept : {Kept::FIRST_ALONE, Kept::BOTH_HINTS, Kept::BOTH_ALONE}) {
      const Proved q = prove_two_links(seed, kept);
      CHECK(q.output_holds, "two links: an output claim does not hold, or is not written as asked");
      CHECK(two_links_c[0] == c1 && two_links_c[1] == c2, "two links: the kept vector and the structured writing differ in an entry");
      CHECK(q.claim == p.claim && q.proof.size() <= p.proof.size() && std::equal(q.proof.begin(), q.proof.end(), p.proof.begin()) &&
                (kept == Kept::BOTH_ALONE || q.proof.size() == p.proof.size()),
            "two links: the proof's bytes hang on how the prover writes its coefficient vectors");
    }
    // A prover that does not know its claims' targets takes the running claim from its first round
    // polynomial (FoldShape::target_from_the_rounds): same bytes, same output claim.
    for (Kept kept : {Kept::FIRST_HINT, Kept::FIRST_ALONE}) {
      const Proved q = prove_two_links(seed, kept, true);
      CHECK(q.output_holds && q.claim == p.claim && q.proof == p.proof, "two links: without the rows' targets the prover sends other bytes, or leaves another claim");
    }
    {  // factors_held, with FIRST_ALONE so that no fold fixes a variable of its claim on the host
      const Proved q = prove_two_links(seed, Kept::FIRST_ALONE, false, true);
      CHECK(q.factors == 2 && q.output_holds && q.claim == p.claim && q.proof == p.proof, "two links: with the written-out factors in the engine's hands the prover sends other bytes, or leaves another claim");
    }
    // Host code that needs an entry of a factor (`Tensor::held`) or a term (`EngineTerm`) that only
    // the engine stores ("holds", "knows" in the messages) must abort, and so must a verifier's
    // statement (`Statement`, blocks/iop/relation.h) that is given a claim with one.
    if (seed == 200) {
      auto held_tensor = [](TableId id, u64 length) {  // 4 entries of the table under a choice variable that picks the odd indices
        Tensor t;
        t.push_dense(2);
        t.set_held(id, 0, length);
        t.insert_choice(0, 1, 1);
        return t;
      };
      CHECK(is_zero(held_tensor(5, 4).at(0)) && is_zero(held_tensor(5, 4).at(6)), "outside the picked bits a tensor with a factor in the engine's hands is not 0");
      CHECK(aborts_saying("only the engine holds", [&] { (void)held_tensor(5, 4).at(3); }), "an entry was read of a factor that only the engine holds");
      CHECK(aborts_saying("only the engine holds", [&] {
              Tensor t = held_tensor(5, 4);
              const E g[2] = {E::one(), E::one()};
              t.fix_low(g, 2);
            }), "a variable was fixed in a factor that only the engine holds");
      CHECK(!aborts_saying("", [&] {  // the choice variable alone: no entry is needed
              Tensor t = held_tensor(5, 4);
              const E g = E::one();
              t.fix_low(&g, 1);
              if (!t.is_held() || t.vars() != 2) std::abort();
            }), "fixing a choice variable of a tensor with a factor in the engine's hands ended the process");
      CHECK(aborts_saying("only the engine holds", [&] {
              Form f;
              f.add(held_tensor(5, 4));
              const E g[2] = {E::one(), E::one()};
              f.fix_low(g, 2);
            }), "a variable was fixed in a form's factor that only the engine holds");
      {  // `mle`: with dense variables, and for a factor of one table entry (no dense variable).
         // Fixing variables never reaches such a factor: without a check of its own, mle() was seen
         // to return the tensor's value without the factor.
        const E x[3] = {E::one(), E::one(), E::one()};
        CHECK(aborts_saying("only the engine holds", [&] { (void)held_tensor(5, 4).mle(x); }), "the value at a point was given of a factor that only the engine holds");
        Tensor one_entry;
        one_entry.set_held(5, 2, 4);  // entry 2 of the table
        one_entry.insert_choice(0, 1, 1);
        CHECK(one_entry.is_held() && one_entry.dense_vars.empty() && one_entry.vars() == 1, "the test's tensor is no held factor of one entry under one variable: the next line would show nothing");
        CHECK(aborts_saying("only the engine holds", [&] { (void)one_entry.mle(x); }), "the value at a point was given of a factor of one entry that only the engine holds");
      }
      CHECK(aborts_saying("only the engine holds", [&] {
              std::vector<E> out(8, E::zero());
              held_tensor(5, 4).add_to(out.data());
            }), "a factor that only the engine holds was written out");
      CHECK(aborts_saying("only the engine holds", [&] {
              Form f;
              f.add(held_tensor(5, 4));
              f.freeze(3);
            }), "a form with a factor that only the engine holds was frozen");
      CHECK(aborts_saying("the slice does not lie in the vector", [&] { (void)held_tensor(5, 3); }), "a factor that reaches beyond the length it names was taken");
      CHECK(aborts_saying("another length than the factor says", [&] {  // the table has 4 entries
              EngineCpu e;
              const TableId data = e.put(Table(E_DEGREE, 1, 8)), factor = e.put(Table(E_DEGREE, 1, 4));
              (void)e.evaluate(data, held_tensor(factor, 8));
            }), "the engine read a factor whose table is shorter than the tensor says");
      CHECK(aborts_saying("only a prover's engine holds", [&] {  // in a verifier's statement
              Statement st(toy_codes());
              st.has_no_engine();
              const GroupId g = st.new_group("a group", 0);
              Form f;
              f.add(held_tensor(5, 4));
              Row row;
              row.add_form(0, std::move(f));
              st.add_row(g, std::move(row));
            }), "a verifier's statement took a row with a factor that only an engine holds");
      CHECK(aborts_saying("only a prover's engine holds", [&] {  // the same for a row's plain terms
              Statement st(toy_codes());
              st.has_no_engine();
              const GroupId g = st.new_group("a group", 0);
              Row row;
              row.terms.push_back(Term{0, Part::MAIN, held_tensor(5, 4)});
              st.add_row(g, std::move(row));
            }), "a verifier's statement took a row with a term whose factor only an engine holds");
      // No call of has_no_engine() here: VerifierContext's constructor makes it.
      CHECK(aborts_saying("only a prover's engine holds", [&] {  // VerifierContext's own statement
              const Case &t = cases[0];
              const Plan plan = toy_plan(t, toy_shape(t));
              const E none[2] = {E::zero(), E::zero()};
              Reader reader(nullptr, 0);
              Binder binder(1, plan);
              const Public<Codes> bound_codes = bind(binder, Inputs{toy_notices(t), numbers_of(t, 100), {none[0], none[1]}}).codes;
              VerifierSession session;
              VerifierRounds ch(binder, reader, session);
              VerifierContext V(bound_codes, ch);
              const GroupId g = V.st.new_group("a group", 0);
              Form f;
              f.add(held_tensor(5, 4));
              Row row;
              row.add_form(0, std::move(f));
              V.st.add_row(g, std::move(row));
            }), "a verifier's context took a row with a factor that only an engine holds");
      CHECK(aborts_saying("only a prover's engine holds", [&] {  // and in a statement that was moved, as a later chain's context takes it over
              Statement first(toy_codes());
              first.has_no_engine();
              Statement st(std::move(first));
              const GroupId g = st.new_group("a group", 0);
              Form f;
              f.engine_terms.push_back(EngineTerm{0, E::one(), 0, 0});
              Row row;
              row.add_form(0, std::move(f));
              st.add_row(g, std::move(row));
            }), "a verifier's statement took a row with a term that only an engine knows");
      CHECK(aborts_saying("only the engine knows", [&] {
              Form f;
              f.engine_terms.push_back(EngineTerm{0, E::one(), 0, 0});
              (void)f.at(0);
            }), "an entry was read of a form with a term that only the engine knows");
      CHECK(aborts_saying("only the engine knows", [&] {  // under choice variables too, at an index that they pick
              Form f;
              f.engine_terms.push_back(EngineTerm{0, E::one(), 0, 0});
              f.insert_choice(0, 2, 2);
              (void)f.at(2 + 4 * 1);
            }), "an entry inside an engine's term was read under choice variables");
      CHECK(aborts_saying("only the engine knows", [&] {
              Form f;
              f.engine_terms.push_back(EngineTerm{0, E::one(), 0, 0});
              const E g = E::one();
              f.fix_low(&g, 1);
            }), "variables were fixed in a form with a term that only the engine knows");
      CHECK(aborts_saying("only the engine knows", [&] {
              Form f;
              f.engine_terms.push_back(EngineTerm{0, E::one(), 0, 0});
              std::vector<E> out(2, E::zero());
              f.add_to(out.data(), 1);
            }), "a form with a term that only the engine knows was written out");
    }
  }
  CHECK(refused == runs, "something that had to be refused was accepted");
  std::printf("fold_base_test: %s (%zu of %zu refused)\n", failures ? "FAILED" : "ok", refused, runs);
  return failures ? 1 : 0;
}
