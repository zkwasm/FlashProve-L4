// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shared fixtures (`world` in the file's name and the #error text) for the tests of three protocol
// blocks: SEND (sends oracle messages and samples them out of domain), BAT (batches linear claims
// into one), SPLIT (splits a claim on two groups).
//
// These blocks check no claim themselves: they pass on linear claims (`Row`, `GroupRow`: a linear
// form of the encoded messages has a public value), and a proximity test on a group of oracle
// messages (encoded rows under Merkle roots) decides. The tests decide in another way, without
// the protocol's proximity test: the test closing (test_close_prove(), test_close_verify()),
// described below.
// An including test must define BLK_PLANS_BY_HAND_FOR_A_TEST before its first include: without it
// the members of `Plan` (blocks/numbers/plan.h) that plan_test_close() calls are private.
#pragma once
#ifndef BLK_PLANS_BY_HAND_FOR_A_TEST
#error "this world lays plans by hand: say #define BLK_PLANS_BY_HAND_FOR_A_TEST before the first include (numbers/plan.h)"
#endif
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "chain/closed.h"
#include "engine/engine_cpu.h"
#include "iop/bat.h"
#include "iop/context.h"
#include "iop/send.h"
#include "iop/tests/toy_list.h"
#include "numbers/blocks.h"
#include "toy/toy_coins.h"

namespace blk {

inline Setting test_setting() {
  return setting_goldilocks();
}

// ---- an engine that keeps what it commits to; a malicious prover can commit to non-codewords ----
class PlantEngine : public EngineCpu {
 public:
  struct Kept {
    std::vector<OraclePart> parts;
    std::vector<Table> main_tail;     // part by part
    std::vector<ToyRows> main, mask;  // the committed words, part by part; a row over E is three rows over F
    std::unique_ptr<OracleTree> tree;
  };
  // Called for every part (one oracle message of a tree) before it is hashed; may change the words.
  // `commit` counts the calls of commit() from 0.
  std::function<void(size_t commit, size_t part, ToyRows &main, ToyRows &mask)> plant;
  std::vector<Kept> kept;

  static ToyRows encode_rows(const Code &code, const Table &msg, const Table &tail) {
    ToyRows out;
    std::vector<F> m(code.ell), t(code.r);
    for (size_t b = 0; b < msg.rows; b++)
      for (unsigned d = 0; d < msg.deg; d++) {
        for (size_t i = 0; i < code.ell; i++) m[i] = msg.a[(b * code.ell + i) * msg.deg + d];
        for (size_t i = 0; i < code.r; i++) t[i] = tail.a[(b * code.r + i) * msg.deg + d];
        out.emplace_back(code.n());
        encode(code, m.data(), t.data(), out.back().data());
      }
    return out;
  }
  CommitId commit(const std::vector<OraclePart> &parts, const OracleTreeShape &shape, Coins &coins, Digest &root) override {
    Kept k;
    k.parts = parts;
    for (size_t i = 0; i < parts.size(); i++) {
      const OraclePart &p = parts[i];
      ToyRows main, mask;
      Table tail;
      if (p.main != NO_ID) {
        const Table m = get(p.main);
        tail = Table(m.deg, m.rows, p.code.r);
        coins.fill(tail.a.data(), tail.a.size());
        main = encode_rows(p.code, m, tail);
      }
      if (p.mask.rows) mask = encode_rows(p.mask_code, p.mask, p.mask_tail);
      if (plant) plant(kept.size(), i, main, mask);
      k.main.push_back(std::move(main));
      k.mask.push_back(std::move(mask));
      k.main_tail.push_back(std::move(tail));
    }
    k.tree = std::make_unique<OracleTree>(shape, coins);
    std::vector<F> col;
    for (u64 x = 0; x < (u64(1) << shape.log_n); x++) {
      leaf_column(k, x, col);
      k.tree->set_leaf(x, col.data(), col.size());
    }
    root = k.tree->finish();
    kept.push_back(std::move(k));
    return (CommitId)(kept.size() - 1);
  }
  void open(CommitId id, const PositionList &at, Opening &out) override {
    Kept &k = kept[id];
    k.tree->open(at, out, [&](u64 x, std::vector<F> &col) { leaf_column(k, x, col); });
  }
  void close(CommitId) override {}

 private:
  static void column(const std::vector<ToyRows> &parts, u64 x, std::vector<F> &col) {
    for (const ToyRows &p : parts)
      for (const ToyRow &row : p) col.push_back(row[x]);
  }
  // The column of the tree leaf at position x: the main rows' symbols at x, then the mask rows' at
  // short_position() of x. In a tree of mask rows only, x is already their position.
  static void leaf_column(const Kept &k, u64 x, std::vector<F> &col) {
    const OracleTreeShape &s = k.tree->shape();
    col.clear();
    if (s.has_main()) column(k.main, x, col);
    if (s.has_mask()) column(k.mask, s.has_main() ? short_position(s.log_n, s.log_n_zk, x) : x, col);
  }
};

// ---- the messages and random tails that a prover says one oracle message encodes ----
struct Claimed {
  Table main, main_tail, mask, mask_tail;
};

// What the honest prover has encoded in oracle message o. `engine` must be the engine of pr.
inline Claimed honest_claim(ProverContext &pr, PlantEngine &engine, OracleId o) {
  Claimed c;
  const TreeBook::Place at = pr.book.place(o);
  if (pr.behind[o].main != NO_ID) {
    c.main = engine.get(pr.behind[o].main);
    c.main_tail = engine.kept[pr.commit[at.tree]].main_tail[at.part];
  }
  c.mask = pr.behind[o].mask;
  c.mask_tail = pr.behind[o].mask_tail;
  return c;
}

// The left side of a group's linear claim on claimed messages (claims[i] belongs to L.members[i]),
// from the group's main and mask tables written out. The index of an entry in a group's table is
// computed here, not by the library; the first row of each message's block is taken from the
// layout, which send_bat_split_test.cpp compares with expected numbers. A kept message's main rows
// (`L.kept`) are outside the main table; `row.kept` has their coefficients.
inline E group_row_on_claims(const Statement &S, const GroupLayout &L, const GroupRow &row, const std::vector<Claimed> &claims) {
  std::vector<E> main(size_t(1) << (L.k + L.log_ell), E::zero()), mask(size_t(1) << (L.h + L.log_ell_zk), E::zero());
  for (size_t i = 0; i < L.members.size(); i++) {
    const OracleId o = L.members[i];
    if (S.has(o, Part::MAIN) && !L.kept[i])
      for (size_t b = 0; b < claims[i].main.rows; b++)
        for (size_t c = 0; c < claims[i].main.len; c++) main[(L.main_first[i] + b) + (c << L.k)] = claims[i].main.at(b, c);
    if (S.has(o, Part::MASK))
      for (size_t b = 0; b < claims[i].mask.rows; b++)
        for (size_t c = 0; c < claims[i].mask.len; c++) mask[(L.mask_first[i] + b) + (c << L.h)] = claims[i].mask.at(b, c);
  }
  std::vector<E> c(main.size(), E::zero());
  row.c.add_to(c.data(), L.k + L.log_ell);
  BLK_REQUIRE(row.d.size() == mask.size(), "test: the written-out coefficients do not fit the group's mask table");
  E sum = E::zero();
  for (size_t j = 0; j < c.size(); j++) sum += c[j] * main[j];
  for (size_t j = 0; j < mask.size(); j++) sum += row.d[j] * mask[j];
  for (const KeptForm &f : row.kept) {
    const Table &t = claims[L.find(f.oracle)].main;
    std::vector<E> w(t.entries(), E::zero());
    f.c.add_to(w.data(), S.table_bits(f.oracle, Part::MAIN));
    for (u64 j = 0; j < t.entries(); j++) sum += w[j] * t.entry(j);
  }
  return sum;
}

// ---- the test closing ----
// For every oracle message of the group the prover sends, unmasked, each row's message and the
// random tail encoded with it (`Claimed`). The verifier checks the group's one linear claim on
// them, draws positions, and compares the openings there with the encoding of what was sent. The
// closing hides nothing; it is sound in the same way as the protocol's test: a prover passes only
// with codewords that meet the claim and agree with the committed words at the drawn positions.
static const char *const R_TEST_CLOSE = "test closing";

// The test closing's round error is entered as 0: the plan does not bound it.
inline void plan_test_close(Plan &p, u64 scalars, u64 positions, unsigned log_n) {
  p.send_scalars("shown in the clear", scalars);
  PlanRound &r = p.close(R_TEST_CLOSE, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  r.draw_positions = positions;
  r.log_n = log_n;
}

// A group's base code, which is that of its oracle messages with main rows. The positions of the
// test closing range over its length.
inline const Code &group_code(const Statement &S, const GroupLayout &L) {
  for (OracleId o : L.members)
    if (S.oracle(o).shape.main_rows) return S.codes().base[S.oracle(o).shape.code];
  BLK_REQUIRE(false, "test: a group without main rows");
  return S.codes().base[0];
}
inline unsigned group_log_n(const Statement &S, const GroupLayout &L) { return group_code(S, L).log_n; }

// claims[i] belongs to L.members[i]. Returns the positions that were drawn, so that a test can tell
// a prover's luck from a verifier's fault.
inline PositionList test_close_prove(ProverContext &pr, const GroupLayout &L, const std::vector<Claimed> &claims) {
  for (const Claimed &c : claims)
    for (const Table *t : {&c.main, &c.main_tail, &c.mask, &c.mask_tail})
      for (size_t b = 0; b < t->rows; b++)
        for (size_t i = 0; i < t->len; i++) pr.ch.send(t->at(b, i));
  PositionsDraw from = pr.ch.close_positions(R_TEST_CLOSE);
  const PositionList at = from.positions(group_log_n(pr.st, L));
  Opened opened;
  open_oracles(pr, L.members, at, opened);
  return at;
}

struct CloseVerdict {
  bool read = false;       // the proof's bytes could be read and the openings fit the roots
  bool row = false;        // the group's linear claim holds on what was sent
  bool positions = false;  // what was sent encodes to the opened symbols
  PositionList at;         // the positions that the verifier drew
  bool accept() const { return read && row && positions; }
};

inline CloseVerdict test_close_verify(VerifierContext &vf, const GroupLayout &L, const GroupRow &row) {
  CloseVerdict verdict;
  const Statement &S = vf.st;
  std::vector<Claimed> claims(L.members.size());
  bool in_field = true;
  for (size_t i = 0; i < L.members.size(); i++) {
    const OracleShape &s = S.oracle(L.members[i]).shape;
    Claimed &c = claims[i];
    if (s.main_rows) {
      c.main = Table(E_DEGREE, s.main_rows, S.codes().base[s.code].ell);
      c.main_tail = Table(E_DEGREE, s.main_rows, S.codes().base[s.code].r);
    }
    if (s.mask_rows) {
      c.mask = Table(E_DEGREE, s.mask_rows, S.codes().mask[s.mask_code].ell);
      c.mask_tail = Table(E_DEGREE, s.mask_rows, S.codes().mask[s.mask_code].r);
    }
    for (Table *t : {&c.main, &c.main_tail, &c.mask, &c.mask_tail})
      for (size_t b = 0; b < t->rows; b++)
        for (size_t j = 0; j < t->len; j++) {
          E x;
          if (!vf.ch.recv(x)) return verdict;
          t->set(b, j, x);
          // Main rows that are sent over F encode messages and random tails over F.
          if (s.main_deg == 1 && (t == &c.main || t == &c.main_tail)) in_field = in_field && is_zero(x.c[1]) && is_zero(x.c[2]);
        }
  }
  verdict.row = in_field && group_row_on_claims(S, L, row, claims) == row.target;
  PositionsDraw from = vf.ch.close_positions(R_TEST_CLOSE);
  const PositionList at = from.positions(group_log_n(S, L));
  verdict.at = at;
  Opened opened;
  if (!vf.ch.ok() || !read_opened(vf, L.members, at, opened)) return verdict;
  verdict.read = true;
  verdict.positions = true;
  auto row_of = [](const Table &t, size_t b) {
    std::vector<E> v(t.len);
    for (size_t j = 0; j < t.len; j++) v[j] = t.at(b, j);
    return v;
  };
  for (size_t i = 0; i < L.members.size(); i++) {
    const OracleShape &s = S.oracle(L.members[i]).shape;
    for (size_t w = 0; w < at.size(); w++) {
      for (size_t b = 0; b < s.main_rows; b++) {
        const Code &code = S.codes().base[s.code];
        const std::vector<E> m = row_of(claims[i].main, b), t = row_of(claims[i].main_tail, b);
        if (symbol_at(code, at[w], m.data(), t.data()) != opened.of[i][w].main[b]) verdict.positions = false;
      }
      for (size_t b = 0; b < s.mask_rows; b++) {
        const Code &code = S.codes().mask[s.mask_code];
        const std::vector<E> m = row_of(claims[i].mask, b), t = row_of(claims[i].mask_tail, b);
        if (symbol_at(code, mask_position(group_code(S, L), code, at[w]), m.data(), t.data()) != opened.of[i][w].mask[b]) verdict.positions = false;
      }
    }
  }
  return verdict;
}

// ---- small helpers ----
inline Table random_table(unsigned deg, size_t rows, size_t len, Coins &coins) {
  Table t(deg, rows, len);
  coins.fill(t.a.data(), t.a.size());
  return t;
}
inline std::vector<E> random_point(unsigned n, Coins &coins) {
  std::vector<E> z(n);
  for (E &x : z) x = coins.e();
  return z;
}
// A constant ("test" in ASCII) in place of the configuration digest, the hash of an instance's
// public files.
inline Digest test_configuration() {
  Digest d{};
  d.w[0] = 0x74736574;
  return d;
}
// Both sides of a run pass its public inputs (`TestInputs`) to test_bind(), which writes them into
// the hash chain's first query (the hash input fixed before the prover speaks), `stated` as the
// statement and the rest as parameters, and returns handles (`TestBound`). A test's verifier is a
// plain function decide(channel, TestBound, notes) under run_verifier() (blocks/chain/closed.h): it
// gets the handles and the proof's bytes, and nothing else of the test's data.
struct TestInputs {
  Codes codes;
  std::vector<OracleNotice> messages;  // in the order of sending
  std::vector<u64> numbers;            // the test's own, such as its groups and their members
  std::vector<E> stated;               // points, coefficients, targets that are not sent
};
struct TestBound {
  Public<Digest> configuration;  // the first item after the plan
  Public<Codes> codes;
  Public<std::vector<OracleNotice>> sent;
  Public<std::vector<u64>> numbers;
  Public<std::vector<E>> stated;
};
inline TestBound test_bind(Binder &b, const TestInputs &in) {
  TestBound x;
  x.configuration = b.configuration(test_configuration());
  x.codes = b.parameter("codes", in.codes), x.sent = b.parameter("oracle messages", in.messages), x.numbers = b.parameter("numbers", in.numbers);
  x.stated = b.statement("stated", in.stated);
  return x;
}
constexpr u64 TEST_LABEL = 1;
// The prover's side of a run: the same first query, made by test_bind() too.
struct TestBinder {
  Binder b;
  TestBound x;
  const Public<Codes> &codes;
  const Public<std::vector<OracleNotice>> &sent;
  TestBinder(const Plan &plan, const TestInputs &in) : b(TEST_LABEL, plan), x(test_bind(b, in)), codes(x.codes), sent(x.sent) {}
  TestBinder(const TestBinder &) = delete;  // codes and sent refer into x
  TestBinder &operator=(const TestBinder &) = delete;
  TestBinder(const Plan &plan, const Codes &c, const std::vector<OracleNotice> &messages, const std::vector<u64> &numbers = {}, const std::vector<E> &stated = {})
      : TestBinder(plan, TestInputs{c, messages, numbers, stated}) {}
};

// ---- notes (`Notes`): what a verifier lets its test see; no decision reads them ----
inline void note_number(Notes *notes, const std::string &what, u64 x) {
  if (notes) notes->emplace_back(what, E::from(from_u64(x)));
}
// Notes a group's linear claim in full: `target`, then the coefficients on the mask table, on the
// main table and on each kept oracle message (`GroupLayout::kept`) that the claim reads.
inline void note_group_row(Notes *notes, const std::string &what, const Statement &S, const GroupLayout &L, const GroupRow &row) {
  if (!notes) return;
  notes->emplace_back(what + ": target", row.target);
  for (const E &d : row.d) notes->emplace_back(what + ": on the mask table", d);
  auto written = [&](const std::string &where, const Form &c, unsigned vars) {
    std::vector<E> v(size_t(1) << vars, E::zero());
    c.add_to(v.data(), vars);
    for (const E &e : v) notes->emplace_back(what + where, e);
  };
  written(": on the main table", row.c, L.k + L.log_ell);
  for (const KeptForm &k : row.kept) written(": on kept member " + std::to_string(k.oracle), k.c, S.table_bits(k.oracle, Part::MAIN));
}
inline void note_verdict(Notes *notes, const std::string &what, const CloseVerdict &v) {
  note_number(notes, what + ": read", v.read), note_number(notes, what + ": row", v.row), note_number(notes, what + ": positions", v.positions);
  for (size_t i = 0; i < v.at.size(); i++) note_number(notes, what + ": drawn", v.at[i]);
}
// The values of the notes whose name begins with `what`, in their order.
inline std::vector<E> noted(const Notes &notes, const std::string &what) {
  std::vector<E> v;
  for (const auto &n : notes)
    if (n.first.rfind(what, 0) == 0) v.push_back(n.second);
  return v;
}
inline bool noted_yes(const Notes &notes, const std::string &what) {
  const std::vector<E> v = noted(notes, what);
  return v.size() == 1 && v[0] == E::one();
}
// For a test that has filled a whole main table with random data: zeroes the pad's entries (they
// mask SEND's sample answers), which send_prove() requires to be zero and then fills. Not for a
// real prover: data stored there by mistake would be erased and send_prove() would not stop.
inline void test_leave_pad_empty(const Codes &codes, const OracleShape &s, Table &main) {
  if (s.answers == 0 || s.mask_rows > 0) return;
  for (const PadPlace &p : Send::pad(codes, s)) main.set_entry(p.index, E::zero());
}
// A chain's salt for a prover's channel: 32 random bytes.
inline Digest test_bytes(Coins &coins) {
  u8 b[DIGEST_BYTES];
  coins.bytes(b, sizeof b);
  Digest d;
  digest_from_bytes(b, d);
  return d;
}
// same(), three overloads: equal as written, part by part. A `Tensor` whose has_dense() is true
// equals none; a `Form`'s `frozen` and `engine_terms` are not compared. No test calls them.
inline bool same(const Tensor &a, const Tensor &b) {
  return a.scalar == b.scalar && a.los() == b.los() && a.his() == b.his() && !a.has_dense() && !b.has_dense();
}
inline bool same(const Form &a, const Form &b) {  // not `held`: hints do not change the vector
  if (a.tensors.size() != b.tensors.size() || a.entries != b.entries) return false;
  for (size_t i = 0; i < a.tensors.size(); i++)
    if (!same(a.tensors[i], b.tensors[i])) return false;
  return true;
}
inline bool same(const GroupRow &a, const GroupRow &b) {
  if (a.group != b.group || !same(a.c, b.c) || a.d != b.d || a.target != b.target || a.kept.size() != b.kept.size()) return false;
  for (size_t i = 0; i < a.kept.size(); i++)
    if (a.kept[i].oracle != b.kept[i].oracle || !same(a.kept[i].c, b.kept[i].c)) return false;
  return true;
}

// Runs f in a child process. True if the child died by SIGABRT, as after a failed BLK_REQUIRE, and
// its stderr contains `why`: a call out of order must be stopped, and for the reason that the test
// has in mind. What f does to the test's objects is lost with the child.
inline bool stops(const char *why, const std::function<void()> &f) {
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
  for (ssize_t got; (got = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)got);
  ::close(fd[0]);
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "test: waitpid failed");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && said.find(why) != std::string::npos;
}
// True if f returns in a child process: the positive control beside a stops().
inline bool runs(const std::function<void()> &f) {
  std::fflush(nullptr);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "test: fork failed");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    f();
    _exit(0);
  }
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "test: waitpid failed");
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

#define TEST_CHECK(cond)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                              \
    }                                                                        \
  } while (0)

}  // namespace blk
