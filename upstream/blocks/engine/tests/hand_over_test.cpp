// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests EngineCpu: hand-over between threads (let_go(), take_over()) and loans (borrow_tree(): one
// engine reads another's Merkle tree). Rules: blocks/engine/engine.h. Not tested, for example:
// borrow_table() called directly; a lender's drop() or set_entries() of a lent table.
#include <cstdio>
#include <thread>

#include "chain/tests/death.h"
#include "engine/engine_cpu.h"
#include "toy/toy_coins.h"

using namespace blk;

#define CHECK(c) \
  do { \
    if (!(c)) { \
      std::printf("FAILED line %d: %s\n", __LINE__, #c); \
      return 1; \
    } \
  } while (0)

static const Code CODE{7, 16, 9};
static Table random_table(Coins &coins, unsigned deg, size_t rows, size_t len) {
  Table t(deg, rows, len);
  coins.fill(t.a.data(), t.a.size());
  return t;
}
static OracleTreeShape shape_of(size_t symbols, unsigned number) {
  OracleTreeShape s;
  s.log_n = CODE.log_n, s.main_symbols = {symbols}, s.mask_symbols = {0}, s.salt_bytes = 40, s.tree_number = number;
  s.run.w[0] = 5, s.parameters.w[0] = 6;
  return s;
}
static CommitId commit(Engine &e, const Table &t, unsigned number, Digest &root, TableId *table = nullptr) {
  std::vector<OraclePart> parts(1);
  parts[0].main = e.put(t), parts[0].code = CODE;
  if (table) *table = parts[0].main;
  ToyCoins coins(4000 + number);
  return e.commit(parts, shape_of(t.rows * t.deg, number), coins, root);
}
static PositionList some_positions() { return PositionList::from_draws({0, 1, 127, 64, 77, 100, 77}, CODE.log_n); }
static bool same(const Opening &a, const Opening &b) {
  return a.main_width == b.main_width && a.mask_width == b.mask_width && a.columns == b.columns && a.salts == b.salts && a.given == b.given;
}
static bool same(const Table &a, const Table &b) { return a.deg == b.deg && a.rows == b.rows && a.len == b.len && a.a == b.a; }
// Outputs of a k-round fold: each round's polynomial q, the folded message, its random tail.
static std::vector<E> folded(Engine &e, const std::vector<FoldPart> &parts, unsigned k, u64 seed) {
  ToyCoins data(seed);
  std::vector<E> z(k + 4), all;
  for (E &x : z) x = data.e();
  Form c;
  c.add(Tensor::eq(z.data(), (unsigned)z.size(), data.e()));
  const FoldId f = e.fold_begin(parts, k, c);
  for (unsigned i = 0; i < k; i++) {
    E q[3];
    e.fold_round(f, q);
    all.insert(all.end(), q, q + 3);
    e.fold_bind(f, data.e());
  }
  Table tail;
  const TableId row = e.fold_end(f, tail);
  const Table message = e.get(row);
  for (size_t j = 0; j < message.len; j++) all.push_back(message.at(0, j));
  for (size_t j = 0; j < tail.len; j++) all.push_back(tail.at(0, j));
  e.drop(row);
  return all;
}

int main() {
  ToyCoins data(1);
  const Table own = random_table(data, 1, 2, CODE.ell), omega = random_table(data, 3, 2, CODE.ell);
  // Reference results: one engine, one thread.
  EngineCpu one;
  Digest r1, r2, s1, s2;
  TableId omega_one;
  const CommitId a_one = commit(one, own, 2, r1), b_one = commit(one, omega, 3, r2, &omega_one);
  Opening o_one, o;
  one.open(b_one, some_positions(), o_one);
  const std::vector<E> f_one = folded(one, {FoldPart{a_one, 0, 0}, FoldPart{b_one, 0, 2}}, 2, 60);

  // A hand-over to another thread and back.
  EngineCpu batch, seat;
  const CommitId theirs = commit(batch, omega, 3, s2), a = commit(seat, own, 2, s1);
  CHECK(r1 == s1 && r2 == s2);
  batch.let_go();
  std::thread([&] { batch.take_over(), batch.open(theirs, some_positions(), o), batch.let_go(); }).join();
  batch.take_over();
  CHECK(same(o, o_one));

  // A loan: the results match the reference.
  const CommitId b = seat.borrow_tree(batch, theirs);
  const std::vector<TableId> under = seat.borrowed_tables(b);
  CHECK(under.size() == 1 && under[0] != NO_ID);
  CHECK(folded(seat, {FoldPart{a, 0, 0}, FoldPart{b, 0, 2}}, 2, 60) == f_one);
  seat.open(b, some_positions(), o);
  CHECK(same(o, o_one) && check_opening(s2, shape_of(6, 3), some_positions(), o));
  CHECK(same(seat.tail(b, 0), one.tail(b_one, 0)) && same(seat.get(under[0]), omega));
  CHECK(seat.entries(under[0], {0, 5, 31}) == one.entries(omega_one, {0, 5, 31}));
  seat.close(b);
  const CommitId again = seat.borrow_tree(batch, theirs); // it can be made again
  seat.close(again);
  // A loan to another thread's engine, while no thread holds the lender.
  batch.let_go();
  std::thread([&] {
    EngineCpu other;
    const CommitId c = other.borrow_tree(batch, theirs);
    other.open(c, some_positions(), o);
    other.close(c);
  }).join();
  batch.take_over();
  CHECK(same(o, o_one));

  // Breaches: each runs in a child process, which must abort with the quoted text in its message.
  CHECK(ends_with("called by a thread that does not hold it", [&] { std::thread([&] { batch.open(theirs, some_positions(), o); }).join(); }));
  CHECK(ends_with("taken over while a thread holds it", [&] { std::thread([&] { batch.take_over(); }).join(); }));
  CHECK(ends_with("called by a thread that does not hold it", [&] { batch.let_go(), batch.open(theirs, some_positions(), o); }));
  CHECK(ends_with("close: another engine has borrowed this tree", [&] { seat.borrow_tree(batch, theirs), batch.close(theirs); }));
  CHECK(ends_with("the tree is lent already", [&] { seat.borrow_tree(batch, theirs), seat.borrow_tree(batch, theirs); }));
  CHECK(ends_with("set_entries: a borrowed table", [&] { seat.set_entries(seat.borrowed_tables(seat.borrow_tree(batch, theirs))[0], {{0, E::one()}}); }));
  CHECK(ends_with("drop: a tree that is not closed stands over this table", [&] { seat.drop(seat.borrowed_tables(seat.borrow_tree(batch, theirs))[0]); }));
  CHECK(ends_with("another thread holds this engine", [&] {
    const CommitId c = seat.borrow_tree(batch, theirs);
    batch.let_go();
    std::thread([&] { batch.take_over(); }).join();
    seat.open(c, some_positions(), o);
  }));
  CHECK(ends_with("it ends while another engine has borrowed a tree of it", [&] {
    auto short_lived = std::make_unique<EngineCpu>();
    Digest r;
    seat.borrow_tree(*short_lived, commit(*short_lived, omega, 5, r));
    short_lived.reset();
  }));
  CHECK(ends_with("from an engine of another kind, or from oneself", [&] { batch.borrow_tree(batch, theirs); }));
  batch.close(theirs), seat.close(a);
  std::printf("hand_over_test: ok (a hand-over, three loans, ten breaches stopped)\n");
  return 0;
}
