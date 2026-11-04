// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests EngineCpu, the reference implementation of Engine (blocks/engine/engine.h: a prover's heavy
// steps), at toy size against direct computation: evaluate(), reshape(), commit(), open() and the
// fold of 2^k rows into one; and table lifetimes during a fold. Other calls are not tested here.
#include <cstdio>

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

static Table random_table(Coins &coins, unsigned deg, size_t rows, size_t len) {
  Table t(deg, rows, len);
  coins.fill(t.a.data(), t.a.size());
  return t;
}

int main() {
  ToyCoins coins(7);
  EngineCpu engine;
  const Code code{7, 16, 9}, mask_code{5, 4, 9};
  CHECK(code_ok(code) && code_ok(mask_code) && mask_fits(code, mask_code));

  // Two oracle messages (`OraclePart`) in one tree: 3 rows over the field F with a mask row; 1 row
  // over its cubic extension E.
  const Table a = random_table(coins, 1, 3, code.ell), b = random_table(coins, 3, 1, code.ell);
  const TableId ta = engine.put(a), tb = engine.put(b);

  // evaluate() on an equality vector, then Tensor::mle(), each against the sum term by term.
  std::vector<E> z(2 + 4);
  for (E &x : z) x = coins.e();
  {
    const Tensor v = Tensor::eq(z.data(), 6, coins.e());
    E want = E::zero();
    for (u64 j = 0; j < a.entries(); j++) want += v.at(j) * a.entry(j);
    CHECK(engine.evaluate(ta, v) == want);
    CHECK(v.mle(z.data()) == [&] {
      E s = E::zero();
      const Tensor e = Tensor::eq(z.data(), 6);
      for (u64 j = 0; j < 64; j++) s += e.at(j) * v.at(j);
      return s;
    }());
  }
  // reshape() keeps every entry at its index (Table::entry()).
  {
    const TableId r = engine.reshape(tb, 2);
    const Table t = engine.get(r);
    CHECK(t.rows == 4 && t.len == 4);
    for (u64 j = 0; j < b.entries(); j++) CHECK(t.entry(j) == b.entry(j));
    engine.drop(r);
  }

  OraclePart pa, pb;
  pa.main = ta, pa.code = code, pa.mask_code = mask_code;
  pa.mask = random_table(coins, 3, 1, mask_code.ell), pa.mask_tail = random_table(coins, 3, 1, mask_code.r);
  pb.main = tb, pb.code = code;
  OracleTreeShape shape;
  shape.log_n = code.log_n, shape.log_n_zk = mask_code.log_n;
  shape.main_symbols = {3, 3}, shape.mask_symbols = {3, 0};
  shape.tree_number = 5;
  u8 run[DIGEST_BYTES];
  coins.bytes(run, sizeof run);
  digest_from_bytes(run, shape.run);
  shape.parameters = shape.run; // any digest that is not zero
  Digest root;
  const CommitId c = engine.commit({pa, pb}, shape, coins, root);
  // An opening at five draws, one repeated; a changed symbol must fail.
  const PositionList at = PositionList::from_draws({3, 100, 35, 3, 67}, code.log_n);
  Opening o;
  engine.open(c, at, o);
  Writer w;
  write_opening(w, shape, at, o);
  CHECK(w.size() == opening_bytes(shape, at));
  Reader r(w.data(), w.size());
  Opening o2;
  CHECK(read_opening(r, shape, at, o2) && r.at_end());
  CHECK(check_opening(root, shape, at, o2));
  o2.columns[1] = o2.columns[1] + F::one();
  CHECK(!check_opening(root, shape, at, o2));

  // Each opened column ends with the mask row's symbol at the short position at[i] mod 2^5.
  for (size_t i = 0; i < at.size(); i++) {
    std::vector<E> m(mask_code.ell), t(mask_code.r);
    for (size_t j = 0; j < m.size(); j++) m[j] = pa.mask.at(0, j);
    for (size_t j = 0; j < t.size(); j++) t[j] = pa.mask_tail.at(0, j);
    const F *s = o.mask_column(i);
    CHECK(symbol_at<E>(mask_code, mask_position(code, mask_code, at[i]), m.data(), t.data()) == E::from(s[0], s[1], s[2]));
  }

  // The fold of a group of 2^3 rows: a in rows 0..3 (row 3 is zero), b in row 4.
  const unsigned k = 3;
  std::vector<E> zz(k + 4);
  for (E &x : zz) x = coins.e();
  Tensor on_a = Tensor::eq(z.data(), 6, coins.e()); // a vector on a's table only
  on_a.insert_choice(2, 1, 0);
  const std::vector<Tensor> cs = {Tensor::eq(zz.data(), k + 4), on_a};
  E claim = E::zero();
  for (const Tensor &v : cs)
    for (u64 j = 0; j < (u64(1) << (k + 4)); j++) {
      const u64 row = j & 7, place = j >> 3;
      claim += v.at(j) * (row < 4 ? a.at(row, place) : row == 4 ? b.at(0, place) : E::zero());
    }
  // Three single entries: one on (row 1, place 3), two on the same index, (row 4, place 0).
  const std::vector<std::pair<u64, E>> entries = {{1 + 8 * 3, coins.e()}, {4, coins.e()}, {4, coins.e()}};
  claim += entries[0].second * a.at(1, 3) + (entries[1].second + entries[2].second) * b.at(0, 0);
  Form form;
  form.tensors = cs, form.entries = entries;
  const FoldId f = engine.fold_begin({FoldPart{c, 0, 0}, FoldPart{c, 1, 4}}, k, form);
  // Each round: q(0) + q(1) = `claim`.
  std::vector<E> gamma;
  for (unsigned j = 0; j < k; j++) {
    E q[3];
    engine.fold_round(f, q);
    CHECK(q[0] + q[0] + q[1] + q[2] == claim);
    gamma.push_back(coins.e());
    claim = q[0] + gamma[j] * (q[1] + gamma[j] * q[2]);
    engine.fold_bind(f, gamma[j]);
  }
  Table tail;
  TableId c_folded = NO_ID;
  const TableId folded = engine.fold_end(f, tail, &c_folded);
  const Table m = engine.get(folded);
  {
    // c_folded is `form` with its row variables fixed at gamma; on the folded row it gives `claim`.
    Form fixed = form;
    fixed.fix_low(gamma.data(), k);
    const Table cf = engine.get(c_folded);
    for (u64 j = 0; j < code.ell; j++) CHECK(cf.at(0, j) == fixed.at(j));
    CHECK(engine.evaluate(folded, fixed) == claim);
    // Second fold: the folded row is row 1 of two; a hint (`Held`) says the whole form is c_folded.
    Form again = fixed;
    again.held.push_back(Held{c_folded, E::one(), 0, again.tensors.size(), 0, again.entries.size(), 0, 0});
    again.insert_choice(0, 1, 1); // on row 1 of two
    again.scale(coins.e());
    OraclePart pc;
    pc.main = folded, pc.code = code;
    OracleTreeShape one;
    one.log_n = code.log_n, one.main_symbols = {3}, one.mask_symbols = {0}, one.tree_number = 6, one.run = shape.run, one.parameters = shape.parameters;
    Digest root2;
    const CommitId c2 = engine.commit({pc}, one, coins, root2);
    const FoldId f2 = engine.fold_begin({FoldPart{c2, 0, 1}}, 1, again); // ends the process if the hint is wrong
    // Engine forbids dropping or writing a table that an unfinished fold reads. EngineCpu reads it
    // only in fold_begin(), yet aborts: callers are tested on EngineCpu, and other engines may read
    // later. ends_with() runs each call in a child process.
    CHECK(ends_with("a fold that has not ended", [&] { engine.drop(c_folded); }));
    CHECK(ends_with("a fold that has not ended", [&] { engine.set_entries(c_folded, {{0, E::one()}}); }));
    E q[3];
    engine.fold_round(f2, q);
    CHECK(q[0] + q[0] + q[1] + q[2] == again.held[0].scale * claim);
    engine.fold_bind(f2, coins.e());
    Table tail2;
    engine.drop(engine.fold_end(f2, tail2));
    // The same for a table that is a tensor's dense factor (Tensor::held), which no hint names.
    const TableId factor = engine.put(Table(E_DEGREE, 1, code.ell));
    Tensor with_factor;
    with_factor.push_dense(ceil_log2(code.ell));
    with_factor.set_held(factor, 0, code.ell);
    with_factor.insert_choice(0, 1, 1);  // on row 1 of two
    Form held_form;
    held_form.add(with_factor);
    const FoldId f3 = engine.fold_begin({FoldPart{c2, 0, 1}}, 1, held_form);
    CHECK(ends_with("a fold that has not ended", [&] { engine.drop(factor); }));
    CHECK(ends_with("a fold that has not ended", [&] { engine.set_entries(factor, {{0, E::one()}}); }));
    engine.fold_round(f3, q);
    engine.fold_bind(f3, coins.e());
    engine.drop(engine.fold_end(f3, tail2));
    CHECK(runs_through([&] { engine.set_entries(factor, {{0, E::one()}}), engine.drop(factor); }));  // the control: after the fold both are allowed
  }
  {
    // `claim` again, from `cs` and `entries`.
    E left = E::zero();
    for (Tensor v : cs) {
      v.fix_low(gamma.data(), k);
      for (u64 j = 0; j < code.ell; j++) left += v.at(j) * m.at(0, j);
    }
    const Tensor eq_gamma = Tensor::eq(gamma.data(), k);
    for (const auto &entry : entries) left += entry.second * eq_gamma.at(entry.first & 7) * m.at(0, entry.first >> k);
    CHECK(left == claim);
  }
  // Opened main symbols, folded like the rows, give the folded row's codeword at that position.
  std::vector<E> eqg(8, E::zero());
  Tensor::eq(gamma.data(), k).add_to(eqg.data());
  for (size_t i = 0; i < at.size(); i++) {
    const F *s = o.main_column(i);
    const E column = eqg[0] * s[0] + eqg[1] * s[1] + eqg[2] * s[2] + eqg[4] * E::from(s[3], s[4], s[5]);
    std::vector<E> mm(code.ell), tt(code.r);
    for (size_t j = 0; j < mm.size(); j++) mm[j] = m.at(0, j);
    for (size_t j = 0; j < tt.size(); j++) tt[j] = tail.at(0, j);
    CHECK(symbol_at<E>(code, at[i], mm.data(), tt.data()) == column);
  }
  std::printf("engine: ok\n");
  return 0;
}
