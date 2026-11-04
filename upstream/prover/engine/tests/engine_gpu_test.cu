// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks that `EngineGpu` gives what `EngineCpu` gives from the same private randomness (`Coins`):
// roots, openings, evaluations, tables. Both implement `Engine` (blocks/engine/engine.h), the
// interface to a prover's heavy steps. Also compares `EngineGpu` on a table in narrow cells
// (`Cell`, `Band`: prover/engine/encode.cuh) with `EngineGpu` on the same table in 64-bit words.
// Exit code 1 if anything differs. Needs a GPU. prover/tests/unit.sh builds it with nvcc and runs
// it; give that script an output directory outside the source tree.
// `Engine`'s calls not made here: search_nonce(), drop(), close(), let_go(), take_over(),
// borrow_table(), borrow_tree(), borrowed_tables(). fold_begin() to fold_end() are compared
// between two `EngineGpu` only, never with `EngineCpu`.
// `Card<T>` (prover/engine/device.cuh): an array in GPU memory.
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/engine_cpu.h"
#include "engine/engine_gpu.cuh"
#include "engine/tests/expand_coins.cuh"
#include "toy/toy_coins.h"
using namespace blk;

// The values of `others` with which main() calls held_factors().
#define HELD_BESIDE {0u, 1u, 2u, 6u}
static int failures = 0;
static unsigned ran = 0; // trees compared by `one_tree` and `one_code` only
// Counts and prints a difference unless `ok`.
static void hold(bool ok, const char *what, unsigned a, unsigned b, unsigned c) {
  if (ok) return;
  failures++;
  std::printf("DIFFERS: %s (case %u %u %u)\n", what, a, b, c);
}
static Table random_table(unsigned deg, size_t rows, size_t len, Coins &coins) {
  Table t(deg, rows, len);
  coins.fill(t.a.data(), t.a.size());
  return t;
}
static bool same(const Opening &a, const Opening &b) {
  return a.main_width == b.main_width && a.mask_width == b.mask_width && a.columns == b.columns && a.salts == b.salts && a.given == b.given;
}

// One tree over three parts (`OraclePart`), on both engines: rows over F and its cubic extension E
// (`deg` 1, 3), two codes of one length (the same code if `ell` is 1), with or without mask rows.
// `EngineGpu` draws `ToyCoins` on the host and `ExpandCoins` in GPU memory. A case without valid
// codes (code_ok()) is skipped silently.
static void one_tree(unsigned log_n, unsigned rate, unsigned variant) {
  ToyCoins data(1000 + log_n * 64 + rate * 8 + variant);
  const size_t ell = (size_t)1 << (log_n - rate);
  const size_t r = log_n - rate >= 3 ? 5 : 1;
  if (ell + r > ((size_t)1 << log_n)) return;
  const Code code{log_n, ell, r}, shorter{log_n, ell / 2 ? ell / 2 : 1, r};
  if (!code_ok(code) || !code_ok(shorter)) return;
  const bool with_mask = variant & 1, many_rows = variant & 2, card_coins = variant & 4;
  const unsigned log_zk = with_mask ? (variant & 8 ? log_n : log_n / 2) : 0;
  const Code mask_code{log_zk, ((size_t)1 << log_zk) / 2 ? ((size_t)1 << log_zk) / 2 : 1, log_zk >= 3 ? (size_t)3 : 0};
  if (with_mask && !code_ok(mask_code)) return;
  ran++;
  EngineCpu cpu;
  EngineGpu gpu;
  const Table t0 = random_table(1, many_rows ? 37 : 3, code.ell, data), t1 = random_table(3, many_rows ? 5 : 1, code.ell, data),
              t2 = random_table(1, 2, shorter.ell, data);
  std::vector<OraclePart> on_cpu(3), on_gpu(3);
  OracleTreeShape shape;
  shape.log_n = log_n, shape.log_n_zk = log_zk, shape.tree_number = 7 + variant;
  shape.salt_bytes = variant & 4 ? 48 : 40;
  shape.stored = variant == 3; // a tree that outlives every run: its leaves are hashed under `Domain::LEAF_STORED_MAIN` and start from `parameters`, not `run`
  data.bytes(reinterpret_cast<u8 *>(shape.run.w), sizeof shape.run.w);
  data.bytes(reinterpret_cast<u8 *>(shape.parameters.w), sizeof shape.parameters.w);
  const Table *tables[3] = {&t0, &t1, &t2};
  const Code codes[3] = {code, code, shorter};
  for (int i = 0; i < 3; i++) {
    on_cpu[i].main = cpu.put(*tables[i]), on_gpu[i].main = gpu.put(*tables[i]);
    on_cpu[i].code = on_gpu[i].code = codes[i];
    shape.main_symbols.push_back(tables[i]->rows * tables[i]->deg);
    size_t mask_rows = with_mask && i != 1 ? 1 + i : 0;
    if (mask_rows) {
      on_cpu[i].mask = on_gpu[i].mask = random_table(3, mask_rows, mask_code.ell, data);
      on_cpu[i].mask_tail = on_gpu[i].mask_tail = random_table(3, mask_rows, mask_code.r, data);
    }
    on_cpu[i].mask_code = on_gpu[i].mask_code = mask_code;
    shape.mask_symbols.push_back(3 * mask_rows);
  }
  Digest root_cpu, root_gpu;
  CommitId c_cpu, c_gpu;
  if (card_coins) {
    ExpandCoins a(99), b(99);
    c_cpu = cpu.commit(on_cpu, shape, a, root_cpu), c_gpu = gpu.commit(on_gpu, shape, b, root_gpu);
  } else {
    ToyCoins a(99), b(99);
    c_cpu = cpu.commit(on_cpu, shape, a, root_cpu), c_gpu = gpu.commit(on_gpu, shape, b, root_gpu);
  }
  hold(root_cpu == root_gpu, "the root", log_n, rate, variant);
  for (unsigned i = 0; i < 3; i++) {
    const Table x = cpu.tail(c_cpu, i), y = gpu.tail(c_gpu, i);
    bool equal = x.a.size() == y.a.size() && x.rows == y.rows && x.len == y.len && x.deg == y.deg;
    for (size_t j = 0; equal && j < x.a.size(); j++) equal = x.a[j] == y.a[j];
    hold(equal, "the encoding randomness", log_n, rate, variant);
  }
  std::vector<u64> draws;
  for (int i = 0; i < 9; i++) {
    u8 w[8];
    data.bytes(w, 8);
    draws.push_back(word_from_bytes(w) & (((u64)1 << log_n) - 1));
  }
  draws.push_back(0), draws.push_back(((u64)1 << log_n) - 1);
  const PositionList list = PositionList::from_draws(draws, log_n);
  Opening o_cpu, o_gpu;
  cpu.open(c_cpu, list, o_cpu), gpu.open(c_gpu, list, o_gpu);
  hold(same(o_cpu, o_gpu), "the opening", log_n, rate, variant);
  hold(check_opening(root_gpu, shape, list, o_gpu), "the opening against the root", log_n, rate, variant);
  // Evaluations, get() and reshape() of each table. Entry (row, place) has the index
  // row + (place << log_rows): the lowest log_rows variables are the row variables.
  for (int i = 0; i < 3; i++) {
    const Table &t = *tables[i];
    const unsigned vars = t.log_rows + ceil_log2(t.len);
    std::vector<E> z(vars);
    for (E &x : z) x = data.e();
    const Tensor v = Tensor::eq(z.data(), vars, data.e());
    hold(cpu.evaluate(on_cpu[i].main, v) == gpu.evaluate(on_gpu[i].main, v), "an evaluation", log_n, rate, variant);
    // A `Form`: v; a Tensor with a dense factor, that is, a list of values indexed by some
    // variables (row variables among them or not); three single entries.
    for (unsigned from : {0u, t.log_rows}) {
      const unsigned dense = vars - from < 3 ? vars - from : 3;
      if (!dense) continue;
      Tensor w;
      w.scalar = data.e();
      for (unsigned k = 0; k < from; k++) w.push(data.e(), data.e());
      w.push_dense(dense);
      for (unsigned k = from + dense; k < vars; k++) w.push(data.e(), data.e());
      auto values = std::make_shared<std::vector<E>>(((size_t)1 << dense) + 2);
      for (E &x : *values) x = data.e();
      w.set_values(values, 1);
      Form f;
      f.add(v), f.add(w);
      f.add_entry(0, data.e()), f.add_entry(((u64)1 << vars) - 1, data.e()), f.add_entry(((u64)1 << vars) / 2, data.e());
      hold(cpu.evaluate(on_cpu[i].main, f) == gpu.evaluate(on_gpu[i].main, f), "the evaluation of a form", log_n, rate, variant);
    }
    const Table back = gpu.get(on_gpu[i].main);
    bool equal = back.a.size() == t.a.size();
    for (size_t j = 0; equal && j < t.a.size(); j++) equal = back.a[j] == t.a[j];
    hold(equal, "a table that comes back", log_n, rate, variant);
    if (vars >= 2) {
      const Table x = cpu.get(cpu.reshape(on_cpu[i].main, vars / 2)), y = gpu.get(gpu.reshape(on_gpu[i].main, vars / 2));
      equal = x.a.size() == y.a.size() && x.rows == y.rows && x.len == y.len;
      for (size_t j = 0; equal && j < x.a.size(); j++) equal = x.a[j] == y.a[j];
      hold(equal, "a table cut anew", log_n, rate, variant);
    }
  }
}

// A tree whose rows share one code (message length a power of two), with salts of several lengths.
// EngineGpu::open() then gets the columns from prover/engine/columns.cuh instead of encoding again.
static void one_code(unsigned log_n, unsigned rate, size_t rows, size_t salt_bytes) {
  ToyCoins data(77 + log_n + rate + rows);
  const Code code{log_n, (size_t)1 << (log_n - rate), 5};
  if (!code_ok(code)) return;
  ran++;
  EngineCpu cpu;
  EngineGpu gpu;
  const Table t = random_table(1, rows, code.ell, data);
  std::vector<OraclePart> a(1), b(1);
  a[0].main = cpu.put(t), b[0].main = gpu.put(t), a[0].code = b[0].code = code;
  OracleTreeShape shape;
  shape.log_n = log_n, shape.main_symbols = {rows}, shape.mask_symbols = {0}, shape.salt_bytes = salt_bytes;
  shape.run.w[0] = 5, shape.parameters.w[0] = 9; // some run's value, some configuration's digest
  ExpandCoins ca(4), cb(4);
  Digest ra, rb;
  const CommitId ia = cpu.commit(a, shape, ca, ra), ib = gpu.commit(b, shape, cb, rb);
  hold(ra == rb, "the root of rows under one code", log_n, rate, (unsigned)rows);
  std::vector<u64> draws = {0, ((u64)1 << log_n) - 1, ((u64)1 << log_n) / 3, 1};
  const PositionList list = PositionList::from_draws(draws, log_n);
  Opening oa, ob;
  cpu.open(ia, list, oa), gpu.open(ib, list, ob);
  hold(same(oa, ob), "the opening of rows under one code", log_n, rate, (unsigned)rows);
}

// Random `Cell::S24` cells, three signed bytes each. Among the first six are the ends of the range:
// all three bytes at -128, at 127, and mixed.
static std::vector<signed char> three_bytes(size_t cells, Coins &coins) {
  std::vector<signed char> b(3 * cells);
  coins.bytes(reinterpret_cast<u8 *>(b.data()), b.size());
  const signed char ends[6][3] = {{-128, -128, -128}, {127, 127, 127}, {-128, 127, -128}, {127, -128, 127}, {0, 0, -128}, {-1, 0, 0}};
  for (size_t j = 0; j < 6 && j < cells; j++)
    for (int k = 0; k < 3; k++) b[3 * j + k] = ends[j][k];
  return b;
}
static F of_three(const signed char *b) { return from_i64((i64)b[0] + (i64)b[1] * (1 << 20) + (i64)b[2] * ((i64)1 << 40)); }

// EngineGpu::adopt() takes a table in place in GPU memory (no copy; later calls read it there). For
// narrow cells it gives the same root and evaluation as put() of the same integers.
static void narrow_cells(size_t rows) {
  const unsigned log_n = 12;
  const Code code{log_n, (size_t)1 << 10, 7};
  std::vector<u8> bytes(rows * code.ell);
  std::vector<short> shorts(rows * code.ell);
  ToyCoins data(5);
  data.bytes(bytes.data(), bytes.size());
  data.bytes(reinterpret_cast<u8 *>(shorts.data()), 2 * shorts.size());
  std::vector<signed char> threes = three_bytes(rows * code.ell, data);
  for (int kind = 0; kind < 4; kind++) {
    Table t(1, rows, code.ell);
    for (size_t j = 0; j < t.a.size(); j++)
      t.a[j] = kind == 0 ? from_u64(bytes[j]) : kind == 1 ? from_u64((unsigned short)shorts[j]) : kind == 2 ? from_i64(shorts[j]) : of_three(&threes[3 * j]);
    Card<u8> card;
    if (kind == 0)
      card.from_host(bytes.data(), bytes.size());
    else if (kind == 3)
      card.from_host(reinterpret_cast<const u8 *>(threes.data()), threes.size());
    else
      card.from_host(reinterpret_cast<const u8 *>(shorts.data()), 2 * shorts.size());
    EngineGpu wide, narrow;
    std::vector<OraclePart> a(1), b(1);
    a[0].main = wide.put(t);
    b[0].main = narrow.adopt(card.get(), kind == 0 ? Cell::U8 : kind == 1 ? Cell::U16 : kind == 2 ? Cell::I16 : Cell::S24, 1, rows, code.ell);
    a[0].code = b[0].code = code;
    OracleTreeShape shape;
    shape.log_n = log_n, shape.main_symbols = {rows}, shape.mask_symbols = {0};
    shape.run.w[0] = 6, shape.parameters.w[0] = 9, shape.salt_bytes = 40; // some run's value, some configuration's digest
    ExpandCoins ca(3), cb(3);
    Digest ra, rb;
    wide.commit(a, shape, ca, ra), narrow.commit(b, shape, cb, rb);
    hold(ra == rb, "the root from narrow cells", kind, 0, 0);
    std::vector<E> z(ceil_log2(rows) + 10);
    for (E &x : z) x = data.e();
    const Tensor v = Tensor::eq(z.data(), (unsigned)z.size());
    hold(wide.evaluate(a[0].main, v) == narrow.evaluate(b[0].main, v), "an evaluation of narrow cells", kind, 0, 0);
  }
}

// adopt() of a table in bands (`Band`: neighbouring rows with one type of cell) gives what put() of
// the same table in 64-bit words gives, both on `EngineGpu`: root, opening, evaluations, fold,
// get(), reshape(), entries(), with_entries(), and the opening after moved(). Root, opening and one
// evaluation are also compared with `EngineCpu`.
// Band x has band_rows[x] rows of cells of type turn[(from + x) % 5].
static void banded(const std::vector<size_t> &band_rows, unsigned number, unsigned from = 0) {
  const unsigned log_n = 12;
  const Code code{log_n, (size_t)1 << 10, 7};
  const Cell turn[5] = {Cell::S24, Cell::WORD, Cell::U16, Cell::U8, Cell::I16};
  size_t rows = 0;
  for (size_t r : band_rows) rows += r;
  ToyCoins data(7 + number);
  Table t(1, rows, code.ell);
  std::vector<Card<u8>> held;
  std::vector<Band> bands;
  size_t first = 0;
  for (size_t x = 0; x < band_rows.size(); x++) {
    const Cell kind = turn[(from + x) % 5];
    const size_t cells = band_rows[x] * code.ell;
    std::vector<u8> raw(cells * cell_bytes(kind));
    if (kind == Cell::S24) {
      const std::vector<signed char> b = three_bytes(cells, data);
      memcpy(raw.data(), b.data(), raw.size());
    } else if (kind == Cell::WORD) {
      data.fill(reinterpret_cast<F *>(raw.data()), cells);
    } else {
      data.bytes(raw.data(), raw.size());
    }
    for (size_t j = 0; j < cells; j++) {
      F &to = t.a[first * code.ell + j];
      short s;
      unsigned short u;
      switch (kind) {
      case Cell::S24: to = of_three(reinterpret_cast<const signed char *>(&raw[3 * j])); break;
      case Cell::WORD: memcpy(&to, &raw[8 * j], 8); break;
      case Cell::U16: memcpy(&u, &raw[2 * j], 2), to = from_u64(u); break;
      case Cell::U8: to = from_u64(raw[j]); break;
      default: memcpy(&s, &raw[2 * j], 2), to = from_i64(s);
      }
    }
    held.emplace_back();
    held.back().from_host(raw.data(), raw.size());
    bands.push_back(Band{held.back().get(), kind, band_rows[x]});
    first += band_rows[x];
  }
  EngineGpu wide, narrow;
  std::vector<OraclePart> a(1), b(1);
  a[0].main = wide.put(t), b[0].main = narrow.adopt(bands, 1, rows, code.ell);
  a[0].code = b[0].code = code;
  OracleTreeShape shape;
  shape.log_n = log_n, shape.main_symbols = {rows}, shape.mask_symbols = {0};
  shape.run.w[0] = 6, shape.parameters.w[0] = 9, shape.salt_bytes = 40; // some run's value, some configuration's digest
  ExpandCoins ca(3), cb(3);
  Digest ra, rb;
  const CommitId ia = wide.commit(a, shape, ca, ra), ib = narrow.commit(b, shape, cb, rb);
  hold(ra == rb, "the root of a table in bands", number, 0, 0);
  const PositionList list = PositionList::from_draws({0, 5, 1000, ((u64)1 << log_n) - 1}, log_n);
  Opening oa, ob;
  wide.open(ia, list, oa), narrow.open(ib, list, ob);
  hold(same(oa, ob), "the opening of a table in bands", number, 0, 0);
  const unsigned log_rows = ceil_log2(rows);
  std::vector<E> z(log_rows + 10);
  for (E &x : z) x = data.e();
  {  // and against `EngineCpu` on the same table with the same randomness, so that the comparison does not rest on a second `EngineGpu` alone
    EngineCpu cpu;
    std::vector<OraclePart> c(1);
    c[0].main = cpu.put(t), c[0].code = code;
    ExpandCoins cc(3);
    Digest rc;
    const CommitId ic = cpu.commit(c, shape, cc, rc);
    Opening oc;
    cpu.open(ic, list, oc);
    hold(rc == rb && same(oc, ob), "the root or the opening of a table in bands is not the CPU engine's", number, 0, 0);
    hold(cpu.evaluate(c[0].main, Tensor::eq(z.data(), (unsigned)z.size())) == narrow.evaluate(b[0].main, Tensor::eq(z.data(), (unsigned)z.size())),
         "an evaluation of a table in bands is not the CPU engine's", number, 0, 0);
  }
  Form form;
  form.tensors.push_back(Tensor::eq(z.data(), (unsigned)z.size()));
  hold(wide.evaluate(a[0].main, form.tensors[0]) == narrow.evaluate(b[0].main, form.tensors[0]), "an evaluation of a table in bands", number, 0, 0);
  first = 0;
  for (size_t r : band_rows) { // the first and the last row of each band, at the first, a middle and the last place
    for (u64 row : {(u64)first, (u64)(first + r - 1)})
      for (u64 place : {(u64)0, (u64)517, (u64)code.ell - 1}) form.entries.emplace_back(row + (place << log_rows), data.e());
    first += r;
  }
  if (rows < ((size_t)1 << log_rows)) form.entries.emplace_back(rows + ((u64)3 << log_rows), data.e()); // a row that is not stored: zeros
  hold(wide.evaluate(a[0].main, form) == narrow.evaluate(b[0].main, form), "many single entries of a table in bands (the fold's road)", number, 0, 0);
  // evaluate(Form) uses `CardFold` (prover/engine/fold.cuh; printed: the fold's road) only for more
  // than two Tensors and entries together (and log_rows <= 10) or for an `EngineTerm`. Otherwise,
  // as here, it reads each entry's cell.
  for (size_t e = 0; e + 1 < form.entries.size(); e += 2) {
    Form few;
    few.entries = {form.entries[e], form.entries[e + 1]};
    hold(wide.evaluate(a[0].main, few) == narrow.evaluate(b[0].main, few), "two single entries of a table in bands (read cell by cell)", number, (unsigned)e, 0);
  }
  {
    Form last;
    last.entries = {form.entries.back()};
    hold(wide.evaluate(a[0].main, last) == narrow.evaluate(b[0].main, last), "one single entry of a table in bands", number, 0, 0);
  }
  // A dense factor (printed: written-out factor) on the lowest row variable and the highest place
  // variable: prover/engine/evaluate.cuh then computes a weight for every cell.
  if (log_rows >= 1) {
    Tensor dense;
    dense.push_dense(1);
    for (unsigned v = 1; v + 1 < z.size(); v++) dense.push(E::one() - z[v], z[v]);
    dense.push_dense(1);
    dense.set_values(std::make_shared<const std::vector<E>>(std::vector<E>{data.e(), data.e(), data.e(), data.e()}), 0);
    hold(wide.evaluate(a[0].main, dense) == narrow.evaluate(b[0].main, dense), "a written-out factor on a row variable, on a table in bands", number, 0, 0);
  }
  // The fold over all row variables: every round's polynomial, the folded message and the folded
  // encoding randomness (`tail`).
  {
    Form c;
    c.tensors.push_back(Tensor::eq(z.data(), (unsigned)z.size()));
    const FoldId fa = wide.fold_begin({FoldPart{ia, 0, 0}}, log_rows, c), fb = narrow.fold_begin({FoldPart{ib, 0, 0}}, log_rows, c);
    bool rounds = true;
    for (unsigned j = 0; j < log_rows; j++) {
      E qa[3], qb[3];
      wide.fold_round(fa, qa), narrow.fold_round(fb, qb);
      rounds = rounds && qa[0] == qb[0] && qa[1] == qb[1] && qa[2] == qb[2];
      const E gamma = data.e();
      wide.fold_bind(fa, gamma), narrow.fold_bind(fb, gamma);
    }
    Table tail_a, tail_b;
    const TableId ma = wide.fold_end(fa, tail_a), mb = narrow.fold_end(fb, tail_b);
    hold(rounds && wide.get(ma).a == narrow.get(mb).a && tail_a.a == tail_b.a, "a fold over a table in bands", number, 0, 0);
  }
  // Each band is copied to a new address before moved(); afterwards the old memory is overwritten.
  {
    std::vector<Card<u8>> there;
    std::vector<const void *> addresses;
    for (size_t x = 0; x < held.size(); x++) {
      there.emplace_back();
      there.back().resize(held[x].size());
      ENGINE_CK(cudaMemcpy(there.back().get(), held[x].get(), held[x].size(), cudaMemcpyDeviceToDevice));
      addresses.push_back(there.back().get());
    }
    narrow.moved(b[0].main, addresses);
    for (Card<u8> &old : held) ENGINE_CK(cudaMemset(old.get(), 0xA5, old.size()));
    Opening again;
    narrow.open(ib, list, again);
    hold(same(oa, again) && narrow.get(b[0].main).a == t.a, "a table in bands after it was moved", number, 0, 0);
    held = std::move(there);
  }
  hold(narrow.get(b[0].main).a == t.a, "a table in bands, read back", number, 0, 0);
  const TableId cut_a = wide.reshape(a[0].main, log_rows + 2), cut_b = narrow.reshape(b[0].main, log_rows + 2);
  hold(wide.get(cut_a).a == narrow.get(cut_b).a, "a table in bands, cut anew", number, 0, 0);
  // with_entries() on the last row only, then on the first and the last row. Its copy has each band
  // with a changed entry in 64-bit words and the other bands as they were.
  for (unsigned which = 0; which < 2; which++) {
    std::vector<std::pair<u64, E>> changed;
    if (which == 0)
      for (u64 place : {(u64)0, (u64)517, (u64)code.ell - 1}) changed.emplace_back((rows - 1) + (place << log_rows), E::from(data.f()));
    else
      changed = {{(u64)0 + ((u64)3 << log_rows), E::from(data.f())}, {(u64)(rows - 1) + ((u64)9 << log_rows), E::from(data.f())}};
    std::vector<u64> at;
    for (const std::pair<u64, E> &e : form.entries) at.push_back(e.first);
    for (const std::pair<u64, E> &c : changed) at.push_back(c.first);
    std::vector<OraclePart> a2 = a, b2 = b;
    a2[0].main = wide.with_entries(a[0].main, changed), b2[0].main = narrow.with_entries(b[0].main, changed);
    hold(wide.entries(a2[0].main, at) == narrow.entries(b2[0].main, at), "entries() of a table in bands with entries replaced", number, which, 0);
    hold(wide.entries(a[0].main, at) == narrow.entries(b[0].main, at), "entries() of a table in bands", number, which, 0);
    hold(wide.get(a2[0].main).a == narrow.get(b2[0].main).a, "with_entries() on a table in bands", number, which, 0);
    hold(narrow.get(b[0].main).a == t.a, "with_entries() leaves a table in bands as it was", number, which, 0);
    ExpandCoins c2a(4), c2b(4);
    Digest r2a, r2b;
    wide.commit(a2, shape, c2a, r2a), narrow.commit(b2, shape, c2b, r2b);
    hold(r2a == r2b && !(r2a == ra), "the root over a table in bands with entries replaced", number, which, 0);
  }
}

static bool same_table(const Table &a, const Table &b) {
  return a.deg == b.deg && a.rows == b.rows && a.len == b.len && a.log_rows == b.log_rows && a.a == b.a;
}
// single_entries() and laid_rows(): both engines, on the same tables and lists.
static void single_entries(unsigned deg, size_t rows, size_t len, unsigned seed) {
  ToyCoins data(70000 + seed);
  EngineCpu cpu;
  EngineGpu gpu;
  const Table t = random_table(deg, rows, len, data);
  const TableId on_cpu = cpu.put(t), on_gpu = gpu.put(t);
  std::vector<u64> at = {0, t.entries() - 1}; // the last may be in a row beyond the real ones, which reads as zero
  std::vector<std::pair<u64, E>> changed;
  for (size_t i = 0; i < 40; i++) {
    u64 w;
    data.bytes(reinterpret_cast<u8 *>(&w), 8);
    at.push_back(w % t.entries());
    const u64 row = (w >> 20) % rows;
    if (i < len) changed.emplace_back(row + (i << t.log_rows), deg == 1 ? E::from(data.f()) : data.e()); // place i: no index comes twice
  }
  const std::vector<E> a = cpu.entries(on_cpu, at), b = gpu.entries(on_gpu, at);
  bool equal = a.size() == b.size();
  for (size_t i = 0; equal && i < a.size(); i++) equal = a[i] == b[i];
  hold(equal, "entries()", deg, (unsigned)rows, (unsigned)len);
  hold(same_table(cpu.get(cpu.with_entries(on_cpu, changed)), gpu.get(gpu.with_entries(on_gpu, changed))), "with_entries()", deg, (unsigned)rows,
       (unsigned)len);
  hold(same_table(cpu.get(on_cpu), gpu.get(on_gpu)) && same_table(gpu.get(on_gpu), t), "with_entries() leaves its table as it was", deg,
       (unsigned)rows, (unsigned)len);
  hold(same_table(cpu.get(cpu.with_entries(on_cpu, {})), gpu.get(gpu.with_entries(on_gpu, {}))), "with_entries() of nothing", deg, (unsigned)rows,
       (unsigned)len);
}
static void laid_rows(size_t row_len, u64 count, u64 first_row, unsigned bits, size_t rows, size_t len) {
  ToyCoins data(71000 + (unsigned)row_len + bits);
  EngineCpu cpu;
  EngineGpu gpu;
  const Table t = random_table(3, 1, row_len, data);
  hold(same_table(cpu.get(cpu.laid(cpu.put(t), count, first_row, bits, rows, len)), gpu.get(gpu.laid(gpu.put(t), count, first_row, bits, rows, len))),
       "laid()", (unsigned)count, (unsigned)first_row, bits);
}
// evaluate(Form) with an `EngineTerm` (scale * a vector that the engine stores), alone or with
// `written` Tensors. The term is zero at every index j with (j & choice_mask) != choice_value; the
// masks used select no, some or all row variables. Both engines must agree, and the value must
// differ from the one without the term.
static void engine_terms(unsigned deg, unsigned log_rows, unsigned log_len, u64 choice_mask, u64 choice_value, unsigned written) {
  ToyCoins data(81000 + deg * 1000 + log_rows * 100 + log_len * 10 + (unsigned)choice_mask);
  const size_t rows = (size_t)1 << log_rows, len = (size_t)1 << log_len;
  const Table t = random_table(deg, rows, len, data);
  const Table w = random_table(3, 1, (rows * len) >> __builtin_popcountll(choice_mask), data);
  EngineCpu cpu;
  EngineGpu gpu;
  const TableId a = cpu.put(t), b = gpu.put(t);
  Form on_cpu;
  std::vector<E> z(log_rows + log_len);
  for (unsigned i = 0; i < written; i++) {
    for (E &x : z) x = data.e();
    on_cpu.tensors.push_back(Tensor::eq(z.data(), (unsigned)z.size()));
  }
  const E without = written ? cpu.evaluate(a, on_cpu) : E::zero();
  Form on_gpu = on_cpu;
  const E scale = data.e();
  on_cpu.engine_terms.push_back(EngineTerm{cpu.put(w), scale, choice_mask, choice_value});
  on_gpu.engine_terms.push_back(EngineTerm{gpu.put(w), scale, choice_mask, choice_value});
  const E one = cpu.evaluate(a, on_cpu), other = gpu.evaluate(b, on_gpu);
  hold(one == other && !(one == without), "evaluate(Form) with a term that only the engine knows", deg * 100 + log_rows * 10 + written, log_len, (unsigned)choice_mask);
}

// A Tensor whose dense factor is read from a table that the engine stores (`Tensor::held`; printed:
// held factor) gives on `EngineGpu` what the same Tensor with the values on the host gives on
// `EngineCpu`: alone, and in a `Form` with `others` further Tensors (up to one: one pass per
// Tensor; more: all together, by prover/engine/fold.cuh).
// The factor is on variables `from` .. `from` + `dense` - 1, with values from entry `first` on.
// `dense` = 0: the factor is the single entry `first`.
static void held_factors(unsigned deg, unsigned log_rows, unsigned log_len, unsigned from, unsigned dense, u64 first, unsigned others) {
  const u64 seed = 91000 + deg * 7919 + log_rows * 641 + log_len * 101 + from * 31 + dense * 7 + first + others * 13;
  ToyCoins data(seed);
  const Table t = random_table(deg, (size_t)1 << log_rows, (size_t)1 << log_len, data);
  const unsigned vars = log_rows + log_len;
  const u64 length = first + ((u64)1 << dense) + 3; // the stretch lies inside a longer table
  const Table w = random_table(3, 1, length, data);
  auto values = std::make_shared<std::vector<E>>(length);
  for (u64 j = 0; j < length; j++) (*values)[j] = w.at(0, j);
  EngineCpu cpu;
  EngineGpu gpu;
  const TableId a = cpu.put(t), b = gpu.put(t), on_card = gpu.put(w);
  auto tensor = [&](bool held) {
    ToyCoins same(seed + 1);
    Tensor x;
    x.scalar = same.e();
    for (unsigned k = 0; k < from; k++) x.push(same.e(), same.e());
    if (dense) x.push_dense(dense);
    for (unsigned k = from + dense; k < vars; k++) x.push(same.e(), same.e());
    if (held)
      x.set_held(on_card, first, length);
    else if (dense)
      x.set_values(values, first);
    else
      x.scalar = x.scalar * (*values)[first];
    return x;
  };
  hold(cpu.evaluate(a, tensor(false)) == gpu.evaluate(b, tensor(true)), "the evaluation of a Tensor with a held factor", deg * 100 + log_rows * 10 + from, dense, (unsigned)first);
  Form written, held;
  written.add(tensor(false)), held.add(tensor(true));
  std::vector<E> z(vars);
  for (unsigned i = 0; i < others; i++) {
    for (E &x : z) x = data.e();
    written.add(Tensor::eq(z.data(), vars)), held.add(Tensor::eq(z.data(), vars));
  }
  hold(cpu.evaluate(a, written) == gpu.evaluate(b, held), "the evaluation of a Form with a held factor", deg * 100 + log_rows * 10 + from, dense, others);
}

// set_entries(): in place, on both engines; in an adopted table what was set outlives moved().
static void set_in_place(unsigned deg) {
  ToyCoins data(72000 + deg);
  const size_t rows = 5, len = 64;
  const Table t = random_table(deg, rows, len, data);
  EngineCpu cpu;
  EngineGpu gpu, keeper;
  const TableId a = cpu.put(t), b = gpu.put(t);
  Card<u64> here, there; // the caller's GPU memory, twice, each filled with t only
  here.from_host(reinterpret_cast<const u64 *>(t.a.data()), t.a.size()), there.from_host(reinterpret_cast<const u64 *>(t.a.data()), t.a.size());
  const TableId c = keeper.adopt(here.get(), Cell::WORD, deg, rows, len);
  std::vector<std::pair<u64, E>> changed;
  for (u64 index : {(u64)4 + (63 << 3), (u64)4 + (62 << 3), (u64)0, (u64)2 + (17 << 3)}) changed.emplace_back(index, deg == 1 ? E::from(data.f()) : data.e());
  cpu.set_entries(a, changed), gpu.set_entries(b, changed), keeper.set_entries(c, changed);
  hold(!same_table(cpu.get(a), t) && same_table(cpu.get(a), gpu.get(b)) && same_table(cpu.get(a), keeper.get(c)), "set_entries()", deg, 0, 0);
  keeper.moved(c, there.get());
  hold(same_table(cpu.get(a), keeper.get(c)), "set_entries() in an adopted table, after moved()", deg, 0, 0);
  changed = {{(u64)4 + (63 << 3), deg == 1 ? E::from(data.f()) : data.e()}}; // the same index once more: the later value counts, also after `moved`
  cpu.set_entries(a, changed), keeper.set_entries(c, changed);
  keeper.moved(c, here.get());
  hold(same_table(cpu.get(a), keeper.get(c)), "set_entries() twice at one place of an adopted table, after moved()", deg, 0, 0);
}

int main() {
  set_in_place(1), set_in_place(3);
  for (unsigned deg : {1u, 3u})
    for (unsigned others : HELD_BESIDE) {
      held_factors(deg, 2, 8, 0, 3, 0, others), held_factors(deg, 2, 8, 2, 4, 5, others), held_factors(deg, 2, 8, 1, 3, 1, others); // dense variables: both row variables and one place variable; place variables; one row variable and two place variables
      held_factors(deg, 0, 9, 4, 5, 17, others), held_factors(deg, 3, 6, 9, 0, 2, others), held_factors(deg, 3, 6, 0, 9, 0, others); // one row; one entry; all variables
    }
  for (unsigned deg : {1u, 3u})
    for (unsigned written : {0u, 1u, 5u}) {
      engine_terms(deg, 0, 6, 0, 0, written), engine_terms(deg, 3, 7, 0, 0, written), engine_terms(deg, 3, 7, 1, 1, written);
      engine_terms(deg, 3, 7, 5, 4, written), engine_terms(deg, 3, 7, 7, 2, written), engine_terms(deg, 2, 12, 2, 0, written);
    }
  for (unsigned deg : {1u, 3u})
    for (size_t rows : {(size_t)1, (size_t)2, (size_t)5, (size_t)8, (size_t)37})
      for (size_t len : {(size_t)1, (size_t)7, (size_t)64, (size_t)1000}) single_entries(deg, rows, len, deg * 100000 + (unsigned)rows * 1000 + (unsigned)len);
  laid_rows(64, 64, 0, 0, 1, 64), laid_rows(64, 64, 0, 2, 4, 16), laid_rows(64, 50, 4, 2, 8, 16), laid_rows(64, 0, 0, 1, 3, 5);
  laid_rows(1000, 777, 2, 1, 5, 400), laid_rows(4096, 4096, 8, 3, 16, 512), laid_rows(4096, 4095, 0, 3, 9, 512);
  for (unsigned log_n : {1u, 2u, 3u, 5u, 7u, 8u, 9u, 11u, 13u, 16u, 17u, 18u})
    for (unsigned rate = 0; rate <= 4 && rate <= log_n; rate++)
      for (unsigned variant = 0; variant < 16; variant++) {
        if (log_n >= 16 && (variant & 2 || rate == 0 || variant >= 8)) continue; // the CPU engine is slow there
        if ((variant & 8) && !(variant & 1)) continue;
        one_tree(log_n, rate, variant);
      }
  for (unsigned log_n : {4u, 9u, 12u, 14u})
    for (unsigned rate : {1u, 2u, 3u})
      for (size_t rows : {(size_t)1, (size_t)8, (size_t)16, (size_t)32, (size_t)45})
        for (size_t salt : {(size_t)40, (size_t)48, (size_t)64}) one_code(log_n, rate, rows, salt);
  // 300 rows: more than EVALUATE_ROWS (256 rows per kernel launch) and SMAC_TERMS (256 terms per
  // sum of 24-bit integers by smac(), prover/field/field_lazy.cuh). 1300 rows: several launches.
  narrow_cells(19), narrow_cells(300), narrow_cells(1300);
  // As prover/engine/lay.cuh stores a table with a pad (random field elements): many `S24` rows and
  // the pad's row in 64-bit words; all five cell types; MOST_BANDS bands; bands of more than
  // EVALUATE_ROWS rows; one band.
  banded({106, 1}, 0), banded({3, 1, 4, 1, 5}, 1), banded({1, 1, 1, 1, 1, 1, 1, 1}, 2), banded({300, 2, 270}, 3), banded({16}, 4);
  banded({3, 1, 4, 2}, 5, 1), banded({38, 1}, 6, 2); // without cells of three bytes; digits of 16 bits and a row of bytes
  ENGINE_CK(cudaDeviceSynchronize());
  std::printf("engine_gpu_test: field %d, hash %d: %u trees compared, %d differences\n", BLK_FIELD, BLK_HASH, ran, failures);
  return failures != 0;
}
