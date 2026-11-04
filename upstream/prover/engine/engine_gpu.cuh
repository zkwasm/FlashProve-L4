// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// EngineGpu: the `Engine` interface of blocks/engine/engine.h on one GPU (`card` in names). It
// stores main tables (the long rows of oracle messages), evaluates linear forms on them, commits to
// them under Merkle trees, opens the trees, folds the rows and searches proof-of-work nonces. Given
// the same private randomness (`Coins`), each call is meant to return, bit for bit, what the CPU
// engine returns (blocks/engine/engine_cpu.h); prover/engine/tests/engine_gpu_test.cu compares
// tables, evaluations, roots and openings. No codeword is stored: of a tree the engine keeps
// pointers into the main tables, each main row's random tail (`tail`: random coefficients encoded
// with the message), the caller's mask rows (short rows of random values) and, depending on
// TreeKeep, salts and digests. open() recomputes the requested columns from the rows. Not in the
// interface: adopt() registers a table in the caller's GPU memory; leave_card() unloads a tree.
// Each of these ends the process: drop() of a table under an unclosed tree; drop() or close() of
// what an unfinished fold reads; a salt length that is not a multiple of 8 bytes; a code longer
// than 2^30 positions; a stream that is not ordered with the default stream (see the constructor's
// two checks); a call from a thread that does not hold the engine (see let_go()).
// Link with -lcublas (used through prover/engine/products.cuh).
#pragma once
#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "engine/card_coins.cuh"
#include "engine/columns.cuh"
#include "engine/device.cuh"
#include "engine/encode.cuh"
#include "engine/engine.h"
#include "engine/entries.cuh"
#include "engine/evaluate.cuh"
#include "engine/fold.cuh"
#include "engine/stream_coins.cuh"
#include "engine/tree.cuh"
#include "hash/work_search.cuh"

namespace blk {


// For reshape(): out[(row * len + place) * deg + d] = coordinate d of the old table's entry number
// row + 2^log_rows * place; 0 if that entry's old row is not a real one. (A table has 2^log_rows
// rows; only the first `rows` of them, the real rows, are stored, and the others read as zero.)
static __global__ void k_reshape(CardBands table, unsigned deg, u64 old_rows, unsigned old_log_rows, unsigned log_rows, u64 len, u64 *out) {
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= (len << log_rows)) return;
  const u64 row = e / len, place = e % len, j = row + (place << log_rows);
  const u64 from_row = j & (((u64)1 << old_log_rows) - 1), from_place = j >> old_log_rows;
  for (unsigned d = 0; d < deg; d++)
    out[e * deg + d] = from_row < old_rows ? table.at(from_row, from_place * deg + d).v : 0;
}
// out[e] = cell e of the table, in row-major order, as a field element's 64-bit word. (A cell
// stores one coordinate of an entry: `Cell`, encode.cuh.)
static __global__ void k_cells_to_words(CardBands table, u64 count, u64 *out) {
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count) out[e] = table.at(e / table.row_cells, e % table.row_cells).v;
}
// out[i] = cell at[i], as above; 0 where at[i] = ~0 (a cell beyond the real rows).
static __global__ void k_pick_cells(CardBands table, const u64 *at, size_t count, u64 *out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = at[i] == ~(u64)0 ? 0 : table.at(at[i] / table.row_cells, at[i] % table.row_cells).v;
}
// out[i * width + k] = in[at[i] * width + k].
static __global__ void k_pick_words(const u64 *in, const u64 *at, size_t count, unsigned width, u64 *out) {
  const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e < count * width) out[e] = in[at[e / width] * width + e % width];
}

// What a tree of n leaves keeps in GPU memory from commit() to close(), besides random tails and
// mask rows. The proof's bytes do not depend on these settings.
struct TreeKeep {
  // false: only the bottom level's digests and the levels from the tenth above it up are kept
  // (`lean` in tree.cuh): about 32 n bytes instead of 64 n. A tree of at most 2^10 leaves keeps
  // all its digests.
  bool nodes = true;
  // false: commit() frees the salts before it returns, and open() regenerates the requested ones: 0
  // bytes instead of salt_bytes * n. This needs `Coins` that are CardStreamCoins
  // (stream_coins.cuh), which can repeat a draw and must then outlive the tree; with any other
  // `Coins` the salts are kept.
  bool salts = true;
  // false: the encoder's work space (16 rows of codewords and the columns' hash states: up to 192 n
  // bytes) returns to the memory pool at the end of every commit(), open() and back_on_card(). In a
  // process with many engines, only those at work then hold a work space.
  bool work = true;
};

class EngineGpu : public Engine {
public:
  explicit EngineGpu(cudaStream_t stream = 0) : stream_(stream) {
    unsigned flags = 0;
    ENGINE_CK(cudaStreamGetFlags(stream, &flags));
    BLK_REQUIRE(!(flags & cudaStreamNonBlocking), "engine: a stream that keeps no order with the default stream");
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM // nvcc defines it under --default-stream per-thread
    BLK_REQUIRE(stream == 0 || stream == cudaStreamPerThread,
                "engine: in a program built with --default-stream per-thread the engine runs on the calling thread's own stream: Card<> takes and "
                "frees on stream 0, which keeps order with no other stream there");
#endif
  }
  ~EngineGpu() override {
    still(); // ends the process if another thread holds the engine
    ENGINE_CK(cudaStreamSynchronize(stream_));
    for (CommitId c = 0; c < commits_.size(); c++)
      if (commits_[c] && commits_[c]->of) commits_[c]->of->lent = false, commits_[c].reset(); // gives back borrowed trees, then tables
    for (const auto &t : tables_)
      if (t && t->lender) t->lender->main_of_mine(t->theirs).lent = false;
    for (const auto &c : commits_) BLK_REQUIRE(!c || !c->lent, "engine: it ends while another engine has borrowed a tree of it");
    for (const auto &t : tables_) BLK_REQUIRE(!t || !t->lent, "engine: it ends while another engine has borrowed a table of it");
  }

  // ---- one thread at a time ----
  // Engine's rule (blocks/engine/engine.h): one thread holds the engine, at first the one that
  // constructed it, and passes it on by let_go(); the next holder calls take_over(). A call from
  // any other thread ends the process (tree_bytes() and tree_bytes_peak() do not check). The
  // reason here: the engine has no lock on its lists, encoder work space and nonce searcher, and
  // under --default-stream per-thread its stream is the calling thread's own. let_go() waits for
  // the work queued on that stream. The `Coins` are passed on separately (blocks/coins/coins.h).
  void let_go() override {
    mine();
    ENGINE_CK(cudaStreamSynchronize(stream_));
    holder_.store(0, std::memory_order_release);
  }
  void take_over() override {
    size_t nobody = 0;
    BLK_REQUIRE(holder_.compare_exchange_strong(nobody, me(), std::memory_order_acq_rel), "engine: taken over while a thread holds it");
  }

  // ---- loans ----
  // Engine's rules for loans (blocks/engine/engine.h), between two EngineGpu: the borrower reads
  // the lender's GPU memory in place. moved() and leave_card() end the process on what is borrowed
  // or lent, as set_entries() does. At borrowing and at each use, either the calling thread holds
  // the lender or no thread does (after let_go() the lender's queued work is done); otherwise the
  // process ends. Four uses are not checked:
  // - the borrower's destructor;
  // - fold_round(), fold_bind() and fold_end() (fold_end() reads the rows and tails that
  //   fold_begin() looked up);
  // - open() of a tree that the borrower committed over a borrowed table (commit() accepts one);
  // - a borrowed table that a `Form` (a coefficient vector, blocks/iop/tensor.h) names as a hint or
  //   an engine term (vectors_of()).
  // If the lender does not keep a tree's salts (TreeKeep::salts), the borrower's open() reads the
  // lender's `Coins`.
  TableId borrow_table(Engine &from, TableId theirs) override {
    mine();
    EngineGpu &lender = of_my_kind(from);
    Main &t = lender.main_of_mine(theirs);
    BLK_REQUIRE(!t.lent && !t.lender, "borrow: the table is lent already, or is itself borrowed");
    auto m = std::make_unique<Main>();
    m->deg = t.deg, m->log_rows = t.log_rows, m->rows = t.rows, m->len = t.len, m->bands = t.bands;
    m->lender = &lender, m->theirs = theirs;
    t.lent = true;
    return keep(std::move(m));
  }
  CommitId borrow_tree(Engine &from, CommitId theirs) override {
    mine();
    EngineGpu &lender = of_my_kind(from);
    Committed &t = lender.committed(theirs);
    BLK_REQUIRE(!t.lent && !t.of, "borrow: the tree is lent already, or is itself borrowed");
    auto c = std::make_unique<Committed>();
    c->shape = t.shape, c->rows = t.rows, c->mask_rows = t.mask_rows, c->lean = t.lean, c->root = t.root;
    c->of = &t, c->lender = &lender;
    for (const OraclePart &p : t.parts) { // codes and main table only; masks stay with the lender
      OraclePart mine_of_it;
      mine_of_it.code = p.code, mine_of_it.mask_code = p.mask_code;
      if (p.main != NO_ID) mine_of_it.main = borrow_table(lender, p.main);
      c->parts.push_back(std::move(mine_of_it));
    }
    t.lent = true;
    commits_.push_back(std::move(c));
    return (CommitId)(commits_.size() - 1);
  }
  // This engine's numbers for the main tables of a borrowed tree's parts (`OraclePart`: one oracle
  // message each); NO_ID for a part without main rows.
  std::vector<TableId> borrowed_tables(CommitId id) override {
    mine();
    BLK_REQUIRE(id < commits_.size() && commits_[id] && commits_[id]->of, "engine: no such borrowed tree");
    std::vector<TableId> all;
    for (const OraclePart &p : committed(id).parts) all.push_back(p.main);
    return all;
  }
  // ---- what trees keep in GPU memory ----
  // `nodes` and `salts` apply to trees committed after this call, `work` to every later commit(),
  // open() and back_on_card().
  void keep(const TreeKeep &k) { mine(), keep_ = k; }
  // Bytes of GPU memory that this engine's trees hold (digests, salts, random tails, mask rows):
  // now, and the largest value so far.
  size_t tree_bytes() const { return tree_bytes_; }
  size_t tree_bytes_peak() const { return tree_bytes_peak_; }
  // Bytes of digests and salts that one tree of this shape holds under k. Where k.salts is false
  // no salts are counted, which is right only with CardStreamCoins (see TreeKeep::salts).
  static size_t tree_bytes(const OracleTreeShape &shape, const TreeKeep &k) {
    return CardTree::bytes(shape.tree(), !k.nodes) +
           (k.salts ? shape.salt_bytes << shape.log_n : 0);
  }

  // ---- main tables ----
  TableId put(const Table &t) override {
    mine();
    auto m = std::make_unique<Main>();
    m->deg = t.deg, m->log_rows = t.log_rows, m->rows = t.rows, m->len = t.len;
    static_assert(sizeof(F) == sizeof(u64), "an F is its word");
    m->own.from_host(reinterpret_cast<const u64 *>(t.a.data()), t.a.size());
    m->bands = {Band{m->own.get(), Cell::WORD, m->rows}};
    return keep(std::move(m));
  }
  // Registers a table that is already in GPU memory: rows x len entries of deg cells, in row-major
  // order; a cell may be narrower than a field element (Cell in encode.cuh). The memory stays the
  // caller's. Not checked: its address must be aligned for the cell type (8 bytes for Cell::WORD,
  // 2 for U16 and I16); it must outlive its use here; its cells must be complete for this
  // engine's stream before a call reads them (written on that stream or on the default stream, or
  // by work that the caller has waited for); and they must not change while an unclosed tree covers
  // the table or a fold reads it (set_entries() then ends the process). The same holds for the new
  // address given to moved().
  TableId adopt(const void *cells_on_card, Cell cell, unsigned deg, size_t rows, size_t len) {
    return adopt({Band{cells_on_card, cell, rows}}, deg, rows, len);
  }
  // The same for a table stored in bands (runs of rows with one cell type), at most MOST_BANDS, in
  // row order.
  TableId adopt(const std::vector<Band> &bands, unsigned deg, size_t rows, size_t len) {
    mine();
    BLK_REQUIRE(deg == 1 || deg == 3, "table: entries are in F or in E");
    BLK_REQUIRE(!bands.empty() && bands.size() <= MOST_BANDS, "table: its rows must lie in 1 to 8 bands (runs of rows with one kind of cell)");
    size_t in_bands = 0;
    for (const Band &b : bands) {
      BLK_REQUIRE(b.rows > 0 || rows == 0, "table: a band without rows");
      BLK_REQUIRE(b.cells || !b.rows || !len, "table: a band with rows and no address");
      BLK_REQUIRE((u32)b.cell <= (u32)Cell::S24, "table: no such kind of cell");
      in_bands += b.rows;
    }
    BLK_REQUIRE(in_bands == rows, "table: the bands have another number of rows than the table");
    auto m = std::make_unique<Main>();
    m->deg = deg, m->rows = rows, m->len = len, m->bands = bands;
    m->log_rows = ceil_log2(rows);
    return keep(std::move(m));
  }
  Table get(TableId id) override {
    mine();
    const Main &m = main(id);
    Table t(m.deg, m.rows, m.len);
    if (t.a.empty()) return t;
    Card<u64> words(t.a.size());
    k_cells_to_words<<<blocks_for(t.a.size(), 256), 256, 0, stream_>>>(m.on_card(), t.a.size(), words.get());
    ENGINE_CK(cudaGetLastError());
    ENGINE_CK(cudaStreamSynchronize(stream_));
    words.to_host(reinterpret_cast<u64 *>(t.a.data()));
    return t;
  }
  void drop(TableId id) override {
    mine();
    main(id);
    for (const auto &c : commits_)
      if (c)
        for (const OraclePart &p : c->parts) BLK_REQUIRE(p.main != id, "drop: a tree that is not closed reads this table");
    for (size_t f = 0; f < folds_.size(); f++)
      if (folds_[f])
        for (TableId t : fold_reads_[f].tables) BLK_REQUIRE(t != id, "drop: a fold that has not ended reads this table");
    BLK_REQUIRE(!tables_[id]->lent, "drop: another engine has borrowed this table");
    settle();
    if (tables_[id]->lender) { // ends the loan once this engine's queued reads are done
      ENGINE_CK(cudaStreamSynchronize(stream_));
      tables_[id]->lender->main_of_mine(tables_[id]->theirs).lent = false;
    }
    tables_[id].reset();
  }
  TableId reshape(TableId id, unsigned log_rows) override {
    mine();
    const Main &t = main(id);
    const u64 total = (u64)t.len << t.log_rows;
    BLK_REQUIRE((total >> log_rows) << log_rows == total && (total >> log_rows) > 0, "reshape: the rows do not divide the table");
    auto m = std::make_unique<Main>();
    m->deg = t.deg, m->log_rows = log_rows, m->rows = (size_t)1 << log_rows, m->len = total >> log_rows;
    m->own.resize(total * t.deg);
    m->bands = {Band{m->own.get(), Cell::WORD, m->rows}};
    k_reshape<<<blocks_for(total, 256), 256, 0, stream_>>>(t.on_card(), t.deg, t.rows, t.log_rows, log_rows, m->len, m->own.get());
    ENGINE_CK(cudaGetLastError());
    return keep(std::move(m));
  }
  std::vector<E> entries(TableId id, const std::vector<u64> &at) override { return mine(), entries_of(main(id), at); }
  // A band with a changed entry is copied as 64-bit words; every other band keeps its cell type.
  TableId with_entries(TableId id, const std::vector<std::pair<u64, E>> &changed) override {
    mine();
    const Main &t = main(id);
    std::vector<u64> seen;
    for (const std::pair<u64, E> &c : changed) seen.push_back(c.first);
    std::sort(seen.begin(), seen.end());
    BLK_REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "with_entries: an index comes twice");
    std::vector<size_t> first_row(t.bands.size() + 1, 0); // of each band
    for (size_t b = 0; b < t.bands.size(); b++) first_row[b + 1] = first_row[b] + t.bands[b].rows;
    auto band_of = [&](u64 row) { return (size_t)(std::upper_bound(first_row.begin(), first_row.end(), row) - first_row.begin()) - 1; };
    std::vector<bool> touched(t.bands.size(), false);
    for (const std::pair<u64, E> &c : changed) {
      const u64 row = c.first & (((u64)1 << t.log_rows) - 1), place = c.first >> t.log_rows;
      BLK_REQUIRE(row < t.rows && place < t.len, "with_entries: no such entry (rows beyond the real ones are zero and cannot be set)");
      BLK_REQUIRE(t.deg == E_DEGREE || (c.second.c[1] == F::zero() && c.second.c[2] == F::zero()),
                  "with_entries: an element outside F in a table over F");
      touched[band_of(row)] = true;
    }
    auto m = std::make_unique<Main>();
    m->deg = t.deg, m->log_rows = t.log_rows, m->rows = t.rows, m->len = t.len;
    std::vector<size_t> begins(t.bands.size()); // each band's first word in the new table's memory
    size_t words = 0;
    for (size_t b = 0; b < t.bands.size(); b++) {
      const size_t cells = t.bands[b].rows * t.len * t.deg;
      begins[b] = words, words += touched[b] ? cells : (cells * cell_bytes(t.bands[b].cell) + 7) / 8;
    }
    m->own.resize(words);
    for (size_t b = 0; b < t.bands.size(); b++) {
      const Band &from = t.bands[b];
      const size_t cells = from.rows * t.len * t.deg;
      u64 *to = m->own.get() + begins[b];
      if (touched[b] && from.cell != Cell::WORD) {
        k_cells_to_words<<<blocks_for(cells, 256), 256, 0, stream_>>>(card_bands({from}, (u64)t.len * t.deg), cells, to);
        ENGINE_CK(cudaGetLastError());
      } else if (cells)
        ENGINE_CK(cudaMemcpyAsync(to, from.cells, cells * cell_bytes(from.cell), cudaMemcpyDeviceToDevice, stream_));
      m->bands.push_back(Band{to, touched[b] ? Cell::WORD : from.cell, from.rows});
    }
    std::vector<u64> at, value;
    for (const std::pair<u64, E> &c : changed) {
      const u64 row = c.first & (((u64)1 << t.log_rows) - 1), place = c.first >> t.log_rows;
      const size_t b = band_of(row);
      for (unsigned d = 0; d < t.deg; d++)
        at.push_back(begins[b] + ((row - first_row[b]) * t.len + place) * t.deg + d), value.push_back(c.second.c[d].v);
    }
    if (!at.empty()) {
      Card<u64> where, what;
      where.from_host(at.data(), at.size()), what.from_host(value.data(), value.size());
      k_replace_words<<<blocks_for(at.size(), 128), 128, 0, stream_>>>(m->own.get(), where.get(), what.get(), at.size());
      ENGINE_CK(cudaGetLastError());
      ENGINE_CK(cudaStreamSynchronize(stream_)); // `where` and `what` are freed next
    }
    return keep(std::move(m));
  }
  // Writes in place. In an adopted table, which is the caller's memory, the entries are also
  // remembered and moved() writes them again: a caller that refills the table need not know them.
  void set_entries(TableId id, const std::vector<std::pair<u64, E>> &changed) override {
    mine();
    Main &t = main(id);
    BLK_REQUIRE(!t.lender && !t.lent, "set_entries: a borrowed table, or one that is lent");
    for (const auto &c : commits_)
      if (c)
        for (const OraclePart &p : c->parts) BLK_REQUIRE(p.main != id, "set_entries: a tree that is not closed stands over this table");
    for (size_t f = 0; f < folds_.size(); f++)
      if (folds_[f])
        for (TableId r : fold_reads_[f].tables) BLK_REQUIRE(r != id, "set_entries: a fold that has not ended reads this table");
    std::vector<u64> seen;
    for (const std::pair<u64, E> &c : changed) seen.push_back(c.first);
    std::sort(seen.begin(), seen.end());
    BLK_REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "set_entries: an index comes twice");
    write_entries(t, changed);
    if (!t.own.size())
      for (const std::pair<u64, E> &c : changed) t.set_here[c.first] = c.second;
  }
  // Returns a new table, zero except that entry j < count of the one-row table `id` is at row
  // first_row + (j mod 2^bits), place j div 2^bits.
  TableId laid(TableId id, u64 count, u64 first_row, unsigned bits, size_t rows, size_t len) override {
    mine();
    const Main &t = main(id);
    BLK_REQUIRE(t.row_of_E() && count <= t.len, "laid: one row over E in whole words, and no more entries than it has");
    BLK_REQUIRE(bits < 64 && first_row + ((u64)1 << bits) <= rows && ((count + ((u64)1 << bits) - 1) >> bits) <= len,
                "laid: the block of rows does not lie in the new table, or is too short");
    auto m = std::make_unique<Main>();
    m->deg = E_DEGREE, m->log_rows = ceil_log2(rows), m->rows = rows, m->len = len;
    m->own.resize(rows * len * E_DEGREE);
    m->bands = {Band{m->own.get(), Cell::WORD, rows}};
    ENGINE_CK(cudaMemsetAsync(m->own.get(), 0, rows * len * E_DEGREE * sizeof(u64), stream_));
    if (count) {
      k_lay_row<<<blocks_for(count, 256), 256, 0, stream_>>>(m->own.get(), t.row_of_E(), count, first_row, bits, len);
      ENGINE_CK(cudaGetLastError());
    }
    return keep(std::move(m));
  }
  E evaluate(TableId id, const Tensor &v) override {
    mine();
    const Main &m = main(id);
    return evaluate_on_card(m.bands, m.deg, m.rows, m.log_rows, m.len, v, stream_, factor_of(v));
  }
  // If v has any `engine_terms` (vectors that only the engine stores), or more than two tensors and
  // single entries together on a table of at most 2^10 rows, CardFold (fold.cuh) computes the inner
  // product as the trace of its Gram matrix; a vector that several tensors share is multiplied
  // once. Otherwise each tensor takes a pass over the table, and each single entry a read of its
  // cells. CardFold takes at most 2^10 rows: a Form with engine terms on a larger table ends the
  // process.
  E evaluate(TableId id, const Form &v) override {
    mine();
    const Main &m = main(id);
    BLK_REQUIRE((m.len & (m.len - 1)) == 0 && v.fits(m.log_rows + ceil_log2(m.len)), "evaluate: the vector and the table differ in length");
    BLK_REQUIRE(v.frozen.empty(), "evaluate: a frozen part in a prover's vector"); // only a verifier's Form has them (blocks/iop/tensor.h)
    if ((v.tensors.size() + v.entries.size() > 2 && m.log_rows <= 10) || !v.engine_terms.empty()) {
      const std::vector<const u64 *> held = vectors_of(v, m.log_rows, m.len);
      const Code lengths{1, m.len, 0}; // CardFold reads only a code's message and tail lengths
      std::vector<CardRows> rows;
      m.card_rows(0, nullptr, 0, rows);
      FoldSetting only_the_sum;
      only_the_sum.folded_coefficients = false; // nothing is folded here
      return CardFold(rows, m.log_rows, lengths, v, held, stream_, only_the_sum).inner_product();
    }
    E sum = E::zero();
    for (const Tensor &t : v.tensors) sum += evaluate_on_card(m.bands, m.deg, m.rows, m.log_rows, m.len, t, stream_, factor_of(t));
    if (v.entries.empty()) return sum;
    std::vector<u64> at;
    for (const std::pair<u64, E> &e : v.entries) at.push_back(e.first);
    std::vector<E> there = entries_of(m, at);
    for (size_t i = 0; i < v.entries.size(); i++) sum += v.entries[i].second * there[i];
    wipe(there.data(), sizeof(E) * there.size());
    return sum;
  }

  // ---- oracles ----
  // Draws from `coins` in the order that Engine::commit() prescribes, on which equality with the
  // CPU engine's tree rests: first the random tails of the main rows, part by part; then one salt
  // for each leaf.
  CommitId commit(const std::vector<OraclePart> &parts, const OracleTreeShape &shape, Coins &coins, Digest &root) override {
    mine();
    BLK_REQUIRE(parts.size() == shape.main_symbols.size() && parts.size() == shape.mask_symbols.size(),
                "commit: the shape speaks of another number of parts");
    BLK_REQUIRE(shape.valid() && shape.salt_bytes % 8 == 0, "commit: not a shape, or a salt length that is not a multiple of 8 bytes");
    CardCoins *card = dynamic_cast<CardCoins *>(&coins);
    CardStreamCoins *look = dynamic_cast<CardStreamCoins *>(&coins), *streams = keep_.salts ? nullptr : look; // look: for the check at this call's end
    std::vector<u32> tail_stream(parts.size(), 0);
    auto c = std::make_unique<Committed>();
    c->parts = parts, c->shape = shape, c->lean = !keep_.nodes;
    for (size_t i = 0; i < parts.size(); i++) {
      const OraclePart &p = parts[i];
      Card<u64> tail;
      size_t main_rows = 0, mask_rows = 0;
      if (p.main != NO_ID) {
        const Main &m = main(p.main);
        BLK_REQUIRE(m.len == p.code.ell && p.code.log_n == shape.log_n && code_ok(p.code),
                    "commit: a main row does not fit its code or its tree");
        draw(tail, m.rows * p.code.r * m.deg, coins, card);
        if (look) tail_stream[i] = look->last_stream();
        main_rows = m.rows * m.deg;
      }
      if (p.mask.rows) {
        BLK_REQUIRE(p.mask.deg == E_DEGREE && p.mask_tail.deg == E_DEGREE, "commit: mask rows are over E");
        BLK_REQUIRE(p.mask.len == p.mask_code.ell && p.mask_tail.len == p.mask_code.r && p.mask_tail.rows == p.mask.rows &&
                        p.mask_code.log_n == shape.log_n_zk,
                    "commit: a mask row does not fit the mask code or its tree");
        BLK_REQUIRE(code_ok(p.mask_code), "commit: the mask code's numbers are no code");
        mask_rows = p.mask.rows * E_DEGREE;
      }
      BLK_REQUIRE(main_rows == shape.main_symbols[i] && mask_rows == shape.mask_symbols[i], "commit: rows other than the shape says");
      c->tail.push_back(std::move(tail));
    }
    const size_t salt_words = shape.salt_bytes / 8;
    c->coins = streams;
    draw_bytes(c->salt, ((size_t)1 << shape.log_n) * salt_words, coins, card); // a salt for each leaf
    const u32 salt_stream = look ? look->last_stream() : 0;
    if (streams) c->salt_stream = salt_stream;
    lay(*c);
    root = c->root = make(*c, true);
    // Self-check of the randomness in GPU memory (prover/engine/stream_coins.cuh explains it), only
    // with CardStreamCoins: held_card() queues a comparison of every main row's random tail and of
    // every leaf's salt with the same values generated a second time; the caller reads the outcome,
    // as counts, from CardStreamCoins::bound(). Salts that are not kept are compared before their
    // buffer is released: open() regenerates only an opened leaf's salt, which the verifier's
    // Merkle path check tests against the tree; the rest may never be generated again. The expected
    // counts come from `shape` (for tails, times the code's r): if they came from the draw itself,
    // a draw of too few values would shorten the buffer, the generator's list of draws (ledger())
    // and the check alike, and go unseen. The check is not in make(), which back_on_card() calls
    // too: there the new root is compared with the committed one.
    if (look) {
      for (size_t i = 0; i < parts.size(); i++) {
        if (parts[i].main == NO_ID) continue;
        const size_t entries = shape.main_symbols[i] * parts[i].code.r;
        BLK_REQUIRE(c->tail[i].size() == entries, "commit: main rows have another number of random entries than shape and code say");
        look->held_card(tail_stream[i], true, c->tail[i].get(), entries, stream_);
      }
      const size_t salts = salt_words << shape.log_n;
      BLK_REQUIRE(c->salt.size() == salts, "commit: the salts have another number of words than the shape says");
      look->held_card(salt_stream, false, c->salt.get(), salts, stream_);
    }
    // Released only now: after the queued comparison, in the order that stream_ keeps with Card's.
    if (c->coins) c->salt.release();
    c->bytes += 8 * c->salt.size(), tree_bytes_ += 8 * c->salt.size(); // the salts that stay
    if (tree_bytes_ > tree_bytes_peak_) tree_bytes_peak_ = tree_bytes_;
    commits_.push_back(std::move(c));
    return (CommitId)(commits_.size() - 1);
  }
  // ---- not in the interface: a tree that leaves GPU memory for a while ----
  // The trees in GPU memory: committed, not closed, not between leave_card() and back_on_card(),
  // not borrowed. For a caller that must free memory and has only the engine. The list includes
  // lent trees and trees that an unfinished fold reads: on these leave_card() ends the process.
  std::vector<CommitId> trees_on_card() const {
    mine();
    std::vector<CommitId> all;
    for (size_t c = 0; c < commits_.size(); c++)
      if (commits_[c] && !commits_[c]->off && !commits_[c]->of) all.push_back((CommitId)c); // a borrowed tree is the lender's to unload
    return all;
  }
  // Frees all GPU memory that the engine holds for this tree: random tails and kept salts are
  // copied to host memory; digests and the GPU copies of mask rows are dropped. Until
  // back_on_card(), close() is the only other call that accepts the tree; the rest end the process.
  // The main tables remain the caller's concern: an adopted table may be freed and filled again
  // meanwhile, also at another address (moved()).
  void leave_card(CommitId id) {
    mine();
    Committed &c = committed(id);
    BLK_REQUIRE(!c.of && !c.lent, "leave_card: a borrowed tree, or one that is lent");
    for (size_t f = 0; f < folds_.size(); f++)
      if (folds_[f])
        for (CommitId t : fold_reads_[f].trees) BLK_REQUIRE(t != id, "leave_card: a fold that has not ended reads this tree's rows");
    settle();
    auto down = [](Card<u64> &from, std::vector<u64> &to) {
      to.resize(from.size());
      from.to_host(to.data());
      from.release();
    };
    c.tail_off.resize(c.tail.size());
    for (size_t i = 0; i < c.tail.size(); i++) down(c.tail[i], c.tail_off[i]);
    down(c.salt, c.salt_off);
    c.mask_words.clear(), c.rows.clear(), c.mask_rows.clear();
    c.tree.release();
    tree_bytes_ -= c.bytes, c.bytes = 0;
    c.off = true;
  }
  // Rebuilds the tree from its rows, tails and salts, at the cost of a commit(). Ends the process
  // if the root differs from commit()'s: the rows are then not the committed ones.
  void back_on_card(CommitId id) {
    mine();
    BLK_REQUIRE(id < commits_.size() && commits_[id] && commits_[id]->off, "back_on_card: no such tree, or one that has not left");
    Committed &c = *commits_[id];
    auto up = [](std::vector<u64> &from, Card<u64> &to) {
      to.from_host(from.data(), from.size());
      wipe(from.data(), 8 * from.size());
      from.clear();
    };
    for (size_t i = 0; i < c.tail.size(); i++) up(c.tail_off[i], c.tail[i]);
    up(c.salt_off, c.salt);
    if (c.coins) {
      c.salt.resize((c.shape.salt_bytes / 8) << c.shape.log_n);
      c.coins->again_words_card(c.salt_stream, 0, c.salt.get(), c.salt.size(), stream_);
    }
    c.off = false;
    lay(c);
    BLK_REQUIRE(make(c) == c.root, "back_on_card: the tree that was made again has another root");
  }
  // Tells the engine that an adopted table is now at new addresses, one for each band. The caller
  // has filled them with the same cells (not checked), except that entries which set_entries()
  // wrote may be missing: they are written again here.
  void moved(TableId id, const void *cells_on_card) { moved(id, std::vector<const void *>{cells_on_card}); }
  void moved(TableId id, const std::vector<const void *> &bands_on_card) {
    mine();
    Main &m = main(id);
    BLK_REQUIRE(!m.own.size(), "moved: the engine's own table");
    BLK_REQUIRE(!m.lender && !m.lent, "moved: a borrowed table, or one that is lent");
    BLK_REQUIRE(bands_on_card.size() == m.bands.size(), "moved: another number of addresses than the table has bands");
    for (size_t f = 0; f < folds_.size(); f++)
      if (folds_[f])
        for (TableId t : fold_reads_[f].tables) BLK_REQUIRE(t != id, "moved: a fold that has not ended reads this table");
    settle();
    for (size_t b = 0; b < m.bands.size(); b++) m.bands[b].cells = bands_on_card[b];
    if (!m.set_here.empty()) write_entries(m, std::vector<std::pair<u64, E>>(m.set_here.begin(), m.set_here.end()));
    for (const auto &c : commits_)
      if (c && !c->off && !c->of) lay(*c);
  }
  // ---- oracles, continued ----
  void open(CommitId id, const PositionList &list, Opening &o) override {
    mine();
    Committed &c = committed(id);
    const OracleTreeShape &s = c.shape;
    (void)opening_given(s, list); // ends the process if `list` is empty or of another log_n
    o = Opening();
    o.main_width = s.main_width(), o.mask_width = s.mask_width();
    // Main and mask symbols are computed for all listed positions, then joined column by column. A
    // mask row is read at the short position (x mod 2^log_n_zk) of each listed position x, so
    // positions that share a short position each get their own copy of the mask symbols.
    // GPU buffers are sized by the number of draws, not by the number of distinct positions (both
    // defined in blocks/tree/positions.h), so their sizes do not depend on the drawn values. The
    // unused end of such a buffer is neither written nor read.
    const std::vector<u64> at = list.all();
    const size_t draws = list.draws().size();
    std::vector<u64> short_at(at.size());
    for (size_t i = 0; i < at.size(); i++) short_at[i] = short_position(s.log_n, s.log_n_zk, at[i]);
    std::vector<F> main_part, mask_part;
    if (s.has_main()) symbols_at(c.rows, s.log_n, at, draws, main_part);
    if (s.has_mask()) symbols_at(c.mask_rows, s.log_n_zk, short_at, draws, mask_part);
    o.columns.resize(at.size() * s.width());
    for (size_t i = 0; i < at.size(); i++) {
      std::copy(main_part.begin() + i * o.main_width, main_part.begin() + (i + 1) * o.main_width, o.columns.begin() + i * s.width());
      std::copy(mask_part.begin() + i * o.mask_width, mask_part.begin() + (i + 1) * o.mask_width, o.columns.begin() + i * s.width() + o.main_width);
    }
    salts_at(kept(c), at, draws, o.salts);
    o.given = kept(c).tree.open(list, stream_); // waits for the stream unless no digest is needed
    if (!keep_.work) encoder_.release();
  }
  Table tail(CommitId id, unsigned part) override {
    mine();
    Committed &c = committed(id);
    BLK_REQUIRE(part < c.parts.size(), "engine: no such part");
    if (c.parts[part].main == NO_ID) return Table();
    const Main &m = main(c.parts[part].main);
    Table t(m.deg, m.rows, c.parts[part].code.r);
    BLK_REQUIRE(t.a.size() == kept(c).tail[part].size(), "tail: the kept randomness has another size");
    ENGINE_CK(cudaStreamSynchronize(stream_));
    kept(c).tail[part].to_host(reinterpret_cast<u64 *>(t.a.data()));
    return t;
  }
  void close(CommitId id) override {
    mine();
    BLK_REQUIRE(id < commits_.size() && commits_[id], "engine: no such tree"); // accepts a tree after `leave_card` too
    for (size_t f = 0; f < folds_.size(); f++)
      if (folds_[f])
        for (CommitId c : fold_reads_[f].trees) BLK_REQUIRE(c != id, "close: a fold that has not ended reads this tree's rows");
    BLK_REQUIRE(!commits_[id]->lent, "close: another engine has borrowed this tree");
    settle();
    if (commits_[id]->of) { // the loan ends: the tree's, then its tables'
      commits_[id]->lender->still();
      ENGINE_CK(cudaStreamSynchronize(stream_)); // this engine's queued reads are then done
      const std::vector<OraclePart> parts = commits_[id]->parts;
      commits_[id]->of->lent = false;
      commits_[id].reset();
      for (const OraclePart &p : parts)
        if (p.main != NO_ID) drop(p.main);
      return;
    }
    tree_bytes_ -= commits_[id]->bytes;
    commits_[id].reset();
  }

  // ---- the fold ----
  // CardFold (fold.cuh) does the work. fold_begin() looks up the rows, their tails and the tables
  // that c refers to.
  FoldId fold_begin(const std::vector<FoldPart> &parts, unsigned k, const Form &c) override {
    mine();
    BLK_REQUIRE(!parts.empty(), "fold: no rows");
    const Code code = committed(parts[0].commit).parts[parts[0].part].code;
    std::vector<CardRows> rows;
    FoldReads reads;
    for (const FoldPart &p : parts) {
      const Committed &cm = committed(p.commit);
      BLK_REQUIRE(p.part < cm.parts.size() && cm.parts[p.part].main != NO_ID && cm.parts[p.part].code == code,
                  "fold: the rows of a group are under one code");
      const Main &m = main(cm.parts[p.part].main);
      m.card_rows(p.first_row, kept(cm).tail[p.part].get(), code.r, rows);
      reads.trees.push_back(p.commit), reads.tables.push_back(cm.parts[p.part].main);
    }
    const std::vector<const u64 *> held = vectors_of(c, k, code.ell, &reads.tables);
    folds_.push_back(std::make_unique<CardFold>(rows, k, code, c, held, stream_));
    fold_length_.push_back(code.ell);
    fold_reads_.push_back(std::move(reads));
    return (FoldId)(folds_.size() - 1);
  }
  void fold_round(FoldId f, E q[3]) override { mine(), folding(f).round(q); }
  void fold_bind(FoldId f, const E &gamma) override { mine(), folding(f).bind(gamma); }
  TableId fold_end(FoldId f, Table &tail, TableId *c_folded = nullptr) override {
    mine();
    CardFold &fold = folding(f);
    auto row = [&] {
      auto m = std::make_unique<Main>();
      m->deg = E_DEGREE, m->rows = 1, m->len = fold_length_[f];
      m->own.resize(m->len * E_DEGREE);
      m->bands = {Band{m->own.get(), Cell::WORD, 1}};
      return m;
    };
    auto message = row(), coefficients = c_folded ? row() : nullptr;
    fold.end(reinterpret_cast<E *>(message->own.get()), tail, coefficients ? reinterpret_cast<E *>(coefficients->own.get()) : nullptr);
    folds_[f].reset();
    if (c_folded) *c_folded = keep(std::move(coefficients));
    return keep(std::move(message));
  }

  // ---- proof of work ----
  // 2^63 - 1 tries: the most that WorkSearcher (blocks/hash/work_search.cuh) takes.
  u64 search_nonce(const ChainPrefix &prefix, unsigned bits) override {
    mine();
    const u64 nonce = searcher_.search(prefix, bits, ((u64)1 << 63) - 1, stream_);
    BLK_REQUIRE(nonce != WORK_NONE, "no nonce passes the work test");
    return nonce;
  }

  // Returns the encoder's work space and the pool's unused memory to the driver. Waits for the GPU.
  void release_work_space() {
    mine();
    encoder_.release();
    card_pool_gives_back();
    pool_trim(); // pooled.cuh's name for the same pool: a second trim, which finds nothing left
  }

private:
  // A main table: 2^log_rows rows of `len` entries, of which the first `rows` are real; `deg`:
  // cells per entry (1: over F, the base field; 3: over E, its cubic extension).
  struct Main {
    unsigned deg = 1, log_rows = 0;
    size_t rows = 0, len = 0;
    std::vector<Band> bands; // cover all real rows, in row order
    Card<u64> own;           // empty if the table is the caller's
    std::map<u64, E> set_here; // adopted table: what `set_entries` wrote; may be private randomness
    EngineGpu *lender = nullptr; // if borrowed: the lender, and the table's number there (`theirs`)
    TableId theirs = NO_ID;
    bool lent = false;           // another engine has borrowed it
    ~Main() {
      for (auto &e : set_here) wipe(&e.second, sizeof e.second);
    }
    CardBands on_card() const { return card_bands(bands, (u64)len * deg); }
    // Appends the table's rows in the form that CardFold takes, one CardRows for each band. They
    // become the rows from first_row on of the group that CardFold reads (a group: a set of oracle
    // messages under one base code; `Group`, blocks/iop/relation.h). tail: the rows' tails, r
    // entries for each row, or null.
    void card_rows(u64 first_row, const u64 *tail, size_t r, std::vector<CardRows> &out) const {
      size_t first = 0;
      for (const Band &b : bands) {
        out.push_back(CardRows{b.cells, b.cell, deg, b.rows, first_row + first, tail ? tail + first * r * deg : nullptr});
        first += b.rows;
      }
    }
    // The table's address if it is one row over E in 64-bit words (Cell::WORD); else null.
    const u64 *row_of_E() const {
      return deg == E_DEGREE && rows == 1 && bands[0].cell == Cell::WORD ? (const u64 *)bands[0].cells : nullptr;
    }
  };
  // The entries at these indices, where index = row + place * 2^log_rows. A row beyond the real
  // ones (row >= m.rows) reads as zero. The entries can be secret (the random pads that mask a
  // proof's answers about a table: see write_entries()), so entries_of() zeroes its own two copies,
  // in GPU memory and in host memory, before their memory is freed. Wiping the returned vector is
  // left to the caller. Neither this function nor evaluate() tests for the 0xFF fill below; the two
  // checks of send_prove() (blocks/iop/send.h) that read through entries() fail on it:
  // own-test-pad-place and own-test-pad-kept.
  std::vector<E> entries_of(const Main &m, const std::vector<u64> &indices) {
    std::vector<E> out(indices.size());
    if (indices.empty()) return out;
    std::vector<u64> at;
    for (u64 j : indices) {
      const u64 row = j & (((u64)1 << m.log_rows) - 1), place = j >> m.log_rows;
      BLK_REQUIRE(place < m.len, "engine: an index beyond the table");
      for (unsigned d = 0; d < m.deg; d++) at.push_back(row < m.rows ? (row * m.len + place) * m.deg + d : ~(u64)0);
    }
    Card<u64> where, cells(at.size());
    where.from_host(at.data(), at.size());
    ENGINE_CK(cudaMemsetAsync(cells.get(), 0xFF, 8 * at.size(), stream_)); // no field element is all ones: if the kernel below never ran, no entry reads as zero or as any value that was written
    k_pick_cells<<<blocks_for(at.size(), 128), 128, 0, stream_>>>(m.on_card(), where.get(), at.size(), cells.get());
    ENGINE_CK(cudaGetLastError());
    ENGINE_CK(cudaStreamSynchronize(stream_));
    std::vector<u64> got(at.size());
    cells.to_host(got.data());
    ENGINE_CK(cudaMemsetAsync(cells.get(), 0, 8 * at.size(), stream_)); // it goes back to the memory pool at the end of this function
    ENGINE_CK(cudaStreamSynchronize(stream_));
    for (size_t i = 0; i < out.size(); i++)
      out[i] = m.deg == 1 ? E::from(F{got[i]}) : E::from(F{got[3 * i]}, F{got[3 * i + 1]}, F{got[3 * i + 2]});
    wipe(got.data(), 8 * got.size());
    return out;
  }
  // Writes entries in place. Each must lie in a real row of Cell::WORD cells, or the process ends.
  // Then reads them back and ends the process if they differ.
  void write_entries(const Main &t, const std::vector<std::pair<u64, E>> &changed) {
    std::vector<size_t> first_row(t.bands.size() + 1, 0); // of each band
    for (size_t b = 0; b < t.bands.size(); b++) first_row[b + 1] = first_row[b] + t.bands[b].rows;
    std::vector<std::vector<u64>> at(t.bands.size()), value(t.bands.size()); // per band: word offsets, new values
    for (const std::pair<u64, E> &c : changed) {
      const u64 row = c.first & (((u64)1 << t.log_rows) - 1), place = c.first >> t.log_rows;
      BLK_REQUIRE(row < t.rows && place < t.len, "set_entries: no such entry (rows beyond the real ones are zero and cannot be set)");
      BLK_REQUIRE(t.deg == E_DEGREE || (c.second.c[1] == F::zero() && c.second.c[2] == F::zero()),
                  "set_entries: an element outside F in a table over F");
      const size_t b = (size_t)(std::upper_bound(first_row.begin(), first_row.end(), row) - first_row.begin()) - 1;
      BLK_REQUIRE(t.bands[b].cell == Cell::WORD, "set_entries: the place lies in a row of narrow cells");
      const u64 written = place;
      for (unsigned d = 0; d < t.deg; d++)
        at[b].push_back(((row - first_row[b]) * t.len + written) * t.deg + d), value[b].push_back(c.second.c[d].v);
    }
    for (size_t b = 0; b < t.bands.size(); b++) {
      if (at[b].empty()) continue;
      Card<u64> where, what;
      where.from_host(at[b].data(), at[b].size()), what.from_host(value[b].data(), value[b].size());
      k_replace_words<<<blocks_for(at[b].size(), 128), 128, 0, stream_>>>((u64 *)const_cast<void *>(t.bands[b].cells), where.get(), what.get(),
                                                                          at[b].size());
      ENGINE_CK(cudaGetLastError());
      // Self-check [own-test-entries-written]: the words are read back from the table and compared
      // with what was sent. The entries can be pads: the random values with which send_prove()
      // (blocks/iop/send.h) masks the answers that a proof gives about a table. If the write were
      // lost, the places would keep what the memory held (zeros, where send_prove() has just
      // checked them: own-test-pad-place); the proof would still verify, and the answers would go
      // out unmasked. The receiving buffer is first filled with 0xFF bytes, and no field element's
      // word is all ones, so the check also fails if the reading kernel never runs.
      // The check does not catch:
      // (1) a value lost in host memory before this call: it compares with what was sent, not with
      //     what was drawn;
      // (2) a wrong place: writing and reading use the same list of places. send_prove() reads its
      //     pads back through entries(), which computes addresses separately (own-test-pad-kept),
      //     but only once, after its own write: the entries that moved() writes again later are
      //     read back by this check alone;
      // (3) a later loss: if the caller refills an adopted table without calling moved(), the
      //     entries are lost unnoticed.
      Card<u64> read(at[b].size());
      ENGINE_CK(cudaMemsetAsync(read.get(), 0xFF, 8 * at[b].size(), stream_));
      k_pick_words<<<blocks_for(at[b].size(), 128), 128, 0, stream_>>>((const u64 *)t.bands[b].cells, where.get(), at[b].size(), 1, read.get());
      ENGINE_CK(cudaGetLastError());
      ENGINE_CK(cudaStreamSynchronize(stream_));
      std::vector<u64> back(at[b].size());
      read.to_host(back.data());
      ENGINE_CK(cudaMemsetAsync(read.get(), 0, 8 * at[b].size(), stream_)); // neither copy of the values goes back to the memory pool
      ENGINE_CK(cudaMemsetAsync(what.get(), 0, 8 * at[b].size(), stream_));
      ENGINE_CK(cudaStreamSynchronize(stream_)); // the three lists are freed as this iteration ends
      u64 differ = 0; // all words are compared, whatever the first ones give
      for (size_t i = 0; i < back.size(); i++) differ |= back[i] ^ value[b][i];
      wipe(back.data(), 8 * back.size()), wipe(value[b].data(), 8 * value[b].size()); // they may be private randomness
      BLK_REQUIRE_NAMED(!differ, "own-test-entries-written",
                        "internal error (own-test-entries-written): entries that were written into a table in GPU memory do not read back as they were sent. The program or the machine has a fault; nothing of this run may be used");
    }
  }
  // The address of the table that holds a Tensor's dense factor (Tensor::held,
  // blocks/iop/tensor.h), or null if no table holds it. Ends the process unless the table is one
  // row over E in 64-bit words, has held_length entries and contains the factor's slice:
  // 2^dense_vars.size() entries from `first` on. Both evaluate() and fold_begin() get the factor
  // through this check.
  const u64 *factor_of(const Tensor &t, std::vector<TableId> *used = nullptr) {
    if (!t.is_held()) return nullptr;
    const Main &m = main(t.held); // ends the process if that table was dropped
    BLK_REQUIRE(m.row_of_E() && m.len == t.held_length && t.dense_vars.size() < 64 && t.first <= m.len &&
                    ((u64)1 << t.dense_vars.size()) <= m.len - t.first,
                "engine: a Tensor's held factor is no row of E in whole words, has another length than the Tensor states, or reaches beyond its table");
    if (used) used->push_back(t.held);
    return m.row_of_E();
  }
  // The pointers that CardFold takes with a coefficient vector c on k row variables, for rows of
  // ell entries: one for each hint (c.held: a table that may be read in place of some of c's
  // parts), then one for each engine term (c.engine_terms: a part of c that only a table stores),
  // then factor_of() for each Tensor of c.tensors with a held factor, in order (`Held`,
  // `EngineTerm`: blocks/iop/tensor.h).
  // A hint's or a term's pointer is its table's address if the table is one row over E in 64-bit
  // words with ell << open entries (`open`: how many of the k row variables are outside
  // choice_mask, the index bits that the hint or term fixes) and choice_mask has no bit from k up;
  // otherwise null. CardFold skips a hint with a null pointer and ends the process on a term with
  // one. used: gets the numbers of the tables whose addresses are returned.
  std::vector<const u64 *> vectors_of(const Form &c, unsigned k, size_t ell, std::vector<TableId> *used = nullptr) {
    std::vector<const u64 *> all;
    auto one = [&](TableId vector, u64 choice_mask) {
      const Main *m = vector < tables_.size() ? tables_[vector].get() : nullptr;
      unsigned open = 0; // the row variables that are not picked
      for (unsigned t = 0; t < k; t++) open += !((choice_mask >> t) & 1);
      const bool fits = m && m->row_of_E() && m->len == ell << open && (choice_mask >> k) == 0;
      all.push_back(fits ? m->row_of_E() : nullptr);
      if (fits && used) used->push_back(vector);
    };
    for (const Held &h : c.held) one(h.vector, h.choice_mask);
    for (const EngineTerm &t : c.engine_terms) one(t.vector, t.choice_mask);
    for (const Tensor &t : c.tensors)
      if (t.is_held()) all.push_back(factor_of(t, used));
    return all;
  }
  struct FoldReads { // what a fold points into until it ends
    std::vector<CommitId> trees;
    std::vector<TableId> tables;
  };
  struct Committed {
    std::vector<OraclePart> parts;
    OracleTreeShape shape;
    std::vector<RowSource> rows;  // main rows, one per coordinate over F, in a column's order
    std::vector<Card<u64>> tail;  // part by part
    std::vector<RowSource> mask_rows;  // mask rows, in the same form and order
    std::vector<Card<u64>> mask_words; // GPU copies of each part's mask rows and their tails
    Card<u64> salt; // one per leaf; if not kept: empty after `commit`, regenerated from these two:
    const CardStreamCoins *coins = nullptr;
    u32 salt_stream = 0;
    CardTree tree;
    // true if TreeKeep::nodes was false at commit() (see there)
    bool lean = false;
    Digest root;
    size_t bytes = 0; // what it holds in GPU memory
    Committed *of = nullptr;     // a borrowed tree: the lender's record, with tails, salts, digests
    EngineGpu *lender = nullptr;
    bool lent = false;           // another engine has borrowed it
    bool off = false; // from `leave_card` to `back_on_card`; tails and kept salts are then in:
    std::vector<std::vector<u64>> tail_off;
    std::vector<u64> salt_off;
    ~Committed() { // private randomness in host memory
      for (std::vector<u64> &t : tail_off) wipe(t.data(), 8 * t.size());
      wipe(salt_off.data(), 8 * salt_off.size());
    }
  };

  // Points c.rows and c.mask_rows at the rows' present addresses. Copies the mask rows to the GPU
  // if they are not there.
  void lay(Committed &c) {
    c.rows.clear(), c.mask_rows.clear();
    const bool masks_there = !c.mask_words.empty();
    size_t word = 0;
    for (size_t i = 0; i < c.parts.size(); i++) {
      const OraclePart &p = c.parts[i];
      if (p.main != NO_ID) {
        const Main &m = main(p.main);
        size_t row = 0;
        for (const Band &band : m.bands)
          for (size_t b = 0; b < band.rows; b++, row++)
            for (unsigned d = 0; d < m.deg; d++)
              c.rows.push_back(RowSource{(const u8 *)band.cells + (b * m.len * m.deg + d) * cell_bytes(band.cell),
                                         c.tail[i].get() + row * p.code.r * m.deg + d, m.deg, m.deg, band.cell, p.code.ell, p.code.r,
                                         p.code.message_at(), p.code.tail_at()});
      }
      if (p.mask.rows) {
        if (!masks_there) {
          Card<u64> message, mask_tail;
          message.from_host(reinterpret_cast<const u64 *>(p.mask.a.data()), p.mask.a.size());
          mask_tail.from_host(reinterpret_cast<const u64 *>(p.mask_tail.a.data()), p.mask_tail.a.size());
          c.mask_words.push_back(std::move(message));
          c.mask_words.push_back(std::move(mask_tail));
        }
        const u64 *message = c.mask_words[word].get(), *mask_tail = c.mask_words[word + 1].get();
        word += 2;
        for (size_t b = 0; b < p.mask.rows; b++)
          for (unsigned d = 0; d < E_DEGREE; d++)
            c.mask_rows.push_back(RowSource{message + b * p.mask.len * E_DEGREE + d, mask_tail + b * p.mask_code.r * E_DEGREE + d, E_DEGREE,
                                            E_DEGREE, Cell::WORD, p.mask_code.ell, p.mask_code.r, p.mask_code.message_at(),
                                            p.mask_code.tail_at()});
      }
    }
  }
  // Encodes the rows, hashes leaves and nodes, and returns the root. c.salt must be filled; it is
  // released at the end if salts are not kept; with `salts_stay` the caller releases and counts it.
  Digest make(Committed &c, bool salts_stay = false) {
    const OracleTreeShape &shape = c.shape;
    const size_t salt_words = shape.salt_bytes / 8;
    c.tree.begin(shape.tree(), c.lean);
    const LeafHash leaves(shape); // only its tag is used; `LeafHash` is in blocks/tree/opening.h
    // A leaf hashes the main rows' symbols at x, the mask rows' symbols at the short position of x,
    // and the salt. In a tree with main rows, the mask rows' codewords, which are short, are
    // computed whole first and copied into the leaves; without main rows, the mask rows are the
    // tree's rows.
    const std::vector<RowSource> &rows = shape.has_main() ? c.rows : c.mask_rows;
    const size_t copied = shape.has_main() ? c.mask_rows.size() : 0;
    Card<u64> short_words(copied << shape.log_n_zk);
    if (copied) encoder_.store(c.mask_rows, shape.log_n_zk, tables_of_codes_, short_words.get(), stream_);
    LeafChain chain{nullptr,
                    c.salt.get(),
                    c.tree.bottom(),
                    (u32)salt_words,
                    leaves.tag,
                    (u32)rows.size(),
                    (u32)(leaf_salt_at<HashFn>(rows.size() + copied, shape.salt_bytes) / 8),
                    short_words.get(),
                    (u32)copied,
                    shape.log_n_zk};
    encoder_.leaves(rows, shape.log_n, tables_of_codes_, chain, stream_);
    const Digest root = c.tree.finish(stream_); // waits for the stream
    if (c.coins && !salts_stay) c.salt.release();
    if (!keep_.work) encoder_.release();
    c.bytes = c.tree.bytes() + (salts_stay ? 0 : 8 * c.salt.size());
    for (const Card<u64> &t : c.tail) c.bytes += 8 * t.size();
    for (const Card<u64> &w : c.mask_words) c.bytes += 8 * w.size();
    tree_bytes_ += c.bytes;
    if (tree_bytes_ > tree_bytes_peak_) tree_bytes_peak_ = tree_bytes_;
    return root;
  }
  // draw(), draw_bytes(): fill `out` from `coins`, on the GPU if they are CardCoins (`card`,
  // card_coins.cuh), otherwise on the host and then copied.
  void draw(Card<u64> &out, size_t n, Coins &coins, CardCoins *card) {
    out.resize(n);
    if (card) return card->fill_card(reinterpret_cast<F *>(out.get()), n, stream_);
    std::vector<F> host(n);
    coins.fill(host.data(), n);
    out.from_host(reinterpret_cast<const u64 *>(host.data()), n);
  }
  void draw_bytes(Card<u64> &out, size_t words, Coins &coins, CardCoins *card) {
    out.resize(words);
    if (card) return card->bytes_card(reinterpret_cast<u8 *>(out.get()), words * 8, stream_);
    std::vector<u64> host(words); // in memory order: each word's low byte first (little-endian)
    coins.bytes(reinterpret_cast<u8 *>(host.data()), words * 8);
    out.from_host(host.data(), words);
  }
  // symbols[i * rows.size() + row] = the symbol of that row at positions[i]; positions may repeat.
  // If most <= 65535 and all rows have the same message length, a power of two, and the same tail
  // length, columns() computes the symbols as inner products with rows of the code's generator
  // matrix (columns.cuh). Otherwise the rows are encoded again as in commit(), except that the last
  // pass (layers 7 to 0, encode.cuh) runs one thread block per listed position and hashes nothing.
  // most: at least positions.size(); buffers are sized by it (see open()).
  void symbols_at(const std::vector<RowSource> &rows, unsigned log_n, const std::vector<u64> &positions, size_t most, std::vector<F> &symbols) {
    const size_t t = positions.size();
    BLK_REQUIRE(t <= most, "engine: more positions than the room that is taken for them");
    Card<u64> at(most), picked(most * rows.size());
    if (t) ENGINE_CK(cudaMemcpy(at.get(), positions.data(), t * sizeof(u64), cudaMemcpyHostToDevice));
    bool one_code = !rows.empty() && (rows[0].ell & (rows[0].ell - 1)) == 0 && most <= 65535;
    for (const RowSource &r : rows) one_code = one_code && r.ell == rows[0].ell && r.r == rows[0].r;
    if (one_code)
      columns(rows, log_n, at.get(), t, picked.get(), stream_, ProductWay::BYTES, 0, most);
    else
      encoder_.gather(rows, log_n, tables_of_codes_, at.get(), t, picked.get(), stream_);
    ENGINE_CK(cudaStreamSynchronize(stream_));
    symbols.resize(t * rows.size());
    if (!symbols.empty()) ENGINE_CK(cudaMemcpy(symbols.data(), picked.get(), symbols.size() * sizeof(u64), cudaMemcpyDeviceToHost));
  }
  // The salts of the leaves at these positions, one after the other. most: as in symbols_at().
  void salts_at(const Committed &c, const std::vector<u64> &positions, size_t most, std::vector<u8> &salts) {
    const size_t t = positions.size(), salt_words = c.shape.salt_bytes / 8;
    BLK_REQUIRE(t <= most, "engine: more positions than the room that is taken for them");
    Card<u64> at(most), picked(most * salt_words);
    if (t) ENGINE_CK(cudaMemcpy(at.get(), positions.data(), t * sizeof(u64), cudaMemcpyHostToDevice));
    if (c.coins)
      c.coins->again_words_at_card(c.salt_stream, at.get(), t, (unsigned)salt_words, (u64)1 << c.shape.log_n, picked.get(), stream_);
    else {
      k_pick_words<<<blocks_for(t * salt_words, 128), 128, 0, stream_>>>(c.salt.get(), at.get(), t, (unsigned)salt_words, picked.get());
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaStreamSynchronize(stream_));
    salts.resize(t * c.shape.salt_bytes);
    if (!salts.empty()) ENGINE_CK(cudaMemcpy(salts.data(), picked.get(), salts.size(), cudaMemcpyDeviceToHost));
  }
  // Memory returns to the pool in the default stream's order (device.cuh). Before a table or a tree
  // is freed or moved, an engine on another stream therefore waits for its own stream's work.
  void settle() {
    if (stream_) ENGINE_CK(cudaStreamSynchronize(stream_));
  }
  TableId keep(std::unique_ptr<Main> m) {
    tables_.push_back(std::move(m));
    return (TableId)(tables_.size() - 1);
  }
  Main &main(TableId t) {
    BLK_REQUIRE(t < tables_.size() && tables_[t], "engine: no such table");
    if (tables_[t]->lender) tables_[t]->lender->still();
    return *tables_[t];
  }
  Main &main_of_mine(TableId t) { // for another engine that borrows or gives back
    BLK_REQUIRE(t < tables_.size() && tables_[t], "engine: no such table");
    return *tables_[t];
  }
  Committed &committed(CommitId c) {
    BLK_REQUIRE(c < commits_.size() && commits_[c], "engine: no such tree");
    BLK_REQUIRE(!commits_[c]->off, "engine: this tree has left the GPU: back_on_card() first");
    if (commits_[c]->of) commits_[c]->lender->still();
    return *commits_[c];
  }
  // The `Committed` with a tree's tails, salts and digests: the lender's, if the tree is borrowed.
  static Committed &kept(Committed &c) { return c.of ? *c.of : c; }
  static const Committed &kept(const Committed &c) { return c.of ? *c.of : c; }
  EngineGpu &of_my_kind(Engine &from) {
    EngineGpu *lender = dynamic_cast<EngineGpu *>(&from);
    BLK_REQUIRE(lender && lender != this, "borrow: from an engine of another kind, or from oneself");
    lender->still();
    return *lender;
  }
  // The calling thread, as a number that is not 0: a hash of its id, so not guaranteed unique.
  static size_t me() { return std::hash<std::thread::id>()(std::this_thread::get_id()) | 1; }
  void mine() const { BLK_REQUIRE(holder_.load(std::memory_order_acquire) == me(), "engine: called by a thread that does not hold it"); }
  // Requires that the calling thread holds this engine or that no thread does. Called on a lender
  // at a borrower's call, and in the destructor. No call can test that a lender still exists;
  // instead, a lender's destructor ends the process while a loan is open.
  void still() const {
    const size_t h = holder_.load(std::memory_order_acquire);
    BLK_REQUIRE(h == 0 || h == me(), "engine: another thread holds this engine (a lender at a borrower's call, or an engine at its end)");
  }
  CardFold &folding(FoldId f) {
    BLK_REQUIRE(f < folds_.size() && folds_[f], "engine: no such fold");
    return *folds_[f];
  }
  cudaStream_t stream_;
  std::atomic<size_t> holder_{me()}; // `me` of the thread that holds the engine; 0: none
  TreeKeep keep_;
  size_t tree_bytes_ = 0, tree_bytes_peak_ = 0;
  std::vector<std::unique_ptr<Main>> tables_;
  std::vector<std::unique_ptr<Committed>> commits_;
  std::vector<std::unique_ptr<CardFold>> folds_;
  std::vector<size_t> fold_length_; // the message length of each fold
  std::vector<FoldReads> fold_reads_;
  WorkSearcher searcher_;
  // the encoder's transform tables, one per codeword length (`CardTable`, encode.cuh)
  CardTable tables_of_codes_;
  SlabEncoder encoder_;
};

} // namespace blk
