// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// HfKeeper presents the weight table T to the library (blocks/, namespace blk) as a committed
// table, blk::IndexHolder, whose rows the protocol steps FOLDEQ and LIN fold and whose columns they
// open (blocks/iop/foldeq.h, blocks/iop/lin.h). T is reached only through the callbacks in HfSeams
// and GPU memory only through those in HfCard, so that a CPU-only test can use this header.
//
// W is the weight matrix, M has W's cells in R long rows, and T is M's rows, encoded with masks.
// A pass over W (`walk` in identifiers) reads all of W once; make_columns may take several.
//   the library's term                   T's term (fields of HfTableParams, callbacks of HfSeams)
//   2^v rows, code length 2^log_n        R = 2^b rows of M, n = 2^ln
//   message of a row, ell = i0 entries   l data cells, then zeros (the gap), up to the row's
//                                        pads, the last P.pads() entries
//   random tail of a row, r = nm entries the masks
//   matrix, matrix_rows x 2^(k_lo + v)   W, m0 rows of k = 2^lk columns; k_lo = lcl = lk - b
//   fold at gamma, block sums            first pass over W: folded row u, omega (make_u, u_cells)
//   opened columns                       make_columns (further passes), show(), hf_blocks_opening()
// T's record (stored answers to samples of its rows, blocks/iop/foldeq.h) has P.answers answers for
// each row. The committer computes it once and publishes it in the key file
// or the leaf file. HfKeeper is given it and computes no answer. The HfSeams implementation adds
// the rows' pads to u and to the opened columns (prover/hash_commitment/hf_open_side.cuh).
#pragma once
#include <cstring>
#include <functional>
#include <vector>

#include <map>

#include "iop/lin.h"

struct K3;  // the GPU's element of the extension field E (prover/common/common.h): three 64-bit
            // words, like blk::E (prover/hash_commitment/hf_keeper.cu asserts the equal size).
            // Pointers to it are passed to HfSeams, never dereferenced here.
typedef K3 Ext;
#include "hash_commitment/hf_open_api.h"

namespace hf {

static_assert(sizeof(blk::E) == 24, "an element of E is three words in both texts");

// T's shape as a committed table. T's Merkle tree (hf_tree.cuh) is hashed as the library hashes a
// stored tree (made once, read by many proofs): leaves under hash number 174 with tree number 0
// (HF_NUM_F), other vertices under 166.
inline blk::IndexShape hf_index_shape(const HfTableParams &P) {
  blk::IndexShape s;
  s.code = blk::Code{(unsigned)P.ln, (size_t)P.i0, (size_t)P.nm};
  s.v = (unsigned)P.b, s.answers = (unsigned)P.answers;
  s.salt_bytes = 4 * SALTW;  // bytes; SALTW counts 32-bit words
  static_assert(HF_NUM_F == 0, "the blocks hash a stored tree under the tree number 0, and so must T's");
  // Only blk::Code defines where a row's coefficients are: mask j at tail_at() + j, data cell i at
  // message_at() + i. The masks start at coefficient 0 and the message at coefficient nm.
  BLK_REQUIRE(s.code.dim() == P.KD, "a row of T has another number of coefficients than the blocks' code of its numbers");
  for (unsigned j = 0; j < s.pads(); j++)
    BLK_REQUIRE(s.pad_place(j / blk::E_DEGREE, j % blk::E_DEGREE) == P.pad_entry(j), "a pad of T stands at another entry of its row's message than the blocks read it at");
  return s;
}
// Returns null, or why the LIN shape S does not fit T as this build makes it. With S.matrix_rows =
// m0 and S.k_lo = lcl, LIN's linear forms stop at entry l = m0 2^lcl of a row's message, never at
// i0: nothing shows that the gap of an untrusted committer's table is zero. Call this function
// before any weight is read: a committer that finds a misfit later has built T for nothing.
inline const char *hf_lin_misfit(const HfTableParams &P, const blk::LinShape &S) {
  const blk::IndexShape s = hf_index_shape(P);
  if (S.fold.index.salt_bytes != s.salt_bytes) return "the parameter file's salt has another length than the leaves of the weight table's tree have in this build (40 bytes)";
  if (!(S.fold.index.code == s.code) || S.fold.index.v != s.v || S.fold.index.answers != s.answers || S.k_lo != (unsigned)P.lcl || S.matrix_rows != P.m0)
    return "the lines describe a test on another table than T";
  return nullptr;
}
inline bool hf_lin_fits(const HfTableParams &P, const blk::LinShape &S) { return !hf_lin_misfit(P, S); }
// The layout of show()'s output for tcols columns, the number that the caller's plan (blk::Plan)
// draws: for LIN, blk::LinShape::fold.columns.
inline HfShown hf_shown(const HfTableParams &P, uint64_t tcols) {
  HfShown s;
  s.tcols = (int)tcols;
  s.c = top_layer(P.ln, tcols);
  s.column_words = 2 * (uint64_t)P.R + SALTW + (uint64_t)DW * (uint64_t)(P.ln - s.c);
  s.sy_words = tcols * (2 * (uint64_t)P.R + SALTW);
  s.shown_words = ((uint64_t)DW << s.c) + tcols * s.column_words;
  return s;
}

inline blk::Digest hf_digest(const uint32_t *words) {
  blk::Digest d;
  for (int i = 0; i < 8; i++) d.w[i] = words[i];
  return d;
}
static_assert(DW == 8, "a digest of T's tree is a digest of the blocks");

// Converts show()'s output (`words`) into the library's opening of a stored tree
// (blocks/tree/opening.h), on the CPU. Input: the top layer (all 2^c vertices of depth c); then for
// each draw, repeats kept, the column's R symbols, its salt and the ln - c siblings on its path,
// lowest first. Output: column and salt for each distinct position, in rising order; then the
// digests that blk::tree_walk() cannot compute from the opened leaves, in its order. Each is a
// sibling on some draw's path, a vertex of the top layer, or a vertex above it, hashed here.
// In messages, "the side" is the HfSeams implementation.
inline blk::Opening hf_blocks_opening(const HfTableParams &P, const HfShown &shown, const blk::OracleTreeShape &tree, const blk::PositionList &at, const uint32_t *words) {
  const std::vector<blk::u64> &draws = at.draws();
  const unsigned ln = (unsigned)P.ln, c = (unsigned)shown.c;
  blk::Opening o;
  o.main_width = P.R, o.mask_width = 0;
  o.columns.assign(at.size() * P.R, blk::F::zero());
  o.salts.assign(at.size() * tree.salt_bytes, 0);
  std::map<blk::u64, blk::Digest> known;  // by tree_index(depth, place)
  for (blk::u64 y = 0; y < ((blk::u64)1 << c); y++) known[blk::tree_index(c, y)] = hf_digest(words + DW * y);
  for (unsigned d = c; d-- > 0;)
    for (blk::u64 y = 0; y < ((blk::u64)1 << d); y++)
      known[blk::tree_index(d, y)] = blk::vertex(tree.tree().vertex_tag(), known[blk::tree_index(d + 1, 2 * y)], known[blk::tree_index(d + 1, 2 * y + 1)]);
  for (size_t a = 0; a < draws.size(); a++) {
    const uint32_t *w = words + ((size_t)DW << c) + a * shown.column_words;
    const size_t i = at.index_of(draws[a]);
    for (size_t r = 0; r < P.R; r++) o.columns[i * P.R + r] = blk::F{w[2 * r] | (uint64_t)w[2 * r + 1] << 32};
    for (size_t b = 0; b < tree.salt_bytes; b++) o.salts[i * tree.salt_bytes + b] = (blk::u8)(w[2 * P.R + b / 4] >> 8 * (b % 4));
    for (unsigned j = 0; j < ln - c; j++) known[blk::tree_index(ln - j, (draws[a] >> j) ^ 1)] = hf_digest(w + 2 * P.R + SALTW + (size_t)DW * j);
  }
  struct Asked {  // for `tree_walk`: computes nothing, records the digests that it asks for
    using Value = char;
    const std::map<blk::u64, blk::Digest> &known;
    std::vector<blk::Digest> &asked;
    Value leaf(size_t) { return 0; }
    bool given(unsigned d, blk::u64 place, Value &v) {
      const auto it = known.find(blk::tree_index(d, place));
      BLK_REQUIRE(it != known.end(), "the side's shown words lack a digest that the blocks' opening holds");
      asked.push_back(it->second);
      return v = 0, true;
    }
    Value node(unsigned, blk::u64, Value, Value) { return 0; }
  } side{known, o.given};
  char root = 0;
  (void)blk::tree_walk(tree.tree(), at, side, root);
  return o;
}

// Callbacks through which HfKeeper uses GPU memory (`card` in identifiers: the GPU). `take`
// allocates and `give` frees; HfKeeper erases nothing before `give` (see row_back()).
struct HfCard {
  std::function<void *(size_t bytes)> take;
  std::function<void(void *)> give;
  std::function<void(void *host, const void *card, size_t bytes)> fetch;  // returns when the bytes are there
  // `store` and `adopt` may be empty: then the folded row passes through host vectors (fold()),
  // which no full-size run can afford. The engine is the caller's blk::Engine, which keeps and
  // encodes the prover's large tables.
  std::function<void(void *card, const void *host, size_t bytes)> store;
  std::function<blk::TableId(const void *card, size_t rows, size_t len)> adopt;  // the engine takes a table over E where it is, without a copy
  // Rule (hf_walk.h): during a pass over the weights the code that opens T holds no borrowed GPU
  // memory. At the first pass HfKeeper holds only omega. Before make_columns(), which makes the
  // further passes, it frees the folded row, and it fills the row again afterwards. The engine does
  // the same for all that it holds: leave() before make_columns(), then moved(table, its new
  // address), then back() (cp_prove sets leave() and back() to do nothing). Set all three or none (only `leave` is tested); without them the folded
  // row stays where it is.
  std::function<void()> leave, back;
  std::function<void(blk::TableId, const void *card)> moved;
  // Memory that keeps its contents during a pass over the weights (hf_walk.h: the slab), for omega,
  // which is allocated before the first pass. If empty, `take` is used.
  std::function<void *(size_t bytes)> take_lasting;
};

// The library calls begin(), then fold_table() or fold(), open() and forget(), in this order
// (blocks/iop/lin.h, blocks/iop/foldeq.h); the order is not checked here. A failed BLK_REQUIRE
// (blocks/base.h) ends the process.
class HfKeeper : public blk::IndexHolder {
 public:
  // P, seams: kept by reference; both must outlive the HfKeeper.
  // commitment: T's root, 32 bytes, which identifies T in the hash chains' first queries. (A leaf's
  // hash covers R, ln, the tree number and the position, and the salt has one length, so a root
  // cannot be opened as columns of another shape.)
  // record: T's record as the key file or the leaf file has it: P.answers answers for each row.
  // shown: the layout for which the HfSeams implementation was built. Not checked: a show() that
  // writes more than shown.shown_words words overruns open()'s buffer.
  // count: the calling program's counter of this table's opened columns; false if the masks do not
  // cover `columns` more. As blk::IndexHolder::begin() requires, it must write the new count before
  // it returns, and the program must make that count durable before fold() or fold_table() is
  // called: a proof that is broken off has spent its columns all the same. HfKeeper checks neither.
  // window: how many entries of u are moved at a time.
  HfKeeper(const HfTableParams &P, const HfShown &shown, HfSeams &seams, HfCard card, std::vector<uint8_t> commitment, blk::IndexRecord record,
           std::function<bool(uint64_t columns)> count, size_t window = (size_t)1 << 22)
      : P_(P), shown_(shown), shape_(hf_index_shape(P)), seams_(seams), card_(std::move(card)), commitment_(std::move(commitment)), record_(std::move(record)),
        count_(std::move(count)), window_(window) {
    BLK_REQUIRE(commitment_.size() == blk::DIGEST_BYTES, "a stored tree's commitment is its root");
    BLK_REQUIRE(!blk::record_fault(shape_, record_), "the record that the keeper is handed has not one answer for each row and seed of T");
    const HfShown own = hf_shown(P, (uint64_t)shown.tcols);
    BLK_REQUIRE(shown.c == own.c && shown.column_words == own.column_words && shown.shown_words == own.shown_words, "the side lays the shown words otherwise than the keeper");
  }

  const blk::IndexShape &shape() const override { return shape_; }
  std::vector<blk::u8> commitment() const override { return commitment_; }
  // The name as blk::IndexKey::name() computes it for the verifier: the record's seeds are derived
  // from it.
  std::vector<blk::u8> name() const override {
    blk::Digest root;
    blk::digest_from_bytes(commitment_.data(), root);
    return blk::index_name(root, shape_, configuration());
  }
  blk::Digest configuration() const override { return hf_digest(P_.configuration); }  // T's tree hashes start from it; LIN compares it with the chain's
  blk::IndexRecord record() override { return record_; }
  bool begin(blk::u64 columns) override {
    BLK_REQUIRE(columns == (blk::u64)shown_.tcols, "the plan draws another number of columns than the side was built for");
    return count_(columns);
  }

  ~HfKeeper() override {
    if (table_) card_.give(table_);
  }

  // The folded row in host vectors, fetched `window_` entries at a time (see HfCard::store).
  void fold(const blk::E *gamma, std::vector<blk::E> &message, std::vector<blk::E> &tail, blk::BlockSums *blocks) override {
    walk(gamma, blocks);
    message.resize(P_.i0), tail.resize(P_.nm);
    void *cells = card_.take(window_ * sizeof(blk::E));
    auto run = [&](uint64_t first, std::vector<blk::E> &to) {  // to[i] = u[first + i]
      for (uint64_t at = 0; at < to.size(); at += window_) {
        const uint64_t count = std::min<uint64_t>(window_, to.size() - at);
        seams_.u_cells(first + at, count, ext(cells, 0));
        card_.fetch(to.data() + at, cells, count * sizeof(blk::E));
      }
    };
    run(shape_.code.message_at(), message), run(shape_.code.tail_at(), tail);
    card_.give(cells);
  }

  // The folded row as a table in GPU memory (layout: fill_the_table()). row_back() or the
  // destructor frees it; the engine must have dropped it by then.
  blk::TableId fold_table(const blk::E *gamma, const blk::FoldedWant &want, blk::BlockSums *blocks) override {
    if (!card_.adopt || !card_.store) return blk::NO_ID;
    BLK_REQUIRE(want.tail_first >= P_.i0 && want.rows * want.len >= want.tail_first + P_.nm && !table_, "the folded row's tail is wanted inside its message or beyond the table, or the row twice");
    walk(gamma, blocks);
    want_ = want;
    fill_the_table();
    return id_ = card_.adopt(table_, want.rows, want.len);
  }

  // Writes to `file` the library's opening of a stored tree, which blk::IndexKey reads; `columns`
  // gets one column for each listed position.
  void open(const blk::PositionList &at, blk::Writer &file, std::vector<blk::F> &columns) override {
    const std::vector<blk::u64> &draws = at.draws();
    const HfShown &shown = shown_;
    // A guard that fails here fails, in cp_prove, inside a translation unit compiled from
    // hf_keeper.cu, on the thread that runs the batch's opening. A test that is not part of this
    // repository plants one and requires that cp_prove then exits with status 1, not by abort().
    BLK_REQUIRE(draws.size() == (size_t)shown.tcols, "the plan draws another number of columns than the side was built for");
    const bool away = table_ && card_.leave;
    if (away) {
      card_.leave();
      card_.give(table_), table_ = nullptr;
    }
    seams_.make_columns(draws.data(), shown.tcols);
    std::vector<uint32_t> words(shown.shown_words);
    seams_.show(words.data());
    if (away) {
      fill_the_table();  // `u_cells` returns the same entries until `forget`
      card_.moved(id_, table_);
      card_.back();
    }
    const blk::OracleTreeShape tree = blk::index_tree(shape_, hf_digest(P_.configuration));
    blk::Opening o = hf_blocks_opening(P_, shown, tree, at, words.data());
    blk::write_opening(file, tree, at, o);
    columns = std::move(o.columns);
  }

  void forget() override { seams_.forget(); }
  // Frees the folded row early. For the caller that runs the opening, once the proximity tests of
  // the shared chain (the hash chain that all proofs of a batch repeat) are over: the engine has
  // then dropped the table and no longer reads it. The contents are hidden as the weights are and
  // are not erased here: the memory's owner zeroes it, as it does before a pass over the weights.
  void row_back() {
    if (table_) card_.give(table_), table_ = nullptr;
  }

 private:
  void fill_the_table() {
    // u's message (i0 entries: data, the gap's zeros, then the folded pads) from
    // entry 0, its nm masks from entry tail_first, zeros elsewhere (blk::FoldedWant).
    const uint64_t all = want_.rows * want_.len, ell = P_.i0, r = P_.nm, B = want_.tail_first;
    table_ = card_.take(all * sizeof(blk::E));
    auto run = [&](uint64_t first, uint64_t count, uint64_t to) {  // entry to + i = u[first + i]
      for (uint64_t at = 0; at < count; at += window_) seams_.u_cells(first + at, std::min<uint64_t>(window_, count - at), ext(table_, to + at));
    };
    auto zero = [&](uint64_t from, uint64_t to) {
      const std::vector<blk::E> zeros(std::min<uint64_t>(window_, to - from), blk::E::zero());
      for (uint64_t at = from; at < to; at += window_) card_.store(static_cast<blk::E *>(table_) + at, zeros.data(), std::min<uint64_t>(window_, to - at) * sizeof(blk::E));
    };
    run(shape_.code.message_at(), ell, 0), zero(ell, B), run(shape_.code.tail_at(), r, B), zero(B + r, all);
    // The folded row is sent as an oracle message. The pad that masks the answer of that message's
    // out-of-domain sample stays 0 here: the step SEND, which sends the message, draws the pad and
    // sets it through the engine, which writes it again at moved().
  }
  static ::Ext *ext(void *cells, uint64_t at) { return reinterpret_cast<::Ext *>(static_cast<blk::E *>(cells) + at); }

  // The first pass over the weights: make_u() computes u and omega = W eq(psi*, .) together, where
  // psi* = (blocks->low, gamma) is LIN's shared point. make_u() takes it as rw, in reverse order:
  // rw[j] = psi*[lk - 1 - j], so that rw[0] goes with the top bit of W's column index. Without
  // `blocks` the low coordinates of psi* (the last lcl entries of rw) are zeros and omega is
  // discarded.
  void walk(const blk::E *gamma, blk::BlockSums *blocks) {
    BLK_REQUIRE(!blocks || (blocks->k_lo == (unsigned)P_.lcl && blocks->count == P_.m0), "the sums over u's blocks are omega: one for each row of W");
    std::vector<blk::E> weight(P_.R), rw(P_.lk);
    blk::eq_table(gamma, (unsigned)P_.b, weight.data());
    for (int j = 0; j < P_.lk; j++) rw[j] = P_.lk - 1 - j >= P_.lcl ? gamma[P_.lk - 1 - j - P_.lcl] : blocks ? blocks->low[P_.lk - 1 - j] : blk::E::zero();
    void *omega = (card_.take_lasting ? card_.take_lasting : card_.take)(P_.m0 * sizeof(blk::E));
    seams_.make_u(reinterpret_cast<const ::Ext *>(weight.data()), reinterpret_cast<const ::Ext *>(rw.data()), ext(omega, 0));
    if (blocks) {
      blocks->sums.resize(P_.m0);
      card_.fetch(blocks->sums.data(), omega, P_.m0 * sizeof(blk::E));
    }
    card_.give(omega);
  }

  const HfTableParams &P_;
  const HfShown shown_;
  blk::IndexShape shape_;
  HfSeams &seams_;
  HfCard card_;
  std::vector<uint8_t> commitment_;
  const blk::IndexRecord record_;
  std::function<bool(uint64_t)> count_;
  size_t window_;
  void *table_ = nullptr;  // the folded row in GPU memory, as the engine has adopted it
  blk::FoldedWant want_;
  blk::TableId id_ = blk::NO_ID;
};

}  // namespace hf
