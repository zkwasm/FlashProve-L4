// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the Merkle tree of blocks/tree/tree.h and the salted tree over columns of symbols of
// blocks/tree/opening.h, against a second implementation written here and with altered openings,
// which must be rejected. main() returns 0 if no CHECK failed.
#include <cstdio>
#include <set>
#include <vector>

#include "chain/tests/death.h"
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

// A small generator for tests (splitmix64: Steele, Lea, Flood,
// "Fast splittable pseudorandom number generators", 2014).
struct TestCoins : Coins {
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

static Digest some_digest(TestCoins &c) {
  u8 b[DIGEST_BYTES];
  c.bytes(b, DIGEST_BYTES);
  Digest d;
  digest_from_bytes(b, d);
  return d;
}
// A PositionList: `draws` random positions, kept as drawn and as a sorted set.
static PositionList some_list(TestCoins &c, unsigned log_n, size_t draws) {
  std::vector<u64> x(draws);
  for (u64 &v : x) v = c.word() & (((u64)1 << log_n) - 1);
  return PositionList::from_draws(x, log_n);
}
static PositionList every_position(unsigned log_n) {
  std::vector<u64> all((size_t)1 << log_n);
  for (size_t x = 0; x < all.size(); x++) all[x] = x;
  return PositionList::from_draws(all, log_n);
}

// ---- a second implementation of the tree's definition; calls nothing of blocks/tree/tree.h ----
static Digest value_again(const TreeShape &s, const std::vector<Digest> &leaves, unsigned d, u64 place) {
  if (d == s.log_n) return leaves[place];
  Hasher h(Tag(s.binding ? Domain::BINDING_NODE : Domain::TREE_NODE, 0, s.start), 0); // tree number 0 and index 0: no tree's number, place or run enters a vertex's hash
  h.push(value_again(s, leaves, d + 1, 2 * place)), h.push(value_again(s, leaves, d + 1, 2 * place + 1));
  return h.finish();
}
// The digests of an opening: the values of the vertices that are on no opened leaf's path to the
// root but whose sibling is, level by level from the leaves up, from left to right within a level.
static std::vector<Digest> given_again(const TreeShape &s, const std::vector<Digest> &leaves, const PositionList &list) {
  std::vector<Digest> out;
  for (unsigned d = s.log_n; d >= 1; d--) {
    std::set<u64> on;
    for (u64 x : list.all()) on.insert(x >> (s.log_n - d));
    for (u64 p : on) // std::set iterates in increasing order
      if (!on.count(p ^ 1)) out.push_back(value_again(s, leaves, d, p ^ 1));
  }
  return out;
}

[[maybe_unused]] static void test_shape(const TreeShape &s, TestCoins &c) {
  CHECK(s.valid());
  std::vector<Digest> leaves;
  for (u64 x = 0; x < ((u64)1 << s.log_n); x++) leaves.push_back(some_digest(c));
  const Tree tree(s, leaves);
  CHECK(tree.root() == value_again(s, leaves, 0, 0));
  for (size_t draws : {(size_t)1, (size_t)2, (size_t)5, (size_t)1 << s.log_n, (size_t)4 << s.log_n}) {
    const PositionList list = some_list(c, s.log_n, draws);
    std::vector<Digest> shown;
    for (u64 x : list.all()) shown.push_back(leaves[x]);
    const std::vector<Digest> given = tree.open(list);
    CHECK(given == given_again(s, leaves, list));
    CHECK(given.size() == tree_given_count(s, list));
    CHECK(tree_check(tree.root(), s, list, shown, given));
    // Each of the following changes makes tree_check() fail.
    for (size_t i = 0; i < given.size(); i++) {
      std::vector<Digest> g = given;
      g[i].w[i % 8] ^= 1;
      CHECK(!tree_check(tree.root(), s, list, shown, g));
    }
    for (size_t i = 0; i < shown.size(); i++) {
      std::vector<Digest> l = shown;
      l[i].w[0] ^= 0x80000000u;
      CHECK(!tree_check(tree.root(), s, list, l, given));
      if (s.log_n > 0) { // a true leaf, of the neighbouring position
        l = shown, l[i] = leaves[list[i] ^ 1];
        CHECK(!tree_check(tree.root(), s, list, l, given));
      }
    }
    std::vector<Digest> g = given, l = shown;
    g.push_back(Digest{});
    CHECK(!tree_check(tree.root(), s, list, shown, g)); // a digest too many
    if (!given.empty()) {
      g = given, g.pop_back();
      CHECK(!tree_check(tree.root(), s, list, shown, g)); // one too few
    }
    if (given.size() >= 2) {
      g = given, std::swap(g[0], g[1]);
      CHECK(!tree_check(tree.root(), s, list, shown, g));
    }
    l.pop_back();
    CHECK(!tree_check(tree.root(), s, list, l, given));
    TreeShape other_start = s;
    other_start.start.w[5] ^= 1;
    if (s.log_n > 0) CHECK(!tree_check(tree.root(), other_start, list, shown, given)); // another `start`; depth 0 has no inner vertex
    TreeShape other_kind = s;
    other_kind.binding = !s.binding;
    if (s.log_n > 0) CHECK(!tree_check(tree.root(), other_kind, list, shown, given)); // a `binding` tree's vertices are hashed under another Domain
    Digest other_root = tree.root();
    other_root.w[7] ^= 1;
    CHECK(!tree_check(other_root, s, list, shown, given));
    // The same leaves and digests under another list of the same size.
    if (s.log_n > 0 && list.size() < ((size_t)1 << s.log_n)) {
      std::vector<u64> moved = list.all();
      for (u64 &x : moved) x ^= 1;
      const PositionList list2 = PositionList::from_draws(moved, s.log_n);
      if (list2 != list) CHECK(!tree_check(tree.root(), s, list2, shown, given));
    }
  }
  CHECK(tree.open(every_position(s.log_n)).empty());
  // Openings of one tree at several lists give no vertex two values in `seen`, a verifier's record
  // (TreeValues) of the vertex values met under one root.
  TreeValues seen;
  for (size_t draws : {(size_t)1, (size_t)3, (size_t)2 << s.log_n, (size_t)2}) {
    const PositionList list = some_list(c, s.log_n, draws);
    std::vector<Digest> shown;
    for (u64 x : list.all()) shown.push_back(leaves[x]);
    CHECK(tree_check(tree.root(), s, list, shown, tree.open(list), &seen) && !seen.clashed());
  }
  // Two values at one vertex under one root would be a hash collision, which no test can produce.
  // Instead, another tree's values are recorded, and this tree's opening is checked against them.
  if (s.log_n > 0) {
    std::vector<Digest> others;
    for (u64 x = 0; x < ((u64)1 << s.log_n); x++) others.push_back(some_digest(c));
    const Tree other(s, others);
    const PositionList list = some_list(c, s.log_n, 2);
    std::vector<Digest> shown, shown_other;
    for (u64 x : list.all()) shown.push_back(leaves[x]), shown_other.push_back(others[x]);
    TreeValues mixed;
    CHECK(tree_check(other.root(), s, list, shown_other, other.open(list), &mixed));
    CHECK(tree_check(tree.root(), s, list, shown, tree.open(list)));          // good by itself
    const size_t held = mixed.vertices();
    bool clash = false;
    CHECK(!tree_check(tree.root(), s, list, shown, tree.open(list), &mixed, &clash) && clash); // rejected: clashes with the other tree's values
    // A rejected opening leaves the record unchanged; the opening accepted before still passes.
    CHECK(!mixed.clashed() && mixed.vertices() == held);
    CHECK(tree_check(other.root(), s, list, shown_other, other.open(list), &mixed, &clash) && !clash && mixed.vertices() == held);
    {
      // An opening that does not lead to its root records nothing, so the honest one then passes.
      TreeValues fresh;
      std::vector<Digest> spoiled = shown;
      spoiled[0].w[0] ^= 1;
      CHECK(!tree_check(tree.root(), s, list, spoiled, tree.open(list), &fresh, &clash) && !clash && fresh.vertices() == 0);
      CHECK(tree_check(tree.root(), s, list, shown, tree.open(list), &fresh, &clash) && fresh.vertices() > 0);
    }
    // The record differs at one vertex whose digest the opening supplies: the sibling leaf.
    TreeValues one;
    const PositionList single = PositionList::from_draws({0}, s.log_n);
    Digest bent = leaves[1];
    bent.w[0] ^= 1;
    CHECK(one.put(s.log_n, 1, bent));
    CHECK(!tree_check(tree.root(), s, single, {leaves[0]}, tree.open(single), &one, &clash) && clash && !one.clashed() && one.vertices() == 1);
  }
}

[[maybe_unused]] static void test_positions() {
  const PositionList d = PositionList::from_draws({6, 1, 6, 5, 1}, 3);
  CHECK((d.all() == std::vector<u64>{1, 5, 6}));
  CHECK((d.draws() == std::vector<u64>{6, 1, 6, 5, 1}));
  CHECK(d.index_of(5) == 1);
  CHECK(d == PositionList::from_draws({1, 5, 6}, 3) && d != PositionList::from_draws({1, 5, 6}, 4));
  CHECK((d.short_positions(2) == std::vector<u64>{1, 2}));
  CHECK((d.short_positions(0) == std::vector<u64>{0}));
}

// ---- columns, salts, bytes ----
struct Rows {
  OracleTreeShape s;
  std::vector<F> main, mask; // main[x * main_width + j], mask[y * mask_width + j]
  Rows(const OracleTreeShape &shape, TestCoins &c) : s(shape) {
    main.resize(s.main_width() << s.log_n);
    mask.resize(s.mask_width() << s.log_n_zk);
    c.fill(main.data(), main.size());
    c.fill(mask.data(), mask.size());
  }
  // Column x: the main rows' symbols at x, then the mask rows' at short position x % 2^log_n_zk.
  // Each mask symbol is thus in 2^(log_n - log_n_zk) columns.
  void column(u64 x, std::vector<F> &col) const {
    const u64 y = x % ((u64)1 << s.log_n_zk);
    col.assign(main.begin() + x * s.main_width(), main.begin() + (x + 1) * s.main_width());
    col.insert(col.end(), mask.begin() + y * s.mask_width(), mask.begin() + (y + 1) * s.mask_width());
  }
};

// Returns whether the verifier accepts the prover's opening. `tamper`: also run the CHECKs below.
static bool run_opening(const OracleTreeShape &s, TestCoins &c, size_t draws, bool tamper) {
  const Rows t(s, c);
  auto fetch = [&](u64 x, std::vector<F> &out) { t.column(x, out); };
  OracleTree tree(s, c);
  std::vector<F> col;
  for (u64 x = 0; x < ((u64)1 << s.log_n); x++) t.column(x, col), tree.set_leaf(x, col.data(), col.size());
  const Digest root = tree.finish();
  const PositionList list = some_list(c, s.log_n, draws);
  Opening o;
  tree.open(list, o, fetch);
  Writer w;
  write_opening(w, s, list, o);
  Reader r(w.data(), w.size());
  Opening got;
  const bool accepted = read_opening(r, s, list, got) && r.at_end() && check_opening(root, s, list, got);
  if (!tamper) return accepted;

  CHECK(w.size() == opening_bytes(s, list));
  // Recomputes the root from the rows and the salts (taken from an opening of every position) with
  // only `Tag` and Hasher of blocks/hash/hash.h: reimplements a leaf's tag, index and content.
  Opening all;
  tree.open(every_position(s.log_n), all, fetch);
  CHECK(all.given.empty());
  auto leaf_again = [&](u64 x) {
    Hasher h(Tag(s.stored ? Domain::LEAF_STORED_MAIN : Domain::LEAF_MAIN, s.tree_number, s.stored ? s.parameters : s.run, s.width(), s.salt_bytes), ((u64)1 << s.log_n) + x);
    std::vector<F> column;
    t.column(x, column);
    for (const F &f : column) h.push_u64(f.v);
    // The salt fits within one block of the hashed string, which begins with a 16-byte label: if it
    // would cross a block boundary, zeros first fill the rest of the current block.
    size_t at = 16 + 8 * column.size();
    if (at % Hasher::BLOCK_BYTES + s.salt_bytes > Hasher::BLOCK_BYTES)
      for (; at % Hasher::BLOCK_BYTES != 0; at += 8) h.push_u64(0);
    h.push(all.salts.data() + x * s.salt_bytes, s.salt_bytes);
    return h.finish();
  };
  std::vector<Digest> leaves;
  for (u64 x = 0; x < ((u64)1 << s.log_n); x++) leaves.push_back(leaf_again(x));
  CHECK(root == value_again(s.tree(), leaves, 0, 0));
  // No two leaves share a salt, not even leaves with the same short position.
  {
    std::set<std::vector<u8>> salts;
    for (u64 x = 0; x < ((u64)1 << s.log_n); x++) salts.insert(std::vector<u8>(all.salts.begin() + x * s.salt_bytes, all.salts.begin() + (x + 1) * s.salt_bytes));
    CHECK(salts.size() == (size_t)1 << s.log_n);
  }
  // Rebuilds the opening's bytes: for each listed position its column and salt, then the digests.
  {
    Writer again;
    for (u64 x : list.all()) {
      t.column(x, col);
      for (const F &f : col) again.put(f);
      again.put_bytes(all.salts.data() + x * s.salt_bytes, s.salt_bytes);
    }
    for (const Digest &d : given_again(s.tree(), leaves, list)) again.put(d);
    CHECK(again.bytes() == w.bytes());
  }
  // Opening::main_column() and Opening::mask_column() return the rows' symbols.
  for (size_t i = 0; i < list.size(); i++) {
    t.column(list[i], col);
    for (size_t j = 0; j < s.main_width(); j++) CHECK(got.main_column(i)[j] == col[j]);
    for (size_t j = 0; j < s.mask_width(); j++) CHECK(got.mask_column(i)[j] == col[s.main_width() + j] && got.mask_column(i)[j] == t.mask[short_position(s.log_n, s.log_n_zk, list[i]) * s.mask_width() + j]);
  }
  // An opening with one flipped bit, in any byte, is rejected, and so is a truncated one.
  for (size_t i = 0; i < w.size(); i++) {
    std::vector<u8> b = w.bytes();
    b[i] ^= (u8)(1u << (i % 8));
    Reader bent(b.data(), b.size());
    CHECK(!(read_opening(bent, s, list, got) && bent.at_end() && check_opening(root, s, list, got)));
  }
  for (size_t cut : {(size_t)0, (size_t)1, w.size() / 2, w.size() - 1}) {
    Reader shorter(w.data(), cut);
    CHECK(!read_opening(shorter, s, list, got));
    CHECK(got.columns.size() == list.size() * s.width() && got.salts.size() == list.size() * s.salt_bytes && got.given.size() == o.given.size()); // full sizes nevertheless
  }
  {
    // The first symbol becomes P, the modulus: not in canonical form.
    std::vector<u8> b = w.bytes();
    for (int k = 0; k < 8; k++) b[k] = (u8)(P >> (8 * k));
    Reader bent(b.data(), b.size());
    CHECK(!read_opening(bent, s, list, got));
  }
  // check_opening() rejects an opening whose vectors have other sizes than shape and list ask for.
  for (int piece = 0; piece < 4; piece++) {
    Opening bad = o;
    if (piece == 0) bad.columns.pop_back();
    if (piece == 1) bad.salts.pop_back();
    if (piece == 2 && !bad.given.empty()) bad.given.pop_back();
    if (piece == 3) bad.given.push_back(Digest{});
    const bool changed = bad.columns.size() != o.columns.size() || bad.salts.size() != o.salts.size() || bad.given.size() != o.given.size();
    CHECK(check_opening(root, s, list, bad) == !changed);
  }
  // A leaf's hash covers `tree_number` and starts from `run`; in a `stored` (long-lived) tree it
  // starts from `parameters`, as every inner vertex's hash does. A changed field fails the check if
  // a hash uses it.
  {
    OracleTreeShape other = s;
    other.run.w[3] ^= 1;
    CHECK(check_opening(root, other, list, o) == s.stored); // a stored tree has no run
    other = s, other.parameters.w[3] ^= 1;
    if (s.stored || s.log_n > 0) CHECK(!check_opening(root, other, list, o));
    other = s, other.tree_number ^= 1;
    CHECK(!check_opening(root, other, list, o));
    other = s, other.stored = !s.stored;
    CHECK(!check_opening(root, other, list, o));
  }
  // After a rejection: the list {0}, which PositionsDraw::positions() (blocks/chain/rounds.h) then
  // returns, and a Reader after refuse(). read_opening() returns false; its opening, zeros at full
  // size, can be read.
  {
    Reader spent(w.data(), w.size());
    spent.refuse();
    const PositionList fixed = PositionList::from_draws({0}, s.log_n);
    CHECK(!read_opening(spent, s, fixed, got));
    CHECK(got.columns.size() == s.width() && got.column(0)[0] == F::zero() && got.mask_column(0) == got.column(0) + s.main_width());
    CHECK(!check_opening(root, s, fixed, got));
  }
  // A list over another range is a programming error, not a bad proof: the process aborts, which
  // ends_with() (blocks/chain/tests/death.h) observes in a child process.
  CHECK(ends_with("does not range over the tree's positions", [&] {
    Reader whole(w.data(), w.size());
    Opening other;
    (void)read_opening(whole, s, PositionList::from_draws({0}, s.log_n + 1), other);
  }));
  return accepted;
}

static OracleTreeShape shape_of(bool has_main, bool has_mask, unsigned log_n, unsigned log_n_zk, size_t salt) {
  OracleTreeShape s;
  s.log_n = log_n, s.log_n_zk = log_n_zk;
  s.main_symbols = has_main ? std::vector<size_t>{3, 0, 2} : std::vector<size_t>{0, 0, 0};
  s.mask_symbols = has_mask ? std::vector<size_t>{3, 6, 0} : std::vector<size_t>{0, 0, 0};
  s.tree_number = 7;
  s.salt_bytes = salt;
  s.run.w[0] = 0x52554E, s.run.w[7] = 1;
  s.parameters.w[0] = 0x504152, s.parameters.w[6] = 2;
  return s;
}

int main() {
  TestCoins c(2026);
  test_positions();
  Digest start{};
  start.w[2] = 0x5A;
  for (unsigned a = 0; a <= 7; a++) test_shape(TreeShape::plain(a, start), c);
  test_shape(TreeShape::plain(4, start, true), c);
  CHECK(!TreeShape::plain(49, start).valid());
  CHECK(!TreeShape::plain(3, Digest{}).valid()); // zeros: `start` was not set

  // Programming errors abort the process; malformed shapes are not valid().
  {
    const OracleTreeShape s = shape_of(true, true, 2, 1, 40);
    const std::vector<F> column(s.width(), F::one());
    auto whole = [&](OracleTree &tree) {
      for (u64 x = 0; x < 4; x++) tree.set_leaf(x, column.data(), column.size());
    };
    CHECK(ends_with("before every leaf was set", [&] {
      OracleTree tree(s, c);
      tree.set_leaf(0, column.data(), column.size());
      tree.finish();
    }));
    CHECK(ends_with("after the tree was finished", [&] {
      OracleTree tree(s, c);
      whole(tree), tree.finish();
      tree.set_leaf(0, column.data(), column.size());
    }));
    CHECK(ends_with("finished twice", [&] {
      OracleTree tree(s, c);
      whole(tree), tree.finish(), tree.finish();
    }));
    CHECK(ends_with("not a leaf of this tree", [&] {
      OracleTree tree(s, c);
      tree.set_leaf(4, column.data(), column.size());
    }));
    CHECK(ends_with("not a leaf of this tree", [&] { // a column without its mask symbols
      OracleTree tree(s, c);
      tree.set_leaf(0, column.data(), s.main_width());
    }));
    CHECK(ends_with("opened before it is finished", [&] {
      OracleTree tree(s, c);
      Opening o;
      tree.open(PositionList::from_draws({1}, 2), o, [&](u64, std::vector<F> &out) { out = column; });
    }));
    CHECK(ends_with("has not the leaf's size", [&] {
      OracleTree tree(s, c);
      whole(tree), tree.finish();
      Opening o;
      tree.open(PositionList::from_draws({1}, 2), o, [&](u64, std::vector<F> &out) { out.assign(s.main_width(), F::one()); });
    }));
    CHECK(ends_with("not the sizes that shape and list ask for", [&] { // an opening that was not made for this list
      Writer w;
      write_opening(w, s, PositionList::from_draws({1}, 2), Opening());
    }));
    CHECK(ends_with("does not range over the tree's positions", [&] { (void)tree_given_count(TreeShape::plain(3, start), PositionList::from_draws({1}, 2)); }));
    CHECK(ends_with("not 2^depth leaves", [&] { Tree(TreeShape::plain(2, start), std::vector<Digest>(3)); }));
    CHECK(ends_with("with k below 64", [&] { PositionList::from_draws({1}, 64); }));
    CHECK(ends_with("at least one draw", [&] { PositionList::from_draws({}, 3); }));
    CHECK(ends_with("a draw outside the range", [&] { PositionList::from_draws({8}, 3); }));
    CHECK(ends_with("longer than the list's range", [&] { PositionList::from_draws({1}, 3).short_positions(4); }));
    CHECK(ends_with("not in the list", [&] { PositionList::from_draws({1}, 3).index_of(2); }));
    OracleTreeShape long_mask = shape_of(true, true, 3, 3, 40);
    long_mask.log_n_zk = 4;
    CHECK(!long_mask.valid());
    CHECK(!shape_of(false, true, 3, 2, 40).valid()); // mask rows only: log_n must equal log_n_zk
    CHECK(!shape_of(false, false, 3, 3, 40).valid());
    OracleTreeShape uneven = shape_of(true, true, 3, 1, 40);
    uneven.mask_symbols.pop_back(); // main_symbols and mask_symbols must have equal sizes
    CHECK(!uneven.valid());
    CHECK(shape_of(true, false, 3, 0, 40).has_main() && !shape_of(true, false, 3, 0, 40).has_mask() && !shape_of(false, true, 2, 2, 40).has_main());
    OracleTreeShape high_number = shape_of(true, false, 3, 0, 40);
    high_number.tree_number = 1u << 16; // a tree's number has 16 bits
    CHECK(!high_number.valid());
    OracleTreeShape no_run = shape_of(true, false, 3, 0, 40);
    no_run.run = Digest{};
    CHECK(!no_run.valid());
    no_run.stored = true; // a stored tree has none
    CHECK(no_run.valid());
    no_run.parameters = Digest{};
    CHECK(!no_run.valid());
  }
  CHECK(!shape_of(true, true, 5, 2, 0).valid() && !shape_of(true, true, 5, 2, SALT_FLOOR_BYTES - 8).valid()); // a salt is required, of at least SALT_FLOOR_BYTES
  CHECK(!shape_of(true, true, 5, 2, SALT_FLOOR_BYTES + 5).valid() && !shape_of(true, true, 5, 2, Hasher::BLOCK_BYTES + 8).valid()); // a multiple of 8; above SALT_MOST_BYTES only Hasher::BLOCK_BYTES + 8 is tried
  {
    OracleTreeShape kept = shape_of(true, true, 5, 2, 40); // a tree that outlives its run
    kept.stored = true;
    CHECK(run_opening(kept, c, 4, true));
  }
  for (size_t salt : {(size_t)40, (size_t)48, (size_t)64})
    for (size_t draws : {(size_t)1, (size_t)4, (size_t)40}) {
      CHECK(run_opening(shape_of(true, true, 5, 2, salt), c, draws, draws == 4));
      CHECK(run_opening(shape_of(true, true, 4, 4, salt), c, draws, draws == 4));
      CHECK(run_opening(shape_of(true, true, 4, 0, salt), c, draws, false));
      CHECK(run_opening(shape_of(true, false, 5, 0, salt), c, draws, draws == 4));
      CHECK(run_opening(shape_of(false, true, 3, 3, salt), c, draws, draws == 4));
    }
  {
    // The tree does not make the copies of a mask symbol agree. A prover can commit to copies that
    // differ; two opened positions sharing a short position then show both. Callers must expect it.
    const OracleTreeShape s = shape_of(true, true, 3, 1, 40);
    std::vector<std::vector<F>> columns(8, std::vector<F>(s.width()));
    for (auto &col : columns) c.fill(col.data(), col.size());
    OracleTree tree(s, c);
    for (u64 x = 0; x < 8; x++) tree.set_leaf(x, columns[x].data(), columns[x].size());
    const Digest root = tree.finish();
    const PositionList list = PositionList::from_draws({1, 3}, 3); // both have the short position 1
    Opening o;
    tree.open(list, o, [&](u64 x, std::vector<F> &out) { out = columns[x]; });
    CHECK(check_opening(root, s, list, o) && o.mask_column(0)[0] != o.mask_column(1)[0]);
  }
  std::printf(failures ? "tree_test: %d FAILURES\n" : "tree_test: all passed\n", failures);
  return failures ? 1 : 0;
}
