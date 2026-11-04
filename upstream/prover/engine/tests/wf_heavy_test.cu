// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests WfGpu and WfProducts (prover/engine/wf_heavy.cuh), the GPU code for the prover's bulk work
// in WF, the committer's range proof by lookups, against the CPU code, on synthetic matrices of 2^4
// and 2^11 columns. WfGpu and WfCpu (blocks/iop/wf.h) must agree, under the same challenges, on
// each lookup's multiplicities (how often each table value occurs among the keys, padding
// excluded), key_roots(), plane_roots(), and the round polynomials and end values of the sumcheck
// at every level of every part's key tree (a lookup's keys are split into parts, WfPart, each with
// its own tree). The table side's sumchecks (LkpStep::TABLE, LEAVES, BITS) are compared only in
// sessions(), in whole proofs at toy size. main() ends with a check for GPU (`card` in identifiers)
// buffers released unzeroed. It begins with a check that a WfKeyMap made without an initialiser has
// no unset byte.
// Build outside the source tree (one command):
//   nvcc -std=c++17 -O2 --gpu-architecture=sm_90 -I blocks -I prover -DBLK_FIELD=2 -DBLK_HASH=1
//     prover/engine/tests/wf_heavy_test.cu -o <somewhere>/wf_heavy_test
// Last line "wf_heavy_test: all passed", exit code 0; after a failed comparison, exit code 1. A
// failed BLK_REQUIRE or CUDA call (ENGINE_CK) aborts the process.
#define BLK_LEFT_BEHIND_LOOK_FOR_A_TEST 1 // `left_unzeroed` then reports to `LeftBehind`
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // for the fixtures (`wf_world`); see blocks/numbers/plan.h
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets the fixtures vouch for a table (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>
#include <new>

#include "engine/engine_cpu.h"
#include "engine/tests/left_behind.cuh"
#include "engine/wf_heavy.cuh"
#include "iop/tests/wf_world.h"
using namespace blk;

// While `look` is set, WfGpu, its LkpPieces and WfProducts allocate from its arena; the matrix and
// the key source never do.
static LeftBehind *look = nullptr;
static CardArena *watched(CardArena *callers) { return look ? look->arena() : callers; }

static int failed = 0, differences = 0;
static bool counting_only = false;  // in a control, differences count but are not failures
static void expect(bool ok, const char *what, unsigned a = 0, unsigned b = 0, unsigned c = 0) {
  if (ok) return;
  differences++;
  if (counting_only) return;
  if (++failed <= 20) std::printf("FAILED: %s (%u, %u, %u)\n", what, a, b, c);
}
// A launch of wf_k_keys() copies all bytes of its WfKeyMap to the GPU.
// This declaration compiles only if every member of WfKeyMap has an initial value.
constexpr WfKeyMap empty_key_map;
static_assert(empty_key_map.digits == 0, "an empty key map has no digits");
// A WfKeyMap that is made without an initialiser, in memory full of the byte A5, must be 0 in every
// byte.
static void key_map_without_initialiser() {
  auto not_zero = [](const unsigned char *bytes) {
    unsigned n = 0;
    for (size_t i = 0; i < sizeof(WfKeyMap); i++) n += bytes[i] != 0;
    return n;
  };
  alignas(WfKeyMap) unsigned char room[sizeof(WfKeyMap)];
  std::memset(room, 0xA5, sizeof room);
  expect(not_zero(room) == sizeof room, "a key map's bytes, a control: the memory was not full of the byte A5");
  const WfKeyMap *map = new (room) WfKeyMap;
  expect(not_zero(reinterpret_cast<const unsigned char *>(map)) == 0, "a WfKeyMap made without an initialiser has bytes that are not 0");
}
// splitmix64 (Steele, Lea, Flood, 2014); for tests only
struct Random {
  u64 s;
  u64 word() {
    u64 z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  E e() { return E::from(from_u64(word()), from_u64(word()), from_u64(word())); }
};

// A test matrix with 2^k_col columns, arranged as in wf_shape_of_units()
// (blocks/frame/wf_units.h): a unit of R rows stores R/2 rows of weights, low digits in its
// first R/2 rows, high digits in the others. An entry is a pair of rows: a weight row's two
// digit rows, or two neighbouring byte-only rows. mu_*, live_*, first_bound: as in WfLookup.
struct MatrixForm {
  unsigned k_col, byte_bits, weight_bits;          // bits of a byte and of a weight, sign included
  std::vector<u64> units;                          // rows of each unit (even)
  u64 bytes_only;                                  // byte-only rows after the units (even)
  std::vector<unsigned> bytes_parts, weight_parts; // log2 of their places
  std::vector<unsigned> first_bound;
  unsigned mu_bytes, live_bytes, mu_weights, live_weights;
};
static void add_run(std::vector<WfRun> &runs, u64 place, u64 row, u64 stride) {
  if (!runs.empty() && runs.back().first_place + runs.back().places == place && runs.back().first_row + runs.back().places == row && runs.back().stride == stride)
    runs.back().places++;
  else
    runs.push_back(WfRun{place, row, 1, stride});
}
// A place gives one key per column. In the lookup "bytes" a place is one row of an entry, chosen by
// the place's top bit as in wf_cut() (blocks/frame/wf_frame.h); in "weights" a place is an entry.
// The parts differ in size; the toy form's last parts are partly padding ("bytes": in each half).
static WfPublic make_public(const MatrixForm &f, u64 &weights, WfMatrix &size) {
  std::vector<std::pair<u64, u64>> entry;
  u64 row = 0;
  for (u64 R : f.units) {
    for (u64 i = 0; i < R / 2; i++) entry.push_back({row + i, row + R / 2 + i});
    row += R;
  }
  weights = entry.size();
  for (u64 i = 0; i < f.bytes_only; i += 2) entry.push_back({row + i, row + i + 1});
  WfPublic x;
  size = WfMatrix{f.k_col, row + f.bytes_only};
  WfLookup bytes, two;
  bytes.name = "bytes", bytes.d = f.byte_bits, bytes.mu = f.mu_bytes, bytes.live = f.live_bytes;
  bytes.digit_weight = {F::one()}, bytes.offset = from_u64(u64(1) << (f.byte_bits - 1)), bytes.first_bound = f.first_bound;
  u64 e = 0;
  for (unsigned L : f.bytes_parts) {
    WfPart p;
    p.log_places = L;
    const u64 half = u64(1) << (L - 1), n = std::min<u64>(half, entry.size() - e);
    for (unsigned q = 0; q < 2; q++)
      for (u64 i = 0; i < n; i++) add_run(p.runs, q * half + i, q ? entry[e + i].second : entry[e + i].first, 0);
    e += n;
    bytes.parts.push_back(p);
  }
  BLK_REQUIRE(e == entry.size(), "the test's parts of the bytes do not cover all entries");
  two.name = "weights", two.d = f.weight_bits, two.mu = f.mu_weights, two.live = f.live_weights;
  two.digit_weight = {F::one(), from_u64(u64(1) << f.byte_bits)}, two.offset = from_u64(u64(1) << (f.weight_bits - 1)), two.first_bound = f.first_bound;
  e = 0;
  for (unsigned L : f.weight_parts) {
    WfPart p;
    p.log_places = L;
    const u64 n = std::min<u64>(u64(1) << L, weights - e);
    for (u64 i = 0; i < n; i++) add_run(p.runs, i, entry[e + i].first, entry[e + i].second - entry[e + i].first);
    e += n;
    two.parts.push_back(p);
  }
  BLK_REQUIRE(e == weights, "the test's parts of the weights do not cover all weights");
  x.lookups = {bytes, two};
  x.message_id = 1;
  return x;
}
static std::vector<signed char> make_matrix(const MatrixForm &f, const WfMatrix &x, u64 seed) {
  Random data{seed};
  const u64 columns = u64(1) << f.k_col;
  std::vector<signed char> M(x.rows * columns);
  const long long B = 1ll << f.byte_bits, W = 1ll << f.weight_bits;
  u64 row = 0;
  for (u64 R : f.units) {
    for (u64 i = 0; i < R / 2; i++)
      for (u64 c = 0; c < columns; c++) {
        long long w = (long long)(data.word() % (u64)W) - W / 2;
        if (i == 0 && c == 0) w = -W / 2;
        if (i == R / 2 - 1 && c == columns - 1) w = W / 2 - 1;
        const long long low = ((w + B / 2) % B + B) % B - B / 2;
        M[(row + i) * columns + c] = (signed char)low, M[(row + R / 2 + i) * columns + c] = (signed char)((w - low) / B);
      }
    row += R;
  }
  for (; row < x.rows; row++)
    for (u64 c = 0; c < columns; c++) M[row * columns + c] = (signed char)((long long)(data.word() % (u64)B) - B / 2);
  return M;
}
static F signed_F(long long v) { return v < 0 ? -from_u64((u64)-v) : from_u64((u64)v); }

static void same_sumcheck(SumcheckProver<E> &gpu, SumcheckProver<E> &cpu, unsigned rounds, Random &coins, unsigned a, unsigned b) {
  for (unsigned t = 0; t < rounds; t++) {
    std::vector<E> x = gpu.round(), y = cpu.round();
    x.resize(3, E::zero()), y.resize(3, E::zero());
    expect(x == y, "a round polynomial differs", a, b, t);
    const E g = coins.e();
    gpu.bind(g), cpu.bind(g);
  }
  expect(gpu.end_values() == cpu.end_values(), "the end values differ", a, b);
}

// Compares WfGpu with WfCpu on one matrix. spoil: 0 none; 1 a weight out of its range; 2 a cell out
// of the byte range (needs byte_bits < 8); 3 (negative control) one cell changed, within its range,
// on the GPU's side only. Under 1 and 2 multiplicities() must return false for the spoiled lookup;
// under 3 some comparison must differ.
static void world(const char *name, const MatrixForm &f, const LkpCut &cut, u64 seed, int spoil = 0) {
  u64 weights = 0;
  WfMatrix size;
  const WfPublic x = make_public(f, weights, size);
  std::vector<signed char> M = make_matrix(f, size, seed), on_card = M;
  const u64 columns = u64(1) << f.k_col;
  if (spoil == 1) on_card[(f.units[0] / 2 + 1) * columns + 3] = M[(f.units[0] / 2 + 1) * columns + 3] = (signed char)(1 << (f.weight_bits - f.byte_bits - 1)), on_card[1 * columns + 3] = M[1 * columns + 3] = 0;
  if (spoil == 2) on_card[(size.rows - 1) * columns + 2] = M[(size.rows - 1) * columns + 2] = (signed char)(1 << (f.byte_bits - 1));
  if (spoil == 3) on_card[(size.rows - 1) * columns + 2] = (signed char)(M[(size.rows - 1) * columns + 2] == 0 ? 1 : 0);
  CardArena *const callers = card_arena();
  const InArena outside(nullptr); // no arena: the matrix, the key source
  Card<signed char> card;
  card.from_host(on_card.data(), on_card.size());
  WfMatrixKeys source(x, size, card.get());
  const InArena in(watched(callers));
  WfGpu gpu(x, size, source, cut);
  const std::vector<signed char> *m = &M;
  WfCpu cpu(x, size, [m, columns](u64 row, u64 column) { return signed_F((*m)[row * columns + column]); });
  Random coins{seed + 1000};
  const int before = differences;
  counting_only = spoil == 3;
  size_t passes = 0;
  for (size_t i = 0; i < x.lookups.size(); i++) {
    std::vector<u64> a, b;
    const bool fits_gpu = gpu.multiplicities(i, a), fits_cpu = cpu.multiplicities(i, b);
    expect(fits_gpu == fits_cpu, "the two sides differ on whether all keys lie in rows", (unsigned)i);
    if (spoil == 1) expect(fits_gpu == (i == 0), "a weight out of its range: the bytes fit, the weights do not", (unsigned)i);
    if (spoil == 2 && i == 0) expect(!fits_gpu, "a cell that is no byte was not noticed");
    if (!fits_gpu || !fits_cpu) continue;
    expect(a == b, "the multiplicities differ", (unsigned)i);
    if (spoil == 1 || spoil == 2) continue;
    LkpProver &p = gpu.prover(i), &q = cpu.prover(i);
    const E alpha = coins.e();
    p.begin(E::zero(), alpha), q.begin(E::zero(), alpha);
    expect(p.key_roots() == q.key_roots(), "the parts' roots differ", (unsigned)i);
    expect(p.plane_roots() == q.plane_roots(), "the planes' roots differ", (unsigned)i);
    const Summand<E> product = lkp_summand_product();
    const std::vector<unsigned> bits = wf_lookups(x, size)[i].part_bits();
    for (unsigned part = 0; part < bits.size(); part++)
      for (unsigned level = 0; level < bits[part]; level++) {
        const std::vector<unsigned> order = level + 1 == bits[part] ? lkp_order_beginning(f.first_bound, level) : lkp_default_order(level);
        std::vector<Weight<E>> vars;
        for (unsigned j = 0; j < level; j++) vars.push_back(Weight<E>::equality(coins.e(), 3));
        const std::unique_ptr<SumcheckProver<E>> s = p.sumcheck(LkpStep::KEYS, part, level, product, vars, order), t = q.sumcheck(LkpStep::KEYS, part, level, product, vars, order);
        same_sumcheck(*s, *t, level, coins, (unsigned)(100 * i + part), level);
      }
    passes += gpu.at_hand()->passes;
  }
  counting_only = false;
  if (!spoil && !cut.plain) expect((gpu.walks_saved > 0) == (f.byte_bits == 8), "parts of the bytes' look-up are counted with the weights' walk where a byte has 8 bits, and only there");
  if (spoil == 3) expect(differences > before, "a control: one cell changed on the GPU's side alone, and nothing differed");
  std::printf("%s: %llu rows, %zu + %zu parts, %zu passes; differences %d; failures so far %d\n", name, (unsigned long long)size.rows, x.lookups[0].parts.size(),
              x.lookups[1].parts.size(), passes, differences - before, failed);
}

// A WfGpu with its matrix, as signed bytes in GPU memory, and its key source.
class OnCard : public WfHeavy {
public:
  OnCard(const WfPublic &x, const WfMatrix &size, const LinMatrix &M) {
    std::vector<signed char> bytes(M.a.size());
    for (size_t i = 0; i < bytes.size(); i++) {
      bool found = false;
      for (int v = -128; v < 128 && !found; v++)
        if (M.a[i] == signed_F(v)) bytes[i] = (signed char)v, found = true;
      BLK_REQUIRE(found, "the test's matrix has an entry that is no signed byte");
    }
    card_.from_host(bytes.data(), bytes.size());
    source_.reset(new WfMatrixKeys(x, size, card_.get(), M.columns, 1));
    heavy_.reset(new WfGpu(x, size, *source_));
  }
  bool multiplicities(size_t lookup, std::vector<u64> &count) override { return heavy_->multiplicities(lookup, count); }
  LkpProver &prover(size_t lookup) override { return heavy_->prover(lookup); }
  WfKeySource &source() { return *source_; }
  const WfGpu *heavy() const { return heavy_.get(); }

private:
  Card<signed char> card_;
  std::unique_ptr<WfMatrixKeys> source_;
  std::unique_ptr<WfGpu> heavy_;
};
// WfProducts on the key source of its own OnCard.
class ProductsOnCard : public MemberHeavy {
public:
  // kept: the session's WfGpu
  ProductsOnCard(const WfPublic &x, const WfMatrix &size, const LinMatrix &M, const WfGpu *kept) : card_(x, size, M), products_(x, size, card_.source(), LkpCut(), kept) {}
  void begin(const std::vector<std::vector<Tensor>> &g) override { products_.begin(g); }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &chi, const std::vector<std::vector<E>> &psi) override {
    return products_.sumcheck(s, vars, chi, psi);
  }
  std::vector<E> combine(const std::vector<E> &a) override { return products_.combine(a); }

private:
  OnCard card_;
  WfProducts products_;
};
// Whole runs of the committer's proof at toy size (wf_session(), blocks/iop/tests/wf_world.h): with
// WfGpu and WfProducts the proof bytes must equal the CPU's under the same seeds, and the verifier
// must accept them; with a weight out of its range both provers must stop without a proof.
static void sessions() {
  EngineCpu engine;
  static OnCard *latest = nullptr;  // the GPU session's; `wf_session` frees it on return
  const WfHeavyMaker on_card = [](const WfPublic &x, const WfMatrix &size, const LinMatrix &M) { return std::unique_ptr<WfHeavy>(latest = new OnCard(x, size, M)); };
  size_t runs = 0;
  for (const WfCase &t : WF_CASES) {
    const WfWorld w = wf_world(t);
    for (int bad = 0; bad < 2; bad++) {
      LinMatrix M = wf_matrix(w.lin.S, 11);
      if (bad) wf_put_digits(M, 3, 5, 0, 2);  // the weight 16; both digits are bytes
      // Each session gets its own table, from the same random values: a session advances its
      // table's count of opened columns.
      TestCoins coins_a(21), coins_b(21);
      IndexCpu index_a(w.lin.S.fold.index, lin_table_rows(w.lin.S, M), coins_a), index_b(w.lin.S.fold.index, lin_table_rows(w.lin.S, M), coins_b);
      const IndexKey key = index_b.key();
      WfSeen a, b, v;
      const LinView cpu = wf_session(w, index_a, index_a.key().rec, M, 41, engine, a);
      const WfPublic *x = &w.wf;
      const MemberHeavyMaker products = [x](const LinShape &S, const LinMatrix &m) { return std::unique_ptr<MemberHeavy>(new ProductsOnCard(*x, wf_matrix_of(S), m, latest->heavy())); };
      const LinView gpu = wf_session(w, index_b, key.rec, M, 41, engine, b, on_card, products);
      expect(a.served == !bad && b.served == !bad, "a session: whether the prover serves", (unsigned)runs);
      expect(cpu.first == gpu.first && cpu.path == gpu.path && cpu.shared == gpu.shared && cpu.second == gpu.second, "a session's views differ", (unsigned)runs);
      if (!bad) expect(!gpu.first.empty() && wf_verify_view(w, key, gpu, v), "the verifier refuses the session made with the GPU's heavy side", (unsigned)runs);
      runs++;
    }
  }
  std::printf("sessions: %zu; failures so far %d\n", runs, failed);
}

// The row weights g of the pending claim (a linear claim on the matrix) that WF leaves for part p
// when the part's key claim is at `point`, built as in wf_pending_row() (blocks/iop/wf.h).
static std::vector<Tensor> weights_at(const WfMatrix &m, const WfLookup &l, const WfPart &p, const std::vector<E> &point) {
  auto e = std::make_shared<std::vector<E>>(1, E::one());
  for (unsigned t = 0; t < p.log_places; t++) {
    const E z = point[m.k_col + t];
    const size_t half = e->size();
    e->resize(2 * half);
    for (size_t i = 0; i < half; i++) (*e)[half + i] = (*e)[i] * z, (*e)[i] = (*e)[i] - (*e)[half + i];
  }
  std::vector<Tensor> g;
  for (const WfRun &r : p.runs)
    for (size_t q = 0; q < l.digit_weight.size(); q++) wf_weights_on_rows(m.k_row(), r.first_row + q * r.stride, e, r.first_place, r.places, E::from(l.digit_weight[q]), g);
  return g;
}
// Checks WfProducts, with LkpCut::plain false and true, against the definition on the CPU:
// W_j = g_j^T M for the pending claims that WF would leave at random points. Only one random
// combination of the W_j, combine(a), is compared.
static void products(const char *name, const MatrixForm &f, LkpCut cut, u64 seed) {
  u64 weights = 0;
  WfMatrix size;
  const WfPublic x = make_public(f, weights, size);
  const std::vector<signed char> M = make_matrix(f, size, seed);
  const u64 columns = u64(1) << f.k_col;
  CardArena *const callers = card_arena();
  const InArena outside(nullptr);
  Card<signed char> card;
  card.from_host(M.data(), M.size());
  WfMatrixKeys source(x, size, card.get());
  const InArena in(watched(callers));
  Random coins{seed + 5};
  std::vector<std::vector<Tensor>> g;
  for (const WfLookup &l : x.lookups)
    for (const WfPart &p : l.parts) {
      std::vector<E> point;
      for (unsigned v = 0; v < size.k_col + p.log_places; v++) point.push_back(coins.e());
      g.push_back(weights_at(size, l, p, point));
    }
  std::vector<E> a, want(columns, E::zero());
  for (size_t j = 0; j < g.size(); j++) a.push_back(coins.e());
  for (size_t j = 0; j < g.size(); j++) {
    std::vector<E> by_row(size_t(1) << size.k_row(), E::zero());
    for (const Tensor &t : g[j]) t.add_to(by_row.data());
    for (u64 row = 0; row < size.rows; row++)
      for (u64 c = 0; c < columns; c++) want[c] += a[j] * by_row[row] * signed_F(M[row * columns + c]);
  }
  for (int plain = 0; plain < 2; plain++) {
    cut.plain = plain != 0;
    WfProducts p(x, size, source, cut);
    p.begin(g);
    expect(p.combine(a) == want, "the products are not those of the definition", (unsigned)plain);
  }
  std::printf("%s: %zu rows on the matrix, fast and plain; failures so far %d\n", name, g.size(), failed);
}

// WfGpu keeps sums for some parts (WfPartSums). A WfProducts that reuses them must give the same
// combination as one that generates every part's keys again, at the points of the parts' key
// claims.
static void kept_sums(const char *name, const MatrixForm &f, const LkpCut &cut, u64 seed) {
  u64 weights = 0;
  WfMatrix size;
  const WfPublic x = make_public(f, weights, size);
  const std::vector<signed char> M = make_matrix(f, size, seed);
  CardArena *const callers = card_arena();
  const InArena outside(nullptr);
  Card<signed char> card;
  card.from_host(M.data(), M.size());
  WfMatrixKeys source(x, size, card.get());
  const InArena in(watched(callers));
  WfGpu gpu(x, size, source, cut);
  Random coins{seed + 7};
  std::vector<std::vector<Tensor>> g;
  size_t kept = 0;
  for (size_t i = 0; i < x.lookups.size(); i++) {
    std::vector<u64> count;
    expect(gpu.multiplicities(i, count), "a key in no row", (unsigned)i);
    LkpProver &p = gpu.prover(i);
    p.begin(E::zero(), coins.e());
    const Summand<E> product = lkp_summand_product();
    const std::vector<unsigned> bits = wf_lookups(x, size)[i].part_bits();
    for (unsigned part = 0; part < bits.size(); part++) {
      std::vector<E> point(bits[part]);
      for (unsigned level = 0; level < bits[part]; level++) {
        const std::vector<unsigned> order = level + 1 == bits[part] ? lkp_order_beginning(f.first_bound, level) : lkp_default_order(level);
        std::vector<Weight<E>> vars;
        for (unsigned j = 0; j < level; j++) vars.push_back(Weight<E>::equality(coins.e(), 3));
        const std::unique_ptr<SumcheckProver<E>> s = p.sumcheck(LkpStep::KEYS, part, level, product, vars, order);
        for (unsigned t = 0; t < level; t++) {
          s->round();
          const E c = coins.e();
          s->bind(c);
          if (level + 1 == bits[part]) point[order[t]] = c;
        }
        s->end_values();
      }
      point[bits[part] - 1] = coins.e();  // LINE's challenge, which joins two end claims
      g.push_back(weights_at(size, x.lookups[i], x.lookups[i].parts[part], point));
      kept += gpu.sums(i, part) != nullptr;
    }
  }
  std::vector<E> a;
  for (size_t j = 0; j < g.size(); j++) a.push_back(coins.e());
  WfProducts on_kept(x, size, source, cut, &gpu), anew(x, size, source, cut);
  bool given_back = false;
  on_kept.after_begin([&gpu, &given_back] { gpu.done(), given_back = true; });  // after `begin` the vectors must not need the sums in `gpu`
  on_kept.begin(g), anew.begin(g);
  expect(given_back && !gpu.sums(0, 0), "the routine behind begin() was not run, or the sums are still there");
  expect(on_kept.combine(a) == anew.combine(a), "the products from the kept sums differ");
  expect(kept > 0, "no part's sums were kept");
  std::printf("%s: %zu rows on the matrix, %zu from kept sums; failures so far %d\n", name, g.size(), kept, failed);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  key_map_without_initialiser();
  // 22 rows: units of 12 and 6 rows, 4 byte-only rows: 11 entries, 9 of them weights. A toy byte
  // has 3 bits, a weight 5. Every part is in memory whole.
  const MatrixForm toy{4, 3, 5, {12, 6}, 4, {4, 2, 2}, {3, 1}, {1, 0}, 4, 9, 3, 0};
  // Printed as "pieces": the committer's ranges (signed bytes; weights in [-4096, 4095]). 144 rows:
  // units of 96 and 40 rows, 8 byte-only rows: 72 entries, 68 of them weights. Parts of 128 and 16
  // places ("bytes"), 64 and 4 ("weights"); the large parts are cut into pieces:
  // LkpCut{piece_places, base, room, rounds, plain} (prover/engine/lkp_pieces.cuh).
  // `nine`: as in wf_shape_of_units().
  const std::vector<unsigned> nine = {2, 1, 0, 5, 4, 3, 8, 7, 6};
  const MatrixForm real{11, 8, 13, {96, 40}, 8, {7, 4}, {6, 2}, nine, 5, 20, 5, 0};
  // At 0 or more, `card_test_byte` (prover/engine/device.cuh) fills GPU buffers at allocation and
  // wherever their owner calls spent(), to expose reads of memory never written.
  for (int byte : {-1, 0x5A, 0xA5}) {
    card_test_byte = byte;
    const unsigned long long before = card_fills;
    sessions();
    world("toy", toy, LkpCut(), 1);
    world("toy, a weight out of its range", toy, LkpCut(), 2, 1);
    world("toy, a cell that is no byte", toy, LkpCut(), 3, 2);
    world("toy, control: one cell changed on the GPU's side", toy, LkpCut(), 4, 3);
    world("pieces", real, LkpCut{5, 13, 12, 3}, 5);
    world("pieces, another cut", real, LkpCut{4, 14, 11, 2}, 6);
    world("pieces, a weight out of its range", real, LkpCut{5, 13, 12, 3}, 7, 1);
    world("pieces, control: one cell changed on the GPU's side", real, LkpCut{5, 13, 12, 3}, 8, 3);
    kept_sums("kept sums, toy", toy, LkpCut(), 12);
    kept_sums("kept sums, pieces", real, LkpCut{5, 13, 12, 3}, 13);
    kept_sums("kept sums, another cut", real, LkpCut{4, 14, 11, 2}, 14);
    world("pieces, the plain way (a walk for the roots)", real, LkpCut{5, 13, 12, 3, true}, 15);
    products("products, toy", toy, LkpCut(), 9);
    products("products, pieces", real, LkpCut{5, 13, 12, 3}, 10);
    products("products, another cut", real, LkpCut{4, 14, 11, 2}, 11);
    expect((card_fills > before + 100) == (byte >= 0), "under a test's byte no more than 100 buffers were filled, or some without it");
    std::printf("all worlds %s: %llu fills\n", byte < 0 ? "as a program runs them" : byte == 0x5A ? "under the byte 5A" : "under the byte A5", card_fills - before);
  }
  card_test_byte = -1;
  // The objects under test must zero each GPU buffer before releasing it, unless they mark it with
  // left_unzeroed(). Six cases (`scenes`) run again on the zero-filled arena of LeftBehind
  // (prover/engine/tests/left_behind.cuh); after each, left() must return 0. sessions() is not
  // among them, and LeftBehind does not check the reasons given to left_unzeroed(). Three controls
  // follow.
  {
    LeftBehind l("wf_heavy_test");
    look = &l;
    struct Scene {
      const char *name;
      std::function<void()> play;
    };
    const std::vector<Scene> scenes = {
        {"left behind: the look-ups, toy", [&] { world("left behind: toy", toy, LkpCut(), 21); }},
        {"left behind: the look-ups, pieces", [&] { world("left behind: pieces", real, LkpCut{5, 13, 12, 3}, 22); }},
        {"left behind: the look-ups, the plain way", [&] { world("left behind: pieces, the plain way", real, LkpCut{5, 13, 12, 3, true}, 23); }},
        {"left behind: a weight out of its range", [&] { world("left behind: pieces, a weight out of its range", real, LkpCut{5, 13, 12, 3}, 24, 1); }},
        {"left behind: kept sums", [&] { kept_sums("left behind: kept sums, pieces", real, LkpCut{5, 13, 12, 3}, 25); }},
        {"left behind: products", [&] { products("left behind: products, pieces", real, LkpCut{5, 13, 12, 3}, 26); }},
    };
    for (const Scene &s : scenes) {
      l.say(s.name);
      const size_t before = l.taken();
      s.play();
      expect(l.taken() > before, "left behind: a scene took nothing from the arena: the look saw nothing");
      expect(l.left() == 0, "left behind: a GPU buffer went back with words that are not zero");
    }
    l.say("the control");
    {
      const InArena in(l.arena());
      Card<u64> forgotten(1000);
      ENGINE_CK(cudaMemset(forgotten.get(), 0x5A, 1000 * sizeof(u64)));
    }
    expect(l.left() == 1000 && l.let() == 0, "left behind, a control: a buffer of 1,000 words that went back unwiped was not seen whole");
    {
      const InArena in(l.arena());
      Card<u64> named(1000);
      left_unzeroed(named, "a control's words");
      ENGINE_CK(cudaMemset(named.get(), 0x5A, 1000 * sizeof(u64)));
    }
    expect(l.left() == 0 && l.let() == 1000, "left behind, a control: a buffer that the source names as left as it is was not let whole");
    {  // what a buffer leaves is counted even if the next buffer in its range zeroes it
      const InArena in(l.arena());
      { Card<u64> forgotten(1000); ENGINE_CK(cudaMemset(forgotten.get(), 0x5A, 1000 * sizeof(u64))); }
      { Card<u64> tidy(1000); ENGINE_CK(cudaMemset(tidy.get(), 0, 1000 * sizeof(u64))); }
    }
    expect(l.left() == 1000, "left behind, a control: leavings in a range that a tidy holder took next were not charged");
    look = nullptr;
  }
  std::printf(failed ? "wf_heavy_test: %d FAILED\n" : "wf_heavy_test: all passed\n", failed);
  return failed ? 1 : 0;
}
