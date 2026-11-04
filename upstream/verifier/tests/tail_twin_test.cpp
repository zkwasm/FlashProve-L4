// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests the verifier's code for the approximate range check (verifier/src/vf_smallness_test.h,
   verifier/src/vf_smallness_test_members.inc) claim by claim, against a CPU prover written here. Printed lines call
   the check the "smallness test", identifiers the `tail`; its parts are defined above `Shape`.

   A cell (one committed small integer) that no lookup covers is bounded only by this check. The
   code under test checks no equation on a proof's values itself. It adds linear claims: a weighted
   sum of committed values equals a public value. They count only through the one equation of the
   sub-proof (the proof for one part of the model; `fork` in identifiers): the sum of all claims,
   each multiplied by a power of the challenge theta. So a run of the verifier program cp_verify on
   altered files does not show which claim is false. Here a synthetic sub-proof consists only of
   the range check, and each claim is evaluated on the prover's data. "Accepted" below means
   `Verdict::accepted`, not a `YES` of cp_verify.

   The check is to show |cell| < 2 D = 2^18, not that a cell is a signed byte: a committed cell of
   200 or of 100,000 is accepted, by design.

   Not run: the closing of the chain segment (the sub-proof's part of a request's hash chain;
   `link` in identifiers and output) and the proximity tests (see `verify`); some shapes of
   matrices and arrays (see `makeWorld`).

   One test fixture (`world` in identifiers and output) commits its arrays in a wrong order, which
   the code must reject (see `main`).

   Exit code 0: every result is as expected; 1: not; 2: the fixture is inconsistent. A failed guard
   of blocks/ (`BLK_REQUIRE`) ends the program by `abort`.
   Build, from verifier/src:
   g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=2 -DBLK_HASH=1 -I .
   -I ../../blocks -I ../../prover ../tests/tail_twin_test.cpp
   -o <a folder of your own>/tail_twin_test */
#include "vf_smallness_test.h"
#include "claims_report.h"

#include "iop/lkp.h"
#include "iop/tests/req_rows_cases.h"

typedef long long i64t;
static int bad = 0;
static blk::E eOf(u64 x) { return blk::E::from(blk::from_u64(x)); }
static blk::E eOfSigned(i64t x) { return x >= 0 ? eOf((u64)x) : blk::E::zero() - eOf((u64)-x); }
static blk::E eOf(const Ex &x) { return blk::E::from(blk::from_u64(eC(x, 0)), blk::from_u64(eC(x, 1)), blk::from_u64(eC(x, 2))); }
static u64 rngState = 7;
static u64 madeUp64() {
  rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
  return rngState >> 11;
}
constexpr i64t DD = 1ll << LBZ;  // D
constexpr u64 BB = 1ull << LRAD; // the radix of a symbol's three lanes

// ---- the synthetic sub-proof's shape ----
/* The range check. A matrix has m rows and k columns of cells; three rows are committed as one
   row of symbols y0 + B y1 + B^2 y2 (three lanes), B = 2^20. A functional is a public weight
   vector phi of length L on one matrix. Its table, in the array U3, has for each column and each
   row segment of L rows the sum of the cells weighted by phi. Every row must be covered by a
   functional, or `selfTestF` rejects.
   After these commitments the verifier draws a matrix, tau entries for each column of cells: 0
   with probability 1/2, otherwise 1 or -1 (`Data::theta`; `F.theta` is the batching challenge).
   Combination u of a row is the sum of its cells, each times its column's entry u. The prover
   commits to each combination + D, D = 2^17, as a low digit of 10 bits and a high digit of 8
   (array ZZ; lookup calls R10z and R8d), so the combination must be in the window [-D, D).
   Claims: a tie for each matrix (the digits against the symbols); a read for each functional and
   row segment (the digits against the table); the lookup's claims that each digit is in its range,
   among them the join (the lookup's keys are the entries of ZZ). The array BZ has the bit planes
   of the lookup's multiplicities. */
struct Shape {
  u32 tau;
  vector<std::pair<u64, u64>> mk;   // each matrix's rows and columns
  vector<std::pair<int, u64>> reads; // each functional's matrix and length
  bool readsLate = false;            // the wrong order (see `main`): U3 comes after ZZ and BZ
};
// One committed array (`tree` in the verifier's identifiers); tree: its index in `HFork::trees`.
struct Decl {
  string name;
  u64 R, K;
  bool overE;
  unsigned moment;
  int tree;
};
// The test fixture. nX4: the number of reads. nU[n]: the length of array U(n+1).
struct World {
  Shape s;
  vector<TMat> mats;
  vector<HFun> fun;
  u64 nU[3] = {0, 0, 0};
  size_t nX4 = 0;
  HTailF2 tl;  // the plan's copy: shapes only
  HFork F;
  Tail2 T;
  vector<Decl> arrays;
  vector<std::pair<size_t, string>> arrayOf;
  blk::ReqLayout layout;
  size_t rows[3] = {0, 0, 0};
  std::unique_ptr<blk::ReqWorld> w;
};

// Places the arrays of one commit point (`moment`) in an oracle message: one block for each row
// and aligned run of 2^b columns, largest first, so that every block is aligned. `rows` is one
// more than the blocks fill, so that no block covers the pad that masks a sample's answer
// (Send::pad(), blocks/iop/send.h).
static void layMessage(const vector<Decl> &all, unsigned moment, size_t message, unsigned logEll, vector<blk::ReqArray> &out, size_t &rows) {
  struct Chunk {
    size_t a;
    u64 i, c0;
    unsigned bits;
  };
  vector<Chunk> chunks;
  const size_t first = out.size();
  for (const Decl &d : all) {
    if (d.moment != moment) continue;
    out.push_back(blk::ReqArray{d.name, message, blk::LaidMatrix{d.R, d.K, {}}});
    for (u64 i = 0; i < d.R; i++)
      for (u64 c0 = 0; c0 < d.K;) {
        unsigned b = 0;
        while (b < logEll && c0 + (2ull << b) <= d.K && !(c0 & ((2ull << b) - 1))) b++;
        chunks.push_back({out.size() - 1, i, c0, b});
        c0 += 1ull << b;
      }
  }
  (void)first;
  std::stable_sort(chunks.begin(), chunks.end(), [](const Chunk &x, const Chunk &y) { return x.bits > y.bits; });
  u64 at = 0;
  for (const Chunk &c : chunks) {
    blk::LaidBlock b;
    b.first_i = c.i, b.bits_i = 0, b.first_c = c.c0, b.bits_c = c.bits, b.first_row = at >> logEll, b.row_bits = 0;
    b.first_place = at & ((1ull << logEll) - 1), b.place_bits = c.bits, b.i_is_low = false;
    out[c.a].laid.blocks.push_back(b);
    at += 1ull << c.bits;
  }
  rows = (size_t)((at + (1ull << logEll) - 1) >> logEll) + 1;
}

// A variant of req_toy_world() (blocks/iop/tests/req_cases.h): one chain segment with this
// sub-proof's three oracle messages and a mask row long enough for its scalars. `target`: the
// plan's error target in bits; makeWorld() passes tau, not the example configurations' 128.
// `readsLate` (see Shape) exchanges the second and the third message.
static std::unique_ptr<blk::ReqWorld> frameOf(unsigned target, unsigned log_n, u64 maskEll, const size_t rows[3], bool readsLate, const blk::ReqSubPlan &sub) {
  using namespace blk;
  Setting set = setting_goldilocks();
  set.target = target;
  std::unique_ptr<ReqWorld> made(new ReqWorld);
  ReqWorld &w = *made;
  Plan plan(set);
  GroupSize g;
  g.name = "links", g.k = ceil_log2(rows[0] + E_DEGREE * rows[readsLate ? 2 : 1] + rows[readsLate ? 1 : 2] + 8), g.h = 0;
  g.main.name = "rows", g.main.log_n = log_n, g.main.ell = u64(1) << (log_n - 2);
  g.mask.name = "mask row", g.mask.log_n = ceil_log2(maskEll) + 3, g.mask.ell = maskEll, g.mask.over_E = true;
  u64 draws = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, draws);
    GroupLoad load;
    load.cube_bits = 40, load.batched_rows = u64(1) << 16;
    g = size_group(set, plan.scope(), g, load);
    if (!g.m) {
      const Analysis a = analysis(set, group_of_m(g, 3));
      printf("the test's world is wrong: no slack holds the group's rounds: target %u; main 2^%u, %llu + %llu; mask 2^%u, %llu + %llu; k %u; at m = 3 the analysis says: %s\n", set.target, g.main.log_n,
             (unsigned long long)g.main.ell, (unsigned long long)g.main.r, g.mask.log_n, (unsigned long long)g.mask.ell, (unsigned long long)g.mask.r, g.k, a.fault ? a.fault : a.bad() ? "bad, no rule named" : "not bad");
      exit(2);
    }
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= draws) break;
    draws = more;
  }
  w.sizes.group = g;
  GroupSize last = g;
  last.name = "last group", last.k = 1;
  for (unsigned n = log_n - 2;; n++) {
    BLK_REQUIRE(n < log_n + 8, "test: no lengths serve the later stages");
    Plan trial(set);
    w.sizes.last = solve_stages(trial, last, draws, {StageChoice{3, 2, n, n - 1}, StageChoice{0, 0, n + 1, n + 1}});
    if (!trial.refused()) break;
  }
  w.shape.last = iopp_stages(w.sizes.last, w.codes);
  w.shape.choice.masks_ride = true;
  auto message = [&](const char *name, size_t r, unsigned deg, size_t mask_rows, unsigned moment) {
    ReqMessage m;
    m.shape.name = name, m.shape.code = 0, m.shape.main_rows = r, m.shape.main_deg = deg, m.shape.mask_rows = mask_rows;
    m.shape.mask_code = 0, m.shape.answers = set.samples;
    m.moment = moment, m.life = 0;
    return m;
  };
  ReqLink link;
  link.name = "the made-up sub-proof";
  link.messages.push_back(message("cells", rows[0], 1, 1, 0));
  link.messages.push_back(readsLate ? message("digits", rows[1], 1, 0, 1) : message("reads", rows[1], E_DEGREE, 0, 1));
  link.messages.push_back(readsLate ? message("reads", rows[2], E_DEGREE, 0, 2) : message("digits", rows[2], 1, 0, 2));
  w.shape.link.push_back(link);
  w.sizes.rows.push_back(1);
  plan_chain_start(plan, "request", true);
  req_plan(plan, w.codes, w.shape, w.sizes, sub);
  w.plan = plan.finish();
  return made;
}

// Builds the fixture. It has no matrix that an earlier chain segment committed to (TMat::kind 1),
// none that another sub-proof reads (txOf >= 0) and, of the arrays U1 to U3, only U3: the branches
// of shapeF() and addTreesF() for these do not run.
static void makeWorld(World &W, const Shape &s, unsigned log_n) {
  W.s = s;
  TAU = (int)s.tau;
  statedTarget.hb = 0;
  for (size_t mu = 0; mu < s.mk.size(); mu++) {
    TMat M;
    M.name = "Y" + std::to_string(mu), M.m = s.mk[mu].first, M.k = s.mk[mu].second, M.kind = 0, M.aId = mu;
    W.mats.push_back(M);
  }
  u64 at = 0;
  for (size_t i = 0; i < s.reads.size(); i++) {
    HFun f;
    f.M = s.reads[i].first, f.L = s.reads[i].second, f.tree = 2, f.at = at, f.name = "read" + std::to_string(i);
    for (u64 j = 0; j < f.L; j++) f.phi.push_back(Ex{madeUp64() % FIELD_P, madeUp64() % FIELD_P, madeUp64() % FIELD_P});
    const TMat &M = W.mats[(size_t)f.M];
    at += M.k * (M.m / f.L), W.nX4 += M.m / f.L;
    W.fun.push_back(f);
  }
  W.nU[2] = at;
  const bool hasU[4] = {false, false, true, false};
  W.tl.shape(W.mats, vector<int>(W.mats.size(), -1), {}, hasU, s.tau);
  W.F.G_ = "toy:";
  W.tl.addTrees(W.F, W.T, W.nU, {}, 0);
  t2Plan(W.F, W.T);
  W.arrayOf.assign(W.F.trees.size(), {0, string()});
  auto declare = [&](int tree, const string &name, unsigned moment) {
    W.arrays.push_back({name, W.F.trees[(size_t)tree].R, W.F.trees[(size_t)tree].K, W.T.nt[(size_t)tree].ec == (u32)CE, moment, tree});
    W.arrayOf[(size_t)tree] = {0, name};
  };
  for (size_t i = 0; i < W.tl.pkMat.size(); i++) declare(W.tl.ix.PK[i], "M" + std::to_string(W.tl.pkMat[i]), 0);
  const bool late = s.readsLate;
  declare(W.tl.ix.U[2], "U3", late ? 2 : 1);
  declare(W.tl.ix.ZZ, "ZZ", late ? 1 : 2);
  declare(W.tl.ix.BZ, "BZ", late ? 1 : 2);
  vector<blk::ReqArray> laid;
  for (unsigned m = 0; m < 3; m++) layMessage(W.arrays, m, m, log_n - 2, laid, W.rows[m]);
  W.layout.link.push_back(laid);
  const HTailF2 *tl = &W.tl;
  const size_t nX4 = W.nX4;
  u64 maskEll = 64;
  while (maskEll < W.tl.smallElems() + 32) maskEll *= 2;
  W.w = frameOf(s.tau, log_n, maskEll, W.rows, late, [tl, nX4, late](blk::Plan &p, size_t, unsigned moment) {
    if (moment == (late ? 0u : 1u)) tl->planRounds(p, "the made-up sub-proof", 0);
    if (moment == (late ? 1u : 2u)) tl->planRounds(p, "the made-up sub-proof", 1);
    if (moment == 2) blk::plan_bat_sigma(p, "the made-up sub-proof, claims", tl->lins(nX4) + tl->equationsF(nX4));
  });
}

// ---- the prover's data ----
// i': a row of symbols. x: off[mu] + i for a low digit (off[mu]: the earlier matrices' rows), mz
// (all matrices' rows) more for a high one.
struct Data {
  vector<vector<i64t>> Y;          // Y[mu][i * k + c]: the cells, as integers
  vector<vector<blk::E>> sym;      // sym[mu][i' * k + c]: what is committed
  vector<blk::E> tab;              // U3
  vector<signed char> theta;       // the matrix, as drawn
  vector<vector<i64t>> psi;        // psi[mu][i * tau + u]: the numbers whose digits are sent
  vector<u64> zz, bits;            // ZZ[u * 2 mz + x]; BZ[j + 32 r]: bit j of multiplicity r
  vector<u64> zzLooked;            // the lookup's keys: ZZ's digits, except under OTHER_KEYS
  vector<blk::E> mask;
};
// What a malicious prover does. main() runs nine provers at tau = 24 and compares the names of the
// false claims with those expected; HALVES runs only in tries(), READ_UNSEEN only in a
// fixture with Shape::readsLate. "Good packing": the symbols are those of the true cells.
enum class Cheat {
  NONE,
  CELL_200,        // below the bound 2 D: must be accepted
  CELL_WIDE,       // one cell = wide; for a combination outside the window the digits of 0 are sent
  READ_FALSE,      // good packing, one table entry + 1
  COMBINATION_OUT, // combination 0 is not computed: the digits of 0 are sent
  ROWS_EXCHANGED,  // two symbol rows exchanged; the digits are the true matrix's
  OTHER_KEYS,      // in ZZ d0 + 1024, d1 - 1: the same number, d0 outside R10z's table; the lookup runs on the true digits
  READS_CANCEL,    // one table entry + 1; another functional's, same matrix and column, - 1
  HALVES,          // two columns hold a / 2 for small odd a; a non-integer combination: digits of 0
  READ_UNSEEN      // one row segment of a table + a vector that the drawn matrix maps to 0
};
constexpr size_t HALVES_IN = 1;  // the matrix that HALVES alters
// The index in U3 of a functional's table entry for column c and row segment sg, written
// separately from the verifier's tabAt().
static u64 tablePlace(const World &W, const HFun &f, u64 c, u64 sg) { return f.at + c * (W.mats[(size_t)f.M].m / f.L) + sg; }
static void cellsOf(const World &W, Data &D, Cheat cheat, i64t wide) {
  const size_t NM = W.mats.size();
  D.Y.resize(NM), D.sym.resize(NM);
  for (size_t mu = 0; mu < NM; mu++) {
    const TMat &M = W.mats[mu];
    D.Y[mu].resize(M.m * M.k);
    for (i64t &y : D.Y[mu]) y = (i64t)(madeUp64() % 255) - 127;
  }
  if (cheat == Cheat::CELL_200) D.Y[0][5 * W.mats[0].k + 1] = 200;
  if (cheat == Cheat::CELL_WIDE) D.Y[0][5 * W.mats[0].k + 1] = wide;
  if (cheat == Cheat::HALVES)  // Y has the numerators: odd numbers a, for the cells a / 2
    for (u64 i = 0; i < W.mats[HALVES_IN].m; i++)
      for (u64 c = 1; c <= 2; c++) D.Y[HALVES_IN][i * W.mats[HALVES_IN].k + c] |= 1;
  auto half = [&](size_t mu, u64 c) { return cheat == Cheat::HALVES && mu == HALVES_IN && (c == 1 || c == 2) ? eOf((FIELD_P + 1) / 2) : eOf(1); };
  for (size_t mu = 0; mu < NM; mu++) {
    const TMat &M = W.mats[mu];
    const u64 mS = (M.m + 2) / 3;
    D.sym[mu].assign(mS * M.k, blk::E::zero());
    for (u64 i = 0; i < M.m; i++)
      for (u64 c = 0; c < M.k; c++) {
        blk::E lane = eOf(1);
        for (u64 l = 0; l < i % 3; l++) lane *= eOf(BB);
        D.sym[mu][(i / 3) * M.k + c] += lane * half(mu, c) * eOfSigned(D.Y[mu][i * M.k + c]);
      }
  }
  if (cheat == Cheat::ROWS_EXCHANGED)
    for (u64 c = 0; c < W.mats[0].k; c++) std::swap(D.sym[0][1 * W.mats[0].k + c], D.sym[0][2 * W.mats[0].k + c]);
  D.tab.assign(W.nU[2], blk::E::zero());
  for (const HFun &f : W.fun) {
    const TMat &M = W.mats[(size_t)f.M];
    for (u64 sg = 0; sg < M.m / f.L; sg++)
      for (u64 c = 0; c < M.k; c++) {
        blk::E v = blk::E::zero();
        for (u64 i = 0; i < f.L; i++) v += eOf(f.phi[i]) * eOfSigned(D.Y[(size_t)f.M][(sg * f.L + i) * M.k + c]);
        D.tab[tablePlace(W, f, c, sg)] = v * half((size_t)f.M, c);
      }
  }
  if (cheat == Cheat::READ_FALSE || cheat == Cheat::READS_CANCEL) D.tab[tablePlace(W, W.fun[0], 1, 0)] += eOf(1);
  if (cheat == Cheat::READS_CANCEL) D.tab[tablePlace(W, W.fun[1], 1, 0)] -= eOf(1);
}
// Once the verifier's matrix is drawn: computes the combinations and their digits. Returns how
// many combinations left the window and were replaced by 0.
static size_t digitsOf(const World &W, Data &D, Cheat cheat) {
  const u32 tau = W.s.tau;
  const u64 mz = W.tl.mz;
  size_t gaveUp = 0;
  D.psi.resize(W.mats.size());
  D.zz.assign((size_t)tau * 2 * mz, 0), D.zzLooked = D.zz;
  u64 column0 = 0;
  for (size_t mu = 0; mu < W.mats.size(); mu++) {
    const TMat &M = W.mats[mu];
    D.psi[mu].assign(M.m * tau, 0);
    for (u64 i = 0; i < M.m; i++)
      for (u32 u = 0; u < tau; u++) {
        i64t p = 0;
        const bool halves = cheat == Cheat::HALVES && mu == HALVES_IN;
        for (u64 c = 0; c < M.k; c++) p += (i64t)D.theta[(size_t)(column0 + c) * tau + u] * D.Y[mu][i * M.k + c] * (halves && c != 1 && c != 2 ? 2 : 1);
        if (halves && (p & 1)) p = 4 * DD;  // not an integer: treated as outside the window
        else if (halves) p /= 2;
        if (cheat == Cheat::COMBINATION_OUT && u == 0) p = 0;
        if (p < -DD || p >= DD) p = 0, gaveUp++;
        D.psi[mu][i * tau + u] = p;
        u64 d0 = (u64)(p + DD) & 1023, d1 = (u64)(p + DD) >> 10;
        D.zzLooked[(size_t)u * 2 * mz + W.tl.off[mu] + i] = d0, D.zzLooked[(size_t)u * 2 * mz + mz + W.tl.off[mu] + i] = d1;
        if (cheat == Cheat::OTHER_KEYS && mu == 0 && i == 3 && u == 1) d0 += 1024, d1 -= 1;
        D.zz[(size_t)u * 2 * mz + W.tl.off[mu] + i] = d0, D.zz[(size_t)u * 2 * mz + mz + W.tl.off[mu] + i] = d1;
      }
    column0 += M.k;
  }
  if (cheat != Cheat::OTHER_KEYS) D.zzLooked = D.zz;
  return gaveUp;
}

// Returns a nonzero vector v over the first matrix's k columns that the drawn matrix maps to 0:
// sum_c theta[c * tau + u] v[c] = 0 for every combination u. k > tau guarantees that v exists.
static vector<blk::E> unseenBy(const vector<signed char> &theta, u64 k, u32 tau) {
  vector<vector<blk::E>> a(tau, vector<blk::E>(k));
  for (u32 u = 0; u < tau; u++)
    for (u64 c = 0; c < k; c++) a[u][c] = eOfSigned(theta[(size_t)c * tau + u]);
  vector<int> pivot(k, -1);  // the row that has a column's leading 1
  size_t rank = 0;
  for (u64 c = 0; c < k && rank < tau; c++) {
    size_t p = rank;
    while (p < tau && a[p][c] == blk::E::zero()) p++;
    if (p == tau) continue;
    std::swap(a[p], a[rank]);
    const blk::E scale = blk::inv(a[rank][c]);
    for (blk::E &x : a[rank]) x *= scale;
    for (size_t q = 0; q < tau; q++) {
      const blk::E f = a[q][c];
      if (q != rank && !(f == blk::E::zero()))
        for (u64 x = 0; x < k; x++) a[q][x] -= f * a[rank][x];
    }
    pivot[c] = (int)rank++;
  }
  u64 free = 0;
  while (free < k && pivot[free] >= 0) free++;
  if (free == k) printf("the test's world is wrong: the matrix has no more columns than combinations\n"), exit(2);
  vector<blk::E> v(k, blk::E::zero());
  v[free] = blk::E::one();
  for (u64 c = 0; c < k; c++)
    if (pivot[c] >= 0) v[c] = blk::E::zero() - a[(size_t)pivot[c]][free];
  return v;
}

// The prover's side of blk::lkp_run(), as in verifier/tests/forest_twin_test.cpp, but with the
// chain segment's mask row and its `MaskLedger` (blocks/iop/mask.h), which gives out each mask
// entry once.
struct Side {
  blk::ProverRounds &c;
  blk::Statement &st;
  blk::MaskLedger &ledger;
  blk::LkpProver &heavy;
  const vector<blk::E> &mask;
  vector<blk::E> power, plane;
  blk::ProverRounds &ch() { return c; }
  blk::Statement &statement() { return st; }
  void tie_factor(const blk::E &) {}
  blk::MaskValues<blk::E> values() {
    const vector<blk::E> *m = &mask;
    return [m](blk::u32, blk::u32 index) { return (*m)[index]; };
  }
  void begin(const blk::E &theta, const blk::E &alpha) {
    heavy.begin(theta, alpha);
    plane = heavy.plane_roots();
    power = plane;
    for (size_t j = plane.size() - 1; j-- > 0;) power[j] = power[j + 1] * power[j + 1] * plane[j];
  }
  bool tie(blk::SumClaim<blk::E> &&claim, const blk::Summand<blk::E> &S, size_t j, blk::PscOutput<blk::E> &out) {
    vector<vector<blk::E>> maps;
    if (j == 0) maps.push_back({heavy.key_roots()[0]});
    maps.push_back({power[j + 1]}), maps.push_back({plane[j]});
    blk::SumcheckCpu<blk::E> inner(S, claim.vars, std::move(maps));
    vector<blk::MaskSlot> fresh = ledger.take(blk::psc_masks(claim, S));
    out = blk::psc_prove<blk::E>(c, std::move(claim), S, inner, std::move(fresh), values());
    return true;
  }
  void note(const char *, const blk::E &) {}
  bool split(blk::SumClaim<blk::E> &&, const blk::Summand<blk::E> &, size_t, blk::PscOutput<blk::E> &) {
    printf("the test's world is wrong: its keys are one part\n"), exit(2);
  }
  bool psc(blk::SumClaim<blk::E> &&claim, const blk::Summand<blk::E> &S, blk::LkpStep step, unsigned part, unsigned level, const vector<unsigned> &order, bool, blk::PscOutput<blk::E> &out) {
    const std::unique_ptr<blk::SumcheckProver<blk::E>> inner = heavy.sumcheck(step, part, level, S, claim.vars, order);
    vector<blk::MaskSlot> fresh = ledger.take(blk::psc_masks(claim, S));
    out = blk::psc_prove<blk::E>(c, std::move(claim), S, *inner, std::move(fresh), values());
    return true;
  }
};

// The lookup's public inputs, and the prover's keys and multiplicities from Data::zzLooked.
struct Lookup {
  blk::LkpPublic fixed;
  vector<blk::E> keys, bits;
};
static Lookup lookupOf(const World &W, Data &D, blk::OracleId planes) {
  const HForest &h = W.tl.f9;
  const Call *calls[2] = {&W.tl.z[1].c, &W.tl.z[2].c};
  const size_t trees = (size_t)1 << h.rK, rows = (size_t)1 << h.d(), nc = (size_t)1 << h.cK;
  const u64 mz2 = 2 * W.tl.mz;
  vector<vector<u64>> value(2);
  vector<vector<char>> real(2);
  for (int q = 0; q < 2; q++) {
    value[q].assign(1ull << calls[q]->n, calls[q]->tauAt(calls[q]->t0)), real[q].assign(1ull << calls[q]->n, 0);
    for (const HPiece &p : W.tl.z[1 + q].pc)
      for (u64 r = 0; r < p.Rp; r++)
        for (u64 c = 0; c < p.Cp; c++) value[q][p.off + r * p.CP + c] = D.zzLooked[(size_t)(p.r0 + r) * mz2 + p.c0 + c], real[q][p.off + r * p.CP + c] = 1;
  }
  Lookup L;
  blk::LkpPublic &fixed = L.fixed;
  fixed.parts = {(unsigned)h.v()}, fixed.d = h.d(), fixed.mu = 5, fixed.g = h.rK, fixed.shifted = h.NC > 1, fixed.read_from = {planes};
  auto paddingRow = [&](int c) { return (u64)c + (calls[c]->t0 << h.cK); };
  for (int c = 0; c < h.NC; c++) fixed.keys.push_back({(u64)h.first[c], (unsigned)h.iota[c], c, 0});
  for (int y = h.NT; y < (int)trees; y++) fixed.keys.push_back({(u64)y, 0, -1, paddingRow(0)});
  fixed.h = h.cK;
  for (size_t c = 0; c < nc; c++) {
    const size_t of = std::min(c, (size_t)h.NC - 1);
    fixed.tables.push_back({(u64)c, (unsigned)calls[of]->k, (unsigned)of, vector<blk::E>()});
  }
  for (int c = 0; c < h.NC; c++) {
    const u64 T = 1ull << h.dT;
    for (u64 b = 0; b < 1ull << h.iota[c]; b++)
      for (u64 x = 0; x < T;) {
        if (real[c][b * T + x]) {
          x++;
          continue;
        }
        u64 n = 0;
        while (x + n < T && !real[c][b * T + x + n]) n++;
        fixed.padding.push_back({0, (u64)h.first[c] + b + (x << h.rK), (unsigned)h.rK, n, paddingRow(c)});
        x += n;
      }
  }
  std::sort(fixed.padding.begin(), fixed.padding.end(), [](const blk::LkpPadding &a, const blk::LkpPadding &b) { return a.first < b.first; });
  fixed.last_key_level_order = blk::lkp_range_order(h.v() - 1, h.rK);
  if (const char *fault = blk::lkp_fault(fixed)) printf("the test's world is wrong: blocks' look-up says of its public inputs: %s\n", fault), exit(2);
  L.keys.assign((size_t)1 << h.v(), blk::E::zero());
  vector<u64> count(rows, 0);
  for (size_t i = 0; i < L.keys.size(); i++) {
    const size_t y = i % trees, x = i / trees;
    if (y >= (size_t)h.NT) continue;
    const size_t c = (size_t)h.callOf[y], place = ((size_t)h.b((int)y) << h.dT) + x;
    L.keys[i] = eOf(value[c][place]);
    if (!real[c][place]) continue;
    u64 counted = value[c][place];
    if (counted >> calls[c]->k) counted = 0;  // outside its table: row 0, so that the counts add up
    count[c + (counted << h.cK)]++;
  }
  L.bits.assign(rows * 32, blk::E::zero());
  D.bits.assign(rows * 32, 0);
  for (size_t r = 0; r < rows; r++)
    for (size_t j = 0; j < 32; j++)
      if ((count[r] >> j) & 1) L.bits[j + 32 * r] = blk::E::one(), D.bits[j + 32 * r] = 1;
  return L;
}

// An array's entries in the prover's data, row by row.
static vector<blk::E> entriesOf(const World &W, const Data &D, const Decl &d) {
  vector<blk::E> v;
  if (d.name == "U3") return D.tab;
  if (d.name == "ZZ") {
    for (u64 x : D.zz) v.push_back(eOf(x));
    return v;
  }
  if (d.name == "BZ") {
    for (u64 x : D.bits) v.push_back(eOf(x));  // bits[X] = BZ[X / d.K][X mod d.K]
    return v;
  }
  for (size_t i = 0; i < W.tl.pkMat.size(); i++)
    if (d.tree == W.tl.ix.PK[i]) return D.sym[W.tl.pkMat[i]];
  printf("the test's world is wrong: no such array\n"), exit(2);
}
static vector<blk::TableId> tablesOf(const World &W, const Data &D, unsigned moment, blk::Engine &engine) {
  const blk::OracleShape &o = W.w->shape.link[0].messages[moment].shape;
  blk::Table t(o.main_deg, o.main_rows, W.w->codes.base[o.code].ell);
  for (size_t a = 0; a < W.arrays.size(); a++) {
    if (W.arrays[a].moment != moment) continue;
    const vector<blk::E> values = entriesOf(W, D, W.arrays[a]);
    const blk::ReqArray *array = W.layout.find(0, W.arrays[a].name);
    if (!array || values.size() != array->laid.size_i * array->laid.size_c) printf("the test's world is wrong: array %s\n", W.arrays[a].name.c_str()), exit(2);
    for (const blk::LaidBlock &b : array->laid.blocks)
      for (u64 c = 0; c < 1ull << b.bits_c; c++) t.set(b.first_row, b.first_place + c, values[b.first_i * array->laid.size_c + b.first_c + c]);
  }
  return {engine.put(t)};
}

struct Proved {
  vector<blk::u8> proof;
  Data data;
  size_t gaveUp = 0;
};
// Written here, on the types of the library blocks/: the prover's data (cells, symbols,
// functionals' tables, combinations, digits, the lookup's inputs). From that library: the lookup's
// prover (blocks/iop/lkp.h), every commitment with its samples (blk::ReqProver, blocks/iop/req.h),
// the rounds and challenges (blk::ProverRounds, blocks/chain/rounds.h). verify() runs on the same
// library, so a fault that both sides share does not show.
static Proved prove(const World &W, blk::u64 seed, Cheat cheat, i64t wide = 0) {
  Proved out;
  Data &D = out.data;
  TAU = (int)W.s.tau;
  rngState = 1000 + seed;
  cellsOf(W, D, cheat, wide);
  blk::EngineCpu engine;
  blk::TestCoins coins(blk::toy_coins_seed(seed));
  blk::Writer proof;
  blk::Binder binder(1, W.w->plan);
  const blk::RowsPublic bound = blk::rows_toy_bind(binder, *W.w, W.layout, seed);
  blk::ProverRounds ch(binder, blk::toy_random_bytes(seed + 2), proof);
  blk::ProverContext P(bound.codes, ch, engine, coins);
  P.pads = blk::toy_pads;
  blk::ReqProver frame(P, bound.shape);
  frame.first(tablesOf(W, D, 0, engine));
  const blk::OracleId masks = frame.where(0, 0);
  for (size_t j = 0; j < W.tl.smallElems(); j++) D.mask.push_back(P.behind[masks].mask.entry(j));
  const bool late = W.s.readsLate;
  if (!late) frame.later(tablesOf(W, D, 1, engine));
  {
    u64 columns = 0;
    for (const TMat &M : W.mats) columns += M.k;
    auto from = ch.close_bits(blk::R_RNG);
    ch.require_combinations(W.s.tau);
    D.theta = from.trits(columns * W.s.tau);
  }
  out.gaveUp = digitsOf(W, D, cheat);
  Lookup L = lookupOf(W, D, blk::OracleId());
  const vector<blk::OracleId> sent = frame.later(tablesOf(W, D, late ? 1 : 2, engine));
  L.fixed.read_from = {sent.at(0)};
  blk::LkpCpu heavy(L.fixed, L.keys, L.bits);
  Side side{ch, P.st, P.st.masks(masks), heavy, D.mask, {}, {}};
  blk::LkpOutput left;
  // The linear claims that lkp_run() leaves are not used: the chain segment is not closed.
  blk::lkp_run(side, L.fixed, [](blk::Row &, blk::Tensor) {}, frame.group(), left);
  (void)left.keys.take();
  if (cheat == Cheat::READ_UNSEEN) {
    if (!late) printf("the test's world is wrong: this prover needs the matrix before it sends the reads\n"), exit(2);
    const vector<blk::E> v = unseenBy(D.theta, W.mats[0].k, W.s.tau);
    for (u64 c = 0; c < W.mats[0].k; c++) D.tab[tablePlace(W, W.fun[0], c, 0)] += v[c];
  }
  if (late) frame.later(tablesOf(W, D, 2, engine));
  {
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(W.tl.lins(W.nX4) + W.tl.equationsF(W.nX4));
    (void)from.draw();
  }
  out.proof = proof.bytes();
  return out;
}

// ---- the verifier's side ----
static Ex exOf(const blk::E &x) { return fromBlkE(x); }
// The prover's data on which the verifier evaluates its claims: the mask entries and, for each
// virtual row (a combination of an array's rows), its weights `cv`, computed here a second time,
// and `D`[x] = sum_r cv[r] array[r][x]. HFork::judgeAll() (verifier/src/vf_subproof_engine.h) rejects if
// the code under test has other weights.
static std::shared_ptr<const HFork::TestProverValues> pairsOf(const World &W, const Data &D, const HFork &F, const HTailF2 &tl) {
  auto values = std::make_shared<HFork::TestProverValues>();
  values->pairs.resize(F.trees.size());
  for (const blk::E &x : D.mask) values->mask.push_back(exOf(x));
  auto add = [&](const Decl &d, const vector<Ex> &cv) {
    const vector<blk::E> a = entriesOf(W, D, d);
    HFork::TestProverValues::TestProverVirtualRow p;
    p.cv = cv, p.D.assign(d.K, eZ());
    for (u64 r = 0; r < d.R; r++)
      if (!isZ(cv[r]))
        for (u64 x = 0; x < d.K; x++) p.D[x] = p.D[x] + cv[r] * exOf(a[r * d.K + x]);
    values->pairs[(size_t)d.tree].push_back(std::move(p));
  };
  u64 Np = 1;
  while (Np < W.s.tau) Np *= 2;
  for (const Decl &d : W.arrays) {
    if (d.name == "U3")
      add(d, {eO()});
    else if (d.name == "ZZ") {
      for (int q = 0; q < 2; q++)
        for (const HPiece &p : tl.z[1 + q].pc) {
          const vector<Ex> e = eqTab(tl.z[1 + q].c.zk.data() + (tl.z[1 + q].c.n - p.b), p.a);
          vector<Ex> cv(d.R, eZ());
          for (u64 r = 0; r < p.Rp; r++) cv[p.r0 + r] = e[r];
          add(d, cv);
        }
      vector<Ex> cv(d.R, eO());
      for (u64 u = 1; u < d.R; u++) cv[u] = cv[u - 1] * F.theta;
      add(d, cv);  // one pair for the reads' and the ties' rows
    } else if (d.name == "BZ") {
      for (int e = 0; e < 2; e++) add(d, eqTab(tl.f9.zb[e].data(), 5));
      add(d, vector<Ex>(d.R, eO()));
    } else {
      vector<Ex> cv(d.R, eO());
      const Ex step = ePow(F.theta, Np);
      for (u64 i = 1; i < d.R; i++) cv[i] = cv[i - 1] * step;
      add(d, cv);
    }
  }
  return values;
}
/* Computes a second time the weights that the ties and the reads put on ZZ, U3 and the symbol
   arrays. Returns "" if they agree with the code's, otherwise where they differ. Evaluating a
   claim by itself cannot show which power of theta (`F.theta`) multiplies it; this comparison
   does.
   In cp_verify `tileSelfTestF` recomputes the ties' weights and the reads' weights on ZZ, not the
   reads' weights on the tables.
   cp_prove numbers the equations in separate code (prover/hiding/tail_f.cuh,
   prover/hiding/walk_calls.cuh), so an error in only one of the two programs makes cp_verify
   reject honest proofs. Seen once: with all reads numbered like the first, cp_verify rejected a
   stored honest proof of a small example configuration at the linear claim of BASE, the proximity
   test's last step.
   The numbering, the same as in `rng_row` (blocks/iop/rng.h) and `powersF`
   (verifier/src/vf_smallness_test_members.inc): let Np be tau rounded up to a power of two, P the planned number of
   claims, and first[mu] the sum over the earlier matrices of Np 2^ceil(log2(symbol rows)). The
   equation of matrix mu, symbol row s and combination u is multiplied by
   theta^(P + first[mu] + s Np + u), that of the q-th read and combination u by
   theta^(P + first[NM] + q Np + u). */
static string weightsHeld(const World &W, const Data &D, const HFork &F, const HTailF2 &tl) {
  const u32 tau = W.s.tau;
  u64 Np = 1;
  while (Np < tau) Np *= 2;
  const u64 P = F.planClaimsF, mz = tl.mz;
  vector<u64> first(W.mats.size() + 1, 0);
  for (size_t mu = 0; mu < W.mats.size(); mu++) first[mu + 1] = first[mu] + (Np << clog2((W.mats[mu].m + 2) / 3));
  vector<vector<Ex>> gam(W.mats.size());
  u64 column0 = 0;
  for (size_t mu = 0; mu < W.mats.size(); column0 += W.mats[mu].k, mu++)
    for (u64 c = 0; c < W.mats[mu].k; c++) {
      Ex s = eZ(), power = eO();
      for (u32 u = 0; u < tau; u++, power = power * F.theta) {
        const int t = D.theta[(size_t)(column0 + c) * tau + u];
        if (t) s = t > 0 ? s + power : s - power;
      }
      gam[mu].push_back(s);
    }
  const HTree &ZZ = F.trees[(size_t)tl.ix.ZZ], &U3 = F.trees[(size_t)tl.ix.U[2]];
  const vector<Ex> &ties = ZZ.rows[HTailF2::zzRowTies()].L, &reads = ZZ.rows[HTailF2::zzRowReads()].L;
  vector<Ex> wantReads(2 * mz, eZ()), wantU(W.nU[2], eZ());
  for (size_t mu = 0; mu < W.mats.size(); mu++) {
    const Ex lead = ePow(F.theta, P + first[mu]);
    for (u64 i = 0; i < W.mats[mu].m; i++) {
      const Ex w = eS(lead * ePow(F.theta, Np * (i / 3)), i % 3 == 0 ? 1 : i % 3 == 1 ? BB : 1) * (i % 3 == 2 ? eB(BB) * eB(BB) : eO());
      if (!(ties[tl.off[mu] + i] == w) || !(ties[mz + tl.off[mu] + i] == eS(w, 1024))) return "the ties' weight on the digits of matrix " + std::to_string(mu) + ", row " + std::to_string(i);
    }
    const vector<Ex> &L = F.trees[(size_t)tl.gtTree[mu]].rows[0].L;
    for (u64 c = 0; c < W.mats[mu].k; c++)
      if (!(L[c] == eZ() - lead * gam[mu][c])) return "the tie's weight on the symbols of matrix " + std::to_string(mu) + ", column " + std::to_string(c);
  }
  u64 q = 0;
  for (size_t mu = 0; mu < W.mats.size(); mu++)
    for (const HFun &f : W.fun) {
      if ((size_t)f.M != mu) continue;
      for (u64 sg = 0; sg < W.mats[mu].m / f.L; sg++, q++) {
        const Ex lead = ePow(F.theta, P + first[W.mats.size()] + q * Np);
        for (u64 i = 0; i < f.L; i++) {
          Ex &lo = wantReads[tl.off[mu] + sg * f.L + i], &hi = wantReads[mz + tl.off[mu] + sg * f.L + i];
          lo = lo + lead * f.phi[i], hi = hi + eS(lead * f.phi[i], 1024);
        }
        for (u64 c = 0; c < W.mats[mu].k; c++) wantU[f.at + c * (W.mats[mu].m / f.L) + sg] = eZ() - lead * gam[mu][c];
      }
    }
  for (u64 x = 0; x < 2 * mz; x++)
    if (!(reads[x] == wantReads[x])) return "the reads' weight on the digits' column " + std::to_string(x);
  for (u64 x = 0; x < W.nU[2]; x++)
    if (!(U3.rows[0].L[x] == wantU[x])) return "the reads' weight on table entry " + std::to_string(x);
  return "";
}
struct Verdict {
  string refused;          // "" or the rejection name
  vector<string> falseIn;  // the sections of the false claims
  bool equation = false;   // the one equation's value on the data is 0
  string apart;            // the result of `weightsHeld`
  u32 claims = 0;
  bool accepted() const { return refused.empty() && falseIn.empty() && equation && apart.empty(); }
  string words() const {
    if (!refused.empty()) return "refused: " + refused;
    const string second = apart.empty() ? "" : "; THE SECOND WRITING DIFFERS: " + apart;
    if (falseIn.empty()) return (equation ? "all claims hold" : "NO CLAIM IS FALSE, YET THE EQUATION FAILS") + second;
    string s = std::to_string(falseIn.size()) + " false:";
    for (size_t i = 0; i < falseIn.size() && i < 4; i++) s += " [" + falseIn[i] + "]";
    return s + (falseIn.size() > 4 ? " .." : "") + second;
  }
};
/* Calls the code under test in the order in which the three types of sub-proof call it
   (verifier/src/vf_layers_check.inc, verifier/src/vf_output_head_check.inc, verifier/src/vf_attention_check.h).
   As in cp_verify, the sub-proof is one chain segment (`ReqLink`, blocks/iop/req.h), here at toy
   lengths, with the programs' three commit points: symbol arrays; tables; digits and bit planes.
   Every round must match a plan whose rounds for the range check come from the verifier's own
   `planRounds`.
   Under `Shape::readsLate` the tables come last in the layout, in the plan and in the calls here.
   Unlike cp_verify, this function does not close the chain segment: `doorH`
   (verifier/src/vf_subproof_segment.h: it passes the one equation to the chain), the closing and the
   proximity tests do not run (verifier/tests/link_door_test.cpp runs them on a synthetic
   equation). Instead the functions that add claims (`HFork::lin`, `t2Lin`) record each claim,
   and `judgeAll` evaluates it on the prover's data. */
static Verdict verify(const World &W, blk::u64 seed, const Proved &p) {
  Verdict out;
  TAU = (int)W.s.tau;
  blk::Binder binder(1, W.w->plan);
  const blk::RowsPublic bound = blk::rows_toy_bind(binder, *W.w, W.layout, seed);
  blk::Reader reader(p.proof.data(), p.proof.size());
  blk::VerifierSession session;
  blk::VerifierRounds channel(binder, reader, session);
  blk::VerifierContext V(bound.codes, channel);
  mb::ChainCheck check(V, mb::ChainShape{bound.shape, bound.layout});
  refuseThrows = true;
  try {
    if (!check.first()) return out.refused = "the link's first message", out;
    LinkCtx C;
    C.check = &check, C.V = &V, C.link = 0, C.arrayOf = &W.arrayOf, C.layout = &channel.get(bound.layout);
    currentSegment = &C;
    HTailF2 tl;
    HFork F;
    Tail2 T;
    F.G_ = "toy:";
    const bool hasU[4] = {false, false, true, false};
    tl.shape(W.mats, vector<int>(W.mats.size(), -1), {}, hasU, W.s.tau);
    tl.addTrees(F, T, W.nU, {}, 0);
    t2Plan(F, T);
    tl.runT = &T;
    F.slots((u32)tl.smallElems());
    Tr tr{V.ch};
    static const u32 received[DW] = {0};
    for (int t : tl.ix.PK) F.rootTaken((size_t)t, received);  // `check.first` has received these
    if (!W.s.readsLate) rootInH(F, tr, {tl.ix.U[2]});
    tl.closeFuns(W.fun);
    u32 slot = 0;
    tl.transcript(F, tr, slot);
    if (W.s.readsLate) rootInH(F, tr, {tl.ix.U[2]});
    if (slot != F.ns) refuse("toy:internal-slot-count");
    F.planClaimsF = (u32)tl.lins(W.nX4);
    t2Batch(F, T, tr, slot);
    F.valuesF = pairsOf(W, p.data, F, tl);
    tl.objectsF(F, W.fun);
    for (u32 j = 0; j < F.ns; j++)
      if (isZ(F.LS[j])) refuse("toy:internal-free-slot");
    tl.funsHeld(F, W.fun);
    out.apart = weightsHeld(W, p.data, F, tl);
    tl.selfTest(F, W.nU);
    tl.coefficients(F);
    tl.rkSelfTest(F);
    trBatchRows(tr, (u64)F.nClaim + tl.equationsLaidF());  // as `doorH` does
    if (F.nClaim != F.planClaimsF) refuse("toy:internal-f-draws");
    F.kapAllTakenF();
    tl.foldAllTakenF(F);
    tl.symShapesHeldF(F, T);
    F.judgeAll();
    for (const HFork::TestClaimFoundFalse &x : F.falseF) out.falseIn.push_back(x.section);
    out.equation = isZ(F.valuesSumF), out.claims = F.nClaim;
    currentSegment = nullptr;
  } catch (Refusal &r) {
    currentSegment = nullptr;
    out.refused = r.name;
  }
  return out;
}

static string firstBad;
static void ask(bool ok, const string &what, const Verdict &v) {
  printf("%s %s: %s\n", ok ? "as expected:" : "NOT AS EXPECTED:", what.c_str(), v.words().c_str());
  if (!ok && !bad++) firstBad = what + ": " + v.words();
}
static bool onlyIn(const Verdict &v, const string &prefix) {
  if (!v.refused.empty() || v.falseIn.empty() || !v.apart.empty()) return false;
  for (const string &s : v.falseIn)
    if (s.rfind(prefix, 0)) return false;
  return true;
}

/* Runs one prover (`which`: honest, wide, halves) at n seeds, at tau = 8, so that the
   draws that let a malicious prover through occur and can be counted. Required at every seed:
   accepted exactly if no combination left the window, or rejected as
   `toy:internal-f-table-read`. In a run without arguments no malicious prover happens to be
   accepted, so that run shows only that they fail; the counts below are from long runs.
   halves: a combination is an integer with probability 1/2, so the check's error of 2^-tau is
   reached: 15 of 4,096 seeds were accepted (2^-8 gives 16).
   wide: a cell of 2^18 passes only if all tau entries of its column are 0. Then the column's
   weight is 0 and `objectsF` rejects as above, which in cp_verify gives `NOT CHECKED`, not `YES`:
   0 of 4,096 seeds were accepted, and all 13 such draws were rejected in this way. So is an honest
   prover with probability (columns) x 2^-tau: negligible at 128, the example configurations' tau.
   Not run here: a cell of 2^17 + 5000 whose combinations above the window are sent with their true
   digits: ties and reads hold, and only the high digit, 256 to 511, is outside its lookup table. It
   must never be accepted; at a seed with no combination below the window, only the lookup on the
   high digit catches it. Negative control: with 9 bits for that lookup table and for the code's
   self-test, this prover is accepted at such seeds. This prover needs a form of the proof library's
   sumcheck prover that goes on past a false claim, which is not part of this repository; it and the
   negative control ran elsewhere, so no test in this repository shows that this table must have
   8 bits. */
static void tries(const World &W, const string &which, blk::u64 from, blk::u64 n) {
  std::map<string, size_t> ended;
  string lucky;
  for (blk::u64 seed = from; seed < from + n; seed++) {
    const Proved p = which == "halves" ? prove(W, seed, Cheat::HALVES) : which == "honest" ? prove(W, seed, Cheat::NONE) : prove(W, seed, Cheat::CELL_WIDE, 1ll << 18);
    const Verdict v = verify(W, seed, p);
    const string how = !v.refused.empty() ? "refused: " + v.refused : v.accepted() ? "ACCEPTED" : "claims false";
    ended[how + (p.gaveUp ? "" : " (every combination stayed in the window)")]++;
    if (v.accepted()) lucky += " " + std::to_string(seed);
    if (v.refused.empty() ? v.accepted() != (p.gaveUp == 0) : v.refused != "toy:internal-f-table-read")
      ask(false, which + ", seed " + std::to_string(seed) + ", " + std::to_string(p.gaveUp) + " combinations left the window", v);
    if (which == "wide" && v.accepted()) ask(false, string("A CELL OF 2^18") + " IS ACCEPTED, seed " + std::to_string(seed), v);
  }
  for (const auto &x : ended) printf("%s: %zu of %llu: %s\n", which.c_str(), x.second, (unsigned long long)n, x.first.c_str());
  printf("%s: accepted at the seeds:%s\n", which.c_str(), lucky.empty() ? " none" : lucky.c_str());
}

int main(int argc, char **argv) {
  printf("0 - 1 = 0x%llx in the verifier's field, 0x%llx in blocks/\n", (unsigned long long)eC(eZ() - eO(), 0), (unsigned long long)(blk::E::zero() - blk::E::one()).c[0].v);
  World W8;
  makeWorld(W8, Shape{8, {{96, 8}, {40, 4}}, {{0, 32}, {0, 96}, {1, 40}}}, 14);
  if (argc == 5 && !strcmp(argv[1], "tries")) {  // tries <prover> <first seed> <seeds>: see `tries`
    tries(W8, argv[2], strtoull(argv[3], nullptr, 10), strtoull(argv[4], nullptr, 10));
    return bad ? 1 : 0;
  }
  World W;
  makeWorld(W, Shape{24, {{96, 8}, {40, 4}}, {{0, 32}, {0, 96}, {1, 40}}}, 14);
  printf("# the world: tau %u; %zu matrices, %llu rows in all; %zu reads; %zu slots; %zu claims, %llu equations behind them; the messages' rows %zu, %zu, %zu\n", W.s.tau, W.mats.size(),
         (unsigned long long)W.tl.mz, W.nX4, W.tl.smallElems(), W.tl.lins(W.nX4), (unsigned long long)W.tl.equationsF(W.nX4), W.rows[0], W.rows[1], W.rows[2]);
  {
    const Proved p = prove(W, 900, Cheat::NONE);
    const Verdict v = verify(W, 900, p);
    ask(v.accepted() && v.claims == W.tl.lins(W.nX4), "the honest prover", v);
  }
  {
    const Verdict v = verify(W, 901, prove(W, 901, Cheat::CELL_200));
    ask(v.accepted(), "a cell of 200: LAWFUL", v);
  }
  {
    const Proved p = prove(W, 902, Cheat::CELL_WIDE, 1ll << 18);
    const Verdict v = verify(W, 902, p);
    ask(p.gaveUp > 0 && v.apart.empty() && v.falseIn == vector<string>{"smallness test: tie of matrix 0", "smallness test: read read0, segment 0", "smallness test: read read1, segment 0"},
        "a cell of 2^18, digits of 0 where the window is left", v);
  }
  {
    const Proved p = prove(W, 910, Cheat::CELL_WIDE, 100000);
    const Verdict v = verify(W, 910, p);
    ask(p.gaveUp == 0 && v.accepted(), "a cell of 100,000: LAWFUL (no combination of its row can leave the window)", v);
  }
  {
    const Verdict v = verify(W, 903, prove(W, 903, Cheat::READ_FALSE));
    ask(v.falseIn == vector<string>{"smallness test: read read0, segment 0"} && v.apart.empty(), "good packing, one table entry + 1", v);
  }
  {
    const Verdict v = verify(W, 905, prove(W, 905, Cheat::COMBINATION_OUT));
    ask(v.falseIn.size() == W.mats.size() + W.nX4 && onlyIn(v, "smallness test: "), "a combination not made: all ties and reads false", v);
  }
  {
    const Verdict v = verify(W, 906, prove(W, 906, Cheat::ROWS_EXCHANGED));
    ask(v.falseIn == vector<string>{"smallness test: tie of matrix 0"} && v.apart.empty(), "two symbol rows exchanged", v);
  }
  // Not run here (it needs the same form of the library's prover): ZZ as under OTHER_KEYS (d0 +
  // 1024, d1 - 1: the same number, d0 outside R10z's table) with the look-up run on ZZ's own
  // digits: then ties and reads are true and the look-up alone is false. No test in this repository
  // tries it.
  {
    const Verdict v = verify(W, 907, prove(W, 907, Cheat::OTHER_KEYS));
    ask(v.falseIn == vector<string>{string("smallness test, ") + FOREST_JOIN_SECTION} && v.apart.empty(), "digits outside their tables, the look-up run on other keys: the join alone is false", v);
  }
  // Two reads whose errors are each other's negatives. Both claims are false whatever powers of
  // theta multiply them; the chain would see only the one equation, which must fail because the
  // powers differ. Two ties cannot be tried in this way: a tie's error depends on the verifier's
  // matrix, which is drawn after the cells are committed. weightsHeld() checks their powers.
  {
    const Verdict v = verify(W, 908, prove(W, 908, Cheat::READS_CANCEL));
    ask(v.falseIn.size() == 2 && onlyIn(v, "smallness test: read ") && !v.equation, "two reads' errors that would cancel under one power: the equation fails", v);
  }
  // The wrong order. The check's error of 2^-tau holds only if the cells and the tables (array U3,
  // oracle message "reads") are committed before the verifier's matrix is drawn. In this fixture
  // (tau = 8; the first matrix has 12 columns) the layout, the plan, prove() and verify() all put
  // the tables after the check's other messages: cells, the matrix, digits, the lookup, tables, the
  // batching challenge. The plan, the channel (blk::VerifierRounds) and rootInH() accept this
  // order. READ_UNSEEN then adds to a table a vector that the drawn matrix maps to 0, which exists
  // whenever a matrix has more columns than there are combinations: every claim holds, and the
  // table is false. This prover and the honest one must both be rejected before the draw, by
  // fixedBeforeTheMatrixF() (verifier/src/vf_smallness_test_members.inc), as `toy:internal-f-matrix-order`.
  // Negative control, in a test that is not part of this repository: with the call of that function
  // removed, both provers are accepted.
  // Not shown: the wrong order in cp_verify and cp_prove themselves; the function's branch for an
  // array of an earlier chain segment, which only honest runs reach (this fixture has one segment).
  {
    World L8;
    makeWorld(L8, Shape{8, {{96, 12}, {40, 4}}, {{0, 32}, {0, 96}, {1, 40}}, true}, 14);
    const Verdict unseen = verify(L8, 921, prove(L8, 921, Cheat::READ_UNSEEN));
    ask(unseen.refused == "toy:internal-f-matrix-order", "reads late, a false read", unseen);
    const Verdict honest = verify(L8, 920, prove(L8, 920, Cheat::NONE));
    ask(honest.refused == "toy:internal-f-matrix-order", "reads late, honest", honest);
  }
  for (const char *which : {"honest", "wide", "halves"}) tries(W8, which, 0, 12);
  if (bad)
    printf("tail_twin_test: FAILED: %d not as expected; the first: %s\n", bad, firstBad.c_str());
  else
    printf("tail_twin_test: ok (9 provers at tau 24, each judged claim by claim; 2 with the reads sent behind the matrix; 36 tries at tau 8)\n");
  return bad ? 1 : 0;
}
