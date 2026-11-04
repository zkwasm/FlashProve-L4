// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests how a sub-proof's claim on the committed weight matrix W becomes coefficients ("weights" in
// the messages) on W's rows and columns. A sub-proof (`Fork` in identifiers) is the argument for
// one part of the forward pass. The verifier's WsLay::build(), wsRowPart(), wsGammaAdd() and
// wsEpsT() (verifier/src/vf_weights_layout.h) are compared with the prover's and committer's layoutW(),
// the prover's gammaW() (prover/common/weight_layout.h) and place(), a third implementation of the
// layout rule, which gives each cell its row and column. An error in the code under test would make
// the verifier check a claim other than the one that the sub-proof left; an error on one side only
// would make honest proofs fail. The test shows only that the three agree on six small synthetic
// shapes with at most 2^12 columns; place() itself follows the comments in those two headers. It
// runs no transcript: the merge itself (wsForkEnd(), wsForkMerge()) is not tested here.
// Build, from verifier/src: g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8
//   -DBLK_FIELD=2 -DBLK_HASH=1 -I . -I ../../blocks -I ../../prover ../tests/weights_twin_test.cpp
//   -o <a folder of your own>/weights_twin_test
#define VF_WITHOUT_LOOKUPS 1
#include "vf_subproof_engine.h"
#include "vf_weights_layout.h"

namespace theirs {  // the prover's and committer's header, compiled with the verifier's field
using std::vector;
using E = Ex;
static const E E0 = eZ(), E1 = eO();
static inline E ebase(u64 x) { return eB(x); }
static inline E esub(const E &a, const E &b) { return a - b; }
static inline E emul(const E &a, const E &b) { return a * b; }
constexpr int SW = 8;  // of MergeW, which nothing here uses
#include "common/weight_layout.h"
}  // namespace theirs

// Synthetic (`madeUp`) values, from a generator with a fixed seed.
static u64 rngState = 11;
static u64 madeUp64() {
  rngState = rngState * 6364136223846793005ull + 1442695040888963407ull;
  return rngState >> 11;
}
static Ex madeUp() { return eB(madeUp64() % 1000003) + eS(eB(madeUp64() % 1000003) * eB(madeUp64() % 1000003), 7); }  // in the base field; the functions under test are linear in each factor
static vector<Ex> madeUpList(size_t n) {
  vector<Ex> v(n);
  for (Ex &x : v) x = madeUp();
  return v;
}
static int failures = 0;
static size_t checks = 0;
// the current test case, for the messages
static string world;
static void expect(bool ok, const string &what) {
  checks++;
  if (!ok && failures++ < 40) printf("FAIL: %s: %s\n", world.c_str(), what.c_str());
}

// A claim says: the numbers of unit u, each times a product of factors, one for each of u's index
// groups (WsGrp), sum to v. A unit is the rows of W for one kind of weight matrix of one layer, for
// the embedding or for the head. W holds a number as base-256 digits (planes), one to a cell, so a
// cell's coefficient is the product of the factors at its indices times 256^plane. The code under
// test splits it into eps~(column) * gamma[row], eps~ being multilinear in the lk column bits (W
// has 2^lk columns); the merge of the claims then binds the columns at a point rm, which leaves
// eps~(rm) * <gamma, rows>.

// The value of a claim's factor f on group g at one index of g, as WsFac and FacW define a factor.
static Ex factorAt(const WsGrp &g, const WsFac &f, u64 index) {
  if (g.cls == 'A') return f.AV[index];
  Ex r = f.HV[index >> g.z];
  for (int b = 0; b < g.z; b++) r = r * (((index >> b) & 1) ? f.pt[g.z - 1 - b] : eO() - f.pt[g.z - 1 - b]);
  return r;
}
// Splits an index of group g by the layout rule. Class E (g.R = o * 2^z, o odd): of the index's z
// low bits, the lowest t (the highest t if g.high) become column bits cb0 to cb0 + t - 1, lowest
// first, and the index's other bits, packed in order, go to the row. Class A: the entire index goes
// to the columns if t > 0, else to the row. The rule is taken from the comments in the two headers
// under test, so the test does not catch an error that both headers share in code and comments.
// g's z, t and cb0 come from WsLay::build() and are checked only against layoutW()'s.
// For the rest of the two headers under test and for every other file of the programs, whether the
// test is independent of it has not been examined.
static void place(const WsGrp &g, u64 index, u64 &columnBits, u64 &rowPart) {
  if (g.cls == 'A') {
    const bool taken = g.t > 0;
    columnBits = taken ? index << g.cb0 : 0, rowPart = taken ? 0 : index;
    return;
  }
  const u64 lo = index & ((1ull << g.z) - 1), hi = index >> g.z;
  const u64 taken = g.high ? lo >> (g.z - g.t) : lo & ((1ull << g.t) - 1), rest = g.high ? lo & ((1ull << (g.z - g.t)) - 1) : lo >> g.t;
  columnBits = taken << g.cb0, rowPart = (hi << (g.z - g.t)) + rest;
}

// A test shape: WsShape's model dimensions. moe: the odd layers have experts. eh: W has the
// embedding and the head.
struct Shape {
  const char *name;
  u64 L, d, dh, dkv, dq, f, fe, NE, V;
  bool moe, eh;
};

int main() {
  // p - 1, to show the field: 0xffffffff00000000 for p = 2^64 - 2^32 + 1.
  printf("0 - 1 = 0x%llx in the verifier's field\n", (unsigned long long)fsub((u64)0, (u64)1));
  // makes refuse() (verifier/src/vf_common.h) throw a Refusal instead of exiting
  refuseThrows = true;
  const vector<Shape> shapes = {
      {"dense layers, sizes with odd parts", 2, 24, 4, 8, 16, 40, 0, 0, 10, false, true},
      {"expert layers, sizes with odd parts", 2, 24, 4, 8, 16, 40, 12, 6, 10, true, true},
      {"powers of two alone", 2, 16, 2, 4, 8, 32, 8, 4, 16, true, true},
      {"odd sizes alone: no bit to give", 1, 3, 1, 5, 7, 9, 0, 0, 3, false, true},
      {"no embedding, no head", 3, 8, 2, 4, 8, 12, 4, 3, 0, true, false},
      {"a size of 1 in every place that takes one", 2, 1, 1, 1, 1, 1, 1, 1, 1, true, true},
  };
  size_t units = 0, cells = 0, empties = 0, worlds = 0;
  for (const Shape &s : shapes)
    for (int lk : {1, 2, 3, 5, 8, 12}) {
      world = string(s.name) + ", " + std::to_string(lk) + " column bits";
      worlds++;
      WsShape mine;
      mine.D = 32, mine.L = s.L, mine.d = s.d, mine.dh = s.dh, mine.dkv = s.dkv, mine.dq = s.dq, mine.f = s.f, mine.fe = s.fe, mine.NE = s.NE, mine.V = s.V, mine.eh = s.eh;
      wsSetMoe(mine, s.moe);
      WsLay lay;
      try {
        lay.build(mine, (u32)lk);
      } catch (const Refusal &r) {
        expect(false, "the verifier's layout refuses the shape: " + r.name);
        continue;
      }
      theirs::ShapeW other;
      other.moeMode = s.moe, other.L = (int)s.L, other.d = s.d, other.dh = s.dh, other.hdR = 64, other.dkv = s.dkv, other.dq = s.dq, other.f = s.f, other.fe = s.fe, other.NE = s.NE, other.V = s.V;
      const theirs::LayW Y = theirs::layoutW(other, lk, 32, 0, s.eh);
      expect(lay.U.size() == Y.U.size() && lay.m0 == Y.m0 && lay.lk == Y.lk, "the numbers of units, of rows or of column bits");
      if (lay.U.size() != Y.U.size()) continue;
      for (size_t ui = 0; ui < lay.U.size(); ui++) {
        const WsUnit &u = lay.U[ui];
        const theirs::UnitW &w = Y.U[ui];
        const string at = "unit " + std::to_string(ui) + " (kind " + std::to_string(u.X) + ", layer " + std::to_string(u.l) + ")";
        units++;
        bool same = (int)u.X == w.X && (int)u.l == w.l && u.nlw == w.nlw && u.ncb == w.ncb && u.rows == w.rows && u.off == w.off && u.g.size() == w.g.size();
        for (size_t gi = 0; same && gi < u.g.size(); gi++) {
          const WsGrp &a = u.g[gi];
          const theirs::GrpW &b = w.g[gi];
          same = a.R == b.R && a.t == b.take && a.cb0 == b.cb0 && a.left == b.left && (a.cls == 'A') == (b.cls == 'A') && (a.cls == 'A' || (a.z == b.z && a.high == (b.cls == 'P')));
        }
        expect(same, at + ": the two layouts");
        if (!same) continue;
        // A claim with synthetic factors, in both implementations' types.
        WsClaim c;
        theirs::ClaimW k;
        c.ui = (u32)ui, k.ui = (int)ui;
        for (const WsGrp &g : u.g) {
          WsFac f;
          if (g.cls == 'A') f.AV = madeUpList(g.R);
          else f.HV = madeUpList(g.o), f.pt = madeUpList(g.z);
          theirs::FacW t;
          t.HV = f.HV, t.pt = f.pt, t.AV = f.AV;
          c.f.push_back(f), k.f.push_back(t);
        }
        try {
          wsClaimShape(lay, c);
        } catch (const Refusal &r) {
          expect(false, at + ": the claim's shape is refused: " + r.name);
          continue;
        }
        // gamma: the verifier's coefficients on W's rows must be ka * weight times the prover's,
        // and 0 outside the unit.
        const Ex ka = madeUp(), weight = madeUp();
        vector<Ex> Gam(lay.m0, eZ());
        wsGammaAdd(Gam, lay, wsRowPart(lay, c, ka), weight);
        const vector<Ex> gamma = theirs::gammaW(w, k);
        bool rows = gamma.size() == u.rows, outside = true;
        for (u64 y = 0; y < lay.m0; y++) {
          if (y >= u.off && y < u.off + u.rows) rows = rows && Gam[y] == ka * weight * gamma[y - u.off];
          else outside = outside && isZ(Gam[y]);
        }
        expect(rows, at + ": the weights on the unit's rows, the verifier's against the prover's");
        expect(outside, at + ": a weight outside the unit's rows");
        if (!rows) continue;
        // eps[X] is eps~ at column X: wsEpsT() at the point of the cube {0,1}^lk that has X's bits,
        // the top bit first.
        vector<Ex> eps(1ull << lk);
        for (u64 X = 0; X < eps.size(); X++) {
          vector<Ex> rm(lk);
          for (int i = 0; i < lk; i++) rm[i] = eB((X >> (lk - 1 - i)) & 1);
          eps[X] = wsEpsT(lay, c, rm);
        }
        // At each cell eps * gamma must be its coefficient; at a (row, column) with no cell, 0.
        vector<char> held(u.rows << lk, 0);
        vector<u64> index(u.g.size(), 0);
        bool every = true, once = true;
        for (bool more = true; more;) {
          Ex product = eO();
          u64 column = 0, row = 0;
          for (size_t gi = u.g.size(); gi-- > 0;) {  // row digits, most significant first: the plane, then the groups from last to first
            u64 bits, part;
            place(u.g[gi], index[gi], bits, part);
            column |= bits, row = row * u.g[gi].left + part;
            product = product * factorAt(u.g[gi], c.f[gi], index[gi]);
          }
          const u64 rowsOfAPlane = u.rows / u.nlw;
          Ex power = eO();
          for (int plane = 0; plane < u.nlw; plane++, power = eS(power, 256)) {
            const u64 y = plane * rowsOfAPlane + row;
            once = once && y < u.rows && column < eps.size() && !held[(y << lk) + column];
            if (!once) break;
            held[(y << lk) + column] = 1;
            every = every && eps[column] * gamma[y] == product * power;
            cells++;
          }
          more = false;
          for (size_t gi = 0; gi < u.g.size() && !more; gi++) {
            if (++index[gi] < u.g[gi].R) more = true;
            else index[gi] = 0;
          }
          if (!once) break;
        }
        expect(once, at + ": two cells in one place, or a cell outside the unit's rows and columns, by this test's reading of the rule");
        expect(every, at + ": eps~(column) x gamma[row] is the product of the factors at the cell's indices");
        bool nothing = true;
        for (u64 y = 0; once && y < u.rows; y++)
          for (u64 X = 0; X < eps.size(); X++)
            if (!held[(y << lk) + X]) nothing = nothing && isZ(eps[X] * gamma[y]), empties++;
        expect(nothing, at + ": a place that holds no cell has a weight");
        // At a synthetic point rm, wsEpsT() must equal the multilinear extension of eps.
        const vector<Ex> rm = madeUpList(lk);
        const vector<Ex> eq = eqTab(rm.data(), lk);  // coordinate 0 belongs to the top bit: column bit lk - 1
        Ex extended = eZ();
        for (u64 X = 0; X < eps.size(); X++) extended = extended + eq[X] * eps[X];
        expect(extended == wsEpsT(lay, c, rm), at + ": wsEpsT at a point is the multilinear extension of its values on the cube");
      }
    }
  if (failures) return printf("weights_twin_test: FAILED (%d of %zu checks)\n", failures, checks), 1;
  printf("weights_twin_test: ok (%zu worlds, %zu units, %zu cells, %zu places without a cell; %zu checks)\n", worlds, units, cells, empties, checks);
  return 0;
}
