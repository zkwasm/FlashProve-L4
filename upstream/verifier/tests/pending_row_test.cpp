// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests `wsPending` (verifier/src/vf_weights_pending_claim.h) on synthetic records and a toy weight layout.
   `wsPending` turns the record of a sub-proof (`Fork` in names: the proof for one part of the
   forward pass), the list of its claims on the weight matrix W, into one pending claim (printed:
   "the row"): a linear claim on W left for LIN, the test on the committed weight table.

   The claim's linear form is computed again, entry by entry ("the text's sum"):
       fv = sum over the record's claims a of ka_a * sum over the rows y of a's unit and all
            columns col of gamma_a[y] * eq(rm, col) * W[off + y][col].
   W has m_0 rows and 2^lk columns; a unit is a run of rows, from row off. gamma_a[y]: see
   `gammaAt`. The first coordinate of the point rm belongs to the highest bit of col.

   Checked for each record: the result's sum equals fv; the row weights are 0 outside the claims'
   units, also from row m_0 on; with an arbitrary mask entry c (the prover's random element) and
   fv' = fv + beta c, the sum + beta c equals the claim's value (`target`) and the sum - beta c
   does not; a claim count one too high, a point one coordinate short, an unknown unit and a LIN
   shape whose `matrix_rows` is one off m_0 are rejected (`refusalOf`) under their names.

   Build, from verifier/src: g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8
   -DBLK_FIELD=2 -DBLK_HASH=1 -I . -I ../../blocks -I ../../prover ../tests/pending_row_test.cpp */
#define VF_SOURCE_DIGEST "0000000000000000000000000000000000000000000000000000000000000000"
#include "vf_smallness_matrices.h"
#include "vf_attention_shape.h"
#include "vf_subproof_engine.h"
#include "vf_smallness_test.h"
#include "vf_norm_digit_rows.h"
#include "vf_weights_pending_claim.h"

static int bad = 0;
// Test data: xorshift64: G. Marsaglia, "Xorshift RNGs", Journal of Statistical Software
// 8 (14), 2003.
static u64 state = 88172645463325252ull;
static u64 next64() {
  state ^= state << 13, state ^= state >> 7, state ^= state << 17;
  return state;
}
static Ex made() { return Ex(next64() >> 4, next64() >> 4, next64() >> 4); }
static vector<Ex> made(size_t n) {
  vector<Ex> v(n);
  for (Ex &x : v) x = made();
  return v;
}
static WsLay toyLayout() {
  WsShape s;
  s.D = 32, s.L = 4, s.d = 64, s.dh = 16, s.dkv = 2 * 16, s.dq = 4 * 16, s.f = 128, s.fe = 64, s.NE = 8;
  s.V = 200, s.eh = 1;
  wsSetMoe(s, true);
  WsLay lay;
  lay.build(s, 8);
  return lay;
}
static WsRecClaim madeClaim(const WsLay &lay, u32 ui) {
  WsRecClaim c;
  c.ui = ui, c.ka = made();
  for (const WsGrp &g : lay.U[ui].g) {
    WsRowFac x;
    if (g.cls == 'E')
      x.HV = made(g.o), x.pl = made((size_t)(g.z - g.t));
    else if (!g.aTaken())
      x.AV = made(g.R);
    c.f.push_back(x);
  }
  return c;
}
// gamma_a[y]: the product of claim c's row factors (`WsRowFac`, verifier/src/vf_weights_layout.h) at the
// mixed-radix digits of y. The first group has the least significant digit; the most significant
// is the limb l (a weight's base-256 digit), factor 256^l.
static Ex gammaAt(const WsLay &lay, const WsRecClaim &c, u64 y) {
  const WsUnit &u = lay.U[c.ui];
  Ex w = eO();
  for (size_t gi = 0; gi < u.g.size(); gi++) {
    const WsGrp &g = u.g[gi];
    const u64 digit = y % g.left;
    y /= g.left;
    if (g.cls == 'E') {
      const int zl = g.z - g.t;
      const u64 hi = digit >> zl, lo = digit & ((1ull << zl) - 1);
      w = w * c.f[gi].HV[hi];
      for (int i = 0; i < zl; i++) w = w * (((lo >> (zl - 1 - i)) & 1) ? c.f[gi].pl[(size_t)i] : eO() - c.f[gi].pl[(size_t)i]);
    } else if (!g.aTaken())
      w = w * c.f[gi].AV[digit];
  }
  for (u64 l = 0; l < y; l++) w = eS(w, 256);
  return w;
}
static string refusalOf(const std::function<void()> &f) {
  refuseThrows = true;
  try {
    f();
  } catch (Refusal &r) {
    refuseThrows = false;
    return r.name;
  }
  refuseThrows = false;
  return "";
}

int main() {
  const WsLay lay = toyLayout();
  blk::LinShape shape; // `wsPending` reads only `matrix_rows` of it
  shape.matrix_rows = lay.m0;
  const unsigned rowBits = shape.k_row();
  const u64 columns = 1ull << lay.lk;
  printf("toy layout: %zu units, m_0 = %llu rows, 2^%d columns\n", lay.U.size(), (unsigned long long)lay.m0, lay.lk);
  size_t cases = 0;
  // Records: one unit (first, last), neighbours, distant units, one unit twice, four units.
  const u32 last = (u32)lay.U.size() - 1;
  const vector<vector<u32>> records = {{0}, {last}, {1, 2}, {0, last}, {3, 3}, {2, 5, 1, last / 2}};
  for (const vector<u32> &units : records) {
    WsForkOut o;
    for (u32 ui : units) o.rec.push_back(madeClaim(lay, ui));
    o.rm = made((size_t)lay.lk);
    vector<Ex> W((size_t)(lay.m0 * columns)); // small integers, as weights' limbs are
    for (Ex &x : W) x = Ex(next64() & 255, 0, 0);
    // fv, the text's sum.
    vector<Ex> eq(columns, eO());
    for (u64 c = 0; c < columns; c++)
      for (int i = 0; i < lay.lk; i++) eq[c] = eq[c] * (((c >> (lay.lk - 1 - i)) & 1) ? o.rm[(size_t)i] : eO() - o.rm[(size_t)i]);
    Ex fv = eZ();
    for (const WsRecClaim &c : o.rec)
      for (u64 y = 0; y < lay.U[c.ui].rows; y++) {
        Ex inner = eZ();
        for (u64 col = 0; col < columns; col++) inner = inner + eq[col] * W[(size_t)((lay.U[c.ui].off + y) * columns + col)];
        fv = fv + c.ka * gammaAt(lay, c, y) * inner;
      }
    const Ex mask = made();
    o.beta = made();
    o.fvp = fv + o.beta * mask;
    const WsPending p = wsPending(lay, o, units.size(), shape);
    // The same sum from the result, by the `blocks/` library's blk::Tensor::at(): the Tensors `g`
    // sum to the row weights; `psi` is rm, bit 0 first.
    const blk::Tensor onColumns = blk::Tensor::eq(p.psi.data(), (unsigned)p.psi.size());
    blk::E sum = blk::E::zero();
    for (u64 y = 0; y < (1ull << rowBits); y++) {
      blk::E g = blk::E::zero();
      for (const blk::Tensor &t : p.g) g += t.at(y);
      bool inside = false;
      for (u32 ui : units) inside = inside || (y >= lay.U[ui].off && y < lay.U[ui].off + lay.U[ui].rows);
      if (!inside && !(g == blk::E::zero())) bad++, printf("a weight outside the claims' units, at row %llu\n", (unsigned long long)y);
      if (y >= lay.m0 || g == blk::E::zero()) continue;
      blk::E inner = blk::E::zero();
      for (u64 col = 0; col < columns; col++) inner += onColumns.at(col) * toBlkE(W[(size_t)(y * columns + col)]);
      sum += g * inner;
    }
    if (!(sum == toBlkE(fv))) bad++, printf("the row's form is not the text's sum (record of %zu claims)\n", units.size());
    if (!(sum + p.beta * toBlkE(mask) == p.target)) bad++, printf("(the form) + beta c is not the target\n");
    if (sum - p.beta * toBlkE(mask) == p.target) bad++, printf("the test's mask is 0: it shows no sign\n");
    cases++;
    // Rejections.
    if (refusalOf([&] { (void)wsPending(lay, o, units.size() + 1, shape); }) != "link-claim-count") bad++, printf("another number of claims is taken\n");
    WsForkOut shortPoint = o;
    shortPoint.rm.pop_back();
    if (refusalOf([&] { (void)wsPending(lay, shortPoint, units.size(), shape); }) != "link-claim-count") bad++, printf("a short point is taken\n");
    WsForkOut unknown = o;
    unknown.rec[0].ui = (u32)lay.U.size();
    if (refusalOf([&] { (void)wsPending(lay, unknown, units.size(), shape); }) != "internal-gamma-rows") bad++, printf("an unknown unit is taken\n");
    for (u64 rows : {lay.m0 - 1, lay.m0 + 1}) {
      blk::LinShape other = shape;
      other.matrix_rows = rows;
      if (refusalOf([&] { (void)wsPending(lay, o, units.size(), other); }) != "internal-gamma-rows") bad++, printf("a matrix of another number of rows is taken\n");
    }
  }
  printf("pending_row_test: %zu records: %s\n", cases, bad ? "FAILED" : "ok");
  return bad ? 1 : 0;
}
