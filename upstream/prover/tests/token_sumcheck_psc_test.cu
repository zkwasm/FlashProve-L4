// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the token sumcheck (`Tsc`, `TscEnd`: prover/hiding/token_sumcheck.cuh) and the output
// head's rounds over the features, which continue it (prover/head/feature_rounds.cuh). Both send
// through namespace `sc_door` (prover/hiding/sumcheck_door.cuh), which runs PSC, the sumcheck with
// a hiding polynomial.
// The GPU proves on synthetic arrays; the last entry of `shapes` is full size (2^10 token
// positions, 5120 features). The host then acts as the verifier, written here from the definitions:
// it compares its challenge point (`coins`, `COIN` in identifiers) with the GPU's; it checks the
// one remaining linear claim (`row` in identifiers), on the mask entries (the prover's random
// values, which the test knows); it removes the inner masks from the end values as sent (`shown`)
// and compares the results with the arrays' multilinear extensions at that point.
// Runs with an injected fault (`Fault`) must be rejected (printed: refused).
// The transcript is a hash chain (blocks/chain/), which writes a proof. Run::message()
// (prover/tests/door_test_host.cuh) reads each message from the proof's bytes and writes it over
// the host's copy of the slots. No fault alters a value after it is sent.
// Build, in prover/: ./build.sh token_sumcheck_psc_test, which compiles this file with nvcc.
#include "common/element_from_hash.h"
#include "hiding/product_test.cuh"
#include "hiding/token_sumcheck.cuh"
#include "hiding/token_plan.h"
#include "head/feature_rounds.cuh"
#include "tests/door_test_host.cuh"

// ---- The verifier's side: the start value (the claimed sum), sent masked, then its sumcheck ----
// After its other rounds PSC has one in a further variable Y, with challenge beta; end value j is
// then sent as v_j + beta c_j: the mask entry c_j is its inner mask.
struct Seen {
  bool challenges = true;        // the host's transcript draws what the GPU drew
  std::vector<T> r, shown, bare; // challenges; end values as sent; end values without their inner masks
  T eps, beta, claim;            // claim: the masked sumcheck's claim after the round in Y
  T masks; // the hiding polynomial at the challenges, plus eps times the start value's mask
  T start; // the start value without its mask
};
// slots, mask: the host's copies of the message slots and their mask entries, from the start
// value's on. z[j]: round j's coordinate of the equality point; each such round sends a polynomial
// q of degree 2.
static Seen follow(Run &R, T *slots, const T *mask, const std::vector<T> &z, int degree_in_y, size_t ends,
                   const std::vector<T> &gpu_r) {
  Seen s;
  size_t at = 0;
  R.message(slots + at, 1);
  const T target = slots[at]; // the start value as sent, with its mask
  s.start = F::sub(slots[at], mask[at]);
  at++;
  const size_t first = at; // index of the sumcheck's first slot
  R.message(slots + at, 1, true);
  s.eps = R.answer(blk::R_PSC_EPSILON, 1, Run::EPSILON)[0];
  T claim = F::add(slots[at++], F::mul(s.eps, target));
  for (size_t j = 0; j < z.size(); j++) {
    // sent: the coefficients of X and X^2; (1 - z) q(0) + z q(1) = claim fixes the constant.
    // q1 and q2 are the GPU's slots, read before message() overwrites them. If the proof's bytes
    // differ, Run::in_slots turns false, which only through() reports: only honest runs call it.
    const T q1 = slots[at], q2 = slots[at + 1], q0 = F::sub(claim, F::mul(z[j], F::add(q1, q2)));
    R.message(slots + at, 2);
    const T r = R.answer(blk::R_PSC_ROUND, 1, Run::COIN)[0];
    s.challenges = s.challenges && same(r, gpu_r[j]);
    s.r.push_back(r);
    claim = F::add(q0, F::mul(r, F::add(q1, F::mul(r, q2))));
    at += 2;
  }
  {
    T q = F::zero(); // sent: Y^1 .. ; q(0) = claim fixes the constant
    R.message(slots + at, degree_in_y);
    s.beta = R.answer(blk::R_PSC_Y, 1, Run::NOT_ZERO)[0];
    for (int k = degree_in_y; k >= 1; k--) q = F::mul(F::add(q, slots[at + k - 1]), s.beta);
    claim = F::add(claim, q);
    at += degree_in_y;
  }
  s.claim = claim;
  R.message(slots + at, (int)ends);
  R.silent(blk::R_PSC_VALUES);
  for (size_t j = 0; j < ends; j++) {
    s.shown.push_back(slots[at + j]);
    s.bare.push_back(F::sub(slots[at + j], F::mul(s.beta, mask[at + j])));
  }
  // The hiding polynomial's coefficients are the mask entries from index `first` on: for each round
  // those of X and X^2, then those of Y^0 .. Y^degree_in_y.
  s.masks = F::mul(s.eps, mask[0]);
  size_t i = first;
  for (size_t j = 0; j < z.size(); j++, i += 2)
    s.masks = F::add(s.masks, F::mul(s.r[j], F::add(mask[i], F::mul(s.r[j], mask[i + 1]))));
  T power = F::one();
  for (int k = 0; k <= degree_in_y; k++, power = F::mul(power, s.beta))
    s.masks = F::add(s.masks, F::mul(power, mask[i + k]));
  if (i + degree_in_y + 1 != at) {
    printf("FAILED: the test's own count of mask entries is wrong\n");
    exit(2);
  }
  return s;
}
// The linear claim that remains: claim = (hiding polynomial at the challenges)
// + eps (mask of the start value) + eps (summand on the end values as sent).
static bool row_holds(const Seen &s, T summand) {
  return same(s.claim, F::add(s.masks, F::mul(s.eps, summand)));
}
// A run with a fault counts as rejected if any one of the host's three checks fails. The comparison
// with the extensions is enough to reject an ARRAY_CHANGED run, so only the FALSE_KEPT runs show
// that the linear claim rejects a fault.
enum Fault { NONE, ARRAY_CHANGED, FALSE_KEPT };

// Proves sum_(t < 2^lt) eq(P, t) sum_m co[m] * map a[m][t] * map b[m][t]. Returns whether the
// host's verifier accepts. kind[m]: which factors of term m are shown (their end values are sent):
// both; the first only, the second being kept (`TscTerm::kept`: its true value at the challenge
// point); or neither. A map shown in several terms has one end value.
enum Kind { BOTH, SECOND_KEPT, NEITHER };
static bool products(int lt, int maps, const std::vector<int> &a, const std::vector<int> &b,
                     const std::vector<Kind> &kind, Fault fault, const std::string &name) {
  const size_t N = (size_t)1 << lt, first = 3;
  const int NT = (int)a.size();
  std::vector<char> is_shown(maps, 0);
  int degree = 0; // of the summand in the shown maps
  for (int m = 0; m < NT; m++) {
    if (kind[m] != NEITHER) is_shown[a[m]] = 1;
    if (kind[m] == BOTH) is_shown[b[m]] = 1;
    degree = std::max(degree, kind[m] == BOTH ? 2 : kind[m] == SECOND_KEPT ? 1 : 0);
  }
  unsigned all_shown = 0;
  for (char x : is_shown) all_shown += x;
  Run M([&](blk::Plan &p) { blk::plan_token_sum(p, "token", (unsigned)lt, all_shown, (unsigned)degree); });
  const std::vector<T> P = elements(lt), co = elements(NT), eP = eq_table(P.data(), lt);
  std::vector<std::vector<T>> map(maps);
  for (auto &v : map) v = elements(N);
  std::vector<T> Fv, Gv;
  T sum = F::zero();
  for (int m = 0; m < NT; m++) {
    Fv.insert(Fv.end(), map[a[m]].begin(), map[a[m]].end());
    Gv.insert(Gv.end(), map[b[m]].begin(), map[b[m]].end());
    for (size_t t = 0; t < N; t++) sum = F::add(sum, F::mul(F::mul(co[m], eP[t]), F::mul(map[a[m]][t], map[b[m]][t])));
  }
  T *dF = up(Fv), *dG = up(Gv), *dco = up(co), *dP = up(P);
  // number[x]: map x's index among the end values, or -1. value[i]: the array whose first entry
  // the rounds turn into end value i.
  std::vector<int> number(maps, -1);
  std::vector<const T *> value;
  std::vector<int> shown_map;
  for (int m = 0; m < NT; m++)
    for (int side = 0; side < (kind[m] == BOTH ? 2 : kind[m] == SECOND_KEPT ? 1 : 0); side++) {
      const int x = side ? b[m] : a[m];
      if (number[x] >= 0) continue;
      number[x] = (int)value.size();
      value.push_back((side ? dG : dF) + (size_t)m * N);
      shown_map.push_back(x);
    }
  std::vector<TscTerm> terms;
  for (int m = 0; m < NT; m++)
    terms.push_back(TscTerm{kind[m] == NEITHER ? -1 : number[a[m]], kind[m] == BOTH ? number[b[m]] : -1, dco + m,
                            kind[m] == SECOND_KEPT ? dG + (size_t)m * N : nullptr});
  Tsc engine;
  engine.init(lt, NT, TscShape{value.size(), degree, 0});
  TscEnd end;
  end.setup(engine.shape, value, terms);
  engine.prove(M.tr, dP, dco, dF, dG, NT, M.d + first);
  if (fault == ARRAY_CHANGED || fault == FALSE_KEPT) { // an end value, or a kept factor, is not what the rounds ran on
    CK(cudaDeviceSynchronize());
    size_t kept_term = 0; // the first term with a kept factor
    while (fault == FALSE_KEPT && kind[kept_term] != SECOND_KEPT) kept_term++;
    T *at = fault == FALSE_KEPT ? dG + kept_term * N : dF;
    const T v = F::add(down(at, 1)[0], F::one());
    CK(cudaMemcpy(at, &v, sizeof v, cudaMemcpyHostToDevice));
  }
  end.run(M.tr);
  M.sent();
  std::vector<T> m = down(M.d, 2 * PAD);
  const size_t n = engine.nmsg();
  const TscOut out = down(engine.out, 1)[0];
  const std::vector<T> gr(out.r, out.r + lt);
  const Seen s = follow(M, m.data() + first, m.data() + PAD + first, P, degree, value.size(), gr);
  if (fault == NONE) {
    hold(same(s.start, sum) && same(out.Cl, sum), name + ": the value sent first is the sum by the definition");
    hold(s.challenges, name + ": the host draws the GPU's challenges");
    hold(M.through(), name + ": both sides' transcripts end alike");
    hold(only_these(m, first, n), name + ": no slot outside the sum-check's is written");
  }
  bool values = true;
  for (size_t j = 0; j < value.size(); j++)
    values = values && same(s.bare[j], extension(map[shown_map[j]].data(), gr.data(), lt));
  T summand = F::zero(); // on the shown values; a factor that is not shown enters with its true value at the point
  for (int t = 0; t < NT; t++)
    summand = F::add(
        summand,
        F::mul(F::mul(co[t], kind[t] == NEITHER ? extension(map[a[t]].data(), gr.data(), lt) : s.shown[number[a[t]]]),
               kind[t] == BOTH ? s.shown[number[b[t]]] : extension(map[b[t]].data(), gr.data(), lt)));
  return s.challenges && row_holds(s, summand) && values;
}

// The output head's form: sum_t eq(P, t) sum_(k < d) A[k][t] B[k][t], over t and then over k.
// k has lam bits (d <= 2^lam: not checked) and is summed without weights: its rounds use the
// equality coordinate 1/2, and every coefficient (`dco`) is 2^-lam.
static bool two_indices(int lt, int lam, int d, Fault fault, const std::string &name) {
  Run M([&](blk::Plan &p) { blk::plan_token_sum(p, "token", (unsigned)lt, 2, 2, (unsigned)lam); });
  const size_t N = (size_t)1 << lt, first = 2;
  const int Pd = 1 << lam;
  const std::vector<T> P = elements(lt), eP = eq_table(P.data(), lt);
  std::vector<T> A = elements((size_t)d * N), B = elements((size_t)d * N);
  const T half = F::inv(F::small(2));
  T weight = F::one(), sum = F::zero();
  for (int i = 0; i < lam; i++) weight = F::mul(weight, half);
  for (int k = 0; k < d; k++)
    for (size_t t = 0; t < N; t++) sum = F::add(sum, F::mul(eP[t], F::mul(A[k * N + t], B[k * N + t])));
  T *dA = up(A), *dB = up(B), *dP = up(P), *dco = up(std::vector<T>(d, weight)), *coins = up(elements(lam));
  Tsc engine;
  engine.init(lt, d, TscShape{2, 2, lam});
  engine.prove(M.tr, dP, dco, dA, dB, d, M.d + first);
  CK(cudaDeviceSynchronize());
  // The two vectors over k, padded with zeros as by k_head_arrays() (prover/head/claims.cuh).
  const std::vector<T> fA = down(dA, (size_t)d * N), fB = down(dB, (size_t)d * N);
  std::vector<T> a(Pd, F::zero()), b(Pd, F::zero());
  for (int k = 0; k < d; k++) a[k] = fA[k * N], b[k] = fB[k * N];
  if (fault == ARRAY_CHANGED) a[d - 1] = F::add(a[d - 1], F::one());
  T *da = up(a), *db = up(b);
  k_head_rounds<<<1, 1024>>>(da, db, lam, lt, M.tr, coins);
  TscEnd end;
  end.setup(engine.shape, {da, db}, {TscTerm{0, 1, nullptr, nullptr}});
  end.run(M.tr);
  M.sent();
  std::vector<T> m = down(M.d, 2 * PAD);
  const size_t n = engine.nmsg();
  const TscOut out = down(engine.out, 1)[0];
  std::vector<T> gr(out.r, out.r + lt), z = P;
  const std::vector<T> rs = down(coins, lam);
  gr.insert(gr.end(), rs.begin(), rs.end());
  z.insert(z.end(), (size_t)lam, half);
  const Seen s = follow(M, m.data() + first, m.data() + PAD + first, z, 2, 2, gr);
  if (fault == NONE) {
    hold(same(s.start, F::mul(weight, sum)), name + ": the value sent first is 2^-lam times the sum by the definition");
    hold(s.challenges, name + ": the host draws the GPU's challenges");
    hold(M.through(), name + ": both sides' transcripts end alike");
    hold(only_these(m, first, n), name + ": no slot outside the sum-check's is written");
  }
  // The extensions at (rs, r): over k with zeros from d on, highest bit first.
  const std::vector<T> ek = eq_table(rs.data(), lam);
  T va = F::zero(), vb = F::zero();
  for (int k = 0; k < d; k++) {
    va = F::add(va, F::mul(ek[k], extension(A.data() + k * N, out.r, lt)));
    vb = F::add(vb, F::mul(ek[k], extension(B.data() + k * N, out.r, lt)));
  }
  return s.challenges && row_holds(s, F::mul(s.shown[0], s.shown[1])) && same(s.bare[0], va) && same(s.bare[1], vb);
}

int main() {
  const std::vector<int> a7 = {0, 0, 1, 2, 2, 3, 3}, b7 = {0, 1, 1, 0, 1, 0, 1}; // the norm's seven row products
  for (int lt : {1, 3, 6, 10, 12}) {
    const std::string at = ", lt " + std::to_string(lt);
    std::string name = "seven products of four maps" + at;
    hold(products(lt, 4, a7, b7, std::vector<Kind>(7, BOTH), NONE, name), name + ": the honest run is accepted");
    hold(!products(lt, 4, a7, b7, std::vector<Kind>(7, BOTH), ARRAY_CHANGED, name),
         name + ": an end value that the rounds did not run on is refused");
    name = "one product" + at;
    hold(products(lt, 2, {0}, {1}, {BOTH}, NONE, name), name + ": the honest run is accepted");
    name = "three products, the second factors kept" + at;
    hold(products(lt, 5, {0, 1, 0}, {2, 3, 4}, std::vector<Kind>(3, SECOND_KEPT), NONE, name),
         name + ": the honest run is accepted");
    hold(!products(lt, 5, {0, 1, 0}, {2, 3, 4}, std::vector<Kind>(3, SECOND_KEPT), FALSE_KEPT, name),
         name + ": a false kept factor in the round in Y is refused");
  }
  for (int lt : {2, 7, 11}) { // every `Kind` in one sumcheck; map 0 is shown in some terms only
    const std::string name = "five products of the three kinds, lt " + std::to_string(lt);
    const std::vector<int> a = {0, 1, 2, 0, 4}, b = {1, 3, 3, 0, 0};
    const std::vector<Kind> kind = {BOTH, SECOND_KEPT, NEITHER, BOTH, NEITHER};
    hold(products(lt, 5, a, b, kind, NONE, name), name + ": the honest run is accepted");
    hold(!products(lt, 5, a, b, kind, ARRAY_CHANGED, name),
         name + ": an end value that the rounds did not run on is refused");
    hold(!products(lt, 5, a, b, kind, FALSE_KEPT, name), name + ": a false kept factor in the round in Y is refused");
  }
  struct {
    int lt, lam, d;
  } shapes[] = {
      {2, 1, 2},     {3, 3, 5},     {5, 4, 16},    {6, 7, 96},    {11, 5, 20},   {3, 10, 1024}, {3, 11, 1025},
      {2, 11, 2048}, {2, 12, 2049}, {2, 12, 4096}, {2, 13, 5120}, {1, 13, 8192}, {10, 13, 5120}};
  for (auto &x : shapes) {
    const std::string name =
        "positions and features, lt " + std::to_string(x.lt) + ", " + std::to_string(x.d) + " features";
    hold(two_indices(x.lt, x.lam, x.d, NONE, name), name + ": the honest run is accepted");
    hold(!two_indices(x.lt, x.lam, x.d, ARRAY_CHANGED, name),
         name + ": a vector changed between the two halves is refused");
  }
  return verdict_line();
}
