// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests two types of sumcheck of the attention sub-proof, SC3 (of degree 3) and SCD (its degree
// varies by round), on small synthetic arrays. Both follow PSC (blocks/iop/psc.h): the sumcheck
// runs on U + eps * summand, where U is a hiding polynomial with mask entries (elements of a
// committed mask row; here synthetic) as coefficients. A last round, in a further variable Y, puts
// an inner mask c on each end value (an array's multilinear extension, its "map", at the
// challenges): value + beta * c is sent. The GPU runs the engines (prover/attn/engines.cuh), whose
// values prover/hiding/sumcheck_door.cuh masks and sends. The host then acts as the verifier (the
// program cp_verify is not run):
//   - it follows the transcript, fills in each round's coefficient that is not sent, and compares
//     the rounds' challenges (but not eps or beta) with the GPU's;
//   - it checks row_holds(), the linear equation that is left, on the mask entries, which it knows;
//   - it removes the inner masks and compares the end values with the maps. (In the sub-proof,
//     linear claims on committed data do this.)
// Each case also runs with the faults of enum Fault; main() checks which of the last two checks
// above then fails.
// As in the prover cp_prove, the transcript is a hash chain (blocks/chain/chain.h); struct Run
// (prover/tests/door_test_host.cuh) holds both sides of it.
// Build in prover/, with nvcc: ./build.sh attention_sumchecks_test
#include "attn/engines.cuh"
#include "attn/plan_rounds.h"
#include "tests/door_test_host.cuh"

// ---- The verifier's side of a sumcheck under plain weights, from the definitions ----
struct Seen {
  bool challenges = true;        // the host's transcript draws what the GPU drew
  std::vector<T> r, shown, bare; // challenges; end values as sent; the same without inner masks
  T eps, beta, claim;            // claim: the masked sumcheck's claim after the round in Y
  T masks;                       // U at the challenges, from the mask entries
  size_t used = 0;               // the slots that the sumcheck has taken
};
struct Verdict {
  bool challenges, row, values; // Seen::challenges; `row_holds`; the end values are the maps'
  bool accepts() const { return challenges && row && values; }
};
// slots, mask: the sumcheck's message slots (its sent scalars) and the mask entries with the same
// numbers, both copied from Run::d (door_test_host.cuh). degree[j]: of round j. start: the public
// claimed sum. Plain weights: a variable is summed as q(0) + q(1). Not checked: that the slots end
// before entry PAD of Run::d, where the mask entries begin.
static Seen follow(Run &R, T *slots, const T *mask, const std::vector<int> &degree, int degree_in_y, size_t ends,
                   T start, const std::vector<T> &gpu_r) {
  Seen s;
  size_t at = 0;
  R.message(slots + at, 1, true);
  s.eps = R.answer(blk::R_PSC_EPSILON, 1, Run::EPSILON)[0];
  T claim = F::add(slots[at++], F::mul(s.eps, start));
  for (size_t j = 0; j < degree.size(); j++) {
    const int d = degree[j];
    std::vector<T> q(d + 1); // sent: X^0, X^2 .. X^d; q(0) + q(1) = claim fixes X^1's coefficient
    q[0] = slots[at];
    for (int k = 2; k <= d; k++) q[k] = slots[at + k - 1];
    T rest = F::add(q[0], q[0]);
    for (int k = 2; k <= d; k++) rest = F::add(rest, q[k]);
    q[1] = F::sub(claim, rest);
    R.message(slots + at, d);
    const T r = R.answer(blk::R_PSC_ROUND, 1)[0];
    s.challenges = s.challenges && same(r, gpu_r[j]);
    s.r.push_back(r);
    claim = power_sum(q, r);
    at += d;
  }
  {
    std::vector<T> q(degree_in_y + 1); // sent: Y^1 .. ; q(0) = claim fixes the constant
    q[0] = claim;
    for (int k = 1; k <= degree_in_y; k++) q[k] = slots[at + k - 1];
    R.message(slots + at, degree_in_y);
    s.beta = R.answer(blk::R_PSC_Y, 1, Run::NOT_ZERO)[0];
    claim = power_sum(q, s.beta);
    at += degree_in_y;
  }
  s.claim = claim;
  R.message(slots + at, (int)ends);
  R.silent(blk::R_PSC_VALUES);
  for (size_t j = 0; j < ends; j++) {
    s.shown.push_back(slots[at + j]);
    s.bare.push_back(F::sub(slots[at + j], F::mul(s.beta, mask[at + j])));
  }
  s.used = at + ends;
  // The coefficients of U are the sumcheck's first mask entries: for each round j those of u_j,
  // X^1 .. X^d, then those of u_Y, Y^0 .. Under plain weights
  // U = sum_j u_j(X_j) prod_(i > j) X_i + u_Y(Y).
  s.masks = F::zero();
  {
    size_t i = at;
    i -= degree_in_y + 1;
    T power = F::one();
    for (int k = 0; k <= degree_in_y; k++, power = F::mul(power, s.beta))
      s.masks = F::add(s.masks, F::mul(power, mask[i + k]));
    T later = F::one();
    for (size_t j = degree.size(); j-- > 0;) {
      i -= degree[j];
      power = later;
      for (int k = 1; k <= degree[j]; k++) {
        power = F::mul(power, s.r[j]);
        s.masks = F::add(s.masks, F::mul(power, mask[i + k - 1]));
      }
      later = F::mul(later, s.r[j]);
    }
    if (i != 0) { // a fault of this test, not a rejection
      printf("FAILED: the test's own count of mask entries is wrong\n");
      exit(2);
    }
  }
  return s;
}
// The equation that is left: claim = U(challenges) + eps * (summand on the end values as sent).
static bool row_holds(const Seen &s, T summand_on_shown) {
  return same(s.claim, F::add(s.masks, F::mul(s.eps, summand_on_shown)));
}

// ---- SC3 ----
// The engine reads entry x of pair i of A and B through a source type; here, field elements.
struct SrcArrays {
  static constexpr bool INT = false, ROW = false;
  const T *a, *b;
  int nv;
  __device__ T A(int i, size_t x) const { return a[((size_t)i << nv) + x]; }
  __device__ T B(int i, size_t x) const { return b[((size_t)i << nv) + x]; }
};
// The same for arrays of small integers, which the engine reads by rows (Ar(), Br(); a row: the
// entries that share x >> nlow) or singly (Ai(), Bi()).
struct SrcIntegers {
  static constexpr bool INT = true, ROW = true;
  const int16_t *a, *b;
  int nv, nlow;
  __device__ const int16_t *Ar(int i, size_t xt) const { return a + ((size_t)i << nv) + (xt << nlow); }
  __device__ const int16_t *Br(int i, size_t xt) const { return b + ((size_t)i << nv) + (xt << nlow); }
  __device__ int Ai(int i, size_t x) const { return a[((size_t)i << nv) + x]; }
  __device__ int Bi(int i, size_t x) const { return b[((size_t)i << nv) + x]; }
};
// ARRAY_CHANGED: in sc3(), an entry of A, after the claimed sum is formed; in scd(), no array but
// an index bit's end value. No fault alters a scalar after it is sent: Run::message() overwrites
// the host's copy of the slots from the proof.
enum Fault { NONE, ARRAY_CHANGED, FALSE_SUM };
static Verdict sc3(int nv, int nlow, int nb, bool integers, Fault fault, const std::string &name) {
  Run M([&](blk::Plan &p) { blk::plan_psc(p, "SC3", blk::attention_sc3((unsigned)nv, (unsigned)nb)); });
  const size_t N = (size_t)1 << nv, first = 7;
  std::vector<T> A = elements(nb * N), B = elements(nb * N), W0 = elements((size_t)1 << (nv - nlow)), wb = elements(nb);
  std::vector<int16_t> iA(nb * N), iB(nb * N);
  if (integers)
    for (size_t x = 0; x < nb * N; x++) {
      iA[x] = (int16_t)(word() % 2001) - 1000, iB[x] = (int16_t)(word() % 2001) - 1000;
      A[x] = F::emb(F::fromi(iA[x])), B[x] = F::emb(F::fromi(iB[x]));
    }
  T sum = F::zero();
  for (int b = 0; b < nb; b++)
    for (size_t x = 0; x < N; x++)
      sum = F::add(sum, F::mul(F::mul(wb[b], W0[x >> nlow]), F::mul(A[b * N + x], B[b * N + x])));
  T *dA = up(A), *dB = up(B), *dW0 = up(W0), *dwb = up(wb), *h4 = up(elements(4)), *r = up(elements(nv)),
    *As = up(elements(nb)), *Bs = up(elements(nb)), *Ws = up(elements(nb));
  const S3Work work{up(elements(nb * N)), up(elements(nb * N)), up(elements(N + 2)), up(elements(N + 2))};
  int16_t *diA = up(iA), *diB = up(iB);
  ga::Eng e;
  auto engine = [&](auto src) {
    a_s3_prep(src, nv, nlow, nb, dwb, dW0, h4, 0, &e);
    if (fault == NONE) hold(same(down(h4, 4)[3], sum), name + ": the claimed sum is the sum by the definition");
    if (fault == ARRAY_CHANGED) {
      A[N / 2 + 1] = F::add(A[N / 2 + 1], F::one());
      iA[N / 2 + 1]++;
      CK(cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(diA, iA.data(), iA.size() * sizeof(int16_t), cudaMemcpyHostToDevice));
    }
    if (fault == FALSE_SUM) {
      sum = F::add(sum, F::one());
      CK(cudaMemcpy(h4 + 3, &sum, sizeof(T), cudaMemcpyHostToDevice));
    }
    a_s3_run(src, nv, nlow, nb, dwb, dW0, work, h4, M.tr, M.d + first, r, As, Bs, Ws, 0, &e);
  };
  if (integers)
    engine(SrcIntegers{diA, diB, nv, nlow});
  else
    engine(SrcArrays{dA, dB, nv});
  M.sent();
  std::vector<T> m = down(M.d, 2 * PAD);
  // The sum of U; three scalars a round, since W A B has degree at most 3 in a variable; two for
  // the round in Y, since the summand has degree 2 in the maps A_b, B_b; the 2 nb end values.
  const size_t n = 1 + 3 * (size_t)nv + 2 + 2 * (size_t)nb;
  const std::vector<T> gr = down(r, nv);
  const Seen s =
      follow(M, m.data() + first, m.data() + PAD + first, std::vector<int>(nv, 3), 2, 2 * (size_t)nb, sum, gr);
  if (fault == NONE) {
    hold(s.challenges, name + ": the host draws the GPU's challenges");
    hold(M.through(), name + ": both sides' transcripts end alike");
    hold(s.used == n && only_these(m, first, n), name + ": no slot outside the sum-check's is written");
    hold(sc_door::slots(sc_door::Form::SUM3, nv, 2 * (size_t)nb, 2) == n, name + ": the door counts the same slots");
  }
  const T W = extension(W0.data(), gr.data(), nv - nlow);
  T on_shown = F::zero();
  bool ends = true;
  for (int b = 0; b < nb; b++) {
    on_shown = F::add(on_shown, F::mul(F::mul(wb[b], W), F::mul(s.shown[2 * b], s.shown[2 * b + 1])));
    ends = ends && same(s.bare[2 * b], extension(A.data() + b * N, gr.data(), nv)) &&
           same(s.bare[2 * b + 1], extension(B.data() + b * N, gr.data(), nv));
  }
  return Verdict{s.challenges, row_holds(s, on_shown), ends};
}

// ---- SCD ----
// Tn = 2^lt tokens, G = 2^lgk key-value heads with HA query heads each, Lam layers. Pair b: layer
// b / HA, query head b % HA of each key-value head.
static Verdict scd(int lt, int lgk, int HA, int Lam, Fault fault, const std::string &name) {
  Run M([&](blk::Plan &p) {
    blk::plan_psc(p, "SCD", blk::attention_scd((unsigned)lt, (unsigned)lgk, (unsigned)(Lam * HA), (unsigned)Lam));
  });
  const size_t Tn = (size_t)1 << lt, G = (size_t)1 << lgk, first = 5;
  const int NP = Lam * HA, rounds = 2 * lt + lgk;
  std::vector<int> ix(Lam * G * HA * Tn);
  for (int &x : ix) x = (int)(word() % Tn);
  std::vector<T> KC = elements(Lam * G * Tn), ri = elements(lt), rg = elements(lgk), om4 = elements(NP);
  const std::vector<T> eri = eq_table(ri.data(), lt), erg = eq_table(rg.data(), lgk);
  auto index = [&](int b, size_t g, size_t i) {
    return ix[((((size_t)(b / HA) * G + g) * HA + b % HA) << lt) + i];
  };
  // Bs[b] = sum_(g, i) eri[i] erg[g] KC[b / HA][g][index(b, g, i)]. In the sub-proof it is an end
  // value of the SC3 before SCD.
  std::vector<T> Bs(NP, F::zero());
  for (int b = 0; b < NP; b++)
    for (size_t g = 0; g < G; g++)
      for (size_t i = 0; i < Tn; i++)
        Bs[b] = F::add(Bs[b], F::mul(F::mul(eri[i], erg[g]), KC[(((size_t)(b / HA) * G + g) << lt) + index(b, g, i)]));
  if (fault == FALSE_SUM) Bs[0] = F::add(Bs[0], F::one());
  T sum = F::zero();
  for (int b = 0; b < NP; b++) sum = F::add(sum, F::mul(om4[b], Bs[b]));
  int *dix = up(ix);
  T *dKC = up(KC), *deri = up(eri), *derg = up(erg), *dom4 = up(om4), *dBs = up(Bs), *rp = up(elements(rounds)),
    *KCs = up(elements(Lam)), *dri = up(ri), *drg = up(rg), *cy = up(elements(2 * lt + 1));
  ga::Eng e;
  a_scd(lt, lgk, HA, Lam, dix, dKC, deri, derg, dom4, dBs, M.tr, M.d + first, rp, KCs, 0, &e);
  CK(cudaDeviceSynchronize());
  const std::vector<T> gr = down(rp, rounds);
  const T *pj = gr.data(), *pg = pj + lt, *pi = pg + lgk;
  // The index bits' maps at (pg, pi), by the definition. (In the sub-proof k_ibs() computes them
  // from a table over committed cells.)
  const std::vector<T> epg = eq_table(pg, lgk), epi = eq_table(pi, lt);
  std::vector<T> ibs((size_t)NP * lt, F::zero());
  for (int b = 0; b < NP; b++)
    for (int k = 0; k < lt; k++)
      for (size_t g = 0; g < G; g++)
        for (size_t i = 0; i < Tn; i++)
          if ((index(b, g, i) >> k) & 1)
            ibs[(size_t)b * lt + k] = F::add(ibs[(size_t)b * lt + k], F::mul(epg[g], epi[i]));
  std::vector<T> true_ibs = ibs;
  if (fault == ARRAY_CHANGED) ibs[1] = F::add(ibs[1], F::one()); // a false index bit at the end
  a_scd_end(lt, lgk, HA, Lam, KCs, up(ibs), dom4, dri, drg, rp, cy, M.tr, 0, &e);
  M.sent();
  std::vector<T> m = down(M.d, 2 * PAD);
  // The summand is eri(i) erg(g) KC(g, j) prod_k eq1(bit_k(g, i), j_k), where eq1(u, y) = u y +
  // (1 - u)(1 - y); each factor is multilinear. A bit of j occurs in KC and in one factor eq1:
  // degree 2. A bit of g occurs in erg, in KC and in the lt index bits: lt + 2. A bit of i occurs
  // in eri and in the lt index bits: lt + 1. The rounds run over j, then g, then i. In the maps
  // whose end values are sent (KC and the lt index bits of a pair) the summand has degree lt + 1.
  const size_t ends = (size_t)Lam + (size_t)NP * lt;
  std::vector<int> degree;
  for (int q = 0; q < lt; q++) degree.push_back(2);
  for (int q = 0; q < lgk; q++) degree.push_back(lt + 2);
  for (int q = 0; q < lt; q++) degree.push_back(lt + 1);
  size_t n = 1 + (size_t)(lt + 1) +
             ends; // the sum of U, the round in Y, the end values; a round of degree d adds d
  for (int d : degree) n += d;
  const Seen s = follow(M, m.data() + first, m.data() + PAD + first, degree, lt + 1, ends, sum, gr);
  if (fault == NONE) {
    hold(s.challenges, name + ": the host draws the GPU's challenges");
    hold(M.through(), name + ": both sides' transcripts end alike");
    hold(s.used == n && only_these(m, first, n), name + ": no slot outside the sum-check's is written");
    hold(a_scd_nout(lt, lgk, Lam, NP) == n, name + ": the engine counts the same slots");
  }
  // The summand above, summed over pairs with weights om4, on the end values as sent.
  T on_shown = F::zero();
  for (int b = 0; b < NP; b++) {
    T t = F::mul(om4[b], s.shown[b / HA]);
    for (int k = 0; k < lt; k++) {
      const T y = pj[lt - 1 - k], u = s.shown[Lam + (size_t)b * lt + k];
      t = F::mul(t, F::add(F::mul(u, y), F::mul(F::sub(F::one(), u), F::sub(F::one(), y))));
    }
    on_shown = F::add(on_shown, t);
  }
  on_shown = F::mul(on_shown, F::mul(extension(eri.data(), pi, lt), extension(erg.data(), pg, lgk)));
  bool values = true;
  for (int lam = 0; lam < Lam; lam++) {
    std::vector<T> point(pg, pg + lgk); // KC[lam] has the index (g | j)
    point.insert(point.end(), pj, pj + lt);
    values = values && same(s.bare[lam], extension(KC.data() + (((size_t)lam * G) << lt), point.data(), lgk + lt));
  }
  for (size_t x = 0; x < true_ibs.size(); x++) values = values && same(s.bare[Lam + x], true_ibs[x]);
  return Verdict{s.challenges, row_holds(s, on_shown), values};
}

int main() {
  // a_s3_prep() and a_s3_run() require nv >= 2 and nb <= ga::MAXBLK (1056) and check neither.
  struct {
    int nv, nlow, nb;
    bool integers;
  } shapes3[] = {{2, 0, 1, false},  {4, 0, 3, false}, {6, 2, 3, false},   {9, 5, 2, false},
                 {11, 5, 5, false}, {7, 7, 2, false}, {4, 0, 300, false}, {3, 1, 257, false},
                 {6, 2, 3, true},   {11, 5, 2, true}, {12, 6, 3, true}};
  for (auto &x : shapes3) {
    const std::string name = "SC3(" + std::to_string(x.nv) + " variables, the weights fixed on the low " +
                             std::to_string(x.nlow) + ", " + std::to_string(x.nb) + " pairs" +
                             (x.integers ? ", integers)" : ")");
    auto run = [&](Fault f) {
      return sc3(x.nv, x.nlow, x.nb, x.integers, f, name);
    };
    hold(run(NONE).accepts(), name + ": the honest run is accepted");
    Verdict v = run(ARRAY_CHANGED);
    hold(!v.row && v.values, name + ": an array changed after the sum was formed: the equation fails");
    v = run(FALSE_SUM);
    hold(!v.row && v.values, name + ": a false claimed sum: the equation fails");
  }
  // lt is at most 10 here, a round's degree at most 12. a_scd() allows lt up to 16; its branch for
  // degrees above 13 is not run by this test.
  struct {
    int lt, lgk, HA, Lam;
  } shapesD[] = {{2, 1, 1, 1}, {3, 1, 2, 2}, {4, 2, 3, 2}, {6, 3, 5, 4}, {10, 1, 2, 1}, {3, 1, 33, 2}, {2, 0, 70, 1}};
  for (auto &x : shapesD) {
    const std::string name = "SCD(lt " + std::to_string(x.lt) + ", lgk " + std::to_string(x.lgk) + ", HA " +
                             std::to_string(x.HA) + ", Lam " + std::to_string(x.Lam) + ")";
    auto run = [&](Fault f) {
      return scd(x.lt, x.lgk, x.HA, x.Lam, f, name);
    };
    hold(run(NONE).accepts(), name + ": the honest run is accepted");
    Verdict v = run(ARRAY_CHANGED);
    hold(!v.row && !v.values,
         name + ": a false index bit at the end: the equation fails, and the value is not the map's");
    v = run(FALSE_SUM);
    hold(!v.row && v.values, name + ": a false claimed sum: the equation fails");
  }
  return verdict_line();
}
