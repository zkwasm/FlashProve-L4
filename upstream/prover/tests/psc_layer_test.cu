// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests two consecutive product layers, proved on the GPU, against the verifier functions of the
// library blocks/, on a proof's hash chain (on the GPU: the transcript object `Tr`). A product
// layer is the sumcheck of sum_x eq(rho, x) L[x] R[x] = C. PtWPL (prover/hiding/product_test.cuh)
// proves it; check() verifies it by psc_verify() (blocks/iop/psc.h: PSC, the sumcheck with a
// hiding polynomial) and line() (blocks/iop/line.h: LINE, two claims on a map's halves into one).
// PSC uses one mask entry, an element of the prover's mask row, for each scalar sent. hw::Walk
// ("the walk", prover/hiding/walk.cuh) batches the prover's linear claims (`row` in identifiers
// and messages); the verifier must arrive at the same coefficients on the mask entries.
//
// The test shows that the GPU's messages are those of the protocol on this shape. It shows nothing
// about hiding (blocks/iop/tests/psc_hiding_test.cpp tests the library's CPU prover for that, over
// small fields) or about a sub-proof that calls the layers. One GPU, no weights. Exit code: the
// number of failed cases. See main() for the optional argument (no file is overwritten) and the
// build with the macro `SLOT_COUNT` defined and not 0. The calls of sc_door::cheat_...() are test
// hooks for a malicious prover, which do nothing here (prover/hiding/sumcheck_door.cuh).
// Build and run, from prover/: tests/unit.sh build tests/psc_layer_test.cu <directory> && <directory>/psc_layer_test [directory]
#include "hiding/product_test.cuh"
#include "hiding/place.cuh"
#include "hiding/walk.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chain/public.h"
#include "iop/line.h"
#include "iop/psc.h"
#include "numbers/blocks.h"
#include "tests/psc_layer_plan.h"

// SEND: the block that sends oracle messages (blocks/iop/send.h).
namespace blk {
struct SendAccess {  // in the protocol only SEND makes receipts; here the mask row is a plain array
  static SampleReceipt make(u32 oracle, u32 first, u32 count) { return SampleReceipt(oracle, first, count); }
};
}  // namespace blk

// The field element T with the operators that the templates of blocks/ need.
struct KT {
  T v;
  static KT zero() { return {F::zero()}; }
  static KT one() { return {F::one()}; }
  friend KT operator+(KT a, KT b) { return {F::add(a.v, b.v)}; }
  friend KT operator-(KT a, KT b) { return {F::sub(a.v, b.v)}; }
  friend KT operator*(KT a, KT b) { return {F::mul(a.v, b.v)}; }
  friend bool operator==(KT a, KT b) { return F::eq(a.v, b.v); }
};
static KT inv(KT a) { return {F::inv(a.v)}; }
// Conversions between T and the library's blk::E.
static T of(const blk::E &x) {
  const u64 c[EDEG] = {x.c[0].v, x.c[1].v, x.c[2].v};
  return ext_from(c);
}
static void to_bytes(const T &x, blk::u8 *bytes) {
  blk::E e;
  for (int k = 0; k < EDEG; k++) e.c[k].v = coord(x, k);
  blk::E_to_bytes(e, bytes);
}
// The channel of psc_verify() and line(): blk::VerifierRounds, with its elements converted to KT.
// It records the challenges that it draws.
struct Channel {
  blk::VerifierRounds &v;
  std::vector<T> drew; // every element, in order
  T eps = F::zero();   // the last result of `close_epsilon`
  KT kept(const blk::E &e) {
    drew.push_back(of(e));
    return {drew.back()};
  }
  bool recv(KT &x) {
    blk::E e;
    const bool ok = v.recv(e);
    x = {of(e)};
    return ok;
  }
  bool recv_hiding_sum(KT &x) {
    blk::E e;
    const bool ok = v.recv_hiding_sum(e);
    x = {of(e)};
    return ok;
  }
  bool ok() const { return v.ok(); }
  KT close_epsilon(const char *name) {
    const KT e = kept(v.close_epsilon(name));
    eps = e.v;
    return e;
  }
  struct Draw {
    blk::AlgebraicDraw d;
    Channel *ch;
    KT draw() { return ch->kept(d.draw()); }
    KT draw_nonzero() { return ch->kept(d.draw_nonzero()); }
  };
  Draw close_algebraic(const char *name) { return Draw{v.close_algebraic(name), this}; }
  void close_empty(const char *name) { v.close_empty(name); }
};

// A second write to a slot, for the test of its counter.
#if SLOT_COUNT
__global__ void k_second_use(T *slot) {
  (void)sc_door::claim(sc_door::Site::FOREST_ROOTS, F::zero(), slot);
}
#endif
// The verifier's first message: the first layer's point.
__global__ void k_point(Tr *tr, T *o, int n) {
  Tr t = *tr;
  tr_point(t, Round::ZERO, o, n);
  *tr = t;
}
static T synthetic(u64 h) {
  const u64 w[3] = {mix64(h + 1) % FIELD_P, mix64(h + 2) % FIELD_P, mix64(h + 3) % FIELD_P}; // FIELD_P: the field's prime
  return {w[0], w[1], w[2]};
}
// The multilinear extension of a (2^mu elements) at the point z; z[0] belongs to the top index bit.
static T value_at(std::vector<T> a, const T *z, int mu) {
  size_t n = a.size();
  for (int i = 0; i < mu; i++) {
    const size_t h = n / 2;
    for (size_t x = 0; x < h; x++) a[x] = F::add(a[x], F::mul(z[i], F::sub(a[x + h], a[x])));
    n = h;
  }
  return a[0];
}

// The next layer's point (tau, r_0, r_1, ..) and its true, unmasked claim (1 - tau) vL + tau vR.
__global__ void k_hand_down(const PtPLOut *o, int mu, T *rho, T *claim) {
  rho[0] = o->tau;
  for (int j = 0; j < mu; j++) rho[1 + j] = o->r[j];
  *claim = F::add(o->vL, F::mul(o->tau, F::sub(o->vR, o->vL)));
}

// The results of one run of the prover. The GPU's message array has planes of PAD_STRIDE elements
// (prover/hiding/sumcheck_door.cuh). Plane 0: the message slots, one for each sent scalar; 1 and 2:
// each slot's mask entry and scratch cell; 3, with SLOT_COUNT=1: its counters.
struct Run {
  std::vector<blk::u8> proof;
  std::vector<T> slots, mask, scratch; // the three planes of the message array, as far as the layers use them
  std::vector<T> drawn;                // every element that the kernels received, in order
  std::vector<T> walk;                 // the walk's coefficients on the mask row
  T claim;                             // the true value of the first layer's sum
  PtPLOut out[2];
};

// What check() records. eps, beta: PSC's challenges; tau: LINE's; theta: the batching challenge.
// (U), (E_j): PSC's linear claims, defined in blocks/iop/psc.h. row[2] is negated, as in the walk.
struct Seen {
  T eps[2], beta[2], tau[2], theta;
  std::vector<T> row[3]; // the coefficients of all mask entries in row (U) of each layer and in the claim that the second layer passes on
  std::vector<T> drew;
};
// Returns "" if the channel accepts the proof and every linear claim holds on `mask` and `arrays`
// (L, R of the first layer, then of the second), else what failed. psc_verify() itself fails only
// if the channel has rejected. After a claim has failed, check() reads on to the proof's end, like
// a verifier that checks its claims later; it returns the first failure unless the channel rejects.
static std::string check(int mu, const blk::Planned &plan, T claim, const std::vector<blk::u8> &proof, const std::vector<T> &mask,
                         const std::vector<T> *const arrays[4], Seen *seen = nullptr) {
  const blk::Summand<KT> S{2, 0, 2, [](const std::vector<KT> &, const std::vector<KT> &y) { return std::vector<KT>{y[0] * y[1]}; }};
  blk::MaskLedger ledger(blk::SendAccess::make(0, 0, (blk::u32)mask.size()));
  blk::Reader reader(proof.data(), proof.size());
  blk::Binder binder(PSC_LAYER_TEST_LABEL, plan);
  (void)binder.configuration(psc_layer_test_configuration());
  blk::VerifierSession session; // one chain, one verdict; it outlives the channel
  blk::VerifierRounds v(binder, reader, session);
  Channel ch{v};
  blk::PointClaim<KT> c{{}, {}, {claim}};
  {
    Channel::Draw from = ch.close_algebraic(blk::R_ZERO);
    for (int j = 0; j < mu; j++) c.point.push_back(from.draw());
  }
  std::string bad;
  blk::u64 rows = 0; // the rows that this run checks against the data and that the batching challenge batches
  KT end[2] = {KT::zero(), KT::zero()}, tau = KT::zero();
  for (int layer = 0; layer < 2; layer++) {
    const int m = mu + layer;
    const std::string where = layer ? ", second layer" : ", first layer";
    blk::SumClaim<KT> sum = blk::SumClaim<KT>::at_point(std::move(c), std::vector<unsigned>(m, 3));
    const size_t n = blk::psc_masks(sum, S);
    if (n != sc_door::layer_slots(m)) return "the number of slots" + where;
    blk::PscOutput<KT> out;
    if (!blk::psc_verify(ch, std::move(sum), S, ledger.take(n), out)) return "the verifier routine" + where;
    KT u = KT::zero();
    for (const auto &t : out.row_masks) u = u + t.coeff * KT{mask[t.slot.index()]};
    if (bad.empty() && !(u == out.row_target)) bad = "row (U)" + where;
    rows++;
    std::vector<T> point;
    for (const KT &x : out.point) point.push_back(x.v);
    for (int j = 0; j < 2 && bad.empty(); j++) {
      const auto &e = out.ends[j];
      end[j] = {value_at(*arrays[2 * layer + j], point.data(), m)};
      if (!(end[j] + e.masks[0].coeff * KT{mask[e.masks[0].slot.index()]} == e.value)) bad = (j ? "row (E1)" : "row (E0)") + where;
    }
    if (seen) {
      seen->eps[layer] = ch.eps, seen->beta[layer] = out.ends[0].masks[0].coeff.v;
      seen->row[layer].assign(mask.size(), F::zero());
      for (const auto &t : out.row_masks) seen->row[layer][t.slot.index()] = F::add(seen->row[layer][t.slot.index()], t.coeff.v);
    }
    c = blk::line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
    tau = c.point.back();
    if (seen) seen->tau[layer] = tau.v;
    std::rotate(c.point.begin(), c.point.end() - 1, c.point.end());  // the new coordinate belongs to the top bit and is bound first
  }
  if (bad.empty()) { // the claim that the second layer passes on: rows (E0) and (E1) of that layer combined under tau
    KT left = (KT::one() - tau) * end[0] + tau * end[1];
    for (const auto &t : c.masks) left = left + t.coeff * KT{mask[t.slot.index()]};
    if (!(left == c.value)) bad = "the claim that the second layer passes on";
  }
  rows++;
  Channel::Draw from = ch.close_algebraic(blk::R_BAT_SIGMA);
  from.d.require_rows(rows); // by this run's own count; the plan says 3
  const KT theta = from.draw();
  if (!v.finish() || !session.accepted()) return std::string("the channel: ") + (v.why() ? v.why() : "the chain was not accepted");
  if (seen) {
    seen->theta = theta.v, seen->drew = ch.drew;
    seen->row[2].assign(mask.size(), F::zero());
    for (const auto &t : c.masks) seen->row[2][t.slot.index()] = F::sub(seen->row[2][t.slot.index()], t.coeff.v);
  }
  return bad;
}

// Writes x as EDEG 64-bit words, in the machine's byte order.
static void put(FILE *f, const T &x) {
  for (int k = 0; k < EDEG; k++) {
    const u64 c = coord(x, k);
    fwrite(&c, 8, 1, f);
  }
}

int main(int argc, char **argv) {
  CK(cudaSetDevice(0));
  g_st = cudaStreamPerThread;
  const size_t planes = (3 + SLOT_COUNT) * PAD_STRIDE;
  T *msg;
  CK(cudaMalloc(&msg, planes * sizeof(T)));
  {
    const long long po = (long long)PAD_STRIDE;
    CK(cudaMemcpyToSymbol(PADOFF, &po, 8));
  }
  sc_door::cheat_from_the_environment();
  Stretch stretch;
  stretch.init(256, 256);
  int cases = 0, failed = 0;
  [[maybe_unused]] int changed = 0, unaltered_accepted = 0, counted = 0;
  // Case mu: layers of mu and mu + 1 variables, n = 4 mu + 12 scalars; the second starts from the
  // claim that the first passes on. mu goes beyond 16 because from 17 variables on PtWPL computes
  // a layer's first rounds differently, from Gram matrices. The arrays and the mask row are
  // synthetic and fixed.
  for (int mu = 1; mu <= 17; mu++) {
    PtWPL wp[2];
    wp[0].init(mu), wp[1].init(mu + 1);
    const size_t N = (size_t)1 << mu, n1 = sc_door::layer_slots(mu), n = n1 + sc_door::layer_slots(mu + 1);
    // a[2], a[3]: the second layer's L and R. Their entrywise product is a[0] followed by a[1], the
    // first layer's L and R.
    std::vector<T> a[4], LR(N);
    a[2].resize(2 * N), a[3].resize(2 * N), a[0].resize(N), a[1].resize(N);
    for (size_t x = 0; x < 2 * N; x++) {
      a[2][x] = synthetic(0x1000000ull * mu + 2 * x);
      a[3][x] = synthetic(0x1000000ull * mu + 2 * x + 1);
      a[x / N][x % N] = F::mul(a[2][x], a[3][x]);
    }
    for (size_t x = 0; x < N; x++) LR[x] = F::mul(a[0][x], a[1][x]);
    const std::vector<T> *const arrays[4] = {&a[0], &a[1], &a[2], &a[3]};
    T *dev[4], *rho_dev, *claim_dev, *LS0;
    for (int i = 0; i < 4; i++) {
      CK(cudaMalloc(&dev[i], a[i].size() * sizeof(T)));
      CK(cudaMemcpy(dev[i], a[i].data(), a[i].size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&rho_dev, 2 * 32 * sizeof(T)));
    CK(cudaMalloc(&claim_dev, 2 * sizeof(T)));
    CK(cudaMalloc(&LS0, n * sizeof(T)));
    blk::Plan laid(blk::setting_of_this_build());
    plan_psc_layer_test(laid, (unsigned)mu);
    const blk::Planned plan = laid.finish();
    // One run of the prover. The host lends the proof's chain to the GPU (Stretch,
    // prover/hiding/transcript_shell.cuh). When it takes the chain back, it replays every round
    // from the scalars sent, and exits unless the rounds are the planned ones and its challenges
    // the kernels'.
    auto run_once = [&] {
      Run r;
      r.mask.resize(n);
      for (size_t i = 0; i < n; i++) r.mask[i] = synthetic(0x77000000ull * mu + i);
      CK(cudaMemset(msg, 0, planes * sizeof(T)));
      CK(cudaMemcpy(msg + PAD_STRIDE, r.mask.data(), n * sizeof(T), cudaMemcpyHostToDevice));
      CK(cudaMemset(LS0, 0, n * sizeof(T)));
      for (PtWPL &w : wp) CK(cudaMemset(w.out, 0, sizeof(PtPLOut)));
      blk::Binder binder(PSC_LAYER_TEST_LABEL, plan);
      (void)binder.configuration(psc_layer_test_configuration());
      blk::Writer proof;
      blk::ProverRounds channel(binder, blk::Digest{}, proof); // the same random bytes in every run, so that two runs can be compared
      stretch.begin(channel, g_st);
      k_point<<<1, 1, 0, g_st>>>(stretch.tr, rho_dev, mu);
      CK(cudaStreamSynchronize(g_st));
      std::vector<T> rho(mu);
      CK(cudaMemcpy(rho.data(), rho_dev, mu * sizeof(T), cudaMemcpyDeviceToHost));
      r.claim = value_at(LR, rho.data(), mu);
      CK(cudaMemcpy(claim_dev, &r.claim, sizeof(T), cudaMemcpyHostToDevice));
      wp[0].prove(stretch.tr, dev[0], dev[1], rho_dev, claim_dev, msg);
      k_hand_down<<<1, 1, 0, g_st>>>(wp[0].out, mu, rho_dev + 32, claim_dev + 1);
      wp[1].prove(stretch.tr, dev[2], dev[3], rho_dev + 32, claim_dev + 1, msg + n1);
      CK(cudaDeviceSynchronize());
      // The walk batches three claims: (U) of each layer, and the claim that the second passes on.
      // The second layer starts from the form that the first passes on, so its (U) contains the
      // first layer's inner masks (the mask entries of its two end values).
      hw::Walk W;
      W.init(4096);
      W.scratch = msg + 2 * PAD_STRIDE;
      hw::Space slots_space;
      slots_space.lam = LS0, slots_space.len = n, slots_space.neg = 1;
      W.sp.push_back(slots_space);
      hw::AF form;
      for (int i = 0; i < 2; i++) {
        std::vector<const T *> challenges;
        for (int j = 0; j < mu + i; j++) challenges.push_back(&wp[i].out->r[j]);
        form = W.layer(W.psc(0, challenges, i ? n1 : 0, 2, 2, &wp[i].out->tau), form);
      }
      W.lin(form);
      W.seal();
      W.finish(stretch.tr, LS0);
      CK(cudaDeviceSynchronize());
      for (const blk::E &x : stretch.end(channel, g_st)) r.drawn.push_back(of(x)); // exits if a round is not the planned one
      channel.finish();
      r.proof.assign(proof.data(), proof.data() + proof.size());
      r.slots.resize(n), r.scratch.resize(n), r.walk.resize(n);
      CK(cudaMemcpy(r.slots.data(), msg, n * sizeof(T), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(r.scratch.data(), msg + 2 * PAD_STRIDE, n * sizeof(T), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(r.walk.data(), LS0, n * sizeof(T), cudaMemcpyDeviceToHost));
      for (int i = 0; i < 2; i++) CK(cudaMemcpy(&r.out[i], wp[i].out, sizeof(PtPLOut), cudaMemcpyDeviceToHost));
      return r;
    };
    sc_door::cheat_again();  // a test hook, empty in this repository; before each case's first run
    const Run first = run_once();
    [[maybe_unused]] const bool struck = sc_door::cheat_struck();
    const Run second = run_once();
    Seen seen;
    std::string why = check(mu, plan, first.claim, first.proof, first.mask, arrays, &seen);
    // The scalars follow the chain's random bytes in the proof, one after the other.
    const size_t head = blk::DIGEST_BYTES;
    if (why.empty() && first.proof.size() != head + n * blk::E_BYTES) why = "the proof's length is not that of the chain's random bytes and the layers' scalars";
    for (size_t i = 0; i < n && why.empty(); i++) {
      blk::u8 bytes[blk::E_BYTES];
      to_bytes(first.slots[i], bytes);
      if (memcmp(bytes, first.proof.data() + head + i * blk::E_BYTES, blk::E_BYTES)) why = "the proof holds another scalar than the slot: number " + std::to_string(i);
    }
    if (why.empty() && seen.drew.size() != first.drawn.size()) why = "the verifier draws another number of elements than the kernels received";
    for (size_t i = 0; i < seen.drew.size() && why.empty(); i++)
      if (!F::eq(seen.drew[i], first.drawn[i])) why = "the verifier draws another element than the kernels received: number " + std::to_string(i);
    for (int i = 0; i < 2 && why.empty(); i++) {
      const size_t at = i ? n1 : 0;
      if (!F::eq(seen.tau[i], first.out[i].tau)) why = "tau differs between the GPU and the host";
      else if (!F::eq(first.scratch[at], seen.eps[i])) why = "eps is not where the walk reads it";
      else if (!F::eq(first.scratch[at + 1 + 2 * (mu + i)], seen.beta[i])) why = "beta is not where the walk reads it";
    }
    if (why.empty() && check(mu, plan, F::add(first.claim, F::one()), first.proof, first.mask, arrays).empty()) why = "a false claim is accepted";
    if (why.empty()) {
      std::vector<blk::u8> proof = first.proof;
      proof[mu % head] ^= 1;
      if (check(mu, plan, first.claim, proof, first.mask, arrays).empty()) why = "a changed byte of the chain's random bytes is accepted";
    }
    // A scalar is changed in the proof's bytes, because these are all that the verifier reads of
    // the prover's messages.
    for (size_t i = 0; i < n && why.empty(); i++) {
      std::vector<blk::u8> proof = first.proof;
      std::vector<T> mask = first.mask;
      to_bytes(F::add(first.slots[i], F::one()), proof.data() + head + i * blk::E_BYTES);
      mask[i] = F::add(mask[i], F::one());
      changed++;
      if (check(mu, plan, first.claim, proof, first.mask, arrays).empty()) why = "a changed scalar is accepted: number " + std::to_string(i);
      else if (check(mu, plan, first.claim, first.proof, mask, arrays).empty()) why = "a changed mask entry is accepted: number " + std::to_string(i);
    }
    if (why.empty() && first.proof != second.proof) why = "a second run gives another proof";
    // Built with the macro `SLOT_COUNT` defined as 1 (tests/unit.sh --slot-count): every slot of
    // the layers must have been written once and the next four slots not at all.
#if SLOT_COUNT
    if (why.empty()) {  // the counters of the second run: run_once() has cleared all planes before it
      std::vector<T> counts(n + 4);
      auto written = [&](size_t i) { return coord(counts[i], 0); };  // the first of a slot's three counters
      CK(cudaMemcpy(counts.data(), msg + 3 * PAD_STRIDE, counts.size() * sizeof(T), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < counts.size() && why.empty(); i++)
        if (written(i) != (i < n ? 1u : 0u)) why = "slot " + std::to_string(i) + " was written " + std::to_string(written(i)) + " times";
      k_second_use<<<1, 1, 0, g_st>>>(msg + n / 2);
      CK(cudaStreamSynchronize(g_st));
      CK(cudaMemcpy(counts.data(), msg + 3 * PAD_STRIDE, counts.size() * sizeof(T), cudaMemcpyDeviceToHost));
      if (why.empty() && written(n / 2) != 2) why = "a second use of a slot is not counted";
      counted += why.empty();
    }
#endif
    if (why.empty()) {
      bool first_in_second = false;
      for (size_t i = 0; i < n && why.empty(); i++) { // row c has the factor theta^c
        const T want = F::add(seen.row[0][i], F::mul(seen.theta, F::add(seen.row[1][i], F::mul(seen.theta, seen.row[2][i]))));
        if (!F::eq(first.walk[i], want)) why = "the walk's coefficient of mask entry " + std::to_string(i);
        if (i < n1 && !F::eq(seen.row[1][i], F::zero())) first_in_second = true;
      }
      if (why.empty() && !first_in_second) why = "the second layer's row has no coefficient on an entry of the first layer";
    }
    // With an existing directory as argument, the test writes the file on which
    // verifier/tests/psc_layer_check.cpp runs `cp_verify`'s product-layer functions and compares
    // their coefficients with the walk's. The file is created, never overwritten: a file of the
    // same name makes the case fail.
    if (argc > 1 && why.empty()) {
      const std::string name = std::string(argv[1]) + "/layers_" + std::to_string(mu) + ".bin";
      FILE *f = fopen(name.c_str(), "wbx"); // "x": the file is created; an existing file is not overwritten and a link at that name is not followed
      if (!f) why = "cannot create " + name + " (the directory must exist and hold no file of that name)";
      else {
        const u64 sizes[3] = {(u64)mu, (u64)n, (u64)first.proof.size()};
        fwrite(sizes, 8, 3, f);
        put(f, first.claim);
        put(f, F::add(first.out[1].vL, F::mul(seen.tau[1], F::sub(first.out[1].vR, first.out[1].vL))));  // the true value of the claim that the second layer passes on
        for (const T &x : first.mask) put(f, x);
        for (const T &x : first.walk) put(f, x);
        fwrite(first.proof.data(), 1, first.proof.size(), f);
        if (fclose(f)) why = "cannot write " + name;
      }
    }
    const bool ok = why.empty();
    printf("two product layers, %d and %d variables: %s%s\n", mu, mu + 1, ok ? "ok" : "FAILS: ", why.c_str());
    fflush(stdout);
    cases++, failed += !ok;
    for (T *d : dev) cudaFree(d);
    cudaFree(rho_dev), cudaFree(claim_dev), cudaFree(LS0);
  }
#if SLOT_COUNT
  printf("psc_layer_test: the slots' counters were read: in %d of %d cases every slot was written once and a second writing was counted\n", counted, cases);
#endif
  printf("psc_layer_test: %d of %d cases passed (%d scalars and as many mask entries changed one at a time)\n", cases - failed, cases, changed);
  return failed;
}
