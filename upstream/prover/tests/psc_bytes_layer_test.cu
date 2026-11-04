// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests PtPL (prover/hiding/product_test.cuh), the GPU prover of one product layer on 16-bit keys,
// against the library's verifier functions on a proof's hash chain: psc_verify() (blocks/iop/psc.h:
// PSC, the sumcheck with a hiding polynomial) and line() (blocks/iop/line.h: LINE, which joins the
// claims on the layer's two halves). PtPL sends through sc_door (prover/hiding/sumcheck_door.cuh).
//
// For 2^(mu + 1) keys, mu at least 16, PtPL proves the claim
//   sum_c eq(rho, c) (cst - key[c]) (cst - key[2^mu + c]) = C.
// Its rounds bind the index bits in the order of range_order(). prover/tests/psc_layer_test.cu
// tests PtWPL, its counterpart on arrays of field elements.
//
// The test shows that PtPL's messages and challenges are those of PSC and LINE for mu = 16 .. 19
// (main() lists the checks). It shows nothing about hiding or about the layer sub-proof that runs
// PtPL (prover/prover/layer_prove.cuh). One GPU, no weights.
// Build and run, from prover/: tests/unit.sh both tests/psc_bytes_layer_test.cu <directory>
#include "hiding/product_test.cuh"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chain/public.h"
#include "iop/line.h"
#include "iop/psc.h"
#include "numbers/blocks.h"
#include "prover/plan_rounds.h"

// SEND: the protocol block that sends oracle messages (blocks/iop/send.h).
namespace blk {
struct SendAccess {  // in the protocol only SEND makes receipts; here the mask row is a plain array
  static SampleReceipt make(u32 oracle, u32 first, u32 count) { return SampleReceipt(oracle, first, count); }
};
}  // namespace blk

// The prover's extension-field element T with the operations that blocks/ requires of a field.
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
static T of(const blk::E &x) {
  const u64 c[EDEG] = {x.c[0].v, x.c[1].v, x.c[2].v};
  return ext_from(c);
}
static void to_bytes(const T &x, blk::u8 *bytes) {
  blk::E e;
  for (int k = 0; k < EDEG; k++) e.c[k].v = coord(x, k);
  blk::E_to_bytes(e, bytes);
}
// blk::VerifierRounds, the library's verifier channel, with elements as KT. It records its draws.
struct Channel {
  blk::VerifierRounds &v;
  std::vector<T> drew; // every element, in order
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
  KT close_epsilon(const char *name) { return kept(v.close_epsilon(name)); }
  struct Draw {
    blk::AlgebraicDraw d;
    Channel *ch;
    KT draw() { return ch->kept(d.draw()); }
    KT draw_nonzero() { return ch->kept(d.draw_nonzero()); }
  };
  Draw close_algebraic(const char *name) { return Draw{v.close_algebraic(name), this}; }
  void close_empty(const char *name) { v.close_empty(name); }
};

// Every chain names a protocol (a label) and a configuration. Both are synthetic values here.
constexpr blk::u64 LABEL = 6;
static blk::Digest configuration() {
  blk::Digest c{};
  c.w[0] = 0x70736279u;
  return c;
}

// The verifier's first message: the layer's point.
__global__ void k_point(Tr *tr, T *o, int n) {
  Tr t = *tr;
  tr_point(t, Round::ZERO, o, n);
  *tr = t;
}
static T synthetic(u64 h) {
  const u64 w[3] = {mix64(h + 1) % FIELD_P, mix64(h + 2) % FIELD_P, mix64(h + 3) % FIELD_P}; // FIELD_P: the field's prime
  return {w[0], w[1], w[2]};
}
// The multilinear value of the array a (2^mu elements) at the point z; z[0] belongs to the top index bit.
static T value_at(std::vector<T> a, const T *z, int mu) {
  size_t n = a.size();
  for (int i = 0; i < mu; i++) {
    const size_t h = n / 2;
    for (size_t x = 0; x < h; x++) a[x] = F::add(a[x], F::mul(z[i], F::sub(a[x + h], a[x])));
    n = h;
  }
  return a[0];
}

static T lift(u64 v) { return {field_from_u64(v), 0, 0}; }

// The results of one run of the prover. The host writes a plan (the list of the chain's rounds),
// opens a proof under it and lends the hash chain to the GPU as the transcript `Tr`
// (Stretch::begin()). k_point() draws the point rho and PtPL runs. Stretch::end() takes the chain
// back: it replays every round against the plan by name, size and number of draws, and the process
// exits at the first mismatch.
struct Run {
  std::vector<blk::u8> proof;
  std::vector<T> slots, mask;
  std::vector<T> drawn; // every element that the kernels received, in order
  T claim;              // the true value of the layer's sum
  PtPLOut out;
};

// o[j]: the index bit that round j of PtPL binds (the range order).
static std::vector<int> range_order(int mu) {
  std::vector<int> o = {2, 1, 0, 5, 4, 3, 8, 7, 6};
  for (int v = mu - 1; v >= 9; v--) o.push_back(v);
  return o;
}
static std::vector<int> plain_order(int mu) {
  std::vector<int> o;
  for (int v = mu - 1; v >= 0; v--) o.push_back(v);
  return o;
}
struct Seen {
  std::vector<T> r;  // the challenges; r[0] belongs to the top bit
  T vL, vR, tau;
  std::vector<T> drew;
};
// psc_verify() and line() run on the proof's bytes through `Channel`, with the coordinates of the
// equality point rho in the order of the rounds. The host checks the linear claims that
// psc_verify() returns (`row` in identifiers and messages; (U) and (E_j) in blocks/iop/psc.h)
// against the mask row and against L and R, the arrays cst - key of the two halves.
// Returns "" if all of this passes, else what rejected the proof. After a claim has failed, check()
// still reads to the proof's end, like a verifier whose claims are checked later; it returns that
// claim, unless the channel rejects the proof too.
static std::string check(int mu, const blk::Planned &plan, const std::vector<int> &order, T claim, const std::vector<blk::u8> &proof,
                         const std::vector<T> &mask, const std::vector<T> &L, const std::vector<T> &R, Seen *seen = nullptr) {
  const blk::Summand<KT> S{2, 0, 2, [](const std::vector<KT> &, const std::vector<KT> &y) { return std::vector<KT>{y[0] * y[1]}; }};
  blk::MaskLedger ledger(blk::SendAccess::make(0, 0, (blk::u32)mask.size()));
  blk::Reader reader(proof.data(), proof.size());
  blk::Binder binder(LABEL, plan);
  (void)binder.configuration(configuration());
  blk::VerifierSession session; // one chain, one verdict; it outlives the channel
  blk::VerifierRounds v(binder, reader, session);
  Channel ch{v};
  std::vector<KT> rho; // coordinate 0 belongs to the top bit
  {
    Channel::Draw from = ch.close_algebraic(blk::R_ZERO);
    for (int j = 0; j < mu; j++) rho.push_back(from.draw());
  }
  blk::PointClaim<KT> c{{}, {}, {claim}};
  for (int j = 0; j < mu; j++) c.point.push_back(rho[mu - 1 - order[j]]);
  blk::SumClaim<KT> sum = blk::SumClaim<KT>::at_point(std::move(c), std::vector<unsigned>(mu, 3));
  const size_t n = blk::psc_masks(sum, S);
  if (n != sc_door::layer_slots(mu) || n != mask.size()) return "the number of slots";
  blk::PscOutput<KT> out;
  if (!blk::psc_verify(ch, std::move(sum), S, ledger.take(n), out)) return "the verifier function";
  std::string bad;
  KT u = KT::zero();
  for (const auto &t : out.row_masks) u = u + t.coeff * KT{mask[t.slot.index()]};
  if (!(u == out.row_target)) bad = "row (U)";
  std::vector<T> point(mu);
  for (int j = 0; j < mu; j++) point[mu - 1 - order[j]] = out.point[j].v;
  T value[2] = {F::zero(), F::zero()};
  for (int j = 0; j < 2 && bad.empty(); j++) {
    const auto &e = out.ends[j];
    value[j] = value_at(j ? R : L, point.data(), mu);
    if (!(KT{value[j]} + e.masks[0].coeff * KT{mask[e.masks[0].slot.index()]} == e.value)) bad = j ? "row (E1)" : "row (E0)";
  }
  c = blk::line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
  if (!v.finish() || !session.accepted()) return std::string("the channel: ") + (v.why() ? v.why() : "the chain was not accepted");
  if (seen) seen->r = point, seen->vL = value[0], seen->vR = value[1], seen->tau = c.point.back().v, seen->drew = ch.drew;
  return bad;
}

// Checked for each mu and each of the four sets of keys:
//   - the proof is the chain's 32 random bytes followed by the 2 mu + 5 scalars that the GPU stored
//     (`Run::slots`);
//   - check() accepts, and every element that it draws is the one that the kernels received;
//   - the challenges that PtPL returns by bit position and its tau (the challenge of line()) are
//     the verifier's, and its two end values are the values of L and R at these challenges;
//   - check() rejects a false claim, a proof with any one scalar changed, a mask row with any one
//     entry changed, and the honest proof with the rounds read in the plain order (top bit first);
//   - a second run gives the same proof.
// A scalar is changed in the proof's bytes, all that the verifier reads of the prover's messages.
int main() {
  CK(cudaSetDevice(0));
  g_st = cudaStreamPerThread;
  const size_t planes = (3 + SLOT_COUNT) * PAD_STRIDE;
  T *msg;
  CK(cudaMalloc(&msg, planes * sizeof(T)));
  {
    const long long po = (long long)PAD_STRIDE;
    CK(cudaMemcpyToSymbol(PADOFF, &po, 8));
  }
  Stretch stretch;
  stretch.init(256, 256);
  static const char *const keys_are[4] = {"synthetic keys", "all keys 0", "all keys 65,535", "two equal halves"};
  int cases = 0, failed = 0, changed = 0;
  for (int mu = 16; mu <= 19; mu++)
    for (int kind = 0; kind < 4; kind++) {
      PtPL pl;
      pl.init(mu);
      const size_t N = (size_t)1 << mu, n = sc_door::layer_slots(mu);
      const T cst = synthetic(0x5100000ull + 16 * mu + kind);
      std::vector<uint16_t> keys(2 * N);
      for (size_t x = 0; x < 2 * N; x++)
        keys[x] = kind == 0 ? (uint16_t)mix64(0x2000000ull * mu + x) : kind == 1 ? 0 : kind == 2 ? 65535 : (uint16_t)mix64(0x3000000ull * mu + x % N);
      std::vector<T> L(N), R(N), LR(N);
      for (size_t x = 0; x < N; x++) {
        L[x] = F::sub(cst, lift(keys[x])), R[x] = F::sub(cst, lift(keys[N + x]));
        LR[x] = F::mul(L[x], R[x]);
      }
      uint16_t *keys_dev;
      T *rho_dev, *claim_dev;
      CK(cudaMalloc(&keys_dev, keys.size() * 2));
      CK(cudaMemcpy(keys_dev, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice));
      CK(cudaMalloc(&rho_dev, 32 * sizeof(T)));
      CK(cudaMalloc(&claim_dev, sizeof(T)));
      blk::Plan written(blk::setting_of_this_build());
      blk::plan_zero(written, "layer's point", (unsigned)mu);
      blk::plan_product_layer(written, "layer", (unsigned)mu);
      const blk::Planned plan = written.finish();
      auto run_once = [&] {
        Run r;
        r.mask.resize(n);
        for (size_t i = 0; i < n; i++) r.mask[i] = synthetic(0x79000000ull * mu + 1000 * kind + i);
        CK(cudaMemset(msg, 0, planes * sizeof(T)));
        CK(cudaMemcpy(msg + PAD_STRIDE, r.mask.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        CK(cudaMemset(pl.out, 0, sizeof(PtPLOut)));
        blk::Binder binder(LABEL, plan);
        (void)binder.configuration(configuration());
        blk::Writer proof;
        blk::ProverRounds channel(binder, blk::Digest{}, proof); // the same random bytes in every run, so that two runs can be compared
        stretch.begin(channel, g_st);
        k_point<<<1, 1, 0, g_st>>>(stretch.tr, rho_dev, mu);
        CK(cudaStreamSynchronize(g_st));
        std::vector<T> rho(mu);
        CK(cudaMemcpy(rho.data(), rho_dev, mu * sizeof(T), cudaMemcpyDeviceToHost));
        r.claim = value_at(LR, rho.data(), mu);
        CK(cudaMemcpy(claim_dev, &r.claim, sizeof(T), cudaMemcpyHostToDevice));
        pl.prove(stretch.tr, keys_dev, rho_dev, cst, claim_dev, msg);
        CK(cudaDeviceSynchronize());
        for (const blk::E &x : stretch.end(channel, g_st)) r.drawn.push_back(of(x)); // exits if a round is not the planned one
        channel.finish();
        r.proof.assign(proof.data(), proof.data() + proof.size());
        r.slots.resize(n);
        CK(cudaMemcpy(r.slots.data(), msg, n * sizeof(T), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&r.out, pl.out, sizeof(PtPLOut), cudaMemcpyDeviceToHost));
        return r;
      };
      const Run first = run_once(), second = run_once();
      const std::vector<int> order = range_order(mu);
      Seen seen;
      std::string why = check(mu, plan, order, first.claim, first.proof, first.mask, L, R, &seen);
      const size_t head = blk::DIGEST_BYTES;
      if (why.empty() && first.proof.size() != head + n * blk::E_BYTES) why = "the proof's length is not that of the chain's random bytes and the layer's scalars";
      for (size_t i = 0; i < n && why.empty(); i++) {
        blk::u8 bytes[blk::E_BYTES];
        to_bytes(first.slots[i], bytes);
        if (memcmp(bytes, first.proof.data() + head + i * blk::E_BYTES, blk::E_BYTES)) why = "the proof holds another scalar than the slot: number " + std::to_string(i);
      }
      if (why.empty() && seen.drew.size() != first.drawn.size()) why = "the verifier draws another number of elements than the kernels received";
      for (size_t i = 0; i < seen.drew.size() && why.empty(); i++)
        if (!F::eq(seen.drew[i], first.drawn[i])) why = "the verifier draws another element than the kernels received: number " + std::to_string(i);
      for (int j = 0; j < mu && why.empty(); j++)
        if (!F::eq(seen.r[j], first.out.r[j])) why = "the layer's challenge of coordinate " + std::to_string(j) + " is not the verifier's";
      if (why.empty() && (!F::eq(seen.vL, first.out.vL) || !F::eq(seen.vR, first.out.vR))) why = "the layer's end values are not the arrays' values at the point";
      if (why.empty() && !F::eq(seen.tau, first.out.tau)) why = "tau differs between the GPU and the host";
      if (why.empty() && check(mu, plan, order, F::add(first.claim, F::one()), first.proof, first.mask, L, R).empty()) why = "a false claim is accepted";
      if (why.empty()) { // same plan, bytes, mask row and claim: only the order of the point's coordinates differs, and row (U) must notice it
        const std::string by = check(mu, plan, plain_order(mu), first.claim, first.proof, first.mask, L, R);
        if (by != "row (U)") why = by.empty() ? "the rounds read in the plain order are accepted" : "the rounds read in the plain order are rejected by " + by + ", not by row (U)";
      }
      for (size_t i = 0; i < n && why.empty(); i++) {
        std::vector<blk::u8> proof = first.proof;
        std::vector<T> mask = first.mask;
        to_bytes(F::add(first.slots[i], F::one()), proof.data() + head + i * blk::E_BYTES);
        mask[i] = F::add(mask[i], F::one());
        changed++;
        if (check(mu, plan, order, first.claim, proof, first.mask, L, R).empty()) why = "a changed scalar is accepted: number " + std::to_string(i);
        else if (check(mu, plan, order, first.claim, first.proof, mask, L, R).empty()) why = "a changed mask entry is accepted: number " + std::to_string(i);
      }
      if (why.empty() && first.proof != second.proof) why = "a second run gives another proof";
      printf("a product layer on keys of 16 bits, %d variables, %s: %s%s\n", mu, keys_are[kind], why.empty() ? "ok" : "FAILS: ", why.c_str());
      fflush(stdout);
      cases++, failed += !why.empty();
      CK(cudaFree(keys_dev));
      CK(cudaFree(rho_dev));
      CK(cudaFree(claim_dev));
    }
  printf("psc_bytes_layer_test: %d of %d cases passed (%d scalars and as many mask entries changed one at a time)\n", cases - failed, cases, changed);
  return failed ? 1 : 0;
}
