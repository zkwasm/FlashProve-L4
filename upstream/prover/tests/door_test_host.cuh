// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Helpers for stand-alone GPU tests, such as those of single sumchecks: test data, copies to and
// from the GPU, multilinear extensions computed from the definition, and struct Run, the message
// array and the transcript of one run of the kernels under test.
// Include it after prover/hiding/transcript.cuh, which provides `T` (an extension-field element)
// and `Tr` (the transcript). Compile from prover/ with -I../blocks -DBLK_FIELD=2.
#pragma once
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "numbers/blocks.h"
#include "chain/public.h"
#include "chain/rounds.h"

static int cases = 0, failed = 0;
static void hold(bool good, const std::string &what) {
  cases++;
  if (good) return;
  failed++;
  printf("FAILS: %s\n", what.c_str());
}
static int verdict_line() {
  printf("%s: %d of %d checks hold, at field %d\n", failed ? "FAILED" : "ok", cases - failed, cases,
         BLK_FIELD); // always 2 (blocks/base.h); the expected lines in SUITE.md end with it
  return failed ? 1 : 0;
}
// A small generator for test data (SplitMix64: Steele, Lea, Flood, "Fast splittable pseudorandom number generators",
// OOPSLA 2014). Not for anything secret.
static u64 state = 1;
static u64 word() {
  u64 z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// Rejection sampling in each coordinate, so that no residue is favoured. `FIELD_P` is the field's
// prime, p = 2^64 - 2^32 + 1.
static T element() {
  u64 k[EDEG];
  for (u64 &x : k) {
    do x = word() >> (64 - FIELD_BITS);
    while (x >= FIELD_P);
  }
  return ext_from(k);
}
static std::vector<T> elements(size_t n) {
  std::vector<T> v(n);
  for (T &x : v) x = element();
  return v;
}
static bool same(T a, T b) {
  bool s = true;
  for (int k = 0; k < EDEG; k++) s = s && coord(a, k) == coord(b, k);
  return s;
}
template <class X> static X *up(const std::vector<X> &v) {
  X *d;
  CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(X)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
  return d;
}
template <class X> static std::vector<X> down(const X *d, size_t n) {
  std::vector<X> v(n);
  CK(cudaMemcpy(v.data(), d, n * sizeof(X), cudaMemcpyDeviceToHost));
  return v;
}
// eq(z, x) for all x of n bits; coordinate 0 of z belongs to the top bit of x.
static std::vector<T> eq_table(const T *z, int n) {
  std::vector<T> e((size_t)1 << n);
  for (size_t x = 0; x < e.size(); x++) {
    T w = F::one();
    for (int v = 0; v < n; v++) w = F::mul(w, ((x >> (n - 1 - v)) & 1) ? z[v] : F::sub(F::one(), z[v]));
    e[x] = w;
  }
  return e;
}
static T extension(const T *table, const T *z, int n) {
  const std::vector<T> e = eq_table(z, n);
  T s = F::zero();
  for (size_t x = 0; x < e.size(); x++) s = F::add(s, F::mul(e[x], table[x]));
  return s;
}
// q[0] + q[1] x + ..
static T power_sum(const std::vector<T> &q, T x) {
  T s = F::zero();
  for (size_t k = q.size(); k-- > 0;) s = F::add(F::mul(s, x), q[k]);
  return s;
}

// Slot: where the kernels write one sent scalar. The mask entry of the same number is PAD elements
// later; the kernels form their masks from these entries.
constexpr long long PAD = 4096; // the length of a plane of the message array (Run::d)

// True if every slot of m (Run::d's first plane, or more) outside [first, first + n) is zero.
static bool only_these(const std::vector<T> &m, size_t first, size_t n) {
  bool good = true;
  for (size_t i = 0; i < (size_t)PAD; i++)
    if (i < first || i >= first + n) good = good && same(m[i], F::zero());
  return good;
}

// Draws Run::closing: see the constructor's closes_.
__global__ void k_closing_draw(Tr *tr, T *o) {
  Tr t = *tr;
  tr_point(t, Round::BAT_SIGMA, o, 1);
  *tr = t;
}

// Order of calls: the constructor; the kernels, on `d` and `tr`; sent(); message(), answer() and
// silent() in the order of the rounds; through().
// The transcript is a hash chain (blocks/chain/chain.h): the constructor starts a proof under a
// plan, the list of the chain's rounds, and lends the chain to the kernels; sent() takes it back.
// Not checked: (1) with `closes`, sent() launches k_closing_draw() in the default stream before it
// synchronises, so kernels in any other stream must have finished by then; (2) a build with
// SLOT_COUNT=1 (prover/tests/unit.sh --slot-count) counts each write to a slot 3 PAD elements
// after the slot, past the end of `d` (slot_count() in prover/hiding/sumcheck_door.cuh).
struct Run {
  T *d = nullptr;   // three planes: slots (zeros), mask entries (test data), scratch
  Tr *tr = nullptr; // what the kernels get
  T *closing = nullptr;
  bool closes = true;
  blk::Plan plan{blk::setting_goldilocks()};
  std::unique_ptr<blk::Binder> binder;
  blk::Writer proof;
  std::unique_ptr<blk::ProverRounds> channel;
  Stretch stretch;
  std::unique_ptr<blk::Reader> reader;
  // What the chains of one verdict share; here there is one. Declared before the verifier's
  // channel (`VerifierRounds`), which it outlives.
  std::unique_ptr<blk::VerifierSession> session;
  std::unique_ptr<blk::VerifierRounds> verifier;
  // fill: adds the run's rounds to the plan, in the order in which the kernels run them. closes_:
  // true if the kernels end with a prover's message; sent() then draws one more element
  // (`closing`), in a round that the constructor adds to the plan. False if their last step is a
  // draw that fill lists.
  explicit Run(const std::function<void(blk::Plan &)> &fill, bool closes_ = true) : closes(closes_) {
    std::vector<T> m(3 * PAD, F::zero());
    for (long long i = 0; i < PAD; i++) m[PAD + i] = element();
    d = up(m);
    closing = up(elements(1));
    const long long pad = PAD;
    CK(cudaMemcpyToSymbol(PADOFF, &pad, sizeof pad));
    fill(plan);
    if (closes) blk::plan_bat_sigma(plan, "the run's end", 1);
    binder = std::make_unique<blk::Binder>(1, plan.finish());
    {
      blk::Digest configuration{}; // every chain names one: made up here
      configuration.w[0] = 0x646f6f72u;
      (void)binder->configuration(configuration);
    }
    blk::Digest bytes{};
    bytes.w[0] = (blk::u32)word();
    channel = std::make_unique<blk::ProverRounds>(*binder, bytes, proof);
    stretch.init(2048, 2048);
    stretch.begin(*channel, cudaStreamPerThread);
    tr = stretch.tr;
  }
  // Ends the prover's side and prepares the verifier's.
  void sent() {
    if (closes) k_closing_draw<<<1, 1>>>(tr, closing);
    CK(cudaDeviceSynchronize());
    (void)stretch.end(*channel, cudaStreamPerThread); // ends the process if the rounds are not the planned ones
    channel->finish();
    reader = std::make_unique<blk::Reader>(proof.data(), proof.size());
    verifier.reset(); // an earlier run's channel goes before its session
    session = std::make_unique<blk::VerifierSession>();
    verifier = std::make_unique<blk::VerifierRounds>(*binder, *reader, *session);
  }
  // The verifier reads the next n scalars that the prover sent. slots: their place in the host's
  // copy of the slots' plane. They are read from the proof's bytes and overwrite that copy, so that
  // the test then computes on what the proof contains; in_slots becomes false if a read fails or a
  // scalar differs from the copy. So a fault injected into the copy only turns in_slots false: it
  // tests none of the verifier's checks.
  // hiding_sum: true for a hiding polynomial's sum, which must come last before an EPSILON round.
  bool in_slots = true;
  void message(T *slots, int n, bool hiding_sum = false) {
    for (int i = 0; i < n; i++) {
      blk::E e;
      in_slots = (hiding_sum ? verifier->recv_hiding_sum(e) : verifier->recv(e)) && in_slots;
      const u64 c[EDEG] = {e.c[0].v, e.c[1].v, e.c[2].v};
      in_slots = in_slots && same(ext_from(c), slots[i]);
      slots[i] = ext_from(c);
    }
  }
  // Returns the verifier's message of n elements. name: the round's name in the plan (blk::R_..).
  // kind: EPSILON fills only the first element, and COIN is the same as POINT.
  // rows: only for a round that batches linear claims (`rows` in the plan): how many the test
  // batches there. Unless the verifier has already rejected, the process ends if the plan's number
  // differs (require_rows()) or if rows is not given in such a round.
  enum Kind { POINT, COIN, EPSILON, NOT_ZERO };
  std::vector<T> answer(const char *name, int n, Kind kind = POINT, long long rows = -1) {
    std::vector<T> v(n);
    auto of = [](const blk::E &e) {
      const u64 c[EDEG] = {e.c[0].v, e.c[1].v, e.c[2].v};
      return ext_from(c);
    };
    if (kind == EPSILON)
      v[0] = of(verifier->close_epsilon(name));
    else {
      blk::AlgebraicDraw from = verifier->close_algebraic(name);
      if (rows >= 0) from.require_rows((blk::u64)rows);
      for (T &x : v) x = of(kind == NOT_ZERO ? from.draw_nonzero() : from.draw());
    }
    return v;
  }
  // A round in which the verifier's message is empty. It still makes a hash query.
  void silent(const char *name) {
    verifier->close_empty(name);
  }
  // True if the verifier's last draw equals `closing` (if `closes`), in_slots is true, nothing was
  // rejected and no byte of the proof is left over.
  bool through() {
    bool good = true;
    if (closes) good = same(answer(blk::R_BAT_SIGMA, 1, POINT, 1)[0], down(closing, 1)[0]); // rows = 1, as the constructor planned this round
    return good && in_slots && verifier->finish();
  }
};
