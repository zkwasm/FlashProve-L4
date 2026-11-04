// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests `Tr`, the sub-proofs' transcript on the GPU (prover/hiding/transcript_shell.cuh), with the
// functions of namespace sc_door (prover/hiding/sumcheck_door.cuh) for PSC, the sumcheck with a
// hiding polynomial (`psc` in identifiers). The transcript is a hash chain, which the host runs
// (blocks/chain/rounds.h) under a plan, the list of its rounds. The host lends the chain to the GPU
// for a stretch of rounds (Stretch::begin()). Stretch::end() takes it back: the host replays each
// round, and the process stops unless the rounds match the plan, by name, and both chains end at
// the same value.
// "By lanes": the first SEND2_N = 4 lanes of a warp call a function together. They share the hash
// only if nothing is open: a round has closed every scalar sent so far and the message that the
// host lent. Otherwise lane 0 runs the round.
// The test shows that the GPU's transcript, sc_door and the plan agree on one format. It shows
// nothing about hiding, speed or a sub-proof's kernels.
// Build and run, from prover/: ./build.sh psc_shell_test && build/psc_shell_test [a-wrong-place]
// With `a-wrong-place` the test injects a fault into itself: the first scalar of a second stretch
// lies one slot after the place where the first stretch ended. The trap in tr_open() must stop the
// kernel. That run ends in a GPU fault on purpose. What the fault does to other work on the same
// GPU is not checked here: use a GPU that nothing else uses.
// BLK_PLANS_BY_HAND_FOR_A_TEST is a switch for tests: it makes public the members of blk::Plan that
// add messages and rounds, which this test calls. It must be defined before the first include. This
// file defines it; no build command does. A program that has it is not for serving: blocks/fence.h
// lists it, puts a marker string into the program and fails a build that defines SERVING_BUILD.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets a test build plans by hand (blocks/numbers/plan.h)
#include "hiding/transcript.cuh"
#include <cstdio>
#include <string>
#include <vector>

#include "chain/public.h"
#include "iop/line.h"
#include "iop/psc.h"
#include "numbers/blocks.h"

// SEND: the block that sends oracle messages (blocks/iop/send.h).
namespace blk {
struct SendAccess {  // in the protocol only SEND makes receipts; here the mask row is a plain array
  static SampleReceipt make(u32 oracle, u32 first, u32 count) { return SampleReceipt(oracle, first, count); }
};
}  // namespace blk

// The prover's field element T with the operators that blocks/iop/ needs.
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
static KT of(const blk::E &x) {
  const u64 c[EDEG] = {x.c[0].v, x.c[1].v, x.c[2].v};
  return {ext_from(c)};
}
// blk::VerifierRounds, returning KT instead of blk::E.
struct Channel {
  blk::VerifierRounds &v;
  bool recv(KT &x) {
    blk::E e;
    const bool ok = v.recv(e);
    x = of(e);
    return ok;
  }
  bool recv_hiding_sum(KT &x) {
    blk::E e;
    const bool ok = v.recv_hiding_sum(e);
    x = of(e);
    return ok;
  }
  bool ok() const { return v.ok(); }
  KT close_epsilon(const char *name) { return of(v.close_epsilon(name)); }
  struct Draw {
    blk::AlgebraicDraw d;
    KT draw() { return of(d.draw()); }
    KT draw_nonzero() { return of(d.draw_nonzero()); }
  };
  Draw close_algebraic(const char *name) { return Draw{v.close_algebraic(name)}; }
  void close_empty(const char *name) { v.close_empty(name); }
};

// A product tree of 2^n leaves, n <= MOST. msg: PADOFF slots for sent scalars, then the mask row,
// then sc_door's scratch space.
constexpr int MOST = 6;
struct Tree {
  const T *level[MOST + 1];  // level[i][x] = level[i + 1][x] * level[i + 1][x + 2^i], x < 2^i
  int n;
  T *msg;
};
// One warp proves the product tree `a` level by level, from the root: a sumcheck in i variables
// and the challenge tau reduce the claim on level i to one on level i + 1. Levels with even i use
// the calls by lanes (except sc_door::begin()), the others the calls for one thread. After the last
// level the kernel draws the batching challenge *theta.
// work: space for 2^(n + 1) elements. got: every challenge that the kernel gets, in order.
__global__ void k_tree(Tr *tr, Tree a, T *work, T *theta, T *got) {
  using namespace sc_door;
  TR_IN_SHARED(t);
  __shared__ T rho[MOST + 1], r[MOST + 1], g[2], claim;
  const int u = threadIdx.x;
  const Site site = Site::LAYER_LATE;
  tr_copy_by_lanes(&t, tr, u, 32);
  if (u == 0) claim = a.level[0][0];
  __syncwarp();
  T *slots = a.msg, *A = work, *B = work + ((size_t)1 << a.n);
  int n_got = 0;  // thread 0's
  for (int i = 0; i < a.n; i++) {
    const bool lanes = (i & 1) == 0;
    if (u == 0) {
      for (int x = 0; x < 1 << i; x++) A[x] = a.level[i + 1][x], B[x] = a.level[i + 1][x + (1 << i)];
      got[n_got++] = begin<Form::EQ2>(site, t, slots, i, [&](int j) { return rho[j]; }, claim, 2);
    }
    __syncwarp();
    for (int j = 0; j < i; j++) {
      const int h = 1 << (i - j - 1);
      if (u == 0) {  // round j's polynomial without equality factors 0 to j: g(0), X^2 coefficient
        g[0] = g[1] = F::zero();
        for (int x = 0; x < h; x++) {
          T w = F::one();
          for (int q = j + 1; q < i; q++) w = F::mul(w, (x >> (i - 1 - q)) & 1 ? rho[q] : F::sub(F::one(), rho[q]));
          g[0] = F::add(g[0], F::mul(w, F::mul(A[x], B[x])));
          g[1] = F::add(g[1], F::mul(w, F::mul(F::sub(A[x + h], A[x]), F::sub(B[x + h], B[x]))));
        }
      }
      __syncwarp();
      if (lanes) {
        if (u < SEND2_N) {
          const T c = round_by_lanes<Form::EQ2>(site, j, &t, g[0], g[1]);
          if (u == 0) r[j] = c;
        }
      } else if (u == 0) {
        r[j] = round<Form::EQ2, sc_door::Draw::COIN>(site, j, t, g);
      }
      __syncwarp();
      if (u == 0) got[n_got++] = r[j];
      if (u == 0)
        for (int x = 0; x < h; x++) {
          A[x] = F::add(A[x], F::mul(r[j], F::sub(A[x + h], A[x])));
          B[x] = F::add(B[x], F::mul(r[j], F::sub(B[x + h], B[x])));
        }
      __syncwarp();
    }
    T tau = F::zero();
    if (lanes) {
      if (u < SEND2_N) tau = layer_ends_by_lanes(site, &t, A[0], B[0]);
    } else if (u == 0) {
      T in_y[2];
      psc::product_in_y<F>(A[0], B[0], inner_mask(t, 0), inner_mask(t, 1), in_y);
      got[n_got++] = further(site, t, in_y);
      const T v[2] = {A[0], B[0]};
      tau = ends<sc_door::Draw::COIN>(site, t, v, 2, Round::LINE);
    }
    if (u == 0) {
      if (lanes) got[n_got++] = t.p.beta;  // `layer_ends_by_lanes` returns only tau
      got[n_got++] = tau;
      claim = F::add(A[0], F::mul(tau, F::sub(B[0], A[0])));
      for (int j = i; j > 0; j--) rho[j] = r[j - 1];  // tau: the top bit's coordinate, bound first
      rho[0] = tau;
    }
    slots += layer_slots(i);
    __syncwarp();
  }
  if (u == 0) tr_point(t, Round::BAT_SIGMA, theta, 1), got[n_got++] = *theta;
  __syncwarp();
  tr_copy_by_lanes(tr, &t, u, 32);
}

// The round by lanes in its three states: the host's message open, a scalar of the kernel open,
// nothing open. Sends slots[0 .. 8), where | is a round's end: two | one, left open, and two | two
// | one, left open at the end. out: the three challenges. One warp.
__global__ void k_three_states(Tr *tr, const T *slots, T *out) {
  TR_IN_SHARED(t);
  const int u = threadIdx.x;
  sc_door::tr_copy_by_lanes(&t, tr, u, 32);
  __syncwarp();
  for (int i = 0; i < 3; i++) {
    const T *at = slots + (i == 0 ? 0 : i == 1 ? 3 : 5);
    if (i == 1 && u == 0) tr_open(t, slots + 2, 1);
    __syncwarp();
    if (u < SEND2_N) {
      const T r = tr_round2_by_lanes(&t, Round::PSC_ROUND, at[0], at[1], at);
      if (u == SEND2_N - 1) out[i] = r;  // a lane other than lane 0
    }
    __syncwarp();
  }
  if (u == 0) tr_open(t, slots + 7, 1);
  __syncwarp();
  sc_door::tr_copy_by_lanes(tr, &t, u, 32);
}
// k_every_length() runs the rounds by lanes for N scalars (lanes_case<N>()): with a challenge
// (printed: an answer) for N = 1, 2, 3, 0 and without one (`empty`) for N = 1, 2, 3; first with
// nothing open, then (`open`) each after a further scalar that the kernel leaves open. Lane 0 runs
// the kernel's first round too, so this test never runs the four-lane path of
// tr_round_by_lanes<1>().
// slots: the 2 x 12 + 7 = 31 scalars in the order of sending. out: the eight challenges. One warp.
template <int N> __device__ void lanes_case(Tr &t, int u, bool open, bool empty, const T *&at, T *&out) {
  if (open) {
    if (u == 0) tr_open(t, at, 1);
    at++;
    __syncwarp();
  }
  if (u < SEND2_N) {
    if constexpr (N > 0)
      if (empty) tr_empty_by_lanes<N>(&t, Round::PSC_VALUES, at, at);
    if (!empty) {
      const T r = tr_round_by_lanes<N>(&t, Round::PSC_ROUND, at, at);
      if (u == SEND2_N - 1) *out = r; // a lane other than lane 0
    }
  }
  at += N;
  if (!empty) out++;
  __syncwarp();
}
__global__ void k_every_length(Tr *tr, const T *slots, T *out) {
  TR_IN_SHARED(t);
  const int u = threadIdx.x;
  sc_door::tr_copy_by_lanes(&t, tr, u, 32);
  __syncwarp();
  for (int open = 0; open < 2; open++) {
    lanes_case<1>(t, u, open, false, slots, out); // the first one closes the lent message (lane 0)
    lanes_case<2>(t, u, open, false, slots, out);
    lanes_case<3>(t, u, open, false, slots, out);
    lanes_case<0>(t, u, open, false, slots, out);
    lanes_case<1>(t, u, open, true, slots, out);
    lanes_case<2>(t, u, open, true, slots, out);
    lanes_case<3>(t, u, open, true, slots, out);
  }
  sc_door::tr_copy_by_lanes(tr, &t, u, 32);
}
// The second stretch after k_three_states(): two scalars and their challenge. One thread.
__global__ void k_one_round(Tr *tr, const T *slots, T *out) {
  Tr t = *tr;
  *out = tr_round<2>(t, Round::PSC_ROUND, slots, slots);
  *tr = t;
}
__global__ void k_minus_one(T *o) {
  *o = F::sub(F::zero(), F::one());
}
static T made_up(u64 h) {
  const u64 w[3] = {mix64(h + 1) % FIELD_P, mix64(h + 2) % FIELD_P, mix64(h + 3) % FIELD_P}; // FIELD_P: the field's prime, 2^64 - 2^32 + 1
  return {w[0], w[1], w[2]};
}
// The multilinear extension of a at z; z[0] goes with the top bit of the index.
static KT top_first(std::vector<KT> a, const std::vector<KT> &z) {
  for (const KT &c : z) {
    const size_t h = a.size() / 2;
    for (size_t x = 0; x < h; x++) a[x] = a[x] + c * (a[x + h] - a[x]);
    a.resize(h);
  }
  return a[0];
}
static blk::Setting setting() { return blk::setting_goldilocks(); }
// The host's scalar has no round of its own: it is open when the chain is lent and joins the
// kernel's first message.
static void host_scalar(blk::Plan &plan) {
  plan.send_scalars("left open by the host", 1);
}
// A chain's first query (the hash input that starts it) contains a configuration digest: in the
// programs a hash of the public files, here an arbitrary value, the same on both sides.
static blk::Digest made_up_configuration() {
  blk::Digest d{};
  d.w[0] = 0x70736373u;
  return d;
}
static blk::Plan plan_of(int n) {
  blk::Plan plan(setting());
  host_scalar(plan);
  for (int i = 0; i < n; i++) {
    blk::PscSize size;
    size.degree.assign(i, 3), size.degree_Y = 2, size.values = 2, size.with_Y = true;
    blk::plan_psc(plan, "level", size);
    blk::plan_line(plan, "level");
  }
  blk::plan_bat_sigma(plan, "rows", (blk::u64)n + 1);
  return plan;
}
static blk::E as_E(const T &x) {
  blk::E e;
  for (int k = 0; k < EDEG; k++) e.c[k].v = coord(x, k);
  return e;
}
// Runs the verifier of blocks/ (blk::psc_verify()) on a proof of k_tree(), and checks the linear
// claims (`row` in identifiers and messages) that it leaves against the mask row and the leaves.
// Returns "", or what failed. *theta: the batching challenge.
static std::string verify(int n, const blk::Planned &plan, const std::vector<blk::u8> &proof, KT stated_root, const std::vector<KT> &leaves,
                          const std::vector<T> &mask, KT *theta) {
  blk::Reader reader(proof.data(), proof.size());
  blk::Binder binder(1, plan);  // the first query covers plan, configuration digest and root
  (void)binder.configuration(made_up_configuration());
  const blk::Public<blk::E> stated = binder.statement("root", as_E(stated_root.v));
  blk::VerifierSession session; // one chain, one verdict
  blk::VerifierRounds v(binder, reader, session);
  const KT root = of(v.get(stated));
  Channel ch{v};
  KT x;
  if (!ch.recv(x)) return "the channel";
  const blk::Summand<KT> S{2, 0, 2, [](const std::vector<KT> &, const std::vector<KT> &y) { return std::vector<KT>{y[0] * y[1]}; }};
  blk::MaskLedger ledger(blk::SendAccess::make(0, 0, (blk::u32)mask.size()));
  blk::PointClaim<KT> c{{}, {}, root};
  std::string bad;
  blk::u64 rows = 0; // linear claims checked so far: what the batching challenge would batch
  for (int i = 0; i < n; i++) {
    blk::SumClaim<KT> claim = blk::SumClaim<KT>::at_point(std::move(c), std::vector<unsigned>(i, 3));
    const size_t fresh = blk::psc_masks(claim, S);
    blk::PscOutput<KT> out;
    if (!blk::psc_verify(ch, std::move(claim), S, ledger.take(fresh), out)) return "the verifier routine, level " + std::to_string(i);
    KT u = KT::zero();
    for (const auto &m : out.row_masks) u = u + m.coeff * KT{mask[m.slot.index()]};
    if (bad.empty() && !(u == out.row_target)) bad = "row (U) of level " + std::to_string(i);
    rows++;
    c = blk::line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
    std::rotate(c.point.begin(), c.point.end() - 1, c.point.end());
  }
  KT left = top_first(leaves, c.point);
  for (const auto &m : c.masks) left = left + m.coeff * KT{mask[m.slot.index()]};
  if (bad.empty() && !(left == c.value)) bad = "the row on the leaves";
  rows++;
  Channel::Draw from = ch.close_algebraic(blk::R_BAT_SIGMA);
  from.d.require_rows(rows); // stops the process unless the plan says n + 1 or `v` has rejected
  const KT th = from.draw();
  if (theta) *theta = th;
  if (!v.finish() || !session.accepted()) return "the channel";
  return bad;
}

int main(int argc, char **argv) {
  const bool wrong_place = argc > 1 && std::string(argv[1]) == "a-wrong-place";
  if (argc > 1 && !wrong_place) return printf("usage: psc_shell_test [a-wrong-place]\n"), 2;
  CK(cudaSetDevice(0));
  {
    // Prints 0 - 1 from a kernel, the host and blocks/, which shows the field of each.
    T *there, here = F::sub(F::zero(), F::one()), fetched;
    CK(cudaMalloc(&there, sizeof(T)));
    k_minus_one<<<1, 1>>>(there);
    CK(cudaMemcpy(&fetched, there, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(there);
    printf("0 - 1 = 0x%llx in a kernel, 0x%llx on the host, 0x%llx in blocks/\n", (unsigned long long)coord(fetched, 0), (unsigned long long)coord(here, 0),
           (unsigned long long)(blk::E::zero() - blk::E::one()).c[0].v);
  }
  T *msg, *work, *theta, *got;
  CK(cudaMalloc(&msg, 3 * PAD_STRIDE * sizeof(T)));
  CK(cudaMalloc(&work, (2 << MOST) * sizeof(T)));
  CK(cudaMalloc(&theta, sizeof(T)));
  CK(cudaMalloc(&got, 256 * sizeof(T)));
  {
    const long long po = (long long)PAD_STRIDE;
    CK(cudaMemcpyToSymbol(PADOFF, &po, 8));
  }
  Stretch stretch;
  stretch.init(256, 256);
  int failures = 0, changed = 0;
  // Trees of 2 to 2^MOST leaves. Required: the kernel got the challenges that the host's replay
  // drew; verify() accepts and draws the same batching challenge; it fails for a false root, and
  // for each copy of the proof with the lowest bit of one byte flipped.
  for (int n = 1; n <= MOST; n++) {
    std::vector<std::vector<T>> level(n + 1);
    level[n].resize((size_t)1 << n);
    for (size_t x = 0; x < level[n].size(); x++) level[n][x] = made_up(1000 * n + x);
    for (int i = n - 1; i >= 0; i--) {
      level[i].resize((size_t)1 << i);
      for (size_t x = 0; x < level[i].size(); x++) level[i][x] = F::mul(level[i + 1][x], level[i + 1][x + level[i].size()]);
    }
    Tree tree{};
    tree.n = n, tree.msg = msg;
    std::vector<T *> held;
    for (int i = 0; i <= n; i++) {
      T *d;
      CK(cudaMalloc(&d, level[i].size() * sizeof(T)));
      CK(cudaMemcpy(d, level[i].data(), level[i].size() * sizeof(T), cudaMemcpyHostToDevice));
      tree.level[i] = d, held.push_back(d);
    }
    size_t slots = 0;
    for (int i = 0; i < n; i++) slots += sc_door::layer_slots(i);
    std::vector<T> mask(slots);
    for (size_t i = 0; i < slots; i++) mask[i] = made_up(77000 * n + i);
    CK(cudaMemset(msg, 0, 3 * PAD_STRIDE * sizeof(T)));
    CK(cudaMemcpy(msg + PAD_STRIDE, mask.data(), slots * sizeof(T), cudaMemcpyHostToDevice));

    const blk::Planned plan = plan_of(n).finish();
    blk::Writer proof;
    blk::Binder binder(1, plan);
    (void)binder.configuration(made_up_configuration());
    (void)binder.statement("root", as_E(level[0][0]));
    blk::ProverRounds ch(binder, blk::Digest{}, proof);
    ch.send(blk::E::one());
    stretch.begin(ch, cudaStreamPerThread);
    k_tree<<<1, 32, 0, cudaStreamPerThread>>>(stretch.tr, tree, work, theta, got);
    CK(cudaGetLastError());
    const std::vector<blk::E> drawn = stretch.end(ch, cudaStreamPerThread);  // ends the process if the chains differ
    ch.finish();
    T theta_there;
    CK(cudaMemcpy(&theta_there, theta, sizeof(T), cudaMemcpyDeviceToHost));

    std::vector<KT> leaves;
    for (const T &x : level[n]) leaves.push_back({x});
    const std::vector<blk::u8> bytes(proof.data(), proof.data() + proof.size());
    KT theta_here = KT::zero();
    std::string why = verify(n, plan, bytes, {level[0][0]}, leaves, mask, &theta_here);
    if (why.empty() && !(theta_here == KT{theta_there})) why = "the verifier draws another batching challenge than the GPU";
    std::vector<T> there(drawn.size());
    CK(cudaMemcpy(there.data(), got, there.size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < drawn.size() && why.empty(); i++)
      if (!(of(drawn[i]) == KT{there[i]})) why = "the kernel got another element than the host's replay drew: number " + std::to_string(i);
    if (why.empty() && verify(n, plan, bytes, KT{level[0][0]} + KT::one(), leaves, mask, nullptr).empty()) why = "a false root is accepted";
    for (size_t i = 0; i < bytes.size() && why.empty(); i++) {
      std::vector<blk::u8> other = bytes;
      other[i] ^= 1;
      changed++;
      if (verify(n, plan, other, {level[0][0]}, leaves, mask, nullptr).empty()) why = "accepted with byte " + std::to_string(i) + " changed";
    }
    printf("a tree of %d leaves on the chain: %s%s (%zu scalars, %zu elements drawn, proof of %zu bytes)\n", 1 << n, why.empty() ? "ok" : "FAILS: ",
           why.c_str(), slots, drawn.size(), bytes.size());
    failures += !why.empty();
    for (T *d : held) cudaFree(d);
  }
  // k_three_states(), a scalar from the host, then k_one_round(), whose round closes a message of
  // four scalars: one left open by the first stretch, the host's, its own two. The GPU's scalars
  // must be in consecutive message slots over both stretches: Stretch::begin() gets the address at
  // which the first ended (Stretch::next_slot); next_slot and Stretch::sent_scalars are checked
  // after each stretch. The verifier must draw the kernels' challenges. With `a-wrong-place` the
  // second stretch starts one slot further on: the trap in tr_open() must stop the kernel, and with
  // it the process. Either way the exit code is 1: the last line is `CK`'s "CUDA error ..." if the
  // trap worked, "FAILS: ..." if not.
  {
    blk::Plan plan(setting());
    host_scalar(plan);
    for (unsigned n : {2u, 3u, 2u, 4u}) {
      plan.send_scalars("scalars", n);
      plan.close(blk::R_PSC_ROUND, blk::RoundKind::ALGEBRAIC, blk::ErrorKind::FIELD, blk::Bound((blk::u64)0), blk::Priced::NOTHING).draw_E = 1;
    }
    std::vector<T> ten(10), there(4);
    for (size_t i = 0; i < 10; i++) ten[i] = made_up(5000 + i);
    CK(cudaMemcpy(msg, ten.data(), 10 * sizeof(T), cudaMemcpyHostToDevice));
    blk::Writer proof;
    blk::Binder binder(2, plan), verifier_binds(2, plan);
    (void)binder.configuration(made_up_configuration()), (void)verifier_binds.configuration(made_up_configuration());
    blk::ProverRounds ch(binder, blk::Digest{}, proof);
    ch.send(blk::E::one());
    stretch.begin(ch, cudaStreamPerThread);
    k_three_states<<<1, 32, 0, cudaStreamPerThread>>>(stretch.tr, msg, work);
    CK(cudaGetLastError());
    (void)stretch.end(ch, cudaStreamPerThread);
    bool places = stretch.next_slot == msg + 8 && stretch.sent_scalars == 8;
    ch.send(blk::E::one());
    stretch.begin(ch, cudaStreamPerThread, stretch.next_slot);
    k_one_round<<<1, 1, 0, cudaStreamPerThread>>>(stretch.tr, msg + 8 + (wrong_place ? 1 : 0), work + 3);
    CK(cudaGetLastError());
    (void)stretch.end(ch, cudaStreamPerThread); // if the kernel has trapped, `CK` exits in here
    if (wrong_place) {
      printf("FAILS: a second stretch whose first scalar stands at another place than where the first ended was NOT STOPPED\n");
      return 1;
    }
    places = places && stretch.next_slot == msg + 10 && stretch.sent_scalars == 2;
    ch.finish();
    CK(cudaMemcpy(there.data(), work, 4 * sizeof(T), cudaMemcpyDeviceToHost));
    blk::Reader reader(proof.data(), proof.size());
    blk::VerifierSession session;
    blk::VerifierRounds v(verifier_binds, reader, session);
    Channel c{v};
    KT x;
    bool good = c.recv(x);
    for (int i = 0; i < 4 && good; i++) {
      for (int j = 0; j < (i == 1 ? 3 : i == 3 ? 4 : 2) && good; j++) good = c.recv(x);
      good = good && c.close_algebraic(blk::R_PSC_ROUND).draw() == KT{there[i]};
    }
    good = good && v.finish() && session.accepted() && places;
    printf("the round by lanes with the host's message open, a scalar open, nothing open; a scalar open at a stretch's end; a second stretch at the place that the first hands on: %s\n",
           good ? "ok" : "FAILS");
    failures += !good;
  }
  // k_every_length(): the verifier must read the scalars sent and draw the kernel's challenges.
  {
    blk::Plan plan(setting());
    for (unsigned open = 0; open < 2; open++) {
      for (unsigned n : {1u, 2u, 3u, 0u}) {
        plan.send_scalars("scalars", n + open);
        plan.close(blk::R_PSC_ROUND, blk::RoundKind::ALGEBRAIC, blk::ErrorKind::FIELD, blk::Bound((blk::u64)0), blk::Priced::NOTHING).draw_E = 1;
      }
      for (unsigned n = 1; n < 4; n++) {
        plan.send_scalars("scalars", n + open);
        plan.close_none(blk::R_PSC_VALUES);
      }
    }
    std::vector<T> all(31), there(8);
    for (size_t i = 0; i < all.size(); i++) all[i] = made_up(7000 + i);
    CK(cudaMemcpy(msg, all.data(), all.size() * sizeof(T), cudaMemcpyHostToDevice));
    blk::Writer proof;
    blk::Binder binder(2, plan), verifier_binds(2, plan);
    (void)binder.configuration(made_up_configuration()), (void)verifier_binds.configuration(made_up_configuration());
    blk::ProverRounds ch(binder, blk::Digest{}, proof);
    stretch.begin(ch, cudaStreamPerThread);
    k_every_length<<<1, 32, 0, cudaStreamPerThread>>>(stretch.tr, msg, work);
    CK(cudaGetLastError());
    (void)stretch.end(ch, cudaStreamPerThread);
    ch.finish();
    CK(cudaMemcpy(there.data(), work, there.size() * sizeof(T), cudaMemcpyDeviceToHost));
    blk::Reader reader(proof.data(), proof.size());
    blk::VerifierSession session;
    blk::VerifierRounds v(verifier_binds, reader, session);
    Channel c{v};
    KT x;
    bool good = true;
    int answer = 0;
    size_t read = 0;
    for (unsigned open = 0; open < 2 && good; open++) {
      for (unsigned n : {1u, 2u, 3u, 0u}) {
        for (unsigned j = 0; j < n + open && good; j++) good = c.recv(x) && x == KT{all[read++]};
        good = good && c.close_algebraic(blk::R_PSC_ROUND).draw() == KT{there[answer++]};
      }
      for (unsigned n = 1; n < 4 && good; n++) {
        for (unsigned j = 0; j < n + open && good; j++) good = c.recv(x) && x == KT{all[read++]};
        c.close_empty(blk::R_PSC_VALUES);
      }
    }
    good = good && read == all.size() && v.finish() && session.accepted();
    printf("rounds by lanes of 0 to 3 scalars with an answer and of 1 to 3 without, with nothing open and with a scalar open: %s\n", good ? "ok" : "FAILS");
    failures += !good;
  }
  if (failures)
    printf("psc_shell_test: FAILED (%d parts of %d, see above)\n", failures, MOST + 2);
  else
    printf("psc_shell_test: ok (%d trees taken back and verified; %d copies with one bit flipped refused)\n", MOST, changed);
  return failures;
}
