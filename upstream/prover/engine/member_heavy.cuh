// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU implementation of MemberHeavy (blocks/iop/member.h): the prover's large computations when a
// request's J pending rows (linear claims on the committed matrix M) become one claim.
// Pending row j has a vector W_j = g_j^T M of 2^k elements of the extension field E, one for each
// column of M (g_j: its coefficients on M's rows). MemberHeavyGpu keeps the W_j in GPU memory, runs
// the sumcheck over them and returns their combination sum_j a_j W_j, the one claim's vector. Host
// code adds the masks and exchanges the messages (psc_prove() in blocks/iop/psc.h).
//
// Each bind() writes tables of half the length, alternating between two buffers each; the full W_j
// are kept, because combine() reads them again. Public source: a sumcheck prover's halved tables:
// Barretenberg's documentation of its class SumcheckProver, "Round 0" and
// "Updating Partial Evaluations in Subsequent Rounds".
// This citation is second-hand: nothing in it is checked against that documentation.
//
// Hidden values. The W_j, their folds, the partial sums and the combination depend on the model's
// weights. Every allocated GPU buffer that has held one is zeroed before it is released: the folds
// in bind() and, with the sums, in ~MemberSumcheckGpu(), the rest in combine() or else in
// ~MemberHeavyGpu(). Not zeroed: member_k_round()'s shared memory, which holds partial sums; the
// other GPU buffers, whose left_unzeroed() calls give the reason; any host copy (round()'s local
// partial sums; the returned vectors, which are the caller's).
//
// One MemberHeavyGpu serves one request. All work runs on the default stream. Under nvcc's
// --default-stream per-thread (prover/build.sh) each thread has its own: one thread must write the
// tables and make every call, destruction included, or the caller must synchronize (not checked).
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "engine/device.cuh"
#include "engine/left_unzeroed.h"
#include "field/field.h"
#include "iop/member.h"

namespace blk {

constexpr unsigned MEMBER_THREADS = 128, MEMBER_BLOCKS_MOST = 256;

// out[j * 2^k + c] = chi[j] * eq(psi_j, c); bit t of c goes with coordinate t.
// psi: J points of k coordinates, one after the other.
__global__ void member_k_weights(const E *chi, const E *psi, unsigned k, u64 all, E *out) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < all; i += (u64)gridDim.x * blockDim.x) {
    const u64 j = i >> k, c = i & ((u64(1) << k) - 1);
    E x = chi[j];
    for (unsigned t = 0; t < k; t++) x *= ((c >> t) & 1) ? psi[j * k + t] : E::one() - psi[j * k + t];
    out[i] = x;
  }
}
// The three coefficients, lowest degree first, of
//   sum over j and b < half of  (f0 + X (f1 - f0)) (w0 + X (w1 - w0)),
// f0 = f[j * stride + 2 b], f1 the next, w alike. `sums` gets one partial triple for each block of
// threads. Not checked: blockDim.x is a power of two from 4 to MEMBER_THREADS.
__global__ void member_k_round(const E *f, const E *w, u64 stride, u64 half, u64 rows, E *sums) {
  __shared__ E part[MEMBER_THREADS][3];
  E c0 = E::zero(), c1 = E::zero(), c2 = E::zero();
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < rows * half; i += (u64)gridDim.x * blockDim.x) {
    const u64 at = (i / half) * stride + 2 * (i % half);
    const E f0 = f[at], df = f[at + 1] - f0, w0 = w[at], dw = w[at + 1] - w0;
    c0 += f0 * w0, c1 += f0 * dw + df * w0, c2 += df * dw;
  }
  part[threadIdx.x][0] = c0, part[threadIdx.x][1] = c1, part[threadIdx.x][2] = c2;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s; s /= 2) {
    if (threadIdx.x < s)
      for (int t = 0; t < 3; t++) part[threadIdx.x][t] += part[threadIdx.x + s][t];
    __syncthreads();
  }
  if (threadIdx.x < 3) sums[3 * blockIdx.x + threadIdx.x] = part[0][threadIdx.x];
}
// to[j * half + b] = x + g (y - x), where x = from[j * stride + 2 b], y the next element
__global__ void member_k_bind(const E *from, u64 stride, u64 half, u64 rows, E g, E *to) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < rows * half; i += (u64)gridDim.x * blockDim.x) {
    const u64 at = (i / half) * stride + 2 * (i % half);
    to[i] = from[at] + g * (from[at + 1] - from[at]);
  }
}
__global__ void member_k_combine(const E *a, const E *w, u64 columns, u64 rows, E *out) {
  for (u64 c = blockIdx.x * (u64)blockDim.x + threadIdx.x; c < columns; c += (u64)gridDim.x * blockDim.x) {
    E x = E::zero();
    for (u64 j = 0; j < rows; j++) x += a[j] * w[j * columns + c];
    out[c] = x;
  }
}
inline unsigned member_blocks(u64 items) { return (unsigned)std::min<u64>(MEMBER_BLOCKS_MOST, std::max<u64>(1, (items + MEMBER_THREADS - 1) / MEMBER_THREADS)); }
inline void member_wipe(Card<E> &x) {
  if (x.size()) ENGINE_CK(cudaMemsetAsync(x.get(), 0, x.size() * sizeof(E), 0));
}

// The sumcheck, without masks, of  sum over columns c of  sum_j chi_j eq(psi_j, c) W_j(c)
// (blocks/iop/batt.h). It reads the caller's tables and must be destroyed before they are released;
// batt_prove() destroys it on return.
// At full size (k = 18, J = 13) a round reads about 160 MB at most, so the kernels are plain loops.
class MemberSumcheckGpu : public SumcheckProver<E> {
 public:
  MemberSumcheckGpu(const E *tables, unsigned k, const std::vector<E> &chi, const std::vector<std::vector<E>> &psi)
      : w_(tables), k_(k), rows_(chi.size()), stride_(u64(1) << k), left_(k) {
    BLK_REQUIRE(k >= 1 && k < 40 && !chi.empty() && psi.size() == chi.size(), "member, GPU: no variable, no row, or not one point for each row");
    std::vector<E> points;
    for (const std::vector<E> &p : psi) {
      BLK_REQUIRE(p.size() == k, "member, GPU: a point has not one coordinate for each variable");
      points.insert(points.end(), p.begin(), p.end());
    }
    Card<E> chi_card, psi_card;
    chi_card.from_host(chi.data(), chi.size()), psi_card.from_host(points.data(), points.size());
    weights_.resize(rows_ << k);
    left_unzeroed(chi_card, "the rows' batching scalars chi_j: powers of the one element that the verifier has drawn (batt_draw)");
    left_unzeroed(psi_card, "the pending rows' points psi_j: words of the rows that are public to that request's client and follow from the chain's draws");
    left_unzeroed(weights_, "chi_j eq(psi_j, c) for every column c: a function of these two alone");
    member_k_weights<<<member_blocks(rows_ << k), MEMBER_THREADS>>>(chi_card.get(), psi_card.get(), k, rows_ << k, weights_.get());
    ENGINE_CK(cudaGetLastError());
    f_ = weights_.get();
    sums_.resize(3 * size_t(MEMBER_BLOCKS_MOST));
  }
  ~MemberSumcheckGpu() override {
    for (Card<E> *x : {&fold_w_[0], &fold_w_[1], &sums_}) member_wipe(*x);
  }
  std::vector<E> round() override {
    BLK_REQUIRE(left_ > 0, "member, GPU: no variable is left");
    const u64 half = u64(1) << (left_ - 1);
    const unsigned blocks = member_blocks(rows_ * half);
    sums_.spent();
    member_k_round<<<blocks, MEMBER_THREADS>>>(f_, w_, stride_, half, rows_, sums_.get());
    ENGINE_CK(cudaGetLastError());
    std::vector<E> part(3 * size_t(blocks)), c(3, E::zero());
    ENGINE_CK(cudaMemcpy(part.data(), sums_.get(), part.size() * sizeof(E), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < part.size(); i++) c[i % 3] += part[i];
    return c;
  }
  void bind(const E &g) override {
    BLK_REQUIRE(left_ > 0, "member, GPU: no variable is left");
    const u64 half = u64(1) << (left_ - 1);
    const int to = int(left_ & 1);
    fold_f_[to].resize(rows_ * half), fold_w_[to].resize(rows_ * half);
    left_unzeroed(fold_f_[to], "the folds of the weights chi_j eq(psi_j, .) at the rounds' challenges: functions of the verifier's messages and the rows' points, public to that request's client");
    member_k_bind<<<member_blocks(rows_ * half), MEMBER_THREADS>>>(f_, stride_, half, rows_, g, fold_f_[to].get());
    member_k_bind<<<member_blocks(rows_ * half), MEMBER_THREADS>>>(w_, stride_, half, rows_, g, fold_w_[to].get());
    ENGINE_CK(cudaGetLastError());
    member_wipe(fold_w_[1 - to]);  // the fold before this one has been read
    f_ = fold_f_[to].get(), w_ = fold_w_[to].get(), stride_ = half, left_--;
  }
  std::vector<E> end_values() override {  // W_j at the challenge point (`kept` maps: not sent)
    BLK_REQUIRE(left_ == 0, "member, GPU: variables are left");
    std::vector<E> v(rows_);
    ENGINE_CK(cudaMemcpy(v.data(), w_, rows_ * sizeof(E), cudaMemcpyDeviceToHost));
    return v;
  }

 private:
  // current tables: chi_j eq(psi_j, .) and W_j
  const E *f_ = nullptr, *w_;
  unsigned k_;
  u64 rows_, stride_;
  unsigned left_;
  Card<E> weights_, fold_f_[2], fold_w_[2], sums_;
};

class MemberHeavyGpu : public MemberHeavy {
 public:
  // Not checked: the caller writes W_j to table(j), on the default stream, before sumcheck() or
  // combine() runs.
  MemberHeavyGpu(unsigned k, size_t rows) : k_(k), rows_(rows), tables_(rows << k) {}
  ~MemberHeavyGpu() override { member_wipe(tables_); }
  E *table(size_t j) {
    BLK_REQUIRE(j < rows_, "member, GPU: no such vector");
    return tables_.get() + (j << k_);
  }
  // When the prover program cp_prove calls this, other threads may still be writing the g_j
  // (chain_member_and_seal() in prover/prover/request_member.cuh).
  void begin(const std::vector<std::vector<Tensor>> &g) override {  // must read only g's size
    BLK_REQUIRE(g.size() == rows_ && !begun_, "member, GPU: not one vector for each pending row, or a second request");
    begun_ = true;
  }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &, const std::vector<Weight<E>> &vars, const std::vector<E> &chi,
                                              const std::vector<std::vector<E>> &psi) override {
    BLK_REQUIRE(begun_ && vars.size() == k_ && chi.size() == rows_, "member, GPU: the sum-check is not about these vectors");
    return std::unique_ptr<SumcheckProver<E>>(new MemberSumcheckGpu(tables_.get(), k_, chi, psi));
  }
  std::vector<E> combine(const std::vector<E> &a) override {
    BLK_REQUIRE(begun_ && a.size() == rows_ && tables_.size(), "member, GPU: not one scalar for each vector, or a second combination");
    const u64 columns = u64(1) << k_;
    Card<E> a_card, out(columns);
    a_card.from_host(a.data(), a.size());
    left_unzeroed(a_card, "the scalars of the combination: with one pending row the one scalar 1; else the kept maps' coefficients in the sum-check's closing row: functions of the verifier's challenges and the rows' points, public to that request's client, which the verifier forms too (batt_verify)");
    member_k_combine<<<member_blocks(columns), MEMBER_THREADS>>>(a_card.get(), tables_.get(), columns, rows_, out.get());
    ENGINE_CK(cudaGetLastError());
    std::vector<E> w(columns);
    out.to_host(w.data());
    member_wipe(out), member_wipe(tables_);
    tables_.release();
    return w;
  }

 private:
  unsigned k_;
  size_t rows_;
  Card<E> tables_;
  bool begun_ = false;
};

}  // namespace blk
