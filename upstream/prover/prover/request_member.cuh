// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The end of a request's first chain (its own hash chain, up to its seal, the digest on which later
// chains build): Prover::chain_member_and_seal(). It batches the claims that the sub-proofs (the
// arguments for the parts of the forward pass) leave on the weight matrix W into one (BATT,
// blocks/iop/batt.h), runs the request's first part of LIN (blocks/iop/lin.h) and seals the chain.
// LIN tests such a claim against the committed weight table; error messages call it "the test on
// the table", and `member` in identifiers here is a request's own part of it.
// blocks/iop/member.h fixes the order of the protocol's messages.
//
// Each sub-proof f with weights (the layer groups', then the output head's) leaves a pending claim
//   < g_f (x) eq(psi_f, .), W > + beta c_f = v          (why this form: blocks/iop/batt_shown.h)
// where (x) is the tensor product, g_f has a coefficient for each row of W and psi_f a coordinate
// for each of its lkW column bits. It leaves three things: U_f = g_f^T W, 2^lkW elements in GPU
// memory; its member words, which give v, beta, psi_f and g_f (member_row_of_words()); and c_f, an
// entry of the request's index-mask message (its oracle message of masks for claims on W;
// ChainRun::index_mask_entry()).
//
// Hidden values: the U_f are linear functions of the weights. MemberHeavyGpu zeroes its copies of
// them (prover/engine/member_heavy.cuh); this file does not erase the sub-proofs' own buffers.
#pragma once
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "engine/member_heavy.cuh"
#include "frame/wf_units.h" // wf_layout_fault, wf_shape_of_layout
#include "iop/member.h"
#include "numbers/round_names.h"
#include "prover/prover.h"

// The layout of W's rows and a claim's row coefficients, as the committer (the program cp_commit)
// has them, over the field type of blocks/: prover/common/weight_layout.h takes the field's names
// from its includer, hence the #include inside this namespace.
namespace wrows {
using std::vector;
using E = blk::E;
using u32 = uint32_t;
using u64 = uint64_t;
static const E E0 = E::zero(), E1 = E::one();
static inline E ebase(u64 x) { return E::from(blk::from_u64(x)); }
static inline E esub(const E &a, const E &b) { return a - b; }
static inline E emul(const E &a, const E &b) { return a * b; }
constexpr int SW = 8;  // of MergeW, which nothing here uses
#include "common/weight_layout.h"
}  // namespace wrows

// out[i] = in[i], converted from the sub-proofs' element type T to blk::E.
__global__ void k_member_table(const T *in, size_t n, blk::E *out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    for (int k = 0; k < EDEG; k++) out[i].c[k].v = coord(in[i], k);
}

// Turns one sub-proof's member words w into its pending claim (`TensorRow`, blocks/iop/batt.h).
// c: the claim's entry of the index-mask message. member() in prover/head/stack.cuh and
// prover/head/hidden_fork.cuh writes the words:
//   v | beta | rm (lk = Y.lk elements) | number of terms | for each term: unit, scalar, row factors
// A unit (`UnitW`) is the rows of W for one kind of weights of one layer, given by its index in
// Y.U. `FacW` (prover/common/weight_layout.h) describes a row factor, one for each index group
// (`GrpW`) of the unit's tensor. A group of extent R gives `take` bits of its index to W's column
// index and the rest to the row index. The factor's words: in class (`cls`) 'A', which gives all
// its bits or none, the R values AV, and only if `take` is 0; in the other classes, R >> z values
// HV (2^z is the largest power of 2 that divides R), then the coordinates pt of those of the
// index's z low bits that stay in the row index, highest bit first.
// psi_f[t] = rm[lk - 1 - t] and
//   g_f = sum over the terms of scalar * gammaW(unit, row factors), on that unit's rows.
// g_f is stored only on the smallest range of rows that contains its units ("stretch" in messages).
// The words come from this program's own kernels: a list that does not parse is a bug.
//
// The words, g_f and psi_f are public only to the request's own verifier. They follow from the
// chain's challenges and, in the embedding's and the head's claims, from the request's tokens. A
// token's index is part of W's column index, so the vectors made from the tokens (kS_psi(),
// prover/head/stack.cuh; kH_AV_tokens(), prover/head/rows.cuh) are not row factors: evaluated at
// rm, they enter the term's scalar. So no size, branch or program exit on the CPU side may depend
// on a value of g_f.
//
// On return g_f is not yet computed: the vector that the claim's Tensors share (g_f is their sum:
// blocks/iop/tensor.h) has its final length, contains zeros, and is being filled by a thread. Until
// the caller has waited on `filling`, nothing may read an entry of the vector; afterwards nothing
// writes one. The thread runs concurrently with BATT and LIN's first part, which read no entry of
// g_f: BATT multiplies only the Tensors' scalars, MemberHeavyGpu has the U_f, and the claim that
// reads g_f (LIN's, on its oracle message Omega) is formed in the second part. A request's g_f
// together cost about two multiplications in E on the CPU for each row of W.
// The thread uses no private randomness, chain, engine or GPU, and shares only the vector and the
// constant wrows::E1. If the program exits meanwhile, the thread ends with the process. If the
// thread runs out of memory, filling.get() rethrows its exception; if it cannot be started,
// std::async() throws here. Nothing catches either, so the terminate handler of
// prover/prover/ends.h ends the program. What the thread frees is zeroed first by this program's
// operator delete (prover/common/zeroing_heap.h), which several threads may call at once.
static inline blk::TensorRow member_row_of_words(const wrows::LayW &Y, const blk::LinShape &S, const std::vector<u32> &w, blk::MaskSlot &&c, std::future<void> &filling) {
  size_t o = 0;
  auto element = [&] {
    BLK_REQUIRE(o + 2 * EDEG <= w.size(), "a sub-proof's member words end too early");
    blk::E x;
    for (int i = 0; i < EDEG; i++) x.c[i].v = w[o + 2 * i] | ((u64)w[o + 2 * i + 1] << 32);
    o += 2 * EDEG;
    return x;
  };
  auto word = [&] {
    BLK_REQUIRE(o < w.size(), "a sub-proof's member words end too early");
    return w[o++];
  };
  const blk::E v = element(), beta = element();
  std::vector<blk::E> psi((size_t)Y.lk);
  for (int i = 0; i < Y.lk; i++) psi[(size_t)(Y.lk - 1 - i)] = element();  // rm[i] is the coordinate of column bit lk - 1 - i
  struct Term {
    blk::E scalar;
    wrows::UnitW unit;  // a copy, so that the thread below does not read Y.U
    wrows::ClaimW claim;
  };
  std::vector<Term> term(word());
  size_t first = Y.m0, last = 0;
  for (Term &t : term) {
    const u32 ui = word();
    BLK_REQUIRE(ui < Y.U.size(), "a sub-proof's member words name a unit that W has not");
    const wrows::UnitW &u = Y.U[ui];
    t.scalar = element();
    t.unit = u;
    t.claim.ui = (int)ui;
    t.claim.f.resize(u.g.size());
    for (size_t i = 0; i < u.g.size(); i++) {  // one factor for each index group of the unit
      const wrows::GrpW &g = u.g[i];
      wrows::FacW &x = t.claim.f[i];
      if (g.cls == 'A') {
        if (!g.take) {
          x.AV.resize(g.R);
          for (blk::E &e : x.AV) e = element();
        }
        continue;
      }
      x.HV.resize(g.R >> g.z);
      for (blk::E &e : x.HV) e = element();
      x.pt.assign((size_t)g.z, blk::E::zero());
      const int b0 = g.cls == 'S' ? g.take : 0, b1 = g.cls == 'S' ? g.z : g.z - g.take;
      for (int b = b1 - 1; b >= b0; b--) x.pt[(size_t)(g.z - 1 - b)] = element();
    }
    first = std::min(first, u.off), last = std::max(last, u.off + u.rows);
  }
  BLK_REQUIRE(o == w.size() && first < last, "a sub-proof's member words have words left over, or no claim");
  // The range must begin and end at an even row: it is then cut into aligned blocks of at least two
  // rows. Each block is a slice of the shared vector, whose entries the CPU code of the engine's
  // fold copies and does not examine (the engine, `Engine` in blocks/engine/engine.h, does the
  // parts of a prover's work that grow with its tables). A block of one row would carry its value
  // as a Tensor's scalar (blocks/iop/member.h, stretch_in_the_cube()), and the fold skips a Tensor
  // whose scalar is 0 (prover/engine/fold.cuh): the sizes of its allocations would then depend on a
  // value of g_f, which this function's comment forbids. Every unit of W has an even number
  // of rows (prover/common/weight_layout.h: rows = nlw * .., nlw 2 or 4). The check reads only
  // public numbers of the layout.
  BLK_REQUIRE(first % 2 == 0 && last % 2 == 0, "a sub-proof's stretch of W's rows begins or ends at an odd row");
  auto values = std::make_shared<std::vector<blk::E>>(last - first, blk::E::zero());
  filling = std::async(std::launch::async, [first, values, term = std::move(term)] {  // explicit: the default policy may defer the work until `get`
    for (const Term &t : term) {  // one thread for a vector: two terms may name the same unit, and their values add up
      const std::vector<blk::E> gamma = wrows::gammaW(t.unit, t.claim);
      for (size_t y = 0; y < t.unit.rows; y++) (*values)[t.unit.off - first + y] += t.scalar * gamma[y];
    }
  });
  // member_weights_on_rows() reads no entry of the vector, because no block has one row. It ends
  // the program if the range is not within the S.matrix_rows rows that LIN counts.
  return blk::member_pending_row(blk::member_weights_on_rows(S, first, std::move(values)), std::move(psi), std::move(c), beta, v);
}

// W's layout, as the committer computes it (prover/hash_commitment/hf_shape.cuh): m0 is the number
// of data rows rounded up to a multiple of D. The prover program cp_prove calls layoutW() nowhere
// else outside that directory.
static inline wrows::LayW member_layout_of_W(const gs::Shape &g, int D) {
  wrows::ShapeW sh;
  sh.moeMode = g.mo, sh.L = g.L, sh.d = (size_t)g.d, sh.dh = (size_t)g.dh, sh.dkv = (size_t)g.dkv, sh.dq = (size_t)g.dq, sh.f = (size_t)g.f, sh.fe = (size_t)g.fe,
  sh.NE = (size_t)g.NE, sh.V = (size_t)g.V;
  return wrows::layoutW(sh, g.lkW, D, 0, g.V ? 1 : 0);
}
// The shape of the lookups in the committer's range proof (WF), computed from that layout by the
// functions that the verifier calls on its own layout (blocks/frame/wf_units.h). Its field
// matrix_rows is W's number of rows. fault: set to null, or to the reason why the layout's units
// are rejected; the returned shape is then empty.
inline blk::WfShape Prover::chain_ranges(unsigned k_col, const char *&fault) {
  const wrows::LayW Y = member_layout_of_W(SHg.s, (int)policy["D"]);
  fault = blk::wf_layout_fault(Y);
  return fault ? blk::WfShape() : blk::wf_shape_of_layout(Y, k_col);
}

// Call it as finish_request() does (request.cuh), after ChainRun::take_back() and finish(): the
// sub-proofs' last proximity test is done and the CPU side holds the chain. s.part.lin is what the
// chain's first query (its first hash input) contains about LIN. The sub-proofs must have taken
// their index-mask entries in the order of f: only the number of entries is checked. Call it on the
// host thread that ran the sub-proofs: k_member_table() and member() rely on that thread's default
// stream to run after their kernels. Sets s.part.lin_first and s.part.seal; on return every g_f is
// complete. SHg.ui(13, Lr) is the index in Y.U of the output head's unit.
inline void Prover::chain_member_and_seal(Slot &s) {
  ChainRun &run = *chain.request;
  blk::ProverContext &P = run.context();
  const blk::LinShape &S = run.channel().get(s.part.lin.shape);
  const size_t J = (size_t)NGr + 1, columns = (size_t)1 << lkW;
  BLK_REQUIRE(S.k_col() == (unsigned)lkW && run.index_slots().size() == J, "the request's end: another number of columns, or not one index-mask entry for each sub-proof with weights");

  static const wrows::LayW Y = member_layout_of_W(SHg.s, (int)policy["D"]);  // fixed by the parameter file
  BLK_REQUIRE(Y.m0 == S.matrix_rows, "the request's end: W's layout has another number of rows than the test on the table counts");  // the verifier rejects the same mismatch (verifier/src/vf_weights_pending_claim.h)

  blk::MemberHeavyGpu heavy((unsigned)lkW, J);
  std::vector<blk::TensorRow> pending;
  std::vector<std::future<void>> filling(J);  // the threads that compute the g_f
  for (size_t f = 0; f < J; f++) {
    const T *table = f < (size_t)NGr ? FK[f].h->GS.Tf : HH.T_H();
    k_member_table<<<256, 256>>>(table, columns, heavy.table(f));
    CK(cudaGetLastError());
    const std::vector<u32> words = f < (size_t)NGr ? FK[f].h->GS.member() : HH.member(SHg.ui(13, Lr));
    pending.push_back(member_row_of_words(Y, S, words, std::move(run.index_slots()[f]), filling[f]));
  }
  run.index_slots().clear();  // every entry that was taken is in a pending claim now
  s.part.lin_first = blk::member_first_prove(P, s.part.lin, run.frame().at_start().at(0), std::move(pending), heavy);
  BLK_REQUIRE(run.channel().seal(blk::R_SEAL, s.part.seal), "a prover's channel does not refuse");
  for (std::future<void> &f : filling) f.get();  // every g_f is complete from here on
  // finish_request() ends the chain (finish_inside_a_file()) right after this call.
}
