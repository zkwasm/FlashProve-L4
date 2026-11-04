// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// What a sub-proof (the argument for one part of the forward pass) passes to the code that closes
// its segment: the coefficients of one equation. A segment (`link` in identifiers) is the
// sub-proof's part of the request's proof; closing it reduces its claims to one
// (blocks/iop/req.h). The equation states that a linear form of the committed arrays and of the
// segment's mask row, whose random elements mask what the sub-proof sends, has a public value. It
// is the sum of the sub-proof's linear claims, each times a power of one batching challenge theta.
// hw::Walk::finish() (prover/hiding/walk.cuh) adds up the coefficients on the GPU; the sub-proof
// then calls hd::close(). One host thread must call clear_async(), finish() and close(): they work
// on that thread's default stream (the build sets nvcc's --default-stream per-thread). Include this
// file after the definitions of T, Tr, CK, u32, hz::TreeN, he::Eqs and hc::TieH.
#pragma once
#include <cstdlib>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "hiding/door_terms.cuh" // RowForm, Pair, Kind, Term
// hd: a sub-proof's equation and the functions that close its segment (prover/hiding/door*.cuh).
namespace hd {
// The equation's public value is not passed: hw::Walk adds up coefficients only and drops the
// claims' constants. For an honest prover it is the form's value on the committed data.
struct Door {
  // One term for each committed array in the equation (door_terms.cuh). Entry (r, c) of the array,
  // counted in the array itself and not in the layout, has the coefficient
  //   sum over the term's pairs (cv, L, c0, c1) with c0 <= c < c1 of cv[r] * L[c]
  // in E, the field's cubic extension (type T). The list is built once, at set-up, and keeps the
  // vectors' addresses, which must not change afterwards. The vectors are on the GPU and hold the
  // coefficients themselves, without a further factor; the sub-proof must have filled them for this
  // proof when it calls close():
  //   U3 (a committed vector over E): one pair, with cv = (1).
  //   ZZ, BZ (the digits of the approximate range check, `tail` in identifiers, and its lookups'
  //     multiplicity bits): one pair for each virtual row sum_r cv[r] A[r][.] of the array A
  //     (HiddenTail::vrows, tail_f.cuh).
  //   M<mu> (symbol matrix mu of the range check; a symbol packs up to three bytes of a column):
  //     one pair. cv = hc::TieH::Er (tie_h.cuh), over the rows of symbols. L, over the columns, has
  //     the matrix's coefficients, with their minus sign, in its tie: the linear claim that relates
  //     the matrix to the digits in ZZ (hw::walk_tail_f(), walk_calls.cuh).
  std::vector<Term> t;
  // This sub-proof's segment number, from 0. The caller sets it before each proof.
  int link = -1;
  // True if the sub-proof leaves a claim on the weights (a layer sub-proof, the output head); false
  // for attention. Nothing reads it.
  bool has_pad = true;
  // On the GPU; set from ch::host.blob() (moments.cuh) before close(). Nothing reads it.
  const u32 *host_words = nullptr;
  // ns + nC entries. Message slots are the sent elements' places in the sub-proof's message array.
  // LS0[j], j < ns, is the coefficient of mask entry j, element j of the segment's mask row. A
  // value that is sent masked at slot j is (what was sent) - factor * (mask entry j), where factor
  // is 1 or a sumcheck's challenge beta (walk.cuh), so hw::k_scat() negates what it adds here
  // (hw::Space::neg = 1); what was sent counts toward the public value.
  // close_by_frame() (door_chain.cuh) reads only these ns entries. No claim adds into the other nC,
  // which stay 0.
  T *LS0 = nullptr;
  // ns: the number of message slots. nC: see LS0. Nothing reads iC.
  size_t ns = 0, nC = 1, iC = 0;
  // Mask entry j is at hS[j], PAD_STRIDE elements (sumcheck_door.cuh) after slot j in the
  // sub-proof's message array. Nothing reads hS.
  const T *hS = nullptr;
  // nLwM: the number of entries of LwM. Matrix mu's vector L begins at LwM + atM[mu].
  T *LwM = nullptr;          // the symbol matrices' column vectors, one after the other
  size_t nLwM = 0;
  std::vector<size_t> atM;
  // Call it before hw::Walk::finish(), which adds into LS0 and LwM. finish() also adds into the
  // vectors L of U3, ZZ and BZ, which this function does not clear.
  void clear_async() const {
    CK(cudaMemsetAsync(LS0, 0, (ns + nC) * sizeof(T), cudaStreamPerThread));
    if (nLwM) CK(cudaMemsetAsync(LwM, 0, nLwM * sizeof(T), cudaStreamPerThread));
  }
  // Call it at set-up, when Tn, num, cv, L, c0, c1 and vecE of every entry of EQ.list are complete
  // (he::Ent, eqs.cuh). It leaves each pair's `form` PLAIN. close_by_frame() rejects such a pair on
  // ZZ or BZ, so HiddenTail::say_forms() (tail_f.cuh) must follow.
  void add_eqs(const he::Eqs &EQ) {
    for (auto &e : EQ.list) {
      Term x;
      x.kind = e.Tn->ec == 1 ? CELLS16 : ELEMENTS, x.R = e.Tn->R, x.K = e.Tn->K, x.data = e.vecE;
      // A vector's term is named by the entry's number: U3 in all three types of sub-proof, as in
      // the segment's plan (ChainHost::plan). The names in hz::TreeN::nm (trees.cuh) differ
      // (U3g<g>, U3, U3H).
      x.nm = x.kind == ELEMENTS ? "U" + std::to_string(e.num + 1) : e.Tn->nm;
      for (size_t i = 0; i < e.cv.size(); i++) x.v.push_back({e.cv[i], e.L[i], e.c0[i], e.c1[i]});
      t.push_back(x);
    }
  }
  // Sets the address and row pitch (in cells) of ZZ or BZ, the term named nm; does nothing if no
  // term has that name. Call it at set-up, after add_eqs().
  void plain_rows(const std::string &nm, const void *cells, size_t ld) {
    for (auto &x : t)
      if (x.nm == nm) x.data = cells, x.ld = ld;
  }
  // Call it at set-up, after h.setup() and when LwM and atM are set. foreign: true if an earlier
  // sub-proof committed the matrix.
  void add_matrix(const hc::TieH &h, const int8_t *cells, bool foreign) {
    Term x;
    x.foreign = foreign;
    x.nm = "M" + std::to_string(h.mu), x.kind = SYMBOLS, x.mu = (int)h.mu, x.R = h.mp, x.K = h.k, x.data = cells,
    x.m = h.m, x.point = h.r, x.nb = h.nb;
    x.v.push_back({h.Er, LwM + atM[h.mu], 0, h.k});
    t.push_back(x);
  }
};
// tr is the request's transcript object on the GPU, a shell over the hash chain that the host
// lends to the GPU (prover/hiding/transcript_shell.cuh). who and g name the sub-proof in messages.
// The function returns 0, or another value after an error. When close_by_frame() returns 0, the
// host has the chain back; ChainHost::first() (prover/prover/chain_host.cuh) lends it again.
typedef int (*CloseFn)(const Door &, Tr *tr, const char *who, int g);
// A pointer, so that this file includes no closing code. It must be installed before the first
// proof: Prover::chain_make() (prover/prover/chain_frame.cuh) does so through hd::link_frame()
// (prover/hiding/door_chain.cuh). While it is null, close() proves nothing: it prints an error,
// returns 0 all the same, and registers an exit handler that ends the program with exit code 23.
// `unclosed` counts such calls.
static CloseFn close_fn = nullptr;
static int unclosed = 0;
static inline int close(const Door &D, Tr *tr, const char *who, int g) {
  if (close_fn) return close_fn(D, tr, who, g);
  printf(
      "internal error: door: %s %d: no closing function is installed, so the "
      "sub-proof's equation was not closed; this run produces no valid proof\n",
      who, g);
  if (!unclosed++)
    atexit([] {
      fprintf(
          stderr,
          "internal error: door: %d sub-proofs ended without a closing function; any files "
          "that this run wrote are not valid proofs\n",
          unclosed);
      fflush(stdout);
      _exit(23);
    });
  return 0;
}
} // namespace hd
