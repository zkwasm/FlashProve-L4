// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Kernels of the output head's sub-proof (prover/head/hidden_fork.cuh) that depend on a request's
   range, directly or through their inputs. The output head is the model's last matrix, Wh[kk][x],
   which turns the final activations into one logit per vocabulary entry x < V; kk < d is a feature.

   A request has a window of NT tokens and two numbers r and e. Its range (`stated` in identifiers)
   is the token positions t with r <= t < e. At each of these the proof compares all logits with the
   logit of the checked token w_t: the window's token t + 1 or, at t = NT - 1, the next token, which
   the prover announces. ew[x] is the sub-proof's coefficient for vocabulary entry x, wx the sum of
   all ew[x], and g[kk] the sum over x of ew[x] Wh[kk][x]. For a w_t outside 0 .. V - 1, t in the
   range or not, `index_below` returns 0 and sets a failure bit of the request.

   The including file must already define u32 (prover/common/common.h), T and F (an element of the
   extension field and that field's operations; prover/hiding/transcript.cuh), `ext_from_signed` (signed
   integer to field element) and i8 (prover/prover/base.cuh), and GRID_LOOP (prover/prover/forward.cuh).
   Public source for the kernels of this file: NVIDIA Technical Blog, "CUDA Pro Tip: Write Flexible
   Kernels with Grid-Stride Loops" (GRID_LOOP): each kernel that uses GRID_LOOP is one such loop. */
#pragma once
#include "common/index_guard.cuh"
// Returns whether t is in the request's range. `rows` points to the words r, e in GPU memory.
__device__ __forceinline__ bool kH_stated(const u32 *rows, int t) {
  return (u32)t >= rows[0] && (u32)t < rows[1];
}
// Computes the matrix U of the sumcheck for the head's matrix product. At a t in the range,
// Gv[kk][t] = wx Wh[kk][w_t] - g[kk], where w_t is w[t]; elsewhere it is 0. Wh (`cells`) and g are
// two limbs, low first, value = low + 256 high: Wh as signed bytes [limb][kk][x], g as [limb][kk].
__global__ void kH_rows_of_U(const i8 *cells, const T *g, const T *wx, const int *w, const u32 *rows, int d, int V,
                             int NT, T *Gv) {
  const T c256 = F::small(256);
  GRID_LOOP(i, (size_t)d * NT) {
    const size_t kk = i / NT;
    const int t = (int)(i % NT);
    const u32 x = index_below(w[t], V, INDEX_WINNER);
    const T by = ext_from_signed((int)cells[kk * V + x] + 256 * (int)cells[((size_t)d + kk) * V + x]);
    Gv[i] = kH_stated(rows, t) ? F::sub(F::mul(wx[0], by), F::add(g[kk], F::mul(g[d + kk], c256))) : F::zero();
  }
}
// Copies e, with 0 outside the request's range. Here e is a vector, not the request's number e.
__global__ void kH_stated_rows(const T *e, const u32 *rows, int NT, T *o) {
  GRID_LOOP(t, NT) o[t] = kH_stated(rows, (int)t) ? e[t] : F::zero();
}
// o[t] = e[t] (or 1 if e is null) at a t in the range whose checked token comes from the
// request's window, that is, at every t in the range except NT - 1; 0 elsewhere.
__global__ void kH_callers_rows(const T *e, const u32 *rows, int NT, T *o) {
  GRID_LOOP(t, NT) o[t] = kH_stated(rows, (int)t) && (int)t < NT - 1 ? (e ? e[t] : F::one()) : F::zero();
}
// Repeats the NT coefficients e, one per token position, until o has n elements.
__global__ void kH_on_every_feature(const T *e, int NT, size_t n, T *o) {
  GRID_LOOP(y, n) o[y] = e[y % NT];
}
// First step of AV[x] = wx (the sum of er[t] over the t < NT with w_t = x) - sR ew[x], the
// public vector in the claim about the head's matrix. sR is the sum of all er[t].
__global__ void kH_AV(const T *ew, const T *sR, int V, T *AV) {
  GRID_LOOP(x, V) AV[x] = F::sub(F::zero(), F::mul(sR[0], ew[x]));
}
// Second step. er[t] is eq(r_H, t) at a t in the range, 0 elsewhere; U's rounds over t end at r_H.
// Launch it with one thread (not checked): several t may share a w_t, so the additions are serial.
__global__ void kH_AV_tokens(const int *w, const T *er, const T *wx, int NT, int V, T *AV) {
  for (int t = 0; t < NT; t++) {
    const u32 x = index_below(w[t], V, INDEX_WINNER);
    AV[x] = F::add(AV[x], F::mul(wx[0], er[t]));
  }
}
// Gathers n elements of v, `stride` apart, from v[0]: in HForkHid::prove(), entry 0 of Gv's rows.
__global__ void kH_every(const T *v, size_t stride, int n, T *o) {
  GRID_LOOP(kk, n) o[kk] = v[kk * stride];
}
