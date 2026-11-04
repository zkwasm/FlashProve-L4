// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For small shapes, WholeCells implements RowCells (prover/hash_commitment/hf_commit_cells.cuh),
// through which the committer program cp_commit reads M. WholeCells keeps all of M in GPU memory,
// rows padded to multiples of 256 bytes. M is the weight matrix W (m0 by k signed bytes)
// rearranged into R long rows.
//
// Include after prover/weights_commitment/common.cuh and
// prover/hash_commitment/hf_commit_walk.h (Piece, WeightWalker).
#pragma once
#include "hash_commitment/commit_lie.cuh" // test hook: empty in this repository
#include "hash_commitment/hf_commit_cells.cuh"
namespace hf {
struct WholeCells : RowCells {
  const HfParams &P;
  int8_t *M = nullptr;
  size_t stride = 0;
  static size_t stride_of(const HfParams &P) { return ((size_t)P.l + 255) & ~(size_t)255; }
  static size_t bytes_of(const HfParams &P) { return P.R * stride_of(P); }
  // The constructor's temporary GPU memory: a panel (rows of W, column-major) for the largest
  // unit (one tensor's rows, or a layer's q and k). WeightWalker's buffers are extra.
  static size_t panel_bytes(const rf::LayW &Y) {
    size_t most = 0;
    for (const rf::UnitW &u : Y.U) most = std::max(most, (size_t)(u.rows + 15) / 16 * 16);
    return most * Y.k;
  }
  // Fills M in one walk(), on the calling thread's current GPU; returns when M is complete.
  // Y: W's layout. um: Y's units as k_cellsT() (prover/weights_commitment/kernels.cuh) reads them.
  // H, G, dh: query heads, key heads, head width.
  // Not checked: `weights` must have Y's units, which size the panel; walk()'s stream must be, or
  // synchronize with, the default stream, where cudaMemset() zeroes M asynchronously.
  WholeCells(const HfParams &P_, const rf::LayW &Y, const std::vector<::g5::UM> &um, long long H, long long G,
             long long dh, WeightWalker &weights)
      : P(P_), stride(stride_of(P_)) {
    if (Y.k != P.k || Y.m0 != P.m0) {
      fprintf(stderr, "cp_commit: internal error: the rows of M are asked for in another shape than the layout's\n");
      exit(4);
    }
    commit_room(&M, bytes_of(P));
    CK(cudaMemset(M, 0, bytes_of(P))); // rows of W that belong to no unit are 0
    int8_t *pan = nullptr;
    const size_t panel = panel_bytes(Y);
    commit_room(&pan, panel);
    cudaEvent_t done;
    CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    weights.walk(WeightWalker::WHOLE_UNITS, [&](const Piece &piece, cudaStream_t st) {
      const rf::UnitW &u = *piece.u;
      const size_t pitch = (size_t)(u.rows + 15) / 16 * 16;
      CK(cudaStreamWaitEvent(st, done, 0)); // one panel, so the callbacks' GPU work is serialized
      ::g5::k_cellsT<<<8192, 256, 0, st>>>(um[piece.unit], H, G, dh, (const short *)piece.w0, (const short *)piece.w1,
                                           (const int *)piece.w0, Y.k, u.off, u.rows, pitch, (i8 *)pan);
      CK(cudaGetLastError());
      commit_lie::panel(pan, pitch, u.rows, u.off, Y.k, st); // test hook: empty in this repository
      // rows_of_panel(): at most 65,535 rows of M.
      for (uint32_t r = 0; r < P.R; r += 32768)
        rows_of_panel(P, pan, pitch, u.rows, u.off, r, (uint32_t)std::min<size_t>(32768, P.R - r), M + r * stride, stride,
                      st);
      CK(cudaEventRecord(done, st));
    });
    CK(cudaEventSynchronize(done));
    CK(cudaEventDestroy(done));
    CK(cudaMemset(pan, 0, panel));
    CK((cudaFree)(pan));
    weights.release();
  }
  // Where R < 16 (R is a power of two) a caller may ask for 16 rows, but only R are allocated:
  // it must not read rows R and above.
  const int8_t *rows(uint32_t r0, int nr, size_t *stride_, cudaStream_t) override {
    if (r0 + (size_t)nr > ((P.R + 15) & ~(size_t)15) || r0 >= P.R) {
      fprintf(stderr, "cp_commit: internal error: M has no such rows\n");
      exit(4);
    }
    *stride_ = stride;
    return M + r0 * stride;
  }
  int most() const override { return (int)P.R; }
  size_t bytes() const override { return M ? bytes_of(P) : 0; }
  void release() override {
    if (!M) return;
    CK(cudaMemset(M, 0, bytes_of(P)));
    CK((cudaFree)(M));
    M = nullptr;
  }
  ~WholeCells() override { release(); }
};
} // namespace hf
