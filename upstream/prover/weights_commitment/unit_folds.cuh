// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Kernels that write weights as the cells of their unit of the committed weight matrix W. A cell,
// one signed byte, stores a limb (balanced base-256 digit) of a weight. A unit (UM, in
// prover/weights_commitment/kernels.cuh) is the range of W's rows that stores one tensor or, if
// its kind UM::X is 0, two.
// Cells that map to no weight are 0. The output o is column-major: k columns of ld cells.
// Weights are read in three forms: byte planes (`PS`), the embedding's limb bytes and the output
// head's int16 weights. wb::BlockWeights (prover/weights_blocks/block_weights.h) stores them, or
// decodes the planes from the model's quantized blocks.
#pragma once
namespace g5 {
// Byte planes for k_cellsP(): w0[s] and w1[s] store the low and high limb of each weight of
// source s. Each array is indexed [x][kk] (output index, input index), pitch[s] bytes per row.
// Only kind 0 has two sources: query weights (0) and key weights (1).
struct PS {
  const i8 *w0[2], *w1[2];
  unsigned long long pitch[2];
};
// Writes the unit's cells. Preconditions, not checked: ld >= u.rows;
// u.X is 0 .. 6, 9 or 11 (a layer's tensor, but not the routed experts';
// layoutW() in prover/common/weight_layout.h lists each kind); u.nlw = 2 limbs.
// Of the model's dimensions S (`SM`, in prover/weights_commitment/kernels.cuh), only H, Gk and dh
// (entries per head) are read.
// The loop form, the grid-stride loop, is public: M. Harris,
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_cellsP(UM u, SM S, PS p, u64 k, u64 ld, i8 *o) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < k * ld; i += (u64)gridDim.x * blockDim.x) {
    const u64 c = i / ld;
    u64 r = i % ld;
    int v = 0;
    if (r < u.rows && !(c >> u.ncb)) {
      long long ix[3] = {0, 0, 0};
      bool ok = true;
      // Decodes the tensor's indices ix[g]; r ends as the limb's number. Class u.cls[g] (`GrpW`,
      // defined in prover/common/weight_layout.h) sets which bits of ix[g] come from column
      // number c; the rest come from r. 'A': all bits if u.take[g] > 0, otherwise none. 'S': the
      // lowest u.take[g] bits. Any other ('P'): bits u.z[g] - u.take[g] to u.z[g] - 1.
      for (int g = 0; g < u.ng; g++) {
        const u64 lf = r % u.left[g];
        r /= u.left[g];
        const u64 col = (c >> u.cb0[g]) & ((1ull << u.take[g]) - 1);
        u64 w;
        if (u.cls[g] == 'A')
          w = u.take[g] ? col : lf;
        else if (u.cls[g] == 'S')
          w = (lf << u.take[g]) | col;
        else {
          const int rem = u.z[g] - u.take[g];
          w = ((lf >> rem) << u.z[g]) | (col << rem) | (lf & ((1ull << rem) - 1));
        }
        if (w >= u.R[g]) {
          ok = false;
          break;
        }
        ix[g] = (long long)w;
      }
      if (ok) {
        int s = 0;
        long long x, kk;
        if (u.X == 0) {
          // ix[0]: S.H query heads, then S.Gk key heads
          kk = ix[2];
          if (ix[0] < S.H)
            x = ix[0] * S.dh + ix[1];
          else if (ix[0] < S.H + S.Gk) {
            s = 1;
            x = (ix[0] - S.H) * S.dh + ix[1];
          } else
            ok = false;
        } else {
          kk = ix[0];
          x = ix[1];
        }
        if (ok) v = (r ? p.w1 : p.w0)[s][x * p.pitch[s] + kk];
      }
    }
    o[i] = (i8)v;
  }
}
// Writes nr rows of the embedding's unit if u.X = 12, otherwise of the output head's unit
// (kind 13), from its row y0. ld >= nr; each column's cells from nr on are 0.
// emb stores limbs, [limb][token < V][x < d]; wh stores int16 weights, [token][kk < d].
// Precondition, not checked: no weight in wh exceeds 32,639, the most that its two limbs can store.
// The indices are decoded as in k_cellsP().
// The loop form, the grid-stride loop, is public (cited at k_cellsP()).
__global__ void k_cellsEH(UM u, long long V, long long d, const i8 *emb, const short *wh, u64 k, u64 y0, u64 nr, u64 ld,
                          i8 *o) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < k * ld; i += (u64)gridDim.x * blockDim.x) {
    const u64 c = i / ld, yy = i % ld;
    u64 r = y0 + yy;
    int v = 0;
    if (yy < nr && r < u.rows && !(c >> u.ncb)) {
      long long ix[3] = {0, 0, 0};
      bool ok = true;
      for (int g = 0; g < u.ng; g++) {
        const u64 lf = r % u.left[g];
        r /= u.left[g];
        const u64 col = (c >> u.cb0[g]) & ((1ull << u.take[g]) - 1);
        u64 w;
        if (u.cls[g] == 'A')
          w = u.take[g] ? col : lf;
        else if (u.cls[g] == 'S')
          w = (lf << u.take[g]) | col;
        else {
          const int rem = u.z[g] - u.take[g];
          w = ((lf >> rem) << u.z[g]) | (col << rem) | (lf & ((1ull << rem) - 1));
        }
        if (w >= u.R[g]) {
          ok = false;
          break;
        }
        ix[g] = (long long)w;
      }
      if (ok) {
        if (u.X == 12)
          v = emb[((long long)r * V + ix[0]) * d + ix[1]];
        else {
          long long w = wh[ix[0] * d + ix[1]];
          // limb r of w
          int bb = 0;
          for (u64 q = 0; q <= r; q++) {
            bb = (int)(((w + 128) & 255) - 128);
            w = (w - bb) >> 8;
          }
          v = bb;
        }
      }
    }
    o[i] = (i8)v;
  }
}
} // namespace g5
