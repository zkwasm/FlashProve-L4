// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Index rules, first half: where in its tensor the weight at a given row and column of the
// committed weight matrix W is stored (W's layout, UnitW and GrpW: prover/common/weight_layout.h).
// For host code and GPU kernels; the second half is hf_range_run_rules.h. hf_range_windows.cuh uses
// the rules to compute the lookup keys of the committer's range proof (WF): a key is defined by row
// and column of W, but a pass over the weights delivers tensors in chunks (`Piece` in identifiers).
//
// A unit is the rows of W that store one tensor (two if its kind, UnitW::X, is 0: the query and key
// matrices). Each index of the tensor has an index group, which splits the index's value v into a
// field (`take` bits of the column number c, from bit cb0 on) and a row part lf < left. By cls:
//   'S': v = lf * 2^take + field          'A': v = field, or lf if take is 0
//   'P': the field is the highest `take` of v's z low bits, and lf is v's other bits, in order
// A weight row r < FoldUnit::weight_rows is the groups' row parts as one number, group 0's varying
// fastest. So a weight's position in its tensor is (a number from r) + (a number from c).
#pragma once
#include "common/field_host.h" // u32, u64, i64, i8, HD
#include <cstddef>
#include <cstring>
#include <vector>
// Unrolls the next loop, for nvcc. A thread's short arrays, then indexed by constants, can be held in registers:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", cited at common/code_layers.cuh.
#if defined(__CUDACC__)
#define FOLD_UNROLL _Pragma("unroll")
#else
#define FOLD_UNROLL
#endif
namespace hfw {
constexpr int FOLD_DIGITS_MOST = 4; // base-256 digits of a weight: 4 in the embedding, 2 elsewhere

// An index group: GrpW's members under their names there (explained at the top of this file), then
// extent: indices >= extent have no weight (GrpW::R; for kind 0's head index, fold_unit()'s heads);
// step, in elements: position in a FoldSource = FoldUnit::bias + the sum over groups of step * v.
struct FoldGroup {
  u64 left = 1, extent = 1;
  i64 step = 0;
  int cls = 'S', z = 0, take = 0, cb0 = 0;
};
// A unit, or the part of one that fold_restrict() cuts out. Its weights are in columns < 2^ncb.
struct FoldUnit {
  u64 off = 0, rows = 0; // the unit's rows of W are off .. off + rows - 1
  u64 weight_rows = 0;   // rows / digits; digit l of weight row r is row off + r + l * weight_rows
  int digits = 2, ncb = 0, groups = 0;
  FoldGroup g[3];
  // After fold_restrict(), group pg has only the row parts row_from .. row_to - 1 and the field
  // values field_from .. field_from + field_count - 1. pg = -1: an entire unit.
  int pg = -1;
  u64 row_from = 0, row_to = 0, field_from = 0, field_count = 0;
  i64 bias = 0;    // minus the position, in the entire tensor, of the source's first element
  u64 columns = 0; // how many columns below 2^ncb have one of these values in group pg's field
};
// A source of weights. Integers: p[0]. Byte planes: p[l][i] is digit l of the weight at position i.
struct FoldSource {
  const void *p[FOLD_DIGITS_MOST] = {nullptr, nullptr, nullptr, nullptr};
};
// The two forms of a source. digits: how many digits fold_cells() fills in; a unit may have fewer.
template <class Int, int NL> struct FoldIntegers {
  static constexpr int digits = NL;
  static constexpr int cell_bytes = (int)sizeof(Int);
};
template <int NL> struct FoldPlanes {
  static constexpr int digits = NL;
  static constexpr int cell_bytes = 1;
};

// ---- the unit -------------------------------------------------------------------------------------------------------
// fold_unit() describes an entire unit from UnitW's and GrpW's members; groups <= 3 is not checked.
// heads: query and key heads together, read for kind 0 only; there the key matrix must follow the
// query matrix in one buffer. Steps: input index fastest, output row, expert (embedding: vocabulary
// index last), as in the model file and ::g5::k_cellsT() (prover/weights_commitment/kernels.cuh).
struct FoldGroupIn {
  u64 R, left;
  int cls, z, take, cb0;
};
inline FoldUnit fold_unit(int kind, u64 off, u64 rows, int digits, int ncb, int groups, const FoldGroupIn *in,
                          u64 heads) {
  FoldUnit u;
  u.off = off, u.rows = rows, u.digits = digits, u.ncb = ncb, u.groups = groups;
  u.weight_rows = rows / (u64)digits;
  for (int g = 0; g < groups; g++) {
    u.g[g].left = in[g].left, u.g[g].extent = in[g].R;
    u.g[g].cls = in[g].cls, u.g[g].z = in[g].z, u.g[g].take = in[g].take, u.g[g].cb0 = in[g].cb0;
  }
  u.g[0].step = 1, u.g[1].step = (i64)in[0].R;
  if (kind == 0)
    u.g[0].step = (i64)(in[1].R * in[2].R), u.g[1].step = (i64)in[2].R, u.g[2].step = 1, u.g[0].extent = heads;
  if (kind == 7 || kind == 8 || kind == 10) u.g[1].step = (i64)(in[2].R * in[0].R), u.g[2].step = (i64)in[0].R;
  if (kind == 12 || kind == 13) u.g[0].step = (i64)in[1].R, u.g[1].step = 1;
  u.columns = (u64)1 << ncb;
  return u;
}
// The same from a UnitW or from any type with its members.
template <class Unit> inline FoldUnit fold_unit_of(const Unit &w, u64 heads) {
  FoldGroupIn in[3] = {};
  const int groups = (int)w.g.size();
  for (int g = 0; g < groups; g++)
    in[g] = FoldGroupIn{(u64)w.g[g].R, (u64)w.g[g].left, (int)w.g[g].cls, w.g[g].z, w.g[g].take, w.g[g].cb0};
  return fold_unit(w.X, w.off, w.rows, w.nlw, w.ncb, groups, in, heads);
}
// The group whose index a chunk restricts (hf::Piece, hf_commit_walk.h): the expert (kind 7, 8, 10)
// or the vocabulary index (12, embedding; 13, output head). -1: the unit arrives in one chunk.
inline int fold_piece_group(int kind) {
  return kind == 7 || kind == 8 || kind == 10 ? 1 : kind == 12 || kind == 13 ? 0 : -1;
}
// Restricts the entire unit u to the weights with index first .. first + count - 1 in group pg >= 0
// (not checked). The source starts at index `origin` there, 0 in the others. Returns false, with u
// partly changed, for class P or if the range is not a range of row parts times a range of fields.
inline bool fold_restrict(FoldUnit &u, int pg, u64 first, u64 count, u64 origin) {
  FoldGroup &g = u.g[pg];
  const u64 field = (u64)1 << g.take;
  u.pg = pg;
  u.row_from = 0, u.row_to = g.left, u.field_from = 0, u.field_count = field;
  if (g.cls == 'A') {
    if (g.take)
      u.field_from = first, u.field_count = count;
    else
      u.row_from = first, u.row_to = first + count;
  } else if (g.cls == 'S') {
    if (first % field == 0 && count % field == 0)
      u.row_from = first / field, u.row_to = (first + count) / field;
    else if (first / field == (first + count - 1) / field)
      u.row_from = first / field, u.row_to = first / field + 1, u.field_from = first % field, u.field_count = count;
    else
      return false;
  } else {
    return false;
  }
  u.bias = -(i64)origin * g.step;
  u.columns = (((u64)1 << u.ncb) >> g.take) * u.field_count;
  return true;
}
// Cuts first .. first + count - 1 into at most three ranges that fold_restrict() accepts for a
// group of class S: the end of a block of 2^take indices, whole blocks, the start of a block.
inline std::vector<std::pair<u64, u64>> fold_ranges(const FoldGroup &g, u64 first, u64 count) {
  std::vector<std::pair<u64, u64>> out;
  const u64 field = (u64)1 << g.take, end = first + count;
  if (g.cls != 'S' || field == 1) return {{first, count}};
  u64 at = first;
  if (at % field) {
    const u64 to = std::min(end, (at / field + 1) * field);
    out.push_back({at, to - at});
    at = to;
  }
  if ((end - at) / field) {
    out.push_back({at, (end - at) / field * field});
    at += (end - at) / field * field;
  }
  if (at < end) out.push_back({at, end - at});
  return out;
}
// Whether a thread may read `wide` adjacent columns, from a multiple of `wide` on, with one load
// from each of the `planes` pointers: the columns are adjacent in the source (group 0: class S,
// step 1, low log2(wide) column bits), loads are aligned, and u's columns come in runs of `wide`.
inline bool fold_wide(const FoldUnit &u, const FoldSource &src, int planes, int cell_bytes, int wide) {
  const FoldGroup &g = u.g[0];
  if (g.cls != 'S' || g.cb0 != 0 || g.step != 1 || ((u64)1 << g.take) % (u64)wide) return false;
  for (int i = 1; i < u.groups; i++)
    if (u.g[i].step % wide) return false;
  if (u.bias % wide || u.columns % (u64)wide || u.pg == 0) return false;
  for (int l = 0; l < planes; l++)
    if ((size_t)src.p[l] % (size_t)(wide * cell_bytes)) return false;
  return true;
}

// The index of group g that has the row part lf and the field 0.
HD u64 fold_row_bits(const FoldGroup &g, u64 lf) {
  if (g.cls == 'A') return g.take ? 0 : lf;
  if (g.cls == 'S') return lf << g.take;
  const int rem = g.z - g.take;
  return ((lf >> rem) << g.z) | (lf & (((u64)1 << rem) - 1));
}
// ---- cells ----------------------------------------------------------------------------------------------------------
// Copies N bytes. In device code N is 1, 2, 4, 8 or 16, both addresses must be multiples of N, and
// the read is one load with the cache hint that the data will not be read again (__ldcs).
// The ideas are public: NVIDIA, CUDA C++ Programming Guide (__ldcs);
// "CUDA Pro Tip: Increase Performance with Vectorized Memory Access" (NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/).
template <int N> HD void fold_load(const void *from, void *to) {
#ifdef __CUDA_ARCH__
  if constexpr (N == 16) {
    *(uint4 *)to = __ldcs((const uint4 *)from);
  } else if constexpr (N == 8) {
    *(uint2 *)to = __ldcs((const uint2 *)from);
  } else if constexpr (N == 4) {
    *(unsigned *)to = __ldcs((const unsigned *)from);
  } else if constexpr (N == 2) {
    *(unsigned short *)to = __ldcs((const unsigned short *)from);
  } else {
    static_assert(N == 1, "a load has 1, 2, 4, 8 or 16 bytes");
    *(unsigned char *)to = __ldcs((const unsigned char *)from);
  }
#else
  memcpy(to, from, N);
#endif
}
// d[l][q]: balanced base-256 digit l of the weight at position at + q. W > 1 requires fold_wide().
template <int W, class Int, int NL>
HD void fold_cells(FoldIntegers<Int, NL>, const FoldSource &src, i64 at, int (&d)[NL][W]) {
  static_assert(sizeof(Int) * W <= 16, "at most 16 bytes a load");
  alignas(16) Int x[W];
  fold_load<(int)sizeof(Int) * W>((const Int *)src.p[0] + at, x);
  FOLD_UNROLL
  for (int q = 0; q < W; q++) {
    int w = (int)x[q];
    FOLD_UNROLL
    for (int l = 0; l < NL; l++) {
      d[l][q] = (int)(i8)w;
      w = (w - d[l][q]) >> 8;
    }
  }
}
template <int W, int NL> HD void fold_cells(FoldPlanes<NL>, const FoldSource &src, i64 at, int (&d)[NL][W]) {
  FOLD_UNROLL
  for (int l = 0; l < NL; l++) {
    alignas(16) i8 x[W];
    fold_load<W>((const i8 *)src.p[l] + at, x);
    FOLD_UNROLL
    for (int q = 0; q < W; q++) d[l][q] = (int)x[q];
  }
}
} // namespace hfw
