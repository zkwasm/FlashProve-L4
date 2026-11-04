// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// WindowCells implements RowCells, the committer program cp_commit's source of M's rows, for an M
// too large to keep in GPU memory, as WholeCells does, or under -rows-in-cuts. Both must return the
// same bytes (tested on small models only, by a test that is not part of this repository).
// M holds the committed weight matrix W as R rows of signed bytes (cells): M[r][(y << lcl) + j] =
// W[y][(r << lcl) + j], j < 2^lcl. Each call of PartMaker::parts(), which builds parts of the
// weight table's tree, reads all of M, in slabs of 16 rows. At full size the weights fill 806 GB of
// M. They are in page-locked host memory as the model's file stores them (about 237 GB, mostly
// quantized blocks of 256 weights), so each read computes a slab's cells from the stored bytes.
// Sizes here are computed from the synthetic weights' shapes and types (-weights made-up-blocks).
// Cuts and windows. A cut is cut_rows (a power of two) consecutive rows of M: the columns of W
// whose numbers agree in all bits from bit lw = lcl + log2(cut_rows) up. A unit (g5::UM) is the
// rows of W that hold one tensor, or two for kind 0 (u.X = 0: query and key). In the layout
// (prover/common/weight_layout.h) each of a unit's column bits is one bit of one index of its
// tensor: of an index's z low bits (2^z divides its extent), class S gives the columns the lowest,
// class P the highest; class A gives all its bits or none. So in each index v a cut fixes at most
// one run of bits, lo .. hi - 1. The part of the tensor with these bits fixed is the cut's window.
// With the fixed bits removed from every index, cv = ((v >> hi) << lo) | (v mod 2^lo), it is a
// dense tensor with the same index order, copied to the GPU in that form. At full size a cut of 64
// rows takes 8 pieces of 11.8 MB of an ffn_gate_exps or ffn_up_exps tensor (half of the outputs of
// 8 neighbouring experts) and 4 whole experts of an ffn_down_exps tensor.
// A window must not cut through a block: a tensor in blocks has a window only if its fastest index
// keeps runs of at least 256 values (32 for Q8_0, whose own blocks hold 32 weights). The two tensors
// of kind 0 (at full size a cut takes every 32nd input) and a tensor of fewer than whole_below bytes
// have no window either. A tensor without a window is copied to GPU memory whole, once, as it is
// stored. Total sizes: see WINDOW_CUT_ROWS.
// Decoding. For each weight row of a unit (a row offset below weight_rows, the product of the
// u.left[g]) and each column of the slab, k_slab_cells() finds the element in the window or whole
// tensor, converts it with stored_integer() and writes the integer's base-256 digits, each to its
// own row y of W. The rule of stored_integer() grew out of VerInf's rule for converting stored
// weights to integers: J. Petrie, VerInf, source code and README, github.com/JamesPetrie/VerInf,
// commit 89903c96, MIT License; see THIRD_PARTY.md, "VerInf", and NOTICE.
// Staging. Two buffers: while one cut's slabs are decoded from stage[now], the next cut's windows
// are copied into the other on a separate stream; held[b] is the cut in stage[b]. rows() is fastest
// when r0 rises from call to call, restarting at 0 after the last row; any order is correct.
// Hidden values: every buffer here holds weights or, like the fault word, depends on them.
// release() zeroes every GPU buffer except `fault`, which is 0 by then. Include this file after
// prover/weights_commitment/lent_memory.h and prover/weights_commitment/common.cuh.
#pragma once
#include "hash_commitment/commit_lie.cuh"
#include "hash_commitment/hf_commit_cells.cuh"
#include "weights_commitment/fold_weights.cuh"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
namespace hf {
// StoredTensor::type: a gguf::Type, or one of these for integers. Values as in hf::PinnedWeights.
constexpr uint32_t STORED_INT16 = 0x10000, STORED_INT32 = 0x10001;
constexpr size_t STORED_NO_GAINS = ~(size_t)0;
struct StoredTensor {
  uint32_t type = 0;
  size_t cells = 0, inputs = 0, bytes = 0; // inputs: the length of a stored row, the fastest index
  const uint8_t *host = nullptr;           // page-locked, registered as one range
  size_t gains = STORED_NO_GAINS;          // where the row's multipliers begin in StoredModel::gains
  int e = 0;                               // the integers are the values times 2^e, rounded
  long long limit = 4095;
  bool wide() const { return type == STORED_INT32 || limit > 32767; } // int32: the embedding
  // A block of the type holds 2^block_bits() elements. 0: the elements are stored one by one.
  int block_bits() const {
    return type == ::gguf::Q4_K || type == ::gguf::Q5_K || type == ::gguf::Q6_K ? 8 : type == ::gguf::Q8_0 ? 5 : 0;
  }
};
static inline size_t stored_bytes(uint32_t type, size_t cells) {
  switch (type) {
  case ::gguf::Q4_K: return cells / 256 * 144;
  case ::gguf::Q5_K: return cells / 256 * 176;
  case ::gguf::Q6_K: return cells / 256 * 210;
  case ::gguf::Q8_0: return cells / 32 * 34;
  case ::gguf::F16:
  case ::gguf::BF16:
  case STORED_INT16: return cells * 2;
  default: return cells * 4; // binary32, STORED_INT32: hf::PinnedWeights holds no tensor of any other type
  }
}
// The weights as they are stored, and the layout. The pointers' targets must outlive every user.
struct StoredModel {
  const rf::LayW *Y = nullptr;
  const std::vector<::g5::UM> *um = nullptr; // the layout's units as the kernels read them
  std::vector<StoredTensor> tensors;         // two for a unit of kind 0, one for every other
  std::vector<std::pair<int, int>> of_unit;  // a unit's tensors: indices into `tensors`; -1: none
  const std::vector<float> *gains = nullptr;
  long long H = 0, G = 0, dh = 0; // query heads, key heads, a head's width
  // Fault injection (hf_commit_doors.h, FORMF_PLANTED_INTEGER): replaces a unit's first integer.
  bool planted = false;
  size_t planted_unit = 0;
  long long planted_value = 0;
};
// Pinned: hf::PinnedWeights (hf_commit_weights.cuh). A template, so this file need not include it.
template <class Pinned> static inline StoredModel stored_model(const Pinned &w, long long H, long long G) {
  StoredModel m;
  m.Y = &w.shape->Y, m.um = &w.shape->um, m.gains = &w.all_gains, m.of_unit = w.of_unit;
  m.H = H, m.G = G, m.dh = (long long)w.shape->sh.dh;
  for (const auto &t : w.tensors) {
    StoredTensor s;
    s.type = t.type, s.cells = t.cells, s.inputs = t.inputs, s.bytes = t.bytes, s.host = t.host;
    s.gains = t.gains, s.e = t.e, s.limit = t.limit;
    m.tensors.push_back(s);
  }
#ifndef SERVING_BUILD
  m.planted = w.planted.on, m.planted_unit = w.planted.unit, m.planted_value = w.planted.value;
#endif
  return m;
}

// ---- the decoder ----
struct TensorView {
  const uint8_t *bytes = nullptr; // GPU: element 0 of the window, or of the whole tensor
  const float *gains = nullptr;   // GPU: indexed by the file's fastest index; null: none
  uint32_t type = 0;
  int wide = 0;
  double scale = 1; // 2^e
  long long limit = 0;
};
struct UnitView {
  ::g5::UM u;
  TensorView t[2];             // t[1]: kind 0's key matrix
  long long H = 0, Gk = 0;     // kind 0: the numbers of query and key heads
  unsigned long long step[3] = {0, 0, 0}; // an element's offset in t[0] is the sum of cv * step[g]
  int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0}; // index g's fixed bits are lo .. hi - 1; lo = hi: none
  int fast = 0;                // the index that runs fastest in a stored row
  unsigned long long weight_rows = 1;
  int planted = 0;
  long long planted_value = 0;
};
__host__ __device__ __forceinline__ unsigned long long without_fixed_bits(unsigned long long v, int lo, int hi) {
  return ((v >> hi) << lo) | (v & ((1ull << lo) - 1));
}
// Returns the committed integer of element `at` of the view; `fastest` is the element's fastest
// index in the file. The rule, step for step, is that of fold::integers(), with its cast to the
// integers' width. Its ldexp((double)p0, e) is written as a product with 2^e: both are exact in
// binary64 (a binary32 number times a power of two, far from the exponent's limits) and so equal.
__device__ __forceinline__ long long stored_integer(const TensorView &t, u64 at, u64 fastest, unsigned *fault) {
  if (t.type == STORED_INT16) return ((const int16_t *)t.bytes)[at];
  if (t.type == STORED_INT32) return ((const int32_t *)t.bytes)[at];
  const float v = ::fold::value(t.bytes, t.type, at);
  const float p0 = t.gains ? __fmul_rn(t.gains[fastest], v) : v;
  if (!isfinite(v) || !isfinite(p0)) atomicOr(fault, ::fold::NOT_FINITE);
  const long long x = __double2ll_rn((double)p0 * t.scale);
  if (x > t.limit || x < -t.limit) atomicOr(fault, ::fold::OUT_OF_RANGE);
  return t.wide ? (long long)(int)x : (long long)(short)x;
}
// k_slab_cells() writes a unit's cells in columns c0 .. c0 + ncols - 1 of W, M's rows c0 >> lcl ..:
// out[((c - c0) >> lcl) * stride + (y << lcl) + (c mod 2^lcl)] = W[y][c]. It writes only cells that
// hold a weight's digit: `out` must be zeroed first. c0 and ncols: multiples of 2^lcl; ncols also
// of 2^lct. Launch: (ncols >> lct, any number) blocks of a multiple of 2^lct threads. The indices'
// row and column parts are g5::k_cellsT's (prover/weights_commitment/kernels.cuh), which cites
// public sources for the loop forms; a tile's rows are prepared once in shared memory, as there.
constexpr int SLAB_TILE_ROWS = 64;
// Which cells hold a digit depends only on the column, except for kind 0's unused heads: an index
// of class S or P is always below its extent (a multiple of 2^z; the column's bits are below bit
// z), and one of class A is only the column's part, or only the row's, which is below its extent.
__global__ void k_slab_cells(const UnitView V, int lcl, u64 c0, int lct, int8_t *__restrict__ out, size_t stride,
                             unsigned *fault) {
  __shared__ u64 row_at[SLAB_TILE_ROWS];
  __shared__ unsigned row_fast[SLAB_TILE_ROWS], row_head[SLAB_TILE_ROWS];
  const ::g5::UM &u = V.u;
  const u64 c = c0 + ((u64)blockIdx.x << lct) + (threadIdx.x & ((1u << lct) - 1));
  const int first = threadIdx.x >> lct, rows_at_once = blockDim.x >> lct;
  bool there = !(c >> u.ncb);
  u64 col_at = 0;
  unsigned col_fast = 0, col_head = 0;
  for (int g = 0; g < u.ng; g++) {
    const u64 ix = ::g5::index_bits_of_column(u, g, c);
    if (u.cls[g] == 'A' && u.take[g] && ix >= u.R[g]) there = false;
    col_at += without_fixed_bits(ix, V.lo[g], V.hi[g]) * V.step[g];
    if (g == V.fast) col_fast = (unsigned)ix;
    if (g == 0) col_head = (unsigned)ix;
  }
  int8_t *const mine = out + ((c - c0) >> lcl) * stride + (c & ((1ull << lcl) - 1));
  const u64 tiles = (V.weight_rows + SLAB_TILE_ROWS - 1) / SLAB_TILE_ROWS;
  for (u64 tile = blockIdx.y; tile < tiles; tile += gridDim.y) {
    const u64 row0 = tile * SLAB_TILE_ROWS;
    __syncthreads();
    for (int i = threadIdx.x; i < SLAB_TILE_ROWS; i += blockDim.x) {
      u64 r = row0 + i, at = 0;
      for (int g = 0; g < u.ng; g++) {
        const u64 ix = ::g5::index_bits_of_row(u, g, r % u.left[g]);
        at += without_fixed_bits(ix, V.lo[g], V.hi[g]) * V.step[g];
        if (g == V.fast) row_fast[i] = (unsigned)ix;
        if (g == 0) row_head[i] = (unsigned)ix;
        r /= u.left[g];
      }
      row_at[i] = at;
    }
    __syncthreads();
    if (!there) continue;
    const int rows_here = (int)(V.weight_rows - row0 < SLAB_TILE_ROWS ? V.weight_rows - row0 : SLAB_TILE_ROWS);
    for (int i = first; i < rows_here; i += rows_at_once) {
      u64 at = row_at[i] + col_at;
      int which = 0;
      if (u.X == 0) {
        const long long head = (long long)(row_head[i] | col_head);
        if (head >= V.H + V.Gk) continue;
        if (head >= V.H) which = 1, at -= (u64)V.H * V.step[0];
      }
      long long w = stored_integer(V.t[which], at, row_fast[i] | col_fast, fault);
      if (V.planted && !c && !(row0 + i)) w = V.planted_value; // all indices 0: the first element of the first tensor
      const u64 y = u.off + row0 + i;
      for (int digit = 0; digit < u.nlw; digit++) {
        const int d = (int)((w + 128) & 255) - 128;
        w = (w - d) >> 8;
        mine[(y + (u64)digit * V.weight_rows) << lcl] = (int8_t)d;
      }
    }
  }
}

// How a unit's tensors are read.
struct UnitLay {
  bool window = false;           // its one tensor is staged cut by cut; false: copied whole, once
  UnitView view;                 // WindowCells fills t[]; a window's address stays null here
  size_t stage_at = 0, stage_bytes = 0;
  int order[3] = {0, 1, 2};      // the indices from the fastest to the slowest in the stored tensor
  int free_bits[3] = {0, 0, 0};  // how many of index g's lowest column bits are free in a cut
  u64 count[3] = {1, 1, 1};      // the extents of the window, or of the whole tensor
};
// What the shapes and stored types determine: which tensors have windows, their forms, the sizes.
struct WindowPlan {
  int cut_rows = 0, out_rows = 0, lw = 0, cuts = 1;
  size_t stride = 0, stage_bytes = 0, whole_bytes = 0, gains_bytes = 0;
  std::vector<UnitLay> lay;
  static int log2_of(u64 x) {
    int l = 0;
    while ((1ull << l) < x) l++;
    return l;
  }
  static size_t rounded(size_t bytes) { return (bytes + 255) & ~(size_t)255; }
  // The GPU memory of a WindowCells with this plan, except the 4 bytes of `fault`.
  size_t bytes() const { return (size_t)out_rows * stride + gains_bytes + whole_bytes + (cuts > 1 ? 2 : 1) * stage_bytes; }
  // Powers of two: 16 <= out_rows <= cut_rows <= R. out_rows: the largest nr in a call of rows().
  // whole_below: a tensor of fewer bytes has no window. Returns null, or why there is no such plan.
  const char *make(const HfParams &P, const StoredModel &model, int cut_rows_, int out_rows_, size_t whole_below) {
    const rf::LayW &Y = *model.Y;
    cut_rows = cut_rows_, out_rows = out_rows_;
    if (Y.k != P.k || Y.m0 != P.m0) return "the rows of M are asked for in another shape";
    if (cut_rows < 16 || (size_t)cut_rows > P.R || (cut_rows & (cut_rows - 1)) || out_rows < 16 || out_rows % 16 ||
        cut_rows % out_rows)
      return "a cut of M's rows is a power of two from 16 to R, and a call's rows divide it";
    lw = P.lcl + log2_of((u64)cut_rows), cuts = (int)(P.R / (size_t)cut_rows);
    stride = rounded((size_t)P.l);
    gains_bytes = model.gains->size() * 4 + 256;
    stage_bytes = whole_bytes = 0;
    lay.assign(Y.U.size(), UnitLay());
    for (size_t unit = 0; unit < Y.U.size(); unit++) {
      const ::g5::UM &u = (*model.um)[unit];
      UnitLay &L = lay[unit];
      UnitView &V = L.view;
      const size_t first = (size_t)model.of_unit[unit].first;
      const StoredTensor &t = model.tensors[first];
      V.u = u, V.H = model.H, V.Gk = model.G;
      for (int g = 0; g < u.ng; g++) V.weight_rows *= u.left[g];
      if (model.planted && model.planted_unit == unit) V.planted = 1, V.planted_value = model.planted_value;
      // The stored order of the indices (g5::k_cellsT's steps).
      if (u.X == 0) L.order[0] = 2, L.order[1] = 1, L.order[2] = 0;
      else if (u.X == 7 || u.X == 8 || u.X == 10) L.order[0] = 0, L.order[1] = 2, L.order[2] = 1;
      else if (u.X == 12 || u.X == 13) L.order[0] = 1, L.order[1] = 0;
      V.fast = L.order[0];
      // The window that a cut would have.
      bool can = u.X != 0 && t.bytes >= whole_below && cuts > 1, any = false;
      for (int g = 0; g < u.ng; g++) {
        L.count[g] = u.R[g];
        L.free_bits[g] = std::max(0, std::min(u.take[g], lw - u.cb0[g]));
        if (L.free_bits[g] == u.take[g]) continue;
        const int shift = u.cls[g] == 'P' ? u.z[g] - u.take[g] : 0;
        V.lo[g] = shift + L.free_bits[g], V.hi[g] = shift + u.take[g];
        // Class A: the index is the column's part, of take = ceil(log2 R) bits: no bit is above the
        // fixed bits. The window has space for 2^lo index values; those from R up are not copied.
        L.count[g] = u.cls[g] == 'A' ? 1ull << V.lo[g] : (u.R[g] >> V.hi[g]) << V.lo[g];
        any = true;
      }
      if (V.lo[V.fast] != V.hi[V.fast] && V.lo[V.fast] < t.block_bits()) can = false; // it would go through a block
      L.window = can && any;
      if (!L.window)
        for (int g = 0; g < 3; g++) V.lo[g] = V.hi[g] = 0, L.count[g] = g < u.ng ? u.R[g] : 1;
      u64 step = 1;
      for (int j = 0; j < u.ng; j++) V.step[L.order[j]] = step, step *= L.count[L.order[j]];
      if (L.window) {
        L.stage_at = stage_bytes, L.stage_bytes = stored_bytes(t.type, (size_t)step);
        stage_bytes += rounded(L.stage_bytes);
        continue;
      }
      whole_bytes += rounded(t.bytes);
      if (model.of_unit[unit].second >= 0) whole_bytes += rounded(model.tensors[(size_t)model.of_unit[unit].second].bytes);
    }
    return nullptr;
  }
};

struct WindowCells : RowCells {
  const HfParams &P;
  const StoredModel model; // a copy; Y, um, gains and the tensors' host bytes must outlive it
  WindowPlan plan;
  std::vector<const uint8_t *> from;               // by unit: a windowed tensor's host bytes
  std::vector<std::pair<uint8_t *, size_t>> whole; // tensors copied whole to the GPU, with sizes
  float *gains = nullptr;
  unsigned *fault = nullptr;
  uint8_t *stage[2] = {nullptr, nullptr};
  int held[2] = {-1, -1}, now = 0;
  cudaStream_t copies = nullptr;
  cudaEvent_t staged[2] = {}, read[2] = {};
  int8_t *out = nullptr;
  size_t copy_calls = 0; // the number of copy calls in the last `stage_cut`

  static void stop(const char *why) {
    fprintf(stderr, "cp_commit: internal error: %s\n", why);
    exit(4);
  }
  // Arguments: those of WindowPlan::make(). Uses the calling thread's current GPU. Returns when the
  // tensors without a window are copied. Exits with code 4 if make() finds no plan.
  WindowCells(const HfParams &P_, const StoredModel &model_, int cut_rows, int out_rows, size_t whole_below)
      : P(P_), model(model_) {
    if (const char *why = plan.make(P, model, cut_rows, out_rows, whole_below)) stop(why);
    commit_room(&gains, plan.gains_bytes);
    CK(cudaMemcpyAsync(gains, model.gains->data(), model.gains->size() * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
    commit_room(&fault, 4);
    CK(cudaMemsetAsync(fault, 0, 4, cudaStreamPerThread));
    from.assign(plan.lay.size(), nullptr);
    for (size_t unit = 0; unit < plan.lay.size(); unit++) {
      UnitLay &L = plan.lay[unit];
      const size_t first = (size_t)model.of_unit[unit].first;
      if (L.window) {
        L.view.t[0] = view_of(model.tensors[first], nullptr);
        from[unit] = model.tensors[first].host;
        continue;
      }
      L.view.t[0] = view_of(model.tensors[first], laid_whole(first));
      if (model.of_unit[unit].second >= 0) {
        const size_t second = (size_t)model.of_unit[unit].second;
        L.view.t[1] = view_of(model.tensors[second], laid_whole(second));
      }
    }
    commit_room(&out, (size_t)plan.out_rows * plan.stride);
    if (plan.stage_bytes) {
      CK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
      for (int b = 0; b < 2; b++) {
        commit_room(&stage[b], plan.stage_bytes);
        CK(cudaEventCreateWithFlags(&staged[b], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&read[b], cudaEventDisableTiming));
      }
    }
    CK(cudaStreamSynchronize(cudaStreamPerThread)); // `gains` too: its copy from pageable memory can return unfinished
  }
  TensorView view_of(const StoredTensor &t, const uint8_t *bytes) const {
    TensorView v;
    v.bytes = bytes, v.gains = t.gains == STORED_NO_GAINS ? nullptr : gains + t.gains;
    v.type = t.type, v.scale = ldexp(1.0, t.e), v.wide = t.wide(), v.limit = t.limit;
    return v;
  }
  const uint8_t *laid_whole(size_t tensor) {
    const StoredTensor &t = model.tensors[tensor];
    uint8_t *here = nullptr;
    const size_t bytes = WindowPlan::rounded(t.bytes);
    commit_room(&here, bytes);
    CK(cudaMemcpyAsync(here, t.host, t.bytes, cudaMemcpyDefault, cudaStreamPerThread));
    whole.push_back({here, bytes});
    return here;
  }
  // Queues the copies of a cut's windows into stage[b], on the stream `copies`. First the host, not
  // that stream, waits for the last read of stage[b]: so no copy is queued before it can run, and
  // the host runs at most about one cut ahead of the GPU. The decoding that it waits for is the
  // first of its slab's work on the GPU, so the GPU has work left while the host queues more.
  void stage_cut(int cut, int b) {
    CK(cudaEventSynchronize(read[b]));
    const u64 c0 = (u64)cut << plan.lw;
    copy_calls = 0;
    for (size_t unit = 0; unit < plan.lay.size(); unit++) {
      const UnitLay &L = plan.lay[unit];
      if (!L.window) continue;
      const ::g5::UM &u = L.view.u;
      const uint32_t type = L.view.t[0].type;
      // j counts the indices from the fastest. Assumed and not checked: j = 0 is not of class A.
      u64 fixed[3] = {0, 0, 0}, run[3] = {1, 1, 1}, R[3] = {1, 1, 1}, count[3] = {1, 1, 1};
      int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
      for (int j = 0; j < u.ng; j++) {
        const int g = L.order[j];
        lo[j] = L.view.lo[g], hi[j] = L.view.hi[g], R[j] = u.R[g], count[j] = L.count[g];
        if (lo[j] != hi[j])
          fixed[j] = ((c0 >> (u.cb0[g] + L.free_bits[g])) & ((1ull << (hi[j] - lo[j])) - 1)) << lo[j];
        run[j] = lo[j] == hi[j] ? count[j] : 1ull << lo[j]; // consecutive values of the index
      }
      auto with_fixed_bits = [&](int j, u64 cv) { return ((cv >> lo[j]) << hi[j]) | fixed[j] | (cv & ((1ull << lo[j]) - 1)); };
      uint8_t *const to = stage[b] + L.stage_at;
      const uint8_t *const source = from[unit];
      const size_t from_row = stored_bytes(type, R[0]), to_row = stored_bytes(type, count[0]);
      const bool whole_rows = run[0] == R[0], whole_planes = whole_rows && run[1] == R[1];
      for (u64 s = 0; s < count[2]; s += whole_planes ? run[2] : 1)
        for (u64 m = 0; m < count[1]; m += run[1]) {
          // An index of class A may reach its extent before its run ends, or before it begins.
          const u64 from_s = with_fixed_bits(2, s), from_m = with_fixed_bits(1, m);
          if (from_s >= R[2] || from_m >= R[1]) continue;
          const u64 from_first = from_s * R[1] + from_m, to_first = s * count[1] + m;
          const u64 rows = whole_planes ? std::min(run[2], R[2] - from_s) * run[1] : std::min(run[1], R[1] - from_m);
          if (whole_rows) {
            CK(cudaMemcpyAsync(to + to_first * to_row, source + from_first * from_row, rows * from_row, cudaMemcpyDefault,
                               copies));
            copy_calls++;
            continue;
          }
          for (u64 f = 0; f < count[0]; f += run[0]) {
            CK(cudaMemcpy2DAsync(to + to_first * to_row + stored_bytes(type, f), to_row,
                                 source + from_first * from_row + stored_bytes(type, with_fixed_bits(0, f)), from_row,
                                 stored_bytes(type, run[0]), rows, cudaMemcpyDefault, copies));
            copy_calls++;
          }
        }
    }
    CK(cudaEventRecord(staged[b], copies));
    held[b] = cut;
  }
  const int8_t *rows(uint32_t r0, int nr, size_t *stride_, cudaStream_t st) override {
    const uint32_t cut_rows = (uint32_t)plan.cut_rows;
    if (r0 % 16 || nr < 1 || nr > plan.out_rows || r0 + (size_t)nr > P.R || r0 / cut_rows != (r0 + nr - 1) / cut_rows)
      stop("M's rows are asked for across a cut, or M has no such rows");
    if (plan.stage_bytes) {
      const int cut = (int)(r0 / cut_rows);
      if (held[now] != cut) {
        if (held[now ^ 1] != cut) stage_cut(cut, now ^ 1);
        now ^= 1;
      }
      CK(cudaStreamWaitEvent(st, staged[now], 0));
      const int next = (cut + 1) % plan.cuts;
      if (held[now ^ 1] != next) stage_cut(next, now ^ 1);
    }
    CK(cudaMemsetAsync(out, 0, (size_t)nr * plan.stride, st));
    const u64 c0 = (u64)r0 << P.lcl, ncols = (u64)nr << P.lcl;
    const int lct = std::min(8, WindowPlan::log2_of(ncols & (~ncols + 1))); // a tile's columns: at most 256; they divide ncols
    for (const UnitLay &L : plan.lay) {
      if (c0 >> L.view.u.ncb) continue; // the unit has no cell in these columns
      UnitView V = L.view;
      if (L.window) V.t[0].bytes = stage[now] + L.stage_at;
      const u64 tiles = (V.weight_rows + SLAB_TILE_ROWS - 1) / SLAB_TILE_ROWS;
      const dim3 grid((unsigned)(ncols >> lct), (unsigned)std::min<u64>(tiles, 65535));
      k_slab_cells<<<grid, 256, 0, st>>>(V, P.lcl, c0, lct, out, plan.stride, fault);
    }
    CK(cudaGetLastError());
    commit_lie::slab(out, plan.stride, r0, nr, P.lcl, P.m0, st); // tests: no-op in this repository
    if (plan.stage_bytes) CK(cudaEventRecord(read[now], st));
    *stride_ = plan.stride;
    return out;
  }
  // Waits for the GPU and returns the decoder's fault word so far (fold::Fault). Any value but 0 is
  // an internal error: check_the_weights() (hf_commit.cu) has converted the same bytes before.
  unsigned faults() const {
    unsigned word = 0;
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&word, fault, 4, cudaMemcpyDeviceToHost));
    return word;
  }
  int most() const override { return plan.out_rows; }
  size_t bytes() const override { return out ? plan.bytes() : 0; }
  // First reads the fault word: on a decoder's fault the program exits before anything is zeroed.
  void release() override {
    if (!out) return;
    if (const unsigned word = faults()) {
      fprintf(stderr, "cp_commit: internal error: the conversion of the stored weights to integers failed in the table's "
                      "making (fault word %u), after the check of the weights had passed\n", word);
      exit(4);
    }
    for (auto &w : whole) {
      CK(cudaMemset(w.first, 0, w.second));
      CK((cudaFree)(w.first));
    }
    whole.clear();
    for (int b = 0; b < 2; b++) {
      if (stage[b]) {
        CK(cudaMemset(stage[b], 0, plan.stage_bytes));
        CK((cudaFree)(stage[b]));
      }
      stage[b] = nullptr, held[b] = -1;
      if (copies) {
        CK(cudaEventDestroy(staged[b]));
        CK(cudaEventDestroy(read[b]));
      }
    }
    if (copies) CK(cudaStreamDestroy(copies));
    copies = nullptr;
    CK(cudaMemset(out, 0, (size_t)plan.out_rows * plan.stride));
    CK((cudaFree)(out));
    out = nullptr;
    CK(cudaMemset(gains, 0, plan.gains_bytes));
    CK((cudaFree)(gains));
    CK((cudaFree)(fault));
    gains = nullptr, fault = nullptr;
  }
  ~WindowCells() override { release(); }
};

// ---- for the main program (hf_commit.cu) ----
// The arguments for every shape: cuts of 16 rows (the smallest allowed; at full size they take the
// least memory), calls of 16 rows, and no window for a tensor below 4 MiB.
// At full size WindowPlan::make() then gives whole_bytes = 1,077,411,840 (tensors without a window)
// and stage_bytes = 1,856,409,600 (a cut's windows; 7,425,638,400 for cuts of 64 rows): worked out,
// not measured, from the synthetic weights' types (-weights made-up-blocks) and from
// prover/examples/target_full_f2/policy.txt: the model's dimensions and the line `table 11 ... 18`
// (R = 2^11, lcl = 18 - 11). By the same rule, worked out by hand: for a file in Q8_0 a cut's windows
// take 3.3 GB and the tensors without a window 1.7 GB; for one in F16 or BF16, 6.2 GB and 3.1 GB.
constexpr int WINDOW_CUT_ROWS = 16, WINDOW_OUT_ROWS = 16;
constexpr size_t WINDOW_WHOLE_BELOW = (size_t)4 << 20;
// The GPU memory that one WindowCells takes (WindowPlan::bytes()).
template <class Shape, class Pinned> static inline size_t window_cells_bytes(const Shape &shape, const Pinned &weights) {
  const StoredModel model = stored_model(weights, 0, 0);
  WindowPlan plan;
  if (const char *why = plan.make(shape.P, model, (int)std::min<size_t>(WINDOW_CUT_ROWS, shape.P.R), WINDOW_OUT_ROWS,
                                  WINDOW_WHOLE_BELOW))
    WindowCells::stop(why);
  return plan.bytes();
}
// The caller deletes the object; `shape` and `weights` must outlive it. H, G: as in StoredModel.
template <class Shape, class Pinned>
static inline RowCells *window_cells(const Shape &shape, Pinned &weights, long long H, long long G) {
  const StoredModel model = stored_model(weights, H, G);
  return new WindowCells(shape.P, model, (int)std::min<size_t>(WINDOW_CUT_ROWS, shape.P.R), WINDOW_OUT_ROWS,
                         WINDOW_WHOLE_BELOW);
}
} // namespace hf
