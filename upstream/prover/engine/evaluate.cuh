// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The inner product of a table in GPU memory with a Tensor, a vector in product form
// (blocks/iop/tensor.h), in one pass over the table, for EngineGpu::evaluate()
// (prover/engine/engine_gpu.cuh). In identifiers, `card` means the GPU.
//
// A table has at most 2^k rows (absent rows count as zero) of len = 2^a entries, each of deg cells
// (1: an element of the field F; 3: of its cubic extension E). Entry c of row b ("place" c) has the
// index j = b + 2^k c. Bit t of j is variable t: a row variable if t < k, else a place variable.
// A Tensor is v[j] = scalar * prod_t (bit t of j ? hi[t] : lo[t]) = R[b] * A[low bits of c] *
// B[high bits of c]; the host writes out the short vectors R (with the scalar), A and B.
//
// A Tensor may have a dense factor ("written-out factor" in messages), indexed by some variables:
// one more factor for each place or, if a row variable indexes it, for each entry, which is slower.
// A band (prover/engine/encode.cuh) is a run of rows with one kind of cell.
//
// A pair (lo, hi) = (1, 0) or (0, 1) picks its bit's value (Tensor::picks()): v = 0 elsewhere. A
// Tensor on an aligned block of the table picks the bits above the block. Places where a picked
// bit differs are not read (exception: places_of()), nor rows with R[b] = 0.
#pragma once
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine/device.cuh"
#include "engine/encode.cuh"
#include "field/field.h"
#include "field/field_lazy.cuh"
#include "iop/tensor.h"

namespace blk {

constexpr unsigned EVALUATE_THREADS = 256;
constexpr unsigned EVALUATE_ROWS = 256; // most rows per launch. Shared memory a row: 4 bytes for
                                        // its number, 72 for its weight in three forms.

// A Tensor's dense factor for a kernel: values[d], where bit i of d (i < count) is bit var[i] of j.
struct DenseFactor {
  const E *values = nullptr; // null: no such factor
  unsigned count = 0;
  bool on_rows = false; // some var[i] is a row variable
  unsigned char var[48] = {};
  // The whole struct is a kernel argument. C++ leaves the values of padding bytes open, so a member
  // stands in their place.
  unsigned char unused[3] = {};
  __device__ __forceinline__ E at(u64 j) const {
    u64 d = 0;
    for (unsigned i = 0; i < count; i++) d |= ((j >> var[i]) & 1) << i;
    return values[d];
  }
};
static_assert(sizeof(DenseFactor) == sizeof(DenseFactor::values) + sizeof(DenseFactor::count) + sizeof(DenseFactor::on_rows) + sizeof(DenseFactor::var) + sizeof(DenseFactor::unused),
              "DenseFactor must have no padding bytes: resize `unused`");

// The places that a kernel visits: those with the given bits in the place variables that pick. The
// other variables form runs of neighbours; place number i has the bits of i in them, run after run:
// the bits[r] bits of i from bit from[r] up become those of the place from bit to[r] up.
struct Places {
  static constexpr unsigned MOST_RUNS = 6;
  u64 count = 0, fixed = 0; // how many; the given bits
  unsigned runs = 0;
  unsigned char from[MOST_RUNS] = {}, to[MOST_RUNS] = {}, bits[MOST_RUNS] = {}; // of run r < runs
  unsigned char unused[2] = {}; // as in DenseFactor
  __device__ __forceinline__ u64 at(u64 i) const {
    u64 c = fixed;
    for (unsigned r = 0; r < runs; r++) c |= ((i >> from[r]) & (((u64)1 << bits[r]) - 1)) << to[r];
    return c;
  }
};
static_assert(sizeof(Places) == sizeof(Places::count) + sizeof(Places::fixed) + sizeof(Places::runs) + sizeof(Places::from) + sizeof(Places::to) + sizeof(Places::bits) + sizeof(Places::unused),
              "Places must have no padding bytes: resize `unused`");
// a: how many place variables; mask, value: their picks. With more than MOST_RUNS runs: all 2^a
// places; A or B is then 0 where a picked bit differs.
inline Places places_of(unsigned a, u64 mask, u64 value) {
  Places p;
  unsigned taken = 0;
  for (unsigned t = 0; t < a;) {
    if ((mask >> t) & 1) {
      t++;
      continue;
    }
    unsigned n = 0;
    while (t + n < a && !((mask >> (t + n)) & 1)) n++;
    if (p.runs == Places::MOST_RUNS) {
      Places all;
      all.count = (u64)1 << a, all.runs = 1, all.from[0] = 0, all.to[0] = 0, all.bits[0] = (unsigned char)a;
      return all;
    }
    p.from[p.runs] = (unsigned char)taken, p.to[p.runs] = (unsigned char)t, p.bits[p.runs] = (unsigned char)n;
    p.runs++, taken += n, t += n;
  }
  p.count = (u64)1 << taken, p.fixed = value & mask & (((u64)1 << a) - 1);
  return p;
}

template <unsigned DEG>
__global__ void __launch_bounds__(EVALUATE_THREADS)
    k_evaluate(const void *cells, Cell kind, u64 first_row, const u32 *which, u64 rows, unsigned log_rows, u64 len, Places places, const E *on_rows, const E *low,
               const E *high, unsigned log_low, E lift, DenseFactor dense, E *partial) {
  // Sums over `rows` <= EVALUATE_ROWS rows of one band: rows which[b] of the band, with the weights
  // on_rows[b]. cells: the band's first row, which is row first_row of the table. low, high: A, B,
  // indexed by i. Required, not checked: that bound, and blockDim.x == EVALUATE_THREADS.
  //
  // A thread takes a place c, so a warp reads along a row; it sums the rows' weighted entries at c
  // and multiplies by A and B once. Unless a row variable indexes the dense factor, a place's sum
  // is reduced only at its end (prover/field/field_lazy.cuh): a sum of products
  // (lazy::AccE) or, if DEG = 1 and the cells are not WORD (8-byte elements of F), a cheaper sum of
  // (element of E) x (small integer): lazy::smac(), or lazy::imac() for S24 cells (three signed
  // bytes). I16 (signed) cells are raised by 2^15 for smac(), and lift = 2^15 * sum_b on_rows[b] is
  // subtracted at the end.
  __shared__ u32 row_at[EVALUATE_ROWS];
  __shared__ E row_weight[EVALUATE_ROWS];
  __shared__ lazy::SPrep small_weight[EVALUATE_ROWS];  // the same weights as smac takes them
  __shared__ lazy::IPrep signed_weight[EVALUATE_ROWS]; // and as imac takes them
  static_assert(EVALUATE_ROWS >= EVALUATE_THREADS, "the block's sum is formed in the place of the rows' weights");
  for (u64 b = threadIdx.x; b < rows; b += EVALUATE_THREADS) {
    row_at[b] = which[b];
    row_weight[b] = on_rows[b];
    small_weight[b] = lazy::sprep(on_rows[b]);
    signed_weight[b] = lazy::iprep(on_rows[b]);
  }
  __syncthreads();
  E sum = E::zero();
  for (u64 i = (u64)blockIdx.x * EVALUATE_THREADS + threadIdx.x; i < places.count; i += (u64)gridDim.x * EVALUATE_THREADS) {
    const u64 c = places.at(i);
    E inner;
    if (dense.values && dense.on_rows) {
      inner = E::zero();
      for (u64 b = 0; b < rows; b++) {
        const E weight = row_weight[b] * dense.at(first_row + row_at[b] + (c << log_rows));
        const u64 at = row_at[b] * len + c;
        if (DEG == 1)
          inner += weight * cell_at(cells, kind, at);
        else
          inner += weight * E::from(cell_at(cells, kind, at * 3), cell_at(cells, kind, at * 3 + 1), cell_at(cells, kind, at * 3 + 2));
      }
    } else if (DEG == 1 && kind == Cell::S24) {
      // One sum per byte, joined with the bytes' weights 1, 2^20, 2^40.
      // lazy::imac() allows (calls) x |byte| <= 2^29: here EVALUATE_ROWS x 2^7.
      lazy::IAcc s0 = lazy::izero(), s1 = lazy::izero(), s2 = lazy::izero();
#pragma unroll 2
      for (u64 b = 0; b < rows; b++) {
        const signed char *digit = (const signed char *)cells + 3 * (row_at[b] * len + c);
        const lazy::IPrep weight = signed_weight[b];
        lazy::imac(s0, weight, __ldg(digit)), lazy::imac(s1, weight, __ldg(digit + 1)), lazy::imac(s2, weight, __ldg(digit + 2));
      }
      inner = lazy::ired(s0) + lazy::shifted<S24_BITS>(lazy::ired(s1)) + lazy::shifted<2 * S24_BITS>(lazy::ired(s2));
    } else if (DEG == 1 && kind != Cell::WORD) {
      // lazy::smac() allows (calls) x (greatest integer) <= 2^32: here EVALUATE_ROWS x 2^16.
      lazy::SAcc s = lazy::szero();
      if (kind == Cell::U8) {
#pragma unroll 4
        for (u64 b = 0; b < rows; b++) lazy::smac(s, small_weight[b], (u32)__ldg((const u8 *)cells + row_at[b] * len + c));
      } else if (kind == Cell::U16) {
#pragma unroll 4
        for (u64 b = 0; b < rows; b++) lazy::smac(s, small_weight[b], (u32)__ldg((const unsigned short *)cells + row_at[b] * len + c));
      } else {
#pragma unroll 4
        for (u64 b = 0; b < rows; b++) lazy::smac(s, small_weight[b], (u32)((int)__ldg((const short *)cells + row_at[b] * len + c) + 32768));
      }
      inner = lazy::sred(s);
      if (kind == Cell::I16) inner -= lift;
    } else if (DEG == 1) {
      lazy::AccE s = lazy::zero_e();
#pragma unroll 4
      for (u64 b = 0; b < rows; b++) lazy::mac(s, row_weight[b], F{__ldg((const u64 *)cells + row_at[b] * len + c)});
      inner = lazy::reduce(s);
    } else {
      lazy::AccE s = lazy::zero_e();
      for (u64 b = 0; b < rows; b++) {
        const u64 at = (row_at[b] * len + c) * 3;
        lazy::mac(s, row_weight[b], E::from(cell_at(cells, kind, at), cell_at(cells, kind, at + 1), cell_at(cells, kind, at + 2)));
      }
      inner = lazy::reduce(s);
    }
    if (dense.values && !dense.on_rows) inner *= dense.at(c << log_rows);
    sum += inner * (low[i & (((u64)1 << log_low) - 1)] * high[i >> log_low]);
  }
  // The block's sum, by halves, in row_weight's shared memory.
  __syncthreads();
  E *fold = row_weight;
  fold[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned half = EVALUATE_THREADS / 2; half; half /= 2) {
    if (threadIdx.x < half) fold[threadIdx.x] += fold[threadIdx.x + half];
    __syncthreads();
  }
  if (!threadIdx.x) partial[blockIdx.x] = fold[0];
}

// The 2^count products of v's pairs on the `count` variables in vars, without scalar and dense
// factor: in w[j], bit t of j chooses hi (1) or lo (0) of variable vars[t].
inline std::vector<E> tensor_part(const Tensor &v, const unsigned *vars, unsigned count) {
  std::vector<E> w((size_t)1 << count);
  w[0] = E::one();
  for (unsigned t = 0; t < count; t++) {
    const size_t half = (size_t)1 << t;
    const E lo = v.lo(vars[t]), hi = v.hi(vars[t]);
    for (size_t j = 0; j < half; j++) {
      w[j + half] = w[j] * hi;
      w[j] = w[j] * lo;
    }
  }
  return w;
}

// Waits for `stream` unless all weights R[b] are zero.
// A failed BLK_REQUIRE or CUDA call ends the process.
// bands: rows x len entries of deg cells in GPU memory, row by row in each band. log_rows: k.
// held: if v's dense factor is an Engine's table (blocks/engine/engine.h; Tensor::held), that table
// in GPU memory: elements of E in three canonical words each. The caller must have checked that the
// factor fits within it. Without dense variables such a factor is the one entry v.first.
// stream must keep order with stream 0, which Card<> (a GPU buffer, prover/engine/device.cuh) uses;
// not checked.
inline E evaluate_on_card(const std::vector<Band> &bands, unsigned deg, u64 rows, unsigned log_rows, u64 len, const Tensor &v,
                          cudaStream_t stream = 0, const u64 *held = nullptr) {
  const unsigned a = ceil_log2(len);
  BLK_REQUIRE(((u64)1 << a) == len && v.vars() == log_rows + a, "evaluate: the vector and the table differ in length");
  BLK_REQUIRE(rows <= ((u64)1 << log_rows) && (deg == 1 || deg == 3), "evaluate: not a table");
  BLK_REQUIRE(!v.has_dense() || (v.is_held() ? held != nullptr : v.values != nullptr), "evaluate: a vector with a written-out factor that is not at hand");
  if (!rows) return E::zero();
  std::vector<unsigned> vars;
  for (unsigned t = 0; t < log_rows; t++) vars.push_back(t);
  std::vector<E> on_rows = tensor_part(v, vars.data(), log_rows);
  on_rows.resize(rows);
  for (E &x : on_rows) x = x * v.scalar;
  // A and B: the products over the lower and upper half of the variables in the runs of `places`.
  const Places places = places_of(a, v.pick_mask() >> log_rows, v.pick_value() >> log_rows);
  vars.clear();
  for (unsigned r = 0; r < places.runs; r++)
    for (unsigned t = 0; t < places.bits[r]; t++) vars.push_back(log_rows + places.to[r] + t);
  const unsigned log_low = (unsigned)vars.size() / 2;
  const std::vector<E> low = tensor_part(v, vars.data(), log_low), high = tensor_part(v, vars.data() + log_low, (unsigned)vars.size() - log_low);
  const unsigned most = 132 * 8, need = blocks_for(places.count, EVALUATE_THREADS), blocks = need < most ? need : most;
  DenseFactor dense;
  Card<E> slice;
  if (v.has_dense()) {
    BLK_REQUIRE(v.dense_vars.size() <= sizeof dense.var, "evaluate: too many dense variables");
    static_assert(sizeof(E) == 3 * sizeof(u64), "an element of E is its three words");
    if (v.is_held())
      dense.values = reinterpret_cast<const E *>(held) + v.first; // read in place, not copied
    else
      slice.from_host(v.values->data() + v.first, (size_t)1 << v.dense_vars.size()), dense.values = slice.get();
    dense.count = (unsigned)v.dense_vars.size();
    for (unsigned i = 0; i < dense.count; i++) {
      dense.var[i] = (unsigned char)v.dense_vars[i];
      dense.on_rows = dense.on_rows || v.dense_vars[i] < log_rows;
    }
  }
  // One launch for each run of at most EVALUATE_ROWS rows of one band whose weights are not zero;
  // each leaves a sum for each of its thread blocks. A run's `count` rows are which[at ..] of its
  // band, with the weights weight[at ..].
  struct Run {
    const void *band;
    Cell kind;
    u64 band_first, at, count; // band_first: the band's first row in the table
  };
  std::vector<Run> all;
  std::vector<u32> which;
  std::vector<E> weight;
  u64 band_first = 0;
  for (const Band &band : bands) {
    BLK_REQUIRE(band_first + band.rows <= rows && band.rows <= 0xffffffffu, "evaluate: the bands have another number of rows than the table");
    u64 in_run = EVALUATE_ROWS; // a band begins a run
    for (u64 b = 0; b < band.rows; b++) {
      if (is_zero(on_rows[band_first + b])) continue;
      if (in_run == EVALUATE_ROWS) all.push_back(Run{band.cells, band.cell, band_first, which.size(), 0}), in_run = 0;
      which.push_back((u32)b), weight.push_back(on_rows[band_first + b]);
      all.back().count++, in_run++;
    }
    band_first += band.rows;
  }
  BLK_REQUIRE(band_first == rows, "evaluate: the bands have another number of rows than the table");
  const u64 runs = all.size();
  if (!runs) return E::zero();
  // The four short lists go to the GPU in one copy: the elements first, then the rows' numbers.
  std::vector<E> lists(weight);
  lists.insert(lists.end(), low.begin(), low.end());
  lists.insert(lists.end(), high.begin(), high.end());
  const size_t elements = lists.size();
  lists.resize(elements + (which.size() * sizeof(u32) + sizeof(E) - 1) / sizeof(E), E::zero());
  std::memcpy(lists.data() + elements, which.data(), which.size() * sizeof(u32));
  Card<E> card_lists;
  card_lists.from_host(lists.data(), lists.size());
  const E *card_rows = card_lists.get(), *card_low = card_rows + weight.size(), *card_high = card_low + low.size();
  const u32 *card_which = reinterpret_cast<const u32 *>(card_lists.get() + elements);
  Card<E> partial(blocks * runs);
  for (u64 run = 0; run < runs; run++) {
    const Run &r = all[run];
    E lift = E::zero(); // 2^15 times the sum of the run's weights
    for (u64 b = 0; b < r.count; b++) lift += weight[r.at + b];
    lift = lift * from_u64(32768);
    if (deg == 1)
      k_evaluate<1><<<blocks, EVALUATE_THREADS, 0, stream>>>(r.band, r.kind, r.band_first, card_which + r.at, r.count, log_rows, len, places, card_rows + r.at, card_low, card_high,
                                                             log_low, lift, dense, partial.get() + run * blocks);
    else
      k_evaluate<3><<<blocks, EVALUATE_THREADS, 0, stream>>>(r.band, r.kind, r.band_first, card_which + r.at, r.count, log_rows, len, places, card_rows + r.at, card_low, card_high,
                                                             log_low, lift, dense, partial.get() + run * blocks);
    ENGINE_CK(cudaGetLastError());
  }
  std::vector<E> sums(blocks * runs);
  ENGINE_CK(cudaStreamSynchronize(stream));
  partial.to_host(sums.data());
  E total = E::zero();
  for (const E &x : sums) total += x;
  return total;
}
// The same for a table in one band; v.is_held() must be false.
inline E evaluate_on_card(const void *cells, Cell kind, unsigned deg, u64 rows, unsigned log_rows, u64 len, const Tensor &v,
                          cudaStream_t stream = 0) {
  return evaluate_on_card(std::vector<Band>{Band{cells, kind, rows}}, deg, rows, log_rows, len, v, stream);
}

} // namespace blk
