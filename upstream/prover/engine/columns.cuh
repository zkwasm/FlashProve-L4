// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Computes on the GPU the columns of many rows: a column is the rows' symbols (codeword entries) at
// one position (an index into the codewords). Works from each row's message entries and random tail
// (its r random entries), without a stored codeword and without encoding any row again.
// The symbol of a row at position x (symbol_at() in blocks/code/code.h) is
//   message_scale(x) * sum_j message[j] basis_j(x) + tail_scale(x) * sum_j tail[j] basis_j(x),
// and basis_j(x) is the product of the factors f_i(x) (`GeneratorRow::f`) over the set bits i of j.
// For j = s 2^w + p, p < 2^w, basis_j(x) = weight_s(x) * basis_p(x), where weight_s(x) uses the
// factors from f_w on. One vector (basis_p(x)), p < 2^w, for each position thus serves every slab
// (2^w message entries): `CardProducts` (prover/engine/products.cuh) multiplies each slab by these
// vectors; the sums are added, each times its weight.
#pragma once
#include <vector>

#include "code/code.h"
#include "engine/device.cuh"
#include "engine/pooled.cuh"
#include "engine/encode.cuh" // RowSource
#include "engine/products.cuh"
#include "field/field.h"
#include "field/mle.h"

namespace blk {

// Threads per block in every launch in this file.
constexpr unsigned COLUMN_THREADS = 256;

// at_position[i]: scales and factors at the i-th listed position.
static __global__ void k_column_positions(GeneratorRow *at_position, Code code, const u64 *at, size_t listed) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < listed) at_position[i] = generator_at(code, at[i]);
}
// low[i * 2^w + p] = basis_p at the i-th position, w <= 16. For p = h * 2^w_low + t it is basis_t,
// which thread t computes once, times high[h], the product of f[w_low + b] over h's set bits b.
// Required, not checked: blockDim.x == COLUMN_THREADS; blockIdx.y is the position's number i.
static __global__ void k_column_low(F *low, const GeneratorRow *at_position, unsigned w) {
  static_assert(COLUMN_THREADS == 256 && CardProducts::MOST_LOG_SLAB <= 16, "two halves of at most 8 variables, a thread for each entry of a half");
  __shared__ F high[COLUMN_THREADS];
  const F *f = at_position[blockIdx.y].f;
  const unsigned w_low = w < 8 ? w : 8, w_high = w - w_low;
  if (threadIdx.x < (1u << w_high)) high[threadIdx.x] = product_index(f + w_low, w_high, threadIdx.x);
  __syncthreads();
  if (threadIdx.x >= (1u << w_low)) return;
  const F mine = product_index(f, w_low, threadIdx.x);
  for (u32 h = blockIdx.x; h < (1u << w_high); h += gridDim.x) low[((u64)blockIdx.y << w) + ((u64)h << w_low) + threadIdx.x] = mine * high[h];
}
// total[row * listed + i] += sum over g < many of
//   (the weight of slab s + g at position i) * sums[g * n + row * listed + i].
// n: entries of total (rows times listed); more: log2 of the number of slabs.
static __global__ void k_column_add(F *total, const F *sums, const GeneratorRow *at_position, unsigned w, unsigned more, u64 s, u32 many, u64 n, u64 listed) {
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  F x = total[e];
  for (u32 g = 0; g < many; g++) x += product_index(at_position[e % listed].f + w, more, s + g) * sums[g * n + e];
  total[e] = x;
}
// out[i * n_rows + row] = message_scale * total[row * listed + i]
//   + tail_scale * < tail of the row , basis at position i >. One warp for each (row, i).
static __global__ void k_column_finish(u64 *out, const F *total, const RowSource *rows, u32 n_rows, const GeneratorRow *at_position, size_t listed, unsigned log_r) {
  static_assert(COLUMN_THREADS % 32 == 0, "whole warps");
  const u64 e = ((u64)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (e >= (u64)n_rows * listed) return;
  const u64 row = e % n_rows, i = e / n_rows;
  const RowSource r = rows[row];
  const GeneratorRow &g = at_position[i];
  F tail = F::zero();
  for (u64 j = threadIdx.x % 32; j < r.r; j += 32) tail += F{r.tail[j * r.tail_stride]} * product_index(g.f, log_r, j);
  for (unsigned h = 16; h > 0; h /= 2) tail += F{__shfl_down_sync(~0u, tail.v, h)};
  if (threadIdx.x % 32 == 0) out[i * n_rows + row] = (g.message_scale * total[row * listed + i] + g.tail_scale * tail).v;
}
// out[i * rows.size() + k] = symbol of rows[k] at position at[i], i < listed. at, out: GPU memory.
// Checked, or the process ends (BLK_REQUIRE): the rows are under one code of length 2^log_n whose
// message length is a power of two; there are at most 65,535 of them (`CardProducts`).
// Required, not checked: each position is below the code's length, 2^log_n; messages, tails and
// positions were written on the stream or waited for. The call returns when the work is done.
// way: `PLAIN` multiplies in the field; `BYTES`, signed bytes with cuBLAS (as `PLAIN` if w < 2).
// slabs_at_once: slabs per product; no value changes the symbols. Under `BYTES` a row's slab is cut
// into product_bytes() rows of bytes, each up to 2^16 long, and slabs_at_once slabs' rows of bytes
// are in memory together. 0 chooses the number of slabs that gives about COLUMN_BYTE_ROWS rows of
// bytes (200 MB at w = 16), or one slab if a single slab has more.
// room: allocates memory for max(room, listed) positions. EngineGpu::open() (engine_gpu.cuh) lists,
// through symbols_at(), one position for each distinct draw of the verifier, a number that varies
// from run to run, and passes as room the number of draws, repeats included, so that the
// allocations' sizes and order do not vary. Exception: a call, or a piece of it (below), with no
// position allocates nothing.
#ifndef COLUMN_BYTE_ROWS
#define COLUMN_BYTE_ROWS 3072
#endif
inline void columns(const std::vector<RowSource> &rows, unsigned log_n, const u64 *at, size_t listed, u64 *out, cudaStream_t stream,
                    ProductWay way = ProductWay::BYTES, u32 slabs_at_once = 0, size_t room = 0) {
  if (rows.empty() || listed == 0) return;
  room = std::max(room, listed);
  constexpr size_t AT_ONCE = 1024; // positions at once: each takes 16 * 2^w bytes and its sums
  if (room > AT_ONCE) { // pieces: number and a_room depend on room only; positions split evenly
    const size_t pieces = (room + AT_ONCE - 1) / AT_ONCE, a_room = (room + pieces - 1) / pieces;
    for (size_t p = 0, i = 0; p < pieces; p++) {
      const size_t some = (listed - i + (pieces - p) - 1) / (pieces - p); // <= a_room, since listed <= room; 0 only if listed < pieces
      columns(rows, log_n, at + i, some, out + i * rows.size(), stream, way, slabs_at_once, a_room);
      i += some;
    }
    return;
  }
  const OnStream on(stream);
  const Code code{log_n, (size_t)rows[0].ell, (size_t)rows[0].r};
  BLK_REQUIRE(code_ok(code) && ((u64)1 << code.log_ell()) == code.ell, "columns: log_n, ell and r do not describe a code, or ell (a row's message entries) is not 2^k");
  std::vector<ProductRow> cells;
  for (const RowSource &r : rows) {
    BLK_REQUIRE(r.ell == code.ell && r.r == code.r && r.message_at == code.message_at() && r.tail_at == code.tail_at(), "columns: rows under different codes");
    cells.push_back(ProductRow{r.message, 0, r.message_stride, r.cell});
  }
  const unsigned w = std::min(code.log_ell(), CardProducts::MOST_LOG_SLAB), more = code.log_ell() - w;
  const u32 n_rows = (u32)rows.size();
  const u64 n = (u64)n_rows * listed;
  Pooled<RowSource> d_rows;
  d_rows.from_host(rows.data(), rows.size());
  Pooled<GeneratorRow> at_position(room);
  const u64 slabs = (u64)1 << more;
  if (slabs_at_once == 0) {
    u64 byte_rows = 0;
    for (const ProductRow &r : cells) byte_rows += product_bytes(r.cell);
    slabs_at_once = (u32)std::max<u64>(1, COLUMN_BYTE_ROWS / ((byte_rows + 3) / 4 * 4)); // `CardProducts` allocates a slab's rows of bytes rounded up to a multiple of 4
  }
  slabs_at_once = (u32)std::min<u64>(slabs_at_once, slabs);
  Pooled<F> low(room << w), sums((u64)n_rows * room * slabs_at_once), total((u64)n_rows * room); // the kernels write and read only the listed positions' entries
  k_column_positions<<<blocks_for(listed, COLUMN_THREADS), COLUMN_THREADS, 0, stream>>>(at_position.get(), code, at, listed);
  ENGINE_CK(cudaGetLastError());
  k_column_low<<<dim3(w > 11 ? 8 : 1, (unsigned)listed), COLUMN_THREADS, 0, stream>>>(low.get(), at_position.get(), w);
  ENGINE_CK(cudaGetLastError());
  ENGINE_CK(cudaMemsetAsync(total.get(), 0, n * sizeof(F), stream));
  CardProducts products(cells, w, way, stream, slabs_at_once);
  ProductRight right;
  products.prepare(right, low.get(), (u32)listed, (u32)room);
  low.release(); // `prepare` has queued its reads; freed before `slab` allocates the rows' bytes
  for (u64 s = 0; s < slabs; s += slabs_at_once) {
    const u32 many = (u32)std::min<u64>(slabs_at_once, slabs - s);
    products.slab(s << w, many);
    products.multiply(right, sums.get());
    k_column_add<<<blocks_for(n, COLUMN_THREADS), COLUMN_THREADS, 0, stream>>>(total.get(), sums.get(), at_position.get(), w, more, s, many, n, listed);
    ENGINE_CK(cudaGetLastError());
  }
  k_column_finish<<<blocks_for(n * 32, COLUMN_THREADS), COLUMN_THREADS, 0, stream>>>(out, total.get(), d_rows.get(), n_rows, at_position.get(), listed, code.log_r());
  ENGINE_CK(cudaGetLastError());
  ENGINE_CK(cudaStreamSynchronize(stream));
}

} // namespace blk
