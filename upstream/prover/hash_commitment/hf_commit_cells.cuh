// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// RowCells is the interface through which phase A of the committer program cp_commit reads M, the
// weight matrix W (m0 rows, k columns) rearranged into R long rows:
//   M[r][y 2^lcl + c'] = W[y][r 2^lcl + c'], a signed byte, for r < R, y < m0, c' < 2^lcl,
// so a row of M has l = m0 2^lcl cells. Phase A (hf_commit_table.cuh) builds a weight table's
// Merkle tree and adds each row's zero padding and masks (random entries).
// R, m0, lcl, l, k: fields of HfTableParams (hf_table_params.h), the base of HfParams.
//
// Implementations in cp_commit: WholeCells (hf_commit_whole.cuh) keeps all of M in GPU memory;
// WindowCells (hf_commit_windows.cuh) decodes rows from the stored tensors at every call; it is
// for the full-size model, whose M does not fit.
//
// Hidden values: the cells are the weights. release() zeroes an object's GPU memory.
#pragma once
#include "common/setting.h" // TESTENV, which the next file reads
#include "weights_commitment/lent_memory.h" // lent::poison(); it comes before every header that allocates GPU memory
#include "common/common.h" // CK
#include "hash_commitment/hf_params.h"
#include <algorithm>
#include <cstdint>
#include <vector>
namespace hf {
// Allocates GPU memory from the driver: the parentheses bypass the macro cudaMalloc of
// prover/weights_commitment/lent_memory.h. In a test build with the environment variable
// LENT_POISON=<byte>, lent::poison() fills the memory with that byte, so that a read of unwritten
// memory gives different output files under different bytes.
template <class T> static inline void commit_room(T **p, size_t bytes) {
  CK((cudaMalloc)((void **)p, bytes));
  CK(::lent::poison(*p, bytes));
}
// All calls on one object must come from one host thread.
struct RowCells {
  // Returns a GPU pointer p to rows r0 .. r0 + nr - 1 of M and sets *stride:
  // M[r0 + t][i] = p[t * *stride + i] for i < l. r0 must be a multiple of 16 and nr at most most()
  // (WholeCells also accepts 16 where R < 16); rows R and above must not be read. The memory is
  // valid until the next rows() or release(). Writes to it are queued on st; a reader on another
  // stream must order its reads itself.
  virtual const int8_t *rows(uint32_t r0, int nr, size_t *stride, cudaStream_t st) = 0;
  virtual int most() const = 0;     // the most rows that a call may ask for
  virtual size_t bytes() const = 0; // the GPU memory that the object holds
  virtual void release() = 0;       // zeroes and frees it
  virtual ~RowCells() {}
};

// A panel contains some rows of W, with all k columns, as g5::k_cellsT()
// (prover/weights_commitment/kernels.cuh) writes it: pan[c * pitch + a] = W[y0 + a][c], a < rows.
// This kernel transposes the panel's cells into rows r = r0 + blockIdx.z of M, for cl < 2^lcl:
//   out[(r - r0) * stride + ((y0 + a) << lcl) + cl] = pan[((r << lcl) + cl) * pitch + a].
// The loops assume blockDim = (32, 8).
// The technique is public: tiled transposition through shared memory with a padded tile:
// M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013,
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
__global__ void k_rows_of_panel(const int8_t *__restrict__ pan, size_t pitch, size_t rows, uint64_t y0, int lcl,
                                uint32_t r0, int8_t *__restrict__ out, size_t stride) {
  __shared__ int8_t tile[32][33];
  const uint32_t r = r0 + blockIdx.z, width = 1u << lcl;
  const size_t a0 = (size_t)blockIdx.x * 32;
  const uint32_t c0 = blockIdx.y * 32;
  for (int j = threadIdx.y; j < 32; j += 8)
    if (c0 + j < width && a0 + threadIdx.x < rows)
      tile[j][threadIdx.x] = pan[(((size_t)r << lcl) + c0 + j) * pitch + a0 + threadIdx.x];
  __syncthreads();
  for (int j = threadIdx.y; j < 32; j += 8)
    if (a0 + j < rows && c0 + threadIdx.x < width)
      out[(size_t)blockIdx.z * stride + ((y0 + a0 + j) << lcl) + c0 + threadIdx.x] = tile[threadIdx.x][j];
}
// Queues the kernel on st. nr must be at most 65,535 (the grid's z dimension). r0 + nr must be at
// most R: not checked.
static inline void rows_of_panel(const HfParams &P, const int8_t *pan, size_t pitch, size_t rows, uint64_t y0, uint32_t r0,
                                 uint32_t nr, int8_t *out, size_t stride, cudaStream_t st) {
  if (!rows || !nr) return;
  const dim3 grid((unsigned)((rows + 31) / 32), (unsigned)((((size_t)1 << P.lcl) + 31) / 32), nr);
  k_rows_of_panel<<<grid, dim3(32, 8), 0, st>>>(pan, pitch, rows, y0, P.lcl, r0, out, stride);
  CK(cudaGetLastError());
}

} // namespace hf
