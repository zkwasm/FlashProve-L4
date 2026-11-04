// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Inner products of many rows of cells with many vectors over the field F, on the GPU, on one slab
// of 2^log_slab places (indices) at a time. A cell stores one entry of a row as a field element or
// a narrow integer (Cell, prover/engine/encode.cuh). For the slab that begins at place `from`,
// CardProducts::multiply() computes
//   sums[row * cols + col] = sum over p < 2^log_slab of
//                            entry(row, from + p) * vectors[col * 2^log_slab + p].
// For vectors of several lengths, none longer than a slab, see prover/engine/tiled.cuh.
//
// Two ways (ProductWay) give the same field elements. PLAIN uses the field's own arithmetic. BYTES
// cuts every number into signed bytes, has the tensor cores sum all products of bytes exactly in 32
// bits (cuBLAS, through prover/common/blas.h; MOST_LOG_SLAB rules out overflow), and combines the
// sums modulo the field's prime.
//
// The caller's duties on streams, and what released memory still holds: see CardProducts. A failed
// BLK_REQUIRE (blocks/base.h) or ENGINE_CK (prover/engine/device.cuh) ends the process.
//
// Public source: numbers cut into bytes for integer tensor cores, sums reduced afterwards: S. Fan
// et al., "TensorFHE", arXiv:2212.14191, section IV-C.
// Public source: that cut and reduction: H. Ootomo, K. Ozaki, R. Yokota, "DGEMM on Integer Matrix
// Multiplication Unit", arXiv:2306.11975, 4.1 (with cublasGemmEx).
// Public source: the pieces of one side stacked into one matrix: Berthomieu, Graillat, Lesnoff,
// Mary, "Multiword matrix multiplication over large finite fields in floating-point arithmetic",
// arXiv:2601.07508, 3.2.
// Public source: reduction put off as long as nothing overflows: J-G. Dumas, P. Giorgi, C. Pernet,
// "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK packages",
// arXiv:cs/0601133, 2.1.2.
// prover/hiding/gram_products.cuh also cites these four.
// Public source: an operand converted once for many products (slab(), prepare()): Intel,
// "Introducing the new Packed APIs for GEMM" (developer article on Intel MKL 2017),
// https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html.
// Public source: weights reordered once, then reused: oneDNN: Intel, oneAPI Deep Neural Network
// Developer Guide and Reference (2024.0), page "Inference", heading "Cache Weights",
// https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2024-0/inference.html;
// under the same heading, https://oneapi-src.github.io/oneDNN/dev_guide_inference.html.
// Public source: a matrix packed once for later products: A. Sharma, E. Dash, M. Vankadari (AMD),
// "BLAS Extension APIs - GEMM Pack and Compute", slides, BLIS Retreat 2024,
// https://cs.utexas.edu/~flame/BLISRetreat2024/slides/Sharma_BLIS_Retreat_2024.pdf.
// Public source: matrix slices stored for reuse: Ootomo, Ozaki, Yokota (above), section 2.3.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/blas.h"
#include "engine/device.cuh"
#include "engine/pooled.cuh"
#include "engine/encode.cuh" // Cell, cell_at
#include "field/field.h"
#include "field/field_lazy.cuh"

namespace blk {

// A row over F: entry j is cell number first + j * stride of `cells`, which is GPU memory. (A row
// over the extension field E is three of these, with stride 3.)
struct ProductRow {
  const void *cells;
  u64 first;
  u32 stride;
  Cell cell;
};
// The signed bytes that BYTES makes of a cell. WORD: the eight of signed_bytes(). U8, U16: each
// unsigned byte minus 128. I16: the low byte minus 128; the high byte is signed already. S24: its
// three signed bytes as stored, S24_BITS bits apart (product_spacing()). k_product_join() makes up
// for the 128s (product_offset()) from the sum of the vector's entries. Byte i of a row's cells on
// a slab is one row of bytes, a row of cuBLAS's left matrix.
BLK_HD inline unsigned product_bytes(Cell c) { // signed bytes of a cell
  return c == Cell::WORD ? 8 : c == Cell::U8 ? 1 : c == Cell::S24 ? 3 : 2;
}
BLK_HD inline unsigned product_spacing(Cell c) { // byte i of a cell has the weight 2^(i * this)
  return c == Cell::S24 ? S24_BITS : 8;
}
BLK_HD inline u64 product_offset(Cell c) { // the cell's value minus its signed bytes' value, in F
  return c == Cell::WORD || c == Cell::S24 ? 0 : c == Cell::U16 ? 128 * 257 : 128;
}
// d[i] = byte i of lazy::signed_digits(x) (prover/field/field_lazy.cuh), so x = sum of d[i] 256^i
// modulo the field's prime.
BLK_HD inline void signed_bytes(F x, signed char d[8]) {
  const u64 all = lazy::signed_digits(x);
  for (int i = 0; i < 8; i++) d[i] = (signed char)(unsigned char)(all >> (8 * i));
}

constexpr unsigned PRODUCT_THREADS = 256, PRODUCT_COLS_AT_ONCE = 8;

struct ProductRowBytes { // a row's rows of bytes: first .. first + count - 1
  u32 first; // its first row of bytes
  u32 count;
};
// A cell's signed bytes, byte i in the bits from 8 i on. As a signed byte, x - 128 is x with its
// top bit flipped.
__device__ __forceinline__ u64 product_cell_bytes(const ProductRow &row, u64 at) {
  switch (row.cell) {
  case Cell::WORD: return lazy::signed_digits(cell_at(row.cells, row.cell, at));
  case Cell::U8: return (u64)(__ldg((const u8 *)row.cells + at) ^ 0x80u);
  case Cell::S24: {
    const u8 *b = (const u8 *)row.cells + 3 * at;
    return (u64)__ldg(b) | (u64)__ldg(b + 1) << 8 | (u64)__ldg(b + 2) << 16;
  }
  default: return (u64)(__ldg((const unsigned short *)row.cells + at) ^ (row.cell == Cell::I16 ? 0x0080u : 0x8080u)); // I16: the high byte is signed already
  }
}
// Writes the rows' bytes for gridDim.z neighbouring slabs of `width` places, the first at place
// `from`. Byte i of row y at place from + g * width + p goes to
//   bytes[(g * all + where[y].first + i) * width + p]      (all: rows of bytes in a slab).
// Grid: y = row, z = slab g. width must be a multiple of 4 and `bytes` aligned to 4 bytes (neither
// is checked): a thread takes four neighbouring places and stores one word of four bytes for each
// row of bytes. __byte_perm() picks the bytes: each hexadecimal digit of the selector selects one
// of the eight bytes of the two operands (NVIDIA, CUDA Math API, "Integer Intrinsics").
static __global__ void k_product_row_bytes(signed char *bytes, const ProductRow *rows, const ProductRowBytes *where, u64 from, u64 width, u32 all) {
  const u64 p = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (p >= width) return;
  const ProductRow row = rows[blockIdx.y];
  u32 *out = reinterpret_cast<u32 *>(bytes + ((u64)blockIdx.z * all + where[blockIdx.y].first) * width + p);
  const u64 step = width / 4, at = row.first + (from + blockIdx.z * width + p) * row.stride;
  if (row.cell == Cell::S24 && row.stride == 1 && ((size_t)row.cells + 3 * at) % 4 == 0) { // the four cells are three whole words
    const u32 *in = reinterpret_cast<const u32 *>((const u8 *)row.cells + 3 * at);
    const u32 a = __ldg(in), b = __ldg(in + 1), c = __ldg(in + 2);
    out[0] = __byte_perm(__byte_perm(a, b, 0x0630), c, 0x5210);        // bytes 0, 3, 6, 9 of the twelve
    out[step] = __byte_perm(__byte_perm(a, b, 0x0741), c, 0x6210);     // 1, 4, 7, 10
    out[2 * step] = __byte_perm(__byte_perm(a, b, 0x0052), c, 0x7410); // 2, 5, 8, 11
    return;
  }
  u64 d[4];
  for (u32 q = 0; q < 4; q++) d[q] = product_cell_bytes(row, at + q * row.stride);
  const u32 count = product_bytes(row.cell);
  // The loop below takes the same steps, in other statements, as the byte transposition in NVIDIA CUTLASS,
  // include/cutlass/transform/thread/transpose.h (BSD-3-Clause); its five selectors (the four values of pick and
  // 0x5410) are those of that transposition. See THIRD_PARTY.md, "NVIDIA CUTLASS".
#pragma unroll
  for (u32 i = 0; i < 8; i++) {
    if (i >= count) break;
    const u32 pick = (i % 4) | (4 + i % 4) << 4; // byte i % 4 of the first operand, then of the second
    const int half = i < 4 ? 0 : 32;
    out[i * step] = __byte_perm(__byte_perm((u32)(d[0] >> half), (u32)(d[1] >> half), pick), __byte_perm((u32)(d[2] >> half), (u32)(d[3] >> half), pick), 0x5410);
  }
}
// bytes[(col * 8 + i) * len + p] = signed byte i of entry p of vector col. Grid: y = col.
static __global__ void k_product_right_bytes(signed char *bytes, const F *right, u64 len) {
  const u64 p = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= len) return;
  signed char d[8];
  signed_bytes(right[(u64)blockIdx.y * len + p], d);
  for (int i = 0; i < 8; i++) bytes[((u64)blockIdx.y * 8 + i) * len + p] = d[i];
}
// total[col] = the sum of vector col's entries. Block x, of PRODUCT_THREADS threads: vector x.
static __global__ void k_product_totals(F *total, const F *right, u64 len) {
  __shared__ F room[PRODUCT_THREADS];
  F s = F::zero();
  for (u64 p = threadIdx.x; p < len; p += PRODUCT_THREADS) s += right[(u64)blockIdx.x * len + p];
  room[threadIdx.x] = s;
  __syncthreads();
  for (unsigned h = PRODUCT_THREADS / 2; h > 0; h /= 2) {
    if (threadIdx.x < h) room[threadIdx.x] += room[threadIdx.x + h];
    __syncthreads();
  }
  if (threadIdx.x == 0) total[blockIdx.x] = room[0];
}
// Combines the 32-bit sums for a row's bytes i < count and a vector's bytes j < 8 into an element:
//   sum over i, j of (sum over q < many of at[q * step + i + j * ld]) * 2^(spacing * i + 8 * j).
// lazy::join_digit() is specified for 64-bit sums below 2^lazy::JOIN_BITS in size (JOIN_BITS = 61).
// With `many` = 1, as called here, each adds at most eight 32-bit sums of at most 2^30
// (MOST_LOG_SLAB), so it is at most 2^33. With `spacing` 8, sums of equal weight are added first.
__device__ __forceinline__ F product_joined(const int *at, u64 ld, u64 many, u64 step, u32 count, unsigned spacing) {
  F x = F::zero();
  if (spacing == 8) {
    long long by_shift[15];
    for (int s = 0; s < 15; s++) by_shift[s] = 0;
    for (u64 q = 0; q < many; q++)
      for (u32 j = 0; j < 8; j++) // a row's sums for one byte of the vector are neighbours in memory
        for (u32 i = 0; i < count; i++) by_shift[i + j] += at[q * step + j * ld + i];
    for (u32 s = 0; s < count + 7; s++) x += lazy::join_digit(by_shift[s], 8 * (int)s);
    return x;
  }
  long long pair[3][8]; // cells with another spacing have at most three bytes
  for (u32 i = 0; i < 3; i++)
    for (u32 j = 0; j < 8; j++) pair[i][j] = 0;
  for (u64 q = 0; q < many; q++)
    for (u32 j = 0; j < 8; j++)
      for (u32 i = 0; i < count && i < 3; i++) pair[i][j] += at[q * step + j * ld + i];
  for (u32 i = 0; i < count && i < 3; i++)
    for (u32 j = 0; j < 8; j++) x += lazy::join_digit(pair[i][j], (int)(spacing * i + 8 * j));
  return x;
}
// Computes the inner products for the vectors col0 .. col0 + now - 1 of cols, on `many` slabs.
// raw[g * all + where[row].first + i + (c * 8 + j) * ld] is the 32-bit sum on slab g for byte i of
// the row and byte j of vector col0 + c. One thread for each (row, g, c), rows fastest.
static __global__ void k_product_join(F *sums, const int *raw, u64 ld, const ProductRow *rows, const ProductRowBytes *where, u32 n_rows, u32 cols, u32 col0, u32 now,
                                      const F *total, u32 all, u32 many) {
  const u64 e = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= (u64)n_rows * many * now) return;
  const u64 row = e % n_rows, g = e / n_rows % many, c = e / ((u64)n_rows * many), col = col0 + c;
  const ProductRowBytes w = where[row];
  sums[(g * n_rows + row) * cols + col] =
      F{product_offset(rows[row].cell)} * total[col] + product_joined(raw + g * all + w.first + c * 8 * ld, ld, 1, 0, w.count, product_spacing(rows[row].cell));
}
// The same inner products by the field's arithmetic, on one slab. Block (x, y), of PRODUCT_THREADS
// threads: row x, the vectors y * PRODUCT_COLS_AT_ONCE and following.
static __global__ void k_product_plain(F *sums, const ProductRow *rows, u64 from, u64 len, const F *right, u32 cols) {
  __shared__ F room[PRODUCT_THREADS];
  const ProductRow row = rows[blockIdx.x];
  const u32 col0 = blockIdx.y * PRODUCT_COLS_AT_ONCE;
  F s[PRODUCT_COLS_AT_ONCE];
  for (unsigned j = 0; j < PRODUCT_COLS_AT_ONCE; j++) s[j] = F::zero();
  for (u64 p = threadIdx.x; p < len; p += PRODUCT_THREADS) {
    const F cell = cell_at(row.cells, row.cell, row.first + (from + p) * row.stride);
    for (unsigned j = 0; j < PRODUCT_COLS_AT_ONCE && col0 + j < cols; j++) s[j] += cell * right[(u64)(col0 + j) * len + p];
  }
  for (unsigned j = 0; j < PRODUCT_COLS_AT_ONCE && col0 + j < cols; j++) {
    room[threadIdx.x] = s[j];
    __syncthreads();
    for (unsigned h = PRODUCT_THREADS / 2; h > 0; h /= 2) {
      if (threadIdx.x < h) room[threadIdx.x] += room[threadIdx.x + h];
      __syncthreads();
    }
    if (threadIdx.x == 0) sums[(u64)blockIdx.x * cols + col0 + j] = room[0];
    __syncthreads();
  }
}

enum class ProductWay { PLAIN, BYTES };

// Vectors that prepare() has made ready; multiply() can use them any number of times.
struct ProductRight {
  u32 cols = 0;
  u32 room = 0;              // at least cols; `prepare` and `multiply` size their buffers by it
  Pooled<F> words;           // PLAIN: the vectors themselves
  Pooled<signed char> bytes; // BYTES: their signed bytes,
  Pooled<F> total;           //        and each vector's sum
};

// `Card` in names means the GPU. Call prepare() and slab(), then multiply(). The rows' bytes are
// made once for a slab and serve all multiplications on it.
// Streams. All work is put on the stream that the constructor is given. What the caller passes in
// (the rows' cells, the vectors) must be complete for that stream before the call: written on it,
// or waited for. Nothing here can check that, and cells that are not yet written are read as
// whatever the memory holds. The cells are read in slab() if by_bytes(log_slab()) holds, otherwise
// in multiply(). Which streams are ordered with which depends on how the program is built:
// prover/engine/arena.h explains it. With stream 0, blas::blas() puts cuBLAS's work on
// cudaStreamPerThread, which is stream 0 itself in a build with --default-stream per-thread and is
// ordered with the legacy stream 0 otherwise.
// Memory. Work spaces are Pooled (GPU memory, prover/engine/pooled.cuh; OnStream sets their stream)
// and are not cleared when they are released: they hold the rows' cells (as bytes) and inner
// products of them. The one exception is cuBLAS's work space, which release_work() zeroes.
// multiply() borrows the thread's cuBLAS handle and restores its settings (blas::Borrowed).
class CardProducts {
public:
  static constexpr unsigned MOST_LOG_SLAB = 16; // 2^16 products of signed bytes add up to at most 2^30
  size_t raw_words = (size_t)1 << 26; // cap (256 MB) on a cuBLAS call's 32-bit sums: see `multiply`

  // slabs_at_once: the largest number of neighbouring slabs that one call of slab() can select.
  CardProducts(const std::vector<ProductRow> &rows, unsigned log_slab, ProductWay way, cudaStream_t stream, u32 slabs_at_once = 1)
      : n_rows_((u32)rows.size()), log_slab_(log_slab), way_(way), stream_(stream), at_once_(slabs_at_once) {
    const OnStream on(stream);
    BLK_REQUIRE(log_slab <= MOST_LOG_SLAB, "products: a slab of more than 2^16 places");
    BLK_REQUIRE(slabs_at_once >= 1 && slabs_at_once <= 65535, "products: no slab at a time, or more than 65,535");
    BLK_REQUIRE(rows.size() <= 65535, "products: more than 65,535 rows");
    rows_.from_host(rows.data(), rows.size());
    std::vector<ProductRowBytes> where;
    for (const ProductRow &r : rows) {
      where.push_back(ProductRowBytes{byte_rows_, product_bytes(r.cell)});
      byte_rows_ += product_bytes(r.cell);
    }
    where_.from_host(where.data(), where.size());
    if (way_ == ProductWay::BYTES) {
      // cuBLAS's integer tensor-core kernels need a left matrix whose number of rows is a multiple
      // of 4: rows of zeros pad it.
      padded_ = (byte_rows_ + 3) / 4 * 4;
      work_.resize(blas::BLAS_WORKSPACE);
    }
  }
  // For timing reports: if `trace` is true, every step is waited for and its seconds are added up
  // here.
  bool trace = false;
  double seconds_row_bytes = 0, seconds_prepare = 0, seconds_multiply = 0, seconds_join = 0;

  u32 rows() const { return n_rows_; }
  const ProductRow *rows_on_card() const { return rows_.get(); }
  unsigned log_slab() const { return log_slab_; }
  bool by_bytes(unsigned w) const { return way_ == ProductWay::BYTES && w >= 2; } // the integer kernels need lengths (2^w) that are multiples of 4

  // Selects `many` neighbouring slabs, the first beginning at place `from`. Not checked: every row
  // has at least from + many * 2^log_slab entries.
  void slab(u64 from, u32 many = 1) {
    BLK_REQUIRE(!released_, "products: a slab is asked for after the work spaces were given back");
    BLK_REQUIRE(many >= 1 && many <= at_once_, "products: more slabs at a time than the constructor was told");
    from_ = from, many_ = many, turned_ = true;
    if (!by_bytes(log_slab_) || n_rows_ == 0) return; // (on a slab of fewer than four places nothing is multiplied by bytes)
    const u64 width = (u64)1 << log_slab_;
    const double t0 = now();
    if (row_bytes_.size() == 0) { // allocated late: a caller can first release what `prepare` read
      const OnStream on(stream_);
      row_bytes_.resize(((size_t)padded_ * at_once_) << log_slab_);
      ENGINE_CK(cudaMemsetAsync(row_bytes_.get(), 0, row_bytes_.size(), stream_)); // (the rows of zeros; no kernel writes them)
    }
    k_product_row_bytes<<<dim3(blocks_for(width / 4, PRODUCT_THREADS), n_rows_, many), PRODUCT_THREADS, 0, stream_>>>(row_bytes_.get(), rows_.get(), where_.get(), from,
                                                                                                              width, padded_);
    ENGINE_CK(cudaGetLastError());
    seconds_row_bytes += now() - t0;
  }
  u64 from() const { return from_; }

  // vectors: cols vectors of 2^log_slab entries in GPU memory, one after the other. They are read
  // on the stream and need not be kept after that.
  // room: memory is allocated as for this many vectors (0: cols). A caller whose number of vectors
  // is sampled (an opening's list of codeword positions is its draws without repeats) passes the
  // number that depends only on the shape (the number of draws), so that all allocations have the
  // same sizes in every run.
  void prepare(ProductRight &r, const F *vectors, u32 cols, u32 room = 0) {
    const OnStream on(stream_);
    room = std::max(room, cols);
    BLK_REQUIRE(room <= 65535, "products: more than 65,535 vectors");
    r.cols = cols, r.room = room;
    const u64 len = (u64)1 << log_slab_;
    if (cols == 0) return;
    const double t0 = now();
    if (!by_bytes(log_slab_)) {
      r.words.resize(room * len);
      ENGINE_CK(cudaMemcpyAsync(r.words.get(), vectors, cols * len * sizeof(F), cudaMemcpyDeviceToDevice, stream_));
      seconds_prepare += now() - t0;
      return;
    }
    r.bytes.resize((size_t)room * 8 * len);
    r.total.resize(room);
    k_product_right_bytes<<<dim3(blocks_for(len, PRODUCT_THREADS), cols), PRODUCT_THREADS, 0, stream_>>>(r.bytes.get(), vectors, len);
    ENGINE_CK(cudaGetLastError());
    k_product_totals<<<cols, PRODUCT_THREADS, 0, stream_>>>(r.total.get(), vectors, len);
    ENGINE_CK(cudaGetLastError());
    seconds_prepare += now() - t0;
  }
  // Writes sums[(g * rows + row) * cols + col] for the g-th selected slab: slabs * rows * cols
  // elements in GPU memory, in the stream's order. Not checked: r is from this object's prepare().
  // BYTES: each cuBLAS call takes all selected slabs and as many vectors as raw_words allows, at
  // least one. The vectors are the same on every slab, so the slabs' rows of bytes, one slab under
  // the other, are one left matrix. One slab of few rows gives cuBLAS too few tiles to fill the
  // GPU.
  void multiply(const ProductRight &r, F *sums) {
    if (n_rows_ == 0 || r.cols == 0) return;
    BLK_REQUIRE(turned_ && !released_, "products: a multiplication before the first slab, or after the work spaces were given back");
    const OnStream on(stream_);
    const u64 len = (u64)1 << log_slab_;
    if (!by_bytes(log_slab_)) {
      const double t0 = now();
      for (u32 g = 0; g < many_; g++) {
        k_product_plain<<<dim3(n_rows_, (r.cols + PRODUCT_COLS_AT_ONCE - 1) / PRODUCT_COLS_AT_ONCE), PRODUCT_THREADS, 0, stream_>>>(sums + (u64)g * n_rows_ * r.cols, rows_.get(),
                                                                                                                                 from_ + g * len, len, r.words.get(), r.cols);
        ENGINE_CK(cudaGetLastError());
      }
      seconds_multiply += now() - t0;
      return;
    }
    const u64 m = (u64)padded_ * many_;
    BLK_REQUIRE(m <= 0x7fffffffu, "products: more rows of bytes in one product than the library's 32-bit sizes take"); // by default `columns` (columns.cuh) keeps it near COLUMN_BYTE_ROWS
    const u32 at_once = (u32)std::max<u64>(1, std::min<u64>(r.room, raw_words / (m * 8)));
    if (raw_.size() < m * 8 * at_once) raw_.resize(m * 8 * at_once);
    work_.spent();
    const blas::Borrowed borrowed(stream_, work_.get(), work_.size());
    for (u32 col = 0; col < r.cols; col += at_once) {
      const u32 now = std::min(at_once, r.cols - col);
      const double t0 = this->now();
      raw_.spent();
      blas::gemm_tn(borrowed.handle, (int)m, (int)(now * 8), (int)len, (const int8_t *)row_bytes_.get(), len,
                    (const int8_t *)r.bytes.get() + (u64)col * 8 * len, len, raw_.get(), m, false);
      const double t1 = this->now();
      k_product_join<<<blocks_for((u64)n_rows_ * many_ * now, PRODUCT_THREADS), PRODUCT_THREADS, 0, stream_>>>(sums, raw_.get(), m, rows_.get(), where_.get(), n_rows_, r.cols, col,
                                                                                                          now, r.total.get(), padded_, many_);
      ENGINE_CK(cudaGetLastError());
      seconds_multiply += t1 - t0, seconds_join += this->now() - t1;
    }
  }

  // Releases the work spaces. rows_on_card() stays valid; no slab() or multiply() may follow.
  void release_work() {
    if (work_.size()) ENGINE_CK(cudaMemsetAsync(work_.get(), 0, work_.size(), stream_)); // prover/common/blas.h: the calling code zeroes cuBLAS's work space
    row_bytes_.release(), raw_.release(), work_.release();
    released_ = true;
  }
  ~CardProducts() { release_work(); }

  double now() const {
    if (!trace) return 0;
    ENGINE_CK(cudaStreamSynchronize(stream_));
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

private:
  u32 n_rows_, byte_rows_ = 0, padded_ = 0;
  unsigned log_slab_;
  bool turned_ = false, released_ = false; // slab() was called; release_work() was called
  ProductWay way_;
  cudaStream_t stream_;
  u32 at_once_, many_ = 1; // slabs: the most that `slab` may select; the number selected
  u64 from_ = 0;
  Pooled<ProductRow> rows_;
  Pooled<ProductRowBytes> where_;
  Pooled<signed char> row_bytes_;
  Pooled<int> raw_;
  Pooled<char> work_;
};

} // namespace blk
