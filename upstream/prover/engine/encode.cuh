// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU encoder for the rows that a Merkle tree commits to: it encodes up to sixteen rows at a time
// (a slab) and hashes each position's symbols (a column) into that position's tree leaf as they are
// computed, so that SlabEncoder::leaves() stores no codeword. The kernels and SlabEncoder's
// function bodies are in prover/engine/encode_kernels.cuh, which this file includes, so that each
// program compiles a separate copy of them: see "How the kernels are built" below.
// Public source of the scheme: S. Ames, C. Hazay, Y. Ishai, M. Venkitasubramaniam, "Ligero:
// Lightweight Sublinear Arguments Without a Trusted Setup" (ePrint 2022/1608): the rows are
// encoded, the columns of the matrix of codewords are hashed, drawn columns are sent.
//
// A row is a message and a random tail under a Code (blocks/code/code.h). Its codeword is what
// Encoder::evaluate() there computes on n = 2^log_n positions: layers top - 1 down to 0, with
// top = Code::log_dim(); layer L turns values (a, b), 2^L apart, into (a + t b, a - t b) for a
// factor t. The layers from top up only copy and are not computed: a slab's first pass (see
// SlabEncoder) reads coefficient x mod 2^top (FFTW, "Pruned FFTs",
// https://www.fftw.org/pruned.html), with the largest top of the slab's rows. So any rate works.
//
// Public source: layers grouped into passes over GPU memory: S. Kim, W. Jung, J. Park, J. H. Ahn,
// "Accelerating Number Theoretic Transformations for Bootstrappable Homomorphic Encryption on
// GPUs", arXiv 2012.01968, sections V and VI. Public source also for several layers on values held
// in registers, exchanged through shared memory.
// Public source: layers grouped into passes: A. S. Ozcan, E. Savas, "Two Algorithms for Fast GPU
// Implementation of NTT", ePrint 2023/1410, section V.
// Public source: many independent calls of a hash on a GPU (the last pass of leaves(): one thread
// for each position): T. Lu, Y. Chen, Z. Wang, X. Wang, W. Chen, J. Zhang, "BatchZK: A Fully
// Pipelined GPU-Accelerated System for Batch Generation of Zero-Knowledge Proofs", IACR ePrint
// 2024/1862, section 3.1.
// Public source: a transform with its factors in bit-reversed order: P. Longa, M. Naehrig,
// "Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography", ePrint
// 2016/504, section 2.2 and Algorithm 1 (there for the negacyclic transform).
// The kernels' other public sources are cited in prover/engine/encode_kernels.cuh.
//
// The first pass reads cells and tails directly, so that no coefficient is written out first; the
// factors t are transform_table()'s (blocks/code/code.h), one heap for all layers; the hash is the
// leaf hash of blocks/hash/hash.h, on blocks of 16 words.
// The arithmetic of the layers is prover/field/'s (`PassFactors` in
// prover/engine/encode_kernels.cuh): up to four layers run as one transform whose inner factors
// are shifts (prover/field/transform_groups_gl64.cuh). Between two such transforms a value is one
// 64-bit word that need not be canonical.
#pragma once
#include <type_traits>
#include <vector>

#include "code/code.h"
#include "engine/device.cuh"
#include "field/field.h"
#include "field/field_lazy.cuh"
#include "hash/hash.h"
#include "tree/tree.h"

namespace blk {
using lazy::TransformStep;

// How a message's entries are stored in GPU memory.
// S24 is `SYMBOLS` of blocks/iop/req_arrays.h at 20 bits (`S24_BITS`).
enum class Cell : u32 {
  WORD = 0, // canonical words of F, 8 bytes
  U8 = 1,   // integers 0 .. 255
  U16 = 2,  // integers 0 .. 65,535
  I16 = 3,  // integers -32,768 .. 32,767
  S24 = 4,  // three neighbouring signed bytes b0, b1, b2: the integer b0 + 2^20 b1 + 2^40 b2
};
constexpr unsigned S24_BITS = 20;
BLK_HD inline size_t cell_bytes(Cell c) { return c == Cell::WORD ? 8 : c == Cell::U8 ? 1 : c == Cell::S24 ? 3 : 2; }

// A table of rows may be stored in GPU memory as several bands: runs of consecutive whole rows with
// one kind of cell, each run stored row by row at a separate address. So the few rows that hold
// whole field elements do not widen the others.
struct Band {
  const void *cells = nullptr;
  Cell cell = Cell::WORD;
  size_t rows = 0;
};
constexpr unsigned MOST_BANDS = 8;

// One row over F (blocks/field/field.h) as the encoder reads it. A row over E, its cubic extension,
// is three of these, with a stride of 3.
// Not checked here: message_at + ell and tail_at + r are at most the call's 2^log_n.
// Nothing here draws the random tail: the caller supplies it.
struct RowSource {
  const void *message; // in GPU memory: entry j is cell number j * message_stride
  const u64 *tail;     // in GPU memory: entry j is word number j * tail_stride
  u32 message_stride, tail_stride;
  Cell cell;
  u64 ell, r, message_at, tail_at; // the code's numbers: lengths of message and tail, and their first coefficients
};

__device__ __forceinline__ F cell_at(const void *base, Cell kind, u64 i) {
  switch (kind) {
  case Cell::WORD: return F{__ldg((const u64 *)base + i)};
  case Cell::U8: return F{(u64)__ldg((const u8 *)base + i)};
  case Cell::U16: return F{(u64)__ldg((const unsigned short *)base + i)};
  case Cell::S24: {
    const signed char *b = (const signed char *)base + 3 * i;
    return from_i64((i64)__ldg(b) + (i64)__ldg(b + 1) * ((i64)1 << S24_BITS) + (i64)__ldg(b + 2) * ((i64)1 << (2 * S24_BITS)));
  }
  default: return from_i64((i64)__ldg((const short *)base + i));
  }
}
// A table's bands as a kernel argument. `Card` in names: GPU memory. A real row is a stored one; a
// table's other rows are zero.
struct CardBands {
  const void *cells[MOST_BANDS] = {};
  Cell cell[MOST_BANDS] = {};
  u64 first_row[MOST_BANDS + 1] = {}; // first_row[count] = the number of real rows
  unsigned count = 0;
  u64 row_cells = 0; // cells of a row: its entries times the cells of an entry
  // The band of a row, which must be real (not checked).
  __device__ __forceinline__ unsigned band_of(u64 row) const {
    unsigned b = 0;
    while (b + 1 < count && row >= first_row[b + 1]) b++;
    return b;
  }
  // Cell number i of a real row.
  __device__ __forceinline__ F at(u64 row, u64 i) const {
    const unsigned b = band_of(row);
    return cell_at(cells[b], cell[b], (row - first_row[b]) * row_cells + i);
  }
};
inline CardBands card_bands(const std::vector<Band> &bands, u64 row_cells) {
  static_assert(MOST_BANDS == 8, "the number stands in the message below and in EngineGpu::adopt()'s");
  BLK_REQUIRE(!bands.empty() && bands.size() <= MOST_BANDS, "table: its rows must lie in 1 to 8 bands (runs of rows with one kind of cell)");
  CardBands c;
  c.count = (unsigned)bands.size(), c.row_cells = row_cells, c.first_row[0] = 0;
  for (unsigned b = 0; b < c.count; b++) c.cells[b] = bands[b].cells, c.cell[b] = bands[b].cell, c.first_row[b + 1] = c.first_row[b] + bands[b].rows;
  return c;
}
__device__ __forceinline__ F coefficient(const RowSource &s, u64 j) {
  if (j - s.message_at < s.ell) return cell_at(s.message, s.cell, (j - s.message_at) * s.message_stride);
  if (j - s.tail_at < s.r) return F{__ldg(s.tail + (j - s.tail_at) * s.tail_stride)};
  return F::zero();
}

constexpr int SLAB_ROWS = 16; // rows that go through the encoder together

// What the last pass needs to hash the tree leaves. The hashed string of position x, in words of 8
// bytes: two label words (`LABEL_WORDS`, blocks/hash/hash.h), the `width` symbols of this call's
// rows, the copied symbols, zeros up to salt_at, the salt. SlabEncoder::leaves() sets cv itself.
template <class Fn> struct LeafChainT {
  u64 *cv;         // chaining values between slabs: position x's begin at cv + x * (Fn::CVW / 2)
  const u64 *salt; // salt_words words for each position, position by position
  Digest *leaves;  // the finished leaf of position x goes to leaves[x]
  u32 salt_words;
  TagT<Fn> tag;    // of the leaves' hash calls
  u32 width;       // symbols of a column that come from this call's rows
  u32 salt_at;     // word of the hashed string, label words counted, at which the salt begins
                   // (`leaf_salt_at` / 8), at or after the last symbol's end; zeros between
  // Symbols of shorter codewords, copied into every leaf after the rows' own: for i < copied_width,
  // symbol width + i of the leaf of position x is
  // copied[i * 2^log_short + short_position(log_n, log_short, x)] (blocks/code/short_position.h).
  const u64 *copied = nullptr;
  u32 copied_width = 0, log_short = 0;
  BLK_HD u32 words() const { return salt_at + salt_words; } // the hashed string in words of 8 bytes
};
using LeafChain = LeafChainT<HashFn>;

// The transform's tables in GPU memory, one for each length; of() copies one there on first use.
class CardTable {
public:
  const u64 *of(unsigned log_n) {
    if (log_n >= made_.size()) made_.resize(log_n + 1);
    if (!made_[log_n].size()) {
      const auto host = transform_table(log_n);
      static_assert(sizeof(F) == sizeof(u64), "an F is its word");
      made_[log_n].from_host(reinterpret_cast<const u64 *>(host->data()), host->size());
    }
    return made_[log_n].get();
  }

private:
  std::vector<Card<u64>> made_;
};

// How the kernels are built. Reading the source does not vouch for a build of these kernels (for a
// compiler fault, see powers() in prover/field/transform_groups_gl64.cuh), so what a test shows
// holds of the machine code that the test ran, and of no other compilation.
//   Without ENGINE_ENCODER_OBJECT  this file includes the kernels: every program and every test
//                                  compiles a separate copy.
//   With ENGINE_ENCODER_OBJECT     it only declares: a program must link one object that has the
//                                  kernels, and tests and provers built so would run the same
//                                  machine code.
// The inline namespaces below give the two ways different names for the linker, so neither can
// stand in for the other. The source file and the script that make the object are not part of this
// repository, and no script in it builds a program with the macro defined: prover/build.sh and
// prover/tests/unit.sh compile the kernels with each program. So what a test of the encoder shows
// holds only of that test's compilation, not of the copies in `cp_commit` and `cp_prove`.
#ifdef ENGINE_ENCODER_OBJECT
inline namespace encoder_object {
#else
inline namespace encoder_in_this_program {
#endif
// Runs the rows of one tree through the encoder, slab by slab: strided passes of 4 to 8 layers
// each, in a work space of 16 n words, then the last pass: layers 7 to 0 and, in leaves(), the
// hashing. A symbol is written to GPU memory and read back once for each strided pass (none up to
// n = 2^8, at most one up to 2^16, two up to 2^24, three up to the limit of 2^30), and never after
// it is hashed. leaves(), store() and gather() queue kernels on `stream` and do not wait for them.
// `stream` must be stream 0 or keep order with it: the copies into this class's buffers and, unless
// the thread has set an arena, their allocation from and return to CUDA's memory pool are steps of
// stream 0 (`Card`, prover/engine/device.cuh). EngineGpu's constructor
// (prover/engine/engine_gpu.cuh) checks this; SlabEncoder does not.
// release() and the destructor free the buffers without clearing them: partly encoded rows and the
// columns' chaining values stay in GPU memory until they are overwritten.
class SlabEncoder {
public:
  // Hashes each column of these rows, with its copied symbols and salt after it, into chain.leaves.
  void leaves(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, LeafChain chain, cudaStream_t stream = 0);
  // Writes the entire codewords of these rows to out_on_card[row * 2^log_n + x]. For short
  // codewords whose symbols are copied into the leaves of longer trees.
  void store(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, u64 *out_on_card, cudaStream_t stream = 0);
  // Writes the symbols of these rows at the positions at_on_card[i], i < listed, to
  // out_on_card[i * rows.size() + row]. The rows are encoded again. Not checked: the positions are
  // below 2^log_n. With listed = 0 the launch fails and the process ends.
  void gather(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, const u64 *at_on_card, size_t listed, u64 *out_on_card,
              cudaStream_t stream = 0);
  void release() { work_.release(), cv_.release(), rows_.release(); }

  // For tests: the number of kernels of a kind in this build, and how many of them this process has
  // launched. kind 0: a strided pass; 1: the last pass with the hash; 2: the last pass that stores;
  // 3: the last pass at listed positions. If the two numbers are equal, every kernel of that kind
  // has been launched.
  static size_t kernels(int kind);
  static size_t picked(int kind);
  // Which of the two ways built this copy, for a test's last line.
  static const char *made();

private:
  bool upper_layers(const RowSource *sources, int live, int log_n, int real, const u64 *table, cudaStream_t stream);
  Card<u64> work_, cv_;
  Card<RowSource> rows_;
};
} // the inline namespace

} // namespace blk

#ifndef ENGINE_ENCODER_OBJECT
#define ENCODER_FN inline
#include "engine/encode_kernels.cuh"
#undef ENCODER_FN
#endif
