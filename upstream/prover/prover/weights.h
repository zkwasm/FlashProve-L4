// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Weights is the prover's interface to the model's integer weights: byte planes for the forward
// pass, committed cells for the proof, token embeddings and the output head's matrix. An
// implementation is called a source: GeneratedWeights (prover/prover/weights_generated.cuh) and
// wb::BlockWeights (prover/weights_blocks/block_weights.h). All pointers are device pointers.
// Layers are numbered across the model, not within a layer group (`Lam` consecutive layers). A
// cell is one committed signed byte: a balanced base-256 limb of a weight. A kind is a tensor type:
//   0 q and k; 1 v; 2 o; 11 router;
//   gating and up projections: 3, 4 dense layer; 5, 6 shared expert; 7, 8 routed experts;
//   down projection: 9 dense layer or shared expert; 10 routed experts.
#pragma once
#include "prover/forward.cuh"
#include <map>
#include <string>
#include <vector>

struct Weights {
  virtual ~Weights() {}
  // Sets `out` to the two byte planes [row][in] (weight = w0 + 256 * w1) by which the forward pass
  // multiplies at `layer`. Matrix classes: 0, the rows of q, k and v; 1, o; 2, the rows of gating
  // and up; 3, down. Classes 2 and 3 are requested at dense layers only.
  // Prover::g_setup_layout() tests only the result for class 0 at layer 0: if false, the forward
  // pass takes its weights from bind()'s callbacks or else generates them. Neither source returns
  // true: that case is never run.
  virtual bool planes(int matrix_class, int layer, ForwardPass::WP &out) const = 0;
  // Writes part `part` of `parts` equal parts of a layer group's cells of one kind. `bytes` counts
  // all these cells, [limb < 2][layer][expert][input][output]; kind 0's outputs:
  // [index in head][64 heads: q's, k's, zero padding].
  virtual void cells(int kind, int group, size_t bytes, int part, int parts, i8 *out) const = 0;
  // Writes x0[k][t] = Emb[tokens[t]][k] for t < T, and each value's xl balanced limbs (xl is 4 or
  // 5) to x0_cells[limb][k][t].
  virtual void embed(const int *tokens, int T, int xl, ResidualInt *x0, i8 *x0_cells) const = 0;
  // Writes `words` 32-bit words of the embedding's cells [limb < 4][token][k], starting at cell
  // `start` (a multiple of 4).
  virtual void embedding_cells(size_t start, size_t words, u32 *out) const = 0;
  // Returns the output head's matrix [V][d], or null: the prover then generates a synthetic one.
  virtual const int16_t *head() const = 0;

  // Optional functions, for a source with its own storage format.

  // Lets a source set the forward pass's callbacks ForwardPass::wplanes and ForwardPass::wblocks.
  virtual void bind(ForwardPass &forward) const {}
  // Return the address of all cells that cells() or embedding_cells() can write, in the same
  // order, or null: the prover then calls those functions.
  virtual const i8 *cells_in_place(int kind, int group) const { return nullptr; }
  virtual const i8 *embedding_in_place() const { return nullptr; }
  // If false, the prover's buffer for cells() is sized without this kind. A call of cells() for
  // the kind is not prevented and could overrun it; wb::BlockWeights::cells() exits instead.
  virtual bool lays_out_cells(int kind) const { return true; }
  // Returns whether `bytes` (see cells()) agrees with the source's count. If not, set-up fails.
  virtual bool cells_number(int kind, int group, size_t bytes) const { return true; }
  // If true, the prover gets this kind's sums from row_pass(), not from the kind's cells. Other
  // steps may still request the cells.
  virtual bool has_row_pass(int kind, int group) const { return false; }
  // Writes table[limb][layer][expert][input column]: the sum over the outputs of phi[output] times
  // the weight's limb. `T`: an extension-field element. With cut > 0, the outputs form 2^cut equal
  // runs, phi is applied to each run, and a column's 2^cut sums are adjacent in `table`. `scratch`
  // has row_pass_scratch() bytes. If only_layer is not negative, only the sums of that layer's
  // tensor are written, at their positions in the group's table.
  virtual void row_pass(int kind, int group, int cut, const T *phi, T *table, void *scratch, cudaStream_t stream,
                        int only_layer = -1) const {}
  virtual size_t row_pass_scratch() const { return 0; }
};
