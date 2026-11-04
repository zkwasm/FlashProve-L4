// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The callbacks (`HfSeams`; `seam` in identifiers) through which hf::HfKeeper (hf_keeper.h) folds
// and opens a weight table, and the state of one table that they need (`HfHeld`). hf_side_seams()
// (hf_open_side.cuh) implements the callbacks on the GPU, for the committer program `cp_commit` and
// the prover program `cp_prove`. This header uses ::Ext without declaring it: the including file
// declares that type first.
//
// Notation. The sizes m0, k, lk, R, b, l, i0, nm, KD, lc, ln are fields of hf::HfTableParams
// (hf_table_params.h). F is the field modulo p = 2^64 - 2^32 + 1, E its extension of degree 3.
// - W: the weights, m0 rows of k = 2^lk columns. M: the same cells as R = 2^b long rows, by
//   hf::HfTableParams::row_of() and coef_of().
// - A row of M has KD = i0 + nm coefficients: a message of i0 entries (l data cells, then padding
//   zeros, then pads) and nm masks (random values encoded with the message). Row r is encoded to
//   T[r] in F^n,
//   n = 2^ln. T is the weight table; each column of T, with a salt, is a leaf of its Merkle tree.
// - A table has a record (docs/format/09-key-and-leaf-files.md, R13.8). The last
//   hf::HfTableParams::pads() entries of each row's message are pads: random elements of F, at the
//   entries pad_entry(j), derived from the table's mask seed (hf_pads.cuh).
// - u = sum over r < R of fold[r] M[r]: the folded row, KD elements of E.
// - position: a column's index, below n. ln can be 33, so a position is a uint64_t.
#pragma once
#include "hash_commitment/hf_open_weights.h" // HfEl, HfProductTerm, HfZuWeights
#include "hash_commitment/hf_params.h"       // hf::HfParams, hf::HfOpen
#include "code/code.h"                       // blocks/code/code.h: blk::Code, which includers of this header use
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

struct hf_key32; // hf_seed.h: a weight table's mask seed (its making key), 32 bytes

// What the callbacks need for one weight table besides the weights. The caller owns all of it.
// If u_host is null, make_u keeps u in GPU memory; otherwise u_host_elements must be at least KD.
struct HfHeld {
  const hf_key32 *seed = nullptr;  // host: mask seed. hf_side_seams() ends the process if it is null
  // host: a separately stored copy of the mask seed's 32 bytes, under which self-checks recompute
  // masks and salts (hf_mask_own.cuh) and pads (hf_pads.cuh).
  const uint8_t *key_again = nullptr;
  const uint32_t *nodes = nullptr; // host: T's tree from level s up, in hf_tree.cuh's order
  int s = 0;                       // the lowest stored level: each of its nodes covers 2^s columns
  ::Ext *u_host = nullptr;         // pinned host: space for u; zeroed by its owner
  size_t u_host_elements = 0;
};

// ---------------------------------------------------------------------------------------------------------------------
// The callbacks. A callback may return before its queued GPU work is done, unless its comment says
// otherwise or its result is in host memory; the unfinished work is ordered on the calling thread's
// cudaStreamPerThread. make_u and make_columns make passes over the weights (`walk` in identifiers:
// hf_walk.h) and block until these have ended. The callbacks that hf_side_seams() returns exit with
// code 4 on a call out of order (a second make_u before forget(), u_cells before make_u, show()
// before make_columns) and on an index or position beyond its bound.
// "GPU" and "host" say where a pointer points. An element of F is a u64, canonical (below p)
// wherever a buffer passes between files. An element of E is ::Ext: three u64 coordinates, the same
// bytes as blk::E.
// ---------------------------------------------------------------------------------------------------------------------
struct HfSeams {
  // Selects the memory for the GPU allocations that follow. true: borrowed memory, which the caller
  // may take back during a pass over the weights. false: memory that keeps its contents through a
  // pass. Nothing in this repository calls it through HfSeams.
  std::function<void(bool)> short_lived;

  // One pass over the weights: computes u[i] for i < KD, masks and pads included, and keeps it
  // until forget(). i is the coefficient's number: with code = blk::Code{ln, i0, nm}
  // (blocks/code/code.h), message entry j is coefficient code.message_at() + j and mask j is
  // coefficient code.tail_at() + j: the masks from 0, the message from nm. A caller takes both
  // starts from the code.
  // The same pass writes omega[y] = sum over c < k of eq(rw, c) W[y][c], y < m0, to omega (GPU, m0
  // elements, which the caller allocates in memory that keeps its contents through a pass).
  // eq is the multilinear equality polynomial. rw: host, lk elements of E, coordinate 0 for the top
  // bit of c. fold: host, R elements of E (public), fold[r] = eq(gamma, r) for gamma = the first b
  // coordinates of rw; not checked.
  // The self-check [own-test-fold] recomputes the masks' part of u at a few indices; if it fails,
  // make_u sets the failure bit OWN_TEST (prover/common/slot_flags.h).
  // The self-check [own-test-pads] compares the pads with a second derivation (hf_pads.cuh); if it
  // fails, make_u sets OWN_TEST and leaves only zeros in u and omega. Without a mask seed
  // hf_side_seams() ends the process.
  std::function<void(const ::Ext *fold, const ::Ext *rw, ::Ext *omega)> make_u;

  // Copies u[i0_] .. u[i0_ + ni - 1], by the coefficient's number as at make_u, to `out` (GPU, ni
  // elements of E). i0_ + ni <= 2^lc, the least power of two not below KD; i0_ is not the size i0.
  // The padding zeros and the indices from KD on give 0. It may be called any number of times, in
  // any order, between make_u and forget(), and gives the same entries each time. The entries are
  // hidden values: the caller must erase them.
  std::function<void(uint64_t i0_, uint64_t ni, ::Ext *out)> u_cells;

  // One or more passes: computes the columns of T at the drawn positions and keeps what show()
  // needs. I: host, tcols positions in the order drawn, repeats kept (blk::PositionList::draws(),
  // blocks/tree/positions.h).
  std::function<void(const uint64_t *I, int tcols)> make_columns;

  // Writes the opened columns to `out` (host, hf::HfShown::shown_words 32-bit words;
  // hf_table_params.h) and returns when they are there. First the top layer of T's tree: the 2^c
  // digests at depth c = hf::HfShown::c. Then for each a < tcols, in the order drawn: the R symbols
  // T[0][I_a] .. T[R - 1][I_a] and the 5 salt symbols (two words each, low first), then the ln - c
  // sibling digests, lowest level first. No word of a column may be written to `out` before every
  // drawn column's node of the lowest stored level has been recomputed and found equal to the
  // stored one. If a self-check of make_columns has failed (the nodes, the salts, or the pads),
  // show() sets OWN_TEST and writes only zeros.
  std::function<void(uint32_t *out)> show;

  // Erases and frees everything that make_u and make_columns keep, except HfHeld::u_host, and
  // returns when the memory is zero. It comes after the last call of u_cells and show().
  std::function<void()> forget;
};

