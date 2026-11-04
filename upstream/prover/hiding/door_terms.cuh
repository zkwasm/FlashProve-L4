// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Each sub-proof (the argument for one part of the forward pass) ends in one linear equation on
// committed arrays. A Term gives the equation's coefficients on one array: that of entry (r, c) is
// the sum over the Term's Pairs of cv[r] * L[c].
// Every pointer in a Term or a Pair points to GPU memory.
// Separate from prover/hiding/door.cuh so that tests can use Term and Pair without its other types.
// Include prover/hiding/transcript.cuh first: it defines T, an element of the extension field E.
#pragma once
#include <string>
#include <vector>
namespace hd {
// How Pair::cv is a product over the bits of the row number r. The code that fills cv states the
// form. For a Term of Kind CELLS16, close_by_frame() (prover/hiding/door_chain.cuh) passes the
// form, not cv, to the library (blocks/), which takes at most one of a Pair's two vectors in full,
// here L. cv is filled all the same; under the test variable CHAIN_CHECK_FORMS close_by_frame()
// compares it with the form. Other Terms' forms are not read. The verifier's counterpart is
// HRow::rowForm (verifier/src/vf_subproof_engine.h).
enum RowForm {
  PLAIN = 0,    // no form is stated; for a CELLS16 Term `close_by_frame` then returns an error
  EQUALITY = 1, // cv[first + r] = eq(point, r) for r < 2^bits, and 0 on all other rows. point[0]
                // goes with the top bit of r; first is a multiple of 2^bits
  ONES = 2,     // 1 on every row
  SET_BITS = 3  // cv[r] = the product of point[bits - 1 - t] over the set bits t of r, for r < R;
                // bits = ceil(log2 R). For the powers of b, point[j] = b^(2^(bits - 1 - j))
};
// Public source: a vector's nonzero range passed with it: GNU Scientific Library, Reference Manual,
// chapter "Basis Splines", function gsl_bspline_basis,
// https://www.gnu.org/software/gsl/doc/html/bspline.html. GSL returns only that range of the
// vector; here L keeps its full length.
struct Pair {
  const T *cv = nullptr; // R entries, one per row of the array
  const T *L = nullptr;  // K entries, one per column of the array
  // CELLS16 Terms: the sub-proofs take the range from hw::Space::lo and hi (walk.cuh).
  size_t c0 = 0, c1 = 0; // L is zero outside [c0, c1)
  RowForm form = PLAIN;
  const T *point = nullptr; // EQUALITY, SET_BITS: `bits` elements, valid when cv is
  int bits = 0;
  size_t first = 0;         // EQUALITY
};
// ZZ and BZ belong to the sub-proof's approximate range check on committed cells (`HiddenTail`,
// tail_f.cuh).
enum Kind {
  CELLS16 = 0,  // u16 cells: ZZ (the range check's digits), BZ (its lookups' multiplicity bits)
  ELEMENTS = 1, // elements of E, R = 1: the vector U3
  SYMBOLS = 2   // signed byte cells, column-major, three cells per symbol
};
struct Term {
  // The array's name. close_by_frame() looks it up in the layout, unless the Term is foreign.
  std::string nm; // U3, ZZ, BZ, M<mu>
  Kind kind = CELLS16;
  int mu = -1; // SYMBOLS: the matrix's number in the range check's list of matrices
  // SYMBOLS: true if an earlier sub-proof committed the matrix. close_by_frame() then finds the
  // array in this sub-proof's ChainLink::reads (prover/prover/chain_host.cuh), entry n for the n-th
  // foreign Term. `data` is set too.
  bool foreign = false;
  size_t R = 0, K = 0; // the array's numbers of rows and columns (SYMBOLS: ceil(m / 3) symbol rows)
  // The cells' address; only door_values() (door_values.cuh), for tests, reads it.
  // ELEMENTS: const T *, K entries. SYMBOLS: const int8_t *, cell (y, c) at c * m + y. CELLS16:
  // const u16 *, cell (r, c) at r * ld + c; null until the sub-proof calls plain_rows() (door.cuh).
  const void *data = nullptr;
  size_t ld = 0; // CELLS16: the distance between two rows, in cells
  size_t m = 0;  // SYMBOLS: the number of rows of cells
  // SYMBOLS: nb elements. cv is their SET_BITS form on the R symbol rows, with nb as `bits`.
  // hc::TieH::powers() (tie_h.cuh) fills cv and point with powers of one element.
  const T *point = nullptr;
  int nb = 0;
  std::vector<Pair> v;
};
} // namespace hd
