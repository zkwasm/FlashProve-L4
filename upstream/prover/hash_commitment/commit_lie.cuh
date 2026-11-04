// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test hook for a malicious committer (`lie` in identifiers): calls at which a test build lets the
// committer `cp_commit` write false or inconsistent files, for tests of the verifier `cp_verify`.
// In this repository every function here is empty, the identity or a constant, in every build. The
// sources in this folder call seven of the nine, at 20 places. The build that gives them other
// bodies under `COMMIT_CHEAT` (by default 0; no code here reads it) is not part of this repository.
//
// A commitment is about one weight matrix W, which `cp_commit` derives from the weights again at
// the places (readers) that `Reader` lists. A lie is a cell of W, altered for the readers that it
// names. walking() is told which table and reader the next pass over the weights serves; panel()
// and slab() are handed what a reader derived, before the program's code runs on it; passes() is
// asked where a self-check of the program has failed; closing() is called after a key file or a
// leaf file is built; read_the_variable(), at the start of a run; source(), once the weights are
// in host memory. No source of `cp_commit` calls laying_for() or digit(). check_the_weights() in
// prover/hash_commitment/hf_commit.cu calls none of them: the range scan never sees a lie.
//
// It names cudaStream_t and __device__: a file that host-only sources also include includes it
// under #ifdef __CUDACC__.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef COMMIT_CHEAT
#define COMMIT_CHEAT 0
#endif

namespace commit_lie {
// Where `cp_commit` derives W. The anchor is a commitment's first weight table (messages: first
// table); a leaf table is a further table of the same weights (messages: copy). The key file
// (`ivk/key.bin`) has the anchor's root and the committer's proofs about it; a leaf file
// (`ivk/leaf/<k>.bin`; `entry` in identifiers and messages) ties leaf table k to the anchor.
//   TABLE     the encoded rows and their Merkle tree: hf_commit_whole.cuh for a matrix kept whole
//             in GPU memory; otherwise, or under `-rows-in-cuts`, hf_commit_windows.cuh
//   KEYS      the lookup keys of the range proof (the key file's proof, by lookups, of the weights'
//             ranges), written (`lay` in identifiers and messages) by lay_digits() in
//             hf_range_windows.cuh
//   PRODUCTS  the products g^T W for the claims that the range proof leaves on W. The program uses
//             sums that the lookups kept (once_make() in hf_commit_once.cuh)
//   FOLD      a table's first pass over the weights (`walk` in identifiers and messages) when it is
//             opened, the anchor for the key file, a leaf table for its leaf file; it gives the
//             folded row: HfOncePanels::walk() in hf_commit_once.cuh
//   FIBRES    its later passes, for the drawn columns that are opened under the root
//   LINKFOLD, LINKFIBRES  the same passes over the anchor's weights while entry_make() in
//             hf_entry_make.cuh makes a leaf file, which folds and opens both tables
//   RECORD    the pass that computes the answers of a table's record (stored answers to samples
//             of its rows), after the table's tree is built: make_the_record() in hf_commit.cu
// No source of `cp_commit` names KEYS or PRODUCTS.
// Seen by TABLE, KEYS, PRODUCTS, FOLD, FIBRES and RECORD, a lie
// on the anchor gives a consistent key file, false only if the altered weight is out of its range.
// Seen by fewer, it makes parts of that file or of a leaf file disagree: such a set of readers
// leaves as few as one check of the verifier, on the key file or on the leaf file, between the lie
// and a `YES`, and a test can aim at that check.
enum Reader : int { TABLE = 0, KEYS = 1, FOLD = 2, FIBRES = 3, PRODUCTS = 4, LINKFOLD = 5, LINKFIBRES = 6, RECORD = 7, READERS = 8 };

// ---- every build ----
static inline void read_the_variable(bool /* under -test-seed */) {}
static inline void walking(int /* table */, Reader) {}
template <class Where> static inline void source(const Where &) {}
static inline void laying_for(Reader) {}
static inline void panel(int8_t *, size_t /* pitch */, size_t /* rows */, size_t /* first row */, size_t /* columns */, cudaStream_t) {}
static inline void slab(int8_t *, size_t /* stride */, uint32_t /* first row of M */, int /* rows of M */, int /* lcl */, size_t /* rows of W */, cudaStream_t) {}
__device__ __forceinline__ int digit(int /* table */, unsigned long long /* row */, unsigned long long /* column */, int v) { return v; }
static constexpr bool passes(const char * /* the stop's name */) { return false; }
static inline int closing(int /* tables */ = 0) { return 0; }

}  // namespace commit_lie
