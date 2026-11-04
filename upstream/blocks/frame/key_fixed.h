// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Public inputs to the first queries of a key file's three hash chains (Fiat-Shamir transcripts,
// check_key.h; a chain's first query is the hash input that starts it). The one-time chain's first
// query contains all six members of `KeyFixed`, the shared chain's contains `configuration`,
// `codes` and `lin`, and the second chain's contains `configuration` alone, each as public_bytes()
// writes it (see key_public_bytes() for what that leaves out). The first queries also contain
// values that are not here, among them the format's version and the plan (`Binder`), the committed
// table's root and record (`IndexName`), the binding tree's root and the earlier chains' seals.
// The members follow from the public files, some of them through the calling program's
// `ShapesOf` function (see key_public_bytes()), so configure() rejects unusable files, with
// key_fault()'s reason, before a table is committed.
#pragma once
#include "frame/files.h"
#include "iop/lin.h"
#include "iop/send.h"
#include "iop/wf.h"

namespace blk {

// make_key() and check_key() build this struct only from their `Configured` argument (configure.h).
// `ranges`: WF, the range proof on the table's entries. `index_masks`: the index-mask message's
// shape; WF and BATT (joins WF's claims) use its mask entries.
struct KeyFixed {
  Digest configuration{};
  Codes codes;
  LinShape lin;               // the committer's one-time test: a batch of one, at its own distance
  OracleNotice index_masks;
  WfPublic ranges;
  LinBindingShape binding;    // one leaf
};
inline KeyFixed key_fixed(const Sized &z, const Digest &configuration) {
  KeyFixed f;
  f.configuration = configuration;
  f.codes = z.key_codes, f.lin = z.once.S, f.index_masks = z.once.input, f.ranges = z.ranges.x;
  f.binding.log_leaves = 0, f.binding.salt_bytes = z.d.set.salt_bytes;
  return f;
}

// Serializes every member of f, but neither the text labels (`OracleShape::name`,
// `WfLookup::name`) nor the table's test distances (`IndexShape::distances`, in f.lin), the
// distances from a codeword for which the proximity tests on the table are sized. public_bytes()
// leaves both out, and foldeq_columns_verify() compares the distances instead.
// The calling program's `ShapesOf` function (blocks/frame/configure.h) derives some members, such
// as the ranges and the number of rows of the table's matrix M (`LinShape::matrix_rows`), from the
// public files, and the configuration digest covers only the files. So check_key() stores these
// bytes in its `TestedTable` (blocks/iop/foldeq.h), check_entry() copies them into its own, and
// check_entry() and check_proof() end the process if they differ from their own `Configured`
// argument's. Otherwise a table whose key file was checked under laxer ranges could be used in a
// check that states stricter ones.
// The comparison guards one process only. An `UnprovedTable` (blocks/iop/foldeq.h) is a table read
// from a key file or a leaf file (`entry` in identifiers) without the proofs that follow. It gets
// these bytes from its caller's `Configured`, not from a check of the key file. So in
// check_entry_if_tested() and check_proof_if_tested(), which the verifier program cp_verify calls,
// the comparison shows nothing about the key file's check.
inline std::vector<u8> key_public_bytes(const KeyFixed &f) {
  const auto &[configuration, codes, lin, index_masks, ranges, binding] = f;  // as many names as a KeyFixed has members, so that a new member stops the build here
  Writer w;
  w.put(configuration);
  public_bytes(w, codes), public_bytes(w, lin), public_bytes(w, index_masks), public_bytes(w, ranges), public_bytes(w, binding);
  return w.bytes();
}

// Returns null, or why no key file can be made or checked under f, including conditions across
// several fields.
inline const char *key_fault(const KeyFixed &f) {
  if (const char *w = lin_fault(f.codes, f.lin)) return w;
  const WfMatrix matrix = wf_matrix_of(f.lin);  // `WfPublic` does not store the matrix's size
  if (const char *w = wf_fault(f.codes, f.ranges, matrix)) return w;
  if (f.binding.log_leaves != 0) return "a key's binding tree has one leaf";
  if (f.ranges.message_id != 1) return "the multiplicities' message is the second oracle message of the one-time chain";
  const OracleShape &m = f.index_masks.shape;
  if (m.main_rows != 0 || m.mask_rows == 0 || m.answers < 1 || m.mask_code >= f.codes.mask.size()) return "the index-mask message is a sampled message of mask rows alone";
  if (wf_index_mask_entries(f.ranges, matrix) > Send::free_mask_entries(m.mask_rows, f.codes.mask[m.mask_code].ell, m.answers))
    return "the index-mask message has no room for the entries that the ranges and the joining take";
  return nullptr;
}

}  // namespace blk
