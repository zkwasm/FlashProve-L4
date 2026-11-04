// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Headers (`head` in identifiers) of the key, leaf and proof files, the configuration digest, the
   commitment ID (`published_id` in identifiers), and the labels of four hash chains (Fiat-Shamir
   transcripts). The files are described in blocks/frame/FORMAT.md, "4. The files". A chain's label
   and the configuration digest enter its first query, the hash input that fixes the chain's public
   inputs before the prover's first message, so that different public files give different
   challenges. `Binder` (blocks/chain/public.h) builds the first queries of these files' chains, for
   the verifier in blocks/frame/check.h, blocks/frame/check_key.h and blocks/iop/link.h. */
#pragma once
#include <string>
#include <vector>

#include "chain/public.h"
#include "chain/chain.h"
#include "chain/rounds.h"  // SampleSeed
#include "frame/version.h"
#include "hash/hash.h"
#include "iop/message.h"

namespace blk {

// A header is two or three little-endian 32-bit words: its tag (four ASCII characters, the
// file's first bytes), the version word (`SERVED_VERSION`, blocks/frame/version.h) and, in a proof
// file only, the number of the leaf table (a stored copy of the committed table; `entry` in
// identifiers) that the batch was proved against.
inline constexpr u32 tag_of(char a, char b, char c, char d) { return (u32)(u8)a | (u32)(u8)b << 8 | (u32)(u8)c << 16 | (u32)(u8)d << 24; }
inline constexpr u32 TAG_KEY = tag_of('C', 'P', 'I', 'K'), TAG_ENTRY = tag_of('C', 'P', 'E', 'N'), TAG_PROOF = tag_of('C', 'P', 'P', 'F');

inline void put_word(Writer &w, u32 x) {
  const u8 b[4] = {(u8)x, (u8)(x >> 8), (u8)(x >> 16), (u8)(x >> 24)};
  w.put_bytes(b, 4);
}
inline bool get_word(Reader &r, u32 &x) {
  const u8 *b = r.take(4);
  if (!b) return false;
  x = (u32)b[0] | (u32)b[1] << 8 | (u32)b[2] << 16 | (u32)b[3] << 24;
  return true;
}
inline void write_head(Writer &w, u32 tag) { put_word(w, tag), put_word(w, SERVED_VERSION); }
inline void write_head(Writer &w, u32 tag, u32 entry) { write_head(w, tag), put_word(w, entry); }
// Returns nullptr if the tag is `tag` and the version word is this build's; otherwise the reason,
// and every later read from `r` fails.
inline const char *read_head(Reader &r, u32 tag) {
  u32 t = 0, v = 0;
  const char *why = !get_word(r, t) || !get_word(r, v) ? "file-too-short" : t != tag ? "file-of-another-kind" : version_refusal(v);
  if (why) r.refuse();
  return why;
}
// Also reads the leaf table's number, which the caller must check.
inline const char *read_head(Reader &r, u32 tag, u32 &entry) {
  const char *why = read_head(r, tag);
  if (!why && !get_word(r, entry)) why = "file-too-short";
  return why;
}

// Digest of the parameter file's text alone. Hash chains use configuration_digest() instead.
inline Digest parameter_file_digest(const std::string &text) {
  Hasher h(Domain::PARAMETERS, 0);
  h.push((const u8 *)text.data(), text.size());
  return h.finish();
}
/* One digest of every byte of the public files that configure a program: the parameter file and,
   for example, a layout of arrays or public constants of the model. Each can fix a shape, a
   coefficient or a table, which are public inputs of every chain. The caller passes the files in a
   fixed order, the parameter file first; nothing here checks that. Each file's length is hashed
   before its bytes, so that no byte can move from one file into the next. */
inline Digest configuration_digest(const std::vector<std::string> &files) {
  Hasher h(Domain::PARAMETERS, 2);  // index 0: a parameter file's text; 1: a Binder's list of parameters
  h.push_u64(files.size());
  for (const std::string &f : files) h.push_u64(f.size()), h.push((const u8 *)f.data(), f.size());
  return h.finish();
}

/* The commitment ID, by which a client says which commitment it means: one hash, under its own
   domain number, of the configuration digest, the name of the anchor (the commitment's first
   table; `first_table` in identifiers), the seeds of the anchor's record (each a point at which
   its rows are sampled, and a factor: `record_seeds`, blocks/iop/foldeq.h) and the record's
   answers (stored answers to these samples). An empty list ends the input
   (docs/format/FORMAT.md, R4.11). A name (root, configuration digest, shape:
   `index_name`, blocks/iop/foldeq.h) does not show whether its table is the anchor: the caller
   must pass the right one, with the seeds that belong to it (neither is checked here).
   Nothing else enters: not the key file's proofs, which anyone can check, and no leaf table, which
   the LINK proof in its leaf file ties to the anchor. So the ID is the same whichever leaf tables
   exist.
   The ID does not say whose weights the table contains. That an ID belongs to a model of some name
   is its publisher's claim, outside the protocol. */
inline Digest published_id(const Digest &configuration, const std::vector<u8> &first_table_name, const std::vector<SampleSeed> &record_seeds, const std::vector<E> &record_answers) {
  Hasher h(Domain::PUBLISHED_ID, 0);
  h.push(configuration);
  h.push_u64(first_table_name.size()), h.push(first_table_name.data(), first_table_name.size());  // a string of bytes: its length, then the bytes
  u64 elements = 0;  // one flat list: its length, then seed by seed the point's coordinates and its factor (R6.9)
  for (const SampleSeed &s : record_seeds) elements += s.z.size() + 1;
  h.push_u64(elements);
  for (const SampleSeed &s : record_seeds) {
    for (const E &x : s.z) h.push(x);
    h.push(s.beta);
  }
  h.push_u64(record_answers.size());                                                              // a list: its length, then the elements
  for (const E &y : record_answers) h.push(y);
  h.push_u64(0);  // an empty list: its length
  return h.finish();
}

// Labels (integers that say which protocol a chain runs) of a proof file's three chains (the first,
// the shared, the second) and of a key file's first, the one-time chain. A key file's shared and
// second chains have theirs in blocks/iop/lin.h, the LINK proof's chain in blocks/iop/link.h. Each
// of the seven chains of key, leaf and proof files has a label of its own: configure.h asserts it.
inline constexpr u64 LABEL_FIRST_CHAIN = 0x46524d01, LABEL_ONE_TIME_CHAIN = 0x46524d02, LABEL_PROOF_SHARED_CHAIN = 0x46524d03, LABEL_PROOF_SECOND_CHAIN = 0x46524d04;

}  // namespace blk
