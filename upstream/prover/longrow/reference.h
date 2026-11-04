// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A slow CPU reference (namespace `lr`) of how a private random value (`coin` in identifiers) is
// taken from the generator's output: under which key, from which block of a stream, from which pair
// of its words, and what a retry does. Only tests include this file;
// prover/tests/coins_known_host_test.cpp compares cell() with known answers.
// It calls the generator (prover/common/coin_chacha.h) and does not reimplement it. P, the field's
// modulus 2^64 - 2^32 + 1, is that of prover/common/hash_tree_ref.h; the stream numbers and the
// number of cells in a block are namespace lrw's (prover/hiding/long_rows_words.h).
#pragma once
#include "common/coin_chacha.h"
#include "common/hash_tree_ref.h"
#include "longrow/params.h"
#include <array>
namespace lr {
using row_commit_ref::P;

// A private random value is cell i of a numbered stream. The generator expands a stream, one block
// a call, under a blinding context's key. The context is `Key`'s last four fields and
// Params::working_word, the version word; context_key() derives its key from them and the process
// key.
struct Key {
  std::array<u32, COINKEYW> key{}; // the process key (K_proc of prover/common/coin_chacha.h)
  u64 ctr = 0;                     // the blinding context's counter
  std::array<u32, NONCEW> nonce{}; // the blinding context's nonce; all zeros in the programs
  u32 obj = lrw::REQUEST;          // the kind of object that the values are for (lrw::Object)
  u32 mem = 0;                     // the request's batch slot; 0 also where there is no batch
};
// out = the key of k's blinding context: the generator's K_ctx.
static inline void context_key(u32 out[COINKEYW], const Key &k, const Params &p = {}) {
  coin_context_key(out, k.key.data(), (u32)k.ctr, (u32)(k.ctr >> 32), p.working_word, k.obj, k.nonce.data(), k.mem);
}
// The value that try number tr gives for cell i of a stream, with q = i modulo the number of cells
// in a block. Try 0 takes pair q of the output words of the block's call. Try tr >= 1 makes the
// same block's call with the retry word RETRYQ tr + q, and takes the first pair.
static inline u64 cell_try(const Key &k, u32 stream, u64 i, u32 tr, const Params &p = {}) {
  const u64 bk = i / lrw::cells_a_block(stream);
  const u32 q = (u32)(i % lrw::cells_a_block(stream));
  u32 kc[COINKEYW], o[COINOUTW];
  context_key(kc, k, p);
  coin_output(o, kc, stream, bk, tr ? RETRYQ * tr + q : 0);
  const int w = tr ? 0 : 2 * q;
  const u64 v = ((u64)o[w + 1] << 32) | o[w]; // p has 64 bits
  volatile u32 *vk = kc, *vo = o; // the context's key and its output do not stay on the stack
  for (int j = 0; j < COINKEYW; j++) vk[j] = 0;
  for (int j = 0; j < COINOUTW; j++) vo[j] = 0;
  return v;
}
// Cell i of a stream: the value of the first try that gives a value below P (the rule of
// blk::F_from_word(), blocks/field/field.h).
static inline u64 cell(const Key &k, u32 stream, u64 i, const Params &p = {}) {
  for (u32 tr = 0;; tr++) {
    const u64 v = cell_try(k, stream, i, tr, p);
    if (v < P) return v;
  }
}
// salt() returns block I of stream lrw::stream(NUM, lrw::SALT, 0): by the numbering of
// prover/hiding/long_rows_words.h, the salt of leaf I of Merkle tree number NUM.
typedef std::array<u64, lrw::SALT_CELLS> Salt;
static_assert(SESSW == SALTW, "Salt is also the type of a session's salt: block 0 of stream (0, SALT, 0)");
static inline Salt salt(const Key &k, u32 NUM, size_t I, const Params &p = {}) {
  Salt s;
  for (int x = 0; x < lrw::SALT_CELLS; x++)
    s[x] = cell(k, lrw::stream(NUM, lrw::SALT, 0), lrw::SALT_CELLS * (u64)I + x, p);
  return s;
}
} // namespace lr
