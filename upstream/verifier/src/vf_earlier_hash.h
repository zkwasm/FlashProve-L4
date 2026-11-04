// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The verifier's own hash: the compression function F of BLAKE2b (verifier/src/vf_earlier_hash_f.h; RFC 7693)
// with two extra inputs, a domain-separation number num and a length l; on it, H(num; w) hashes a
// list w of 32-bit words. Defined in verifier/HASHES.md, "H(tag; x)". A proof's hash chains and
// Merkle trees use another hash, blocks/hash/hash.h.
// The prover has separate code for F and H (prover/common/blake2b_f.h, prover/common/hash.h). If
// prover and verifier put different digests into the statement, the proof is rejected. Both F's are
// adapted from the BLAKE2 reference implementation (blake2b-ref.c), so a comparison of the two can
// show an error made in adapting that code, but not an error of that code.
#pragma once
#include "vf_widths_and_bounds.h"
#include "vf_earlier_hash_f.h"
// Two 32-bit words make a 64-bit word of F, the low word first.
HD void words64(uint64_t *out, const u32 *w, u32 n64) {
  for (u32 i = 0; i < n64; i++) out[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
}
HD void words32(u32 *out, const uint64_t *x, u32 n64) {
  for (u32 i = 0; i < n64; i++) {
    out[2 * i] = (u32)x[i];
    out[2 * i + 1] = (u32)(x[i] >> 32);
  }
}
// cv := F(cv, m, t, last). Low 64 bits of t: the counter c0 + 2^32 c1. High 64 bits: num + 2^32 l.
HD void earlier_hash_c(u32 cv[SW], const u32 m[BLKW], u32 c0, u32 c1, u32 l, u32 num, bool last) {
  uint64_t h[SW / 2], b[BLKW / 2];
  words64(h, cv, SW / 2);
  words64(b, m, BLKW / 2);
  f2b::F(h, b, (uint64_t)c0 | ((uint64_t)c1 << 32), (uint64_t)num | ((uint64_t)l << 32), last);
  words32(cv, h, SW / 2);
}
// The initial state: that of BLAKE2b with a digest of 64 bytes and no key (RFC 7693, section 3.3).
HD void earlier_hash_initial_state(u32 cv[SW]) {
  uint64_t h[SW / 2];
  for (u32 i = 0; i < SW / 2; i++) h[i] = f2b::IV[i];
  h[0] ^= 0x01010040ull;
  words32(cv, h, SW / 2);
}
// H's state (SW = 16 words: verifier/src/vf_widths_and_bounds.h), with any l instead of L. The L words w[0],
// w[stride], ... are hashed in blocks of BLKW words: at least one, the last padded with zeros.
// Every call takes num and l. The counter counts the list's bytes hashed so far; only the last call
// has `last` set. x is added to c1; no caller passes x. With num = l = x = 0 this is BLAKE2b with a
// 64-byte digest on the list's bytes, little-endian (a test that is not part of this repository
// checks this against Python's hashlib on one list of each length from 0 to 300 words, stride 1).
HD void earlier_hash_h_state_with_length(u32 out[SW], u32 num, u32 l, const u32 *w, u64 L, u64 stride = 1, u32 x = 0) {
  earlier_hash_initial_state(out);
  u64 nb = (L + BLKW - 1) / BLKW;
  if (nb == 0) nb = 1;
  for (u64 b = 0; b < nb; b++) {
    u32 m[BLKW];
    for (u32 i = 0; i < BLKW; i++) {
      const u64 j = b * BLKW + i;
      m[i] = j < L ? w[j * stride] : 0;
    }
    const bool last = b + 1 == nb;
    const u64 fed = last ? 4 * L : 4 * (u64)BLKW * (b + 1);
    earlier_hash_c(out, m, (u32)fed, (u32)(fed >> 32) + x, l, num, last);
  }
}
// The state of H(num; w): l = L, the list's length in words. The verifier calls earlier_hash_h_state_with_length() only
// here. L must be below 2^32, because l has 32 bits. This is not checked, here or by a caller: the
// cast drops the higher bits (the last call's counter still carries 4 L in 64 bits). cp_verify
// hashes these lists: in TH() (verifier/src/vf_common.h), chunks of at most THPW words; in
// verifier/src/cp_verify.cpp, tables of 2^16 words, the RoPE table of T * dh words (T positions, dh
// channels a head), the exponential table, and the text of the lists of lookup calls.
HD void earlier_hash_h_state(u32 out[SW], u32 num, const u32 *w, u64 L, u64 stride = 1) {
  earlier_hash_h_state_with_length(out, num, (u32)L, w, L, stride);
}
// The digest of H(num; w): the first DW words of the state.
HD void earlier_hash_h_digest(u32 out[DW], u32 num, const u32 *w, u64 L, u64 stride = 1) {
  u32 s[SW];
  earlier_hash_h_state(s, num, w, L, stride);
  for (u32 i = 0; i < DW; i++) out[i] = s[i];
}
