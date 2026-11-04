// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// F, the compression function of BLAKE2b, and a count of its calls.
//
// The text below of F and of its parts (the two tables, the rotation, G, a round, the compression) comes from the BLAKE2
// team's reference implementation in C. The head of its file blake2b-ref.c, which its two headers have too:
//
//   BLAKE2 reference source code package - reference C implementations
//
//   Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
//   terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
//   your option.  The terms of these licenses can be found at:
//
//   - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
//   - OpenSSL license   : https://www.openssl.org/source/license.html
//   - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
//
//   More information about the BLAKE2 hash function can be found at
//   https://blake2.net.
//
// Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
// source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
//
// Our changes:
// - Taken from blake2b-ref.c: the tables blake2b_IV (called IV here) and blake2b_sigma, the macros G and ROUND, and
//   blake2b_compress without its first loop (called F here). From blake2-impl.h: rotr64. All else is left out.
// - F is the bare compression function. Where blake2b_compress reads a state struct S and a block of 128 bytes, F has
//   plain arguments: the chaining value h (S->h), the block m as sixteen 64-bit words (so no load64), the two counter
//   words t_low and t_high (S->t[0] and S->t[1]) and the last-block flag last (true stands for S->f[0] = (uint64_t)-1,
//   false for 0). There is no last-node flag (the reference's S->f[1], here always 0).
// - In the reference S->t[1] is the upper half of a count of bytes; here t_high is any word, and vf_earlier_hash.h puts the
//   format's additional words there.
// - F counts its calls.
// - The code is C++ in the namespace f2b. The loops count with int where the reference has size_t, and the spacing is
//   this repository's.
//
// BLAKE2's published specification is RFC 7693 (M-J. Saarinen (ed.), J-P. Aumasson: "The BLAKE2 Cryptographic Hash and
// Message Authentication Code (MAC)", November 2015, https://www.rfc-editor.org/rfc/rfc7693).
// A test that is not part of this repository compares F with the RFC's digest of "abc" and,
// through the format's chained hash, with Python's hashlib on 301 lists.
#pragma once
#include <atomic>
#include <cstdint>

namespace f2b {
// ---- How often this process has called F ----
// F counts its calls in every build, because cp_verify prints its own number of hash calls in its
// accounting. Every hash of vf_earlier_hash.h goes through F. The library's hash has a separate counter
// (blocks/hash/tweak.h, if BLK_COUNT_HASH_CALLS is set), and cp_verify adds the two. Each thread
// has its own count and adds it to a shared total when it ends. calls_total() is that total plus
// the calling thread's count, so it is complete only once all other threads that hashed are joined.
// Counting changes no hash.
constexpr int CALL_NUMBERS = 1024; // left over from a count by number: nothing reads it
inline std::atomic<uint64_t> calls_of_ended_threads{0};
struct CallsOfAThread {
  uint64_t n = 0;
  void hand_in() {
    calls_of_ended_threads.fetch_add(n, std::memory_order_relaxed);
    n = 0;
  }
  ~CallsOfAThread() { hand_in(); }
};
inline thread_local CallsOfAThread calls_of_this_thread;
inline const char *calls_who = "main"; // set by calls_begin(); nothing reads it
inline uint64_t calls_total() { return calls_of_ended_threads.load(std::memory_order_relaxed) + calls_of_this_thread.n; }
// Resets the count and sets `calls_who`. For a child process right after fork(), so that it does
// not count the parent's calls again. Other threads' counts are not reset: such a child has none.
inline void calls_begin(const char *who) {
  calls_of_this_thread.hand_in();
  calls_of_ended_threads.store(0, std::memory_order_relaxed);
  calls_who = who;
}
// blake2b_IV of blake2b-ref.c, under the name IV. RFC 7693, section 2.6: the first 64 bits of the
// fractional parts of the square roots of the first eight primes.
static const uint64_t IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
// blake2b_sigma of blake2b-ref.c. It is the message schedule of RFC 7693, section 2.7: round r of F
// takes the block's sixteen words in the order of row r. Rounds 10 and 11 use the schedule's rows 0
// and 1 again, so rows 10 and 11 here repeat them.
static const uint8_t blake2b_sigma[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3}};
// rotr64 of blake2-impl.h. It is x >>> n of RFC 7693, section 2.3. Here c is 32, 24, 16 or 63
// (section 2.1); the expression is defined only for 0 < c < 64.
static inline uint64_t rotr64(const uint64_t w, const unsigned c) {
  return (w >> c) | (w << (64 - c));
}
// G and ROUND of blake2b-ref.c. G is the mixing function of RFC 7693, section 3.1: it mixes two
// words of the block m, which r and i select through blake2b_sigma, into a, b, c and d, four words
// of the work vector v. Its sums are modulo 2^64. ROUND is round r: with v read as a 4 x 4 matrix,
// row by row, its first four calls of G mix the columns and its last four the diagonals. The names
// m and v exist only inside F, which undefines both macros after its last loop.
#define G(r, i, a, b, c, d)                     \
  do {                                          \
    a = a + b + m[blake2b_sigma[r][2 * i + 0]]; \
    d = rotr64(d ^ a, 32);                      \
    c = c + d;                                  \
    b = rotr64(b ^ c, 24);                      \
    a = a + b + m[blake2b_sigma[r][2 * i + 1]]; \
    d = rotr64(d ^ a, 16);                      \
    c = c + d;                                  \
    b = rotr64(b ^ c, 63);                      \
  } while (0)
#define ROUND(r)                       \
  do {                                 \
    G(r, 0, v[0], v[4], v[8], v[12]);  \
    G(r, 1, v[1], v[5], v[9], v[13]);  \
    G(r, 2, v[2], v[6], v[10], v[14]); \
    G(r, 3, v[3], v[7], v[11], v[15]); \
    G(r, 4, v[0], v[5], v[10], v[15]); \
    G(r, 5, v[1], v[6], v[11], v[12]); \
    G(r, 6, v[2], v[7], v[8], v[13]);  \
    G(r, 7, v[3], v[4], v[9], v[14]);  \
  } while (0)
// The compression function, specified in RFC 7693, section 3.2, with 64-bit words and 12 rounds
// (w = 64 and r = 12 in section 2.1). It updates the chaining value h in place with one block,
// given as its sixteen 64-bit words m. t_low and t_high are the two words of the section's offset
// counter t, and `last` is its flag f. F cannot check the counter or the flag: vf_earlier_hash.h says what
// its callers put there.
static inline void F(uint64_t h[8], const uint64_t m[16], uint64_t t_low, uint64_t t_high, bool last) {
  calls_of_this_thread.n++;
  // Adapted from blake2b_compress of blake2b-ref.c. The arguments h, t_low, t_high and last replace
  // its S->h, S->t[0], S->t[1] and S->f[0].
  uint64_t v[16];
  for (int i = 0; i < 8; ++i) {
    v[i] = h[i];
  }
  v[8] = IV[0];
  v[9] = IV[1];
  v[10] = IV[2];
  v[11] = IV[3];
  v[12] = IV[4] ^ t_low;                     // S->t[0]
  v[13] = IV[5] ^ t_high;                    // S->t[1]
  v[14] = IV[6] ^ (last ? (uint64_t)-1 : 0); // S->f[0], as blake2b_set_lastblock sets it
  v[15] = IV[7];                             // S->f[1] is 0
  ROUND(0);
  ROUND(1);
  ROUND(2);
  ROUND(3);
  ROUND(4);
  ROUND(5);
  ROUND(6);
  ROUND(7);
  ROUND(8);
  ROUND(9);
  ROUND(10);
  ROUND(11);
  for (int i = 0; i < 8; ++i) {
    h[i] = h[i] ^ v[i] ^ v[i + 8];
  }
#undef G
#undef ROUND
}
} // namespace f2b
