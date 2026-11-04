// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// One BLAKE2b compression computed by four neighbouring lanes of a warp and, built on it,
// chain_call_lanes(): one hash call of a round of a Fiat-Shamir hash chain, with the same answer
// as chain_call_cells() of hash.h. For a chain that runs inside a kernel, where a sumcheck round
// waits for its hash: each of the four lanes evaluates G, BLAKE2b's mixing function, a quarter as
// often as one thread would.
//
// The scheme, described in full at BLK_LANES_ROUND: F, BLAKE2b's compression function, works on a
// 4 x 4 array of words whose rows are G's arguments a, b, c and d. Lane q of the four holds column
// q. For G on the diagonals, b stays in its lane, and a, c and d change lanes by shuffles.
// Public sources of the scheme:
//  - the eight uses of G in a round: RFC 7693 (M-J. Saarinen, J-P. Aumasson, "The BLAKE2 Cryptographic Hash and Message Authentication Code
//    (MAC)", November 2015, https://www.rfc-editor.org/rfc/rfc7693), section 3.2;
//  - the four columns side by side, then the four diagonals: S. Neves, J.-P. Aumasson, "Implementing BLAKE with AVX, AVX2, and XOP"
//    (https://www.aumasson.jp/data/papers/NA12.pdf; the text at that address is headed "BLAKE and 256-bit advanced vector extensions"), section 2;
//  - a permutation packed into one number: the same paper, section 4.2, on the instruction vpermq; blake2b_sigma<r>(j) below packs a row in a
//    like way and picks an entry by a shift and a mask;
//  - leaving b in place: https://github.com/sneves/blake2-avx2/pull/4; blake2b_simd's src/avx2.rs has the same scheme (diagonalize,
//    undiagonalize; MIT licence; https://docs.rs/crate/blake2b_simd/latest/source/src/avx2.rs), and so has BLAKE3's SSE and AVX code;
//  - shuffles: NVIDIA Technical Blog, "Using CUDA Warp-Level Primitives" (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).
//
// The text of the compression (the message schedule, G, a round, LaneCall<Blake2bF>::call and ::begin) comes from the BLAKE2 team's reference
// implementation in C. The head of its file blake2b-ref.c, which its two headers have too:
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
// Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and blake2-impl.h, of the BLAKE2 reference
// source code package (reference C implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).
// Our changes. Taken are blake2b_sigma, the macros G and ROUND, blake2b_compress and, for the first chaining value, blake2b_init_param and
// blake2b_init0; blake2b_IV, the rotation and the parameter block's first word are those of blake2b_f.h.
//  - Four neighbouring GPU lanes compute one call of blake2b_compress together. Lane q of the four holds column q of v: its v[0], v[1], v[2]
//    and v[3] are the reference's v[q], v[4 + q], v[8 + q] and v[12 + q]. Its cv.lo and cv.hi are S->h[q] and S->h[4 + q].
//  - call takes plain arguments where blake2b_compress reads *S: cv for S->h, and a tweak for the rest: c0 + 2^32 c1 is S->t[0], num + 2^32 l
//    is S->t[1], and last stands for S->f[0], which is (u64)-1 if last is true (what blake2b_set_lastblock stores) and 0 if not. S->f[1] is
//    always 0 here, and the xor with it is left out. It returns the new chaining value and leaves cv as it was. The argument f has no
//    counterpart there: it holds the lane's place among its four and the mask that __shfl_sync needs.
//  - word(i) takes the place of m[i]. The caller passes the block as a function that returns its words of 64 bits, so the array m and the loop
//    that fills it with load64 are left out.
//  - blake2b_sigma is a function template: blake2b_sigma<r>(j) is blake2b_sigma[r][j]. A row is one 64-bit constant with four bits for each
//    entry, so an entry at a place that depends on q is read without a table in memory.
//  - G is named BLK_LANES_G and ROUND is named BLK_LANES_ROUND, because G and ROUND are too short for macros in a header. BLK_LANES_G differs
//    from G in the two expressions for the block's words, where word(blake2b_sigma<r>(..)) is written for m[blake2b_sigma[r][..]], and in the
//    rotation, where b2_right<c>(w) is written for rotr64(w, c). BLK_LANES_ROUND uses G twice where ROUND uses it eight times, once for the
//    lane's column and once for its diagonal. G's second argument is a variable there, which depends on q. Before and after the second
//    use, three of the lane's four words change lanes.
//  - In place of the loops over i in blake2b_init0 and blake2b_init_param, a lane picks its two words of S->h by q.
#pragma once
#include "hash/hash.h"

namespace blk {
// A lane's place among its four. lanes: the mask of all lanes of the warp that make the call together, in whole fours.
// The lane's number in its warp is taken as threadIdx.x & 31: true in a one-dimensional block.
struct Four {
  unsigned lanes;
  int q, first, before, after, across;
  __device__ __forceinline__ explicit Four(unsigned lanes_) : lanes(lanes_) {
    const int lane = threadIdx.x & 31;
    q = lane & 3, first = lane & ~3;
    before = first | ((q + 3) & 3), after = first | ((q + 1) & 3), across = first | ((q + 2) & 3);
  }
  template <class W> __device__ __forceinline__ W pick(W x0, W x1, W x2, W x3) const {
    return q == 0 ? x0 : q == 1 ? x1 : q == 2 ? x2 : x3;
  }
};
// Adapted from blake2b_sigma of blake2b-ref.c: blake2b_sigma<r>(j) is blake2b_sigma[r][j], for j from 0 to 15. BLK_SIGMA_ROW packs a row into
// 64 bits, entry j in bits 4 j to 4 j + 3. r is a template argument, so the row is a constant of the compiled code, and j may be known only at
// run time.
#define BLK_SIGMA_ROW(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) \
  (((u64)(s0) << 0) | ((u64)(s1) << 4) | ((u64)(s2) << 8) | ((u64)(s3) << 12) |             \
   ((u64)(s4) << 16) | ((u64)(s5) << 20) | ((u64)(s6) << 24) | ((u64)(s7) << 28) |          \
   ((u64)(s8) << 32) | ((u64)(s9) << 36) | ((u64)(s10) << 40) | ((u64)(s11) << 44) |        \
   ((u64)(s12) << 48) | ((u64)(s13) << 52) | ((u64)(s14) << 56) | ((u64)(s15) << 60))
template <int r> __device__ __forceinline__ int blake2b_sigma(const int j) {
  static_assert(0 <= r && r <= 11, "BLAKE2b has twelve rounds");
  constexpr u64 row =
      r ==  0 ? BLK_SIGMA_ROW( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15) :
      r ==  1 ? BLK_SIGMA_ROW(14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3) :
      r ==  2 ? BLK_SIGMA_ROW(11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4) :
      r ==  3 ? BLK_SIGMA_ROW( 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8) :
      r ==  4 ? BLK_SIGMA_ROW( 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13) :
      r ==  5 ? BLK_SIGMA_ROW( 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9) :
      r ==  6 ? BLK_SIGMA_ROW(12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11) :
      r ==  7 ? BLK_SIGMA_ROW(13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10) :
      r ==  8 ? BLK_SIGMA_ROW( 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5) :
      r ==  9 ? BLK_SIGMA_ROW(10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0) :
      r == 10 ? BLK_SIGMA_ROW( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15) :
                BLK_SIGMA_ROW(14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3);
  return (int)(row >> (4 * j)) & 15;
}
#undef BLK_SIGMA_ROW

// Adapted from the macro G of blake2b-ref.c. It reads word from the function that uses it, as G reads m there. r is a constant from 0 to 11.
// As in G, i has no parentheses around it in the body: pass a number or a name, not a sum.
#define BLK_LANES_G(r,i,a,b,c,d)               \
  do {                                         \
    a = a + b + word(blake2b_sigma<r>(2*i+0)); \
    d = b2_right<32>(d ^ a);                   \
    c = c + d;                                 \
    b = b2_right<24>(b ^ c);                   \
    a = a + b + word(blake2b_sigma<r>(2*i+1)); \
    d = b2_right<16>(d ^ a);                   \
    c = c + d;                                 \
    b = b2_right<63>(b ^ c);                   \
  } while(0)

// Adapted from the macro ROUND of blake2b-ref.c: one lane's share of round r. It reads v, word, f, column and diagonal from the function that
// uses it. ROUND applies G first to the four columns of v, in its lines with i = 0 .. 3, then to the four diagonals, with i = 4 .. 7.
// Columns: lane q applies G to the four words that it holds, which is ROUND's line with i = q.
// Diagonals: G's b stays in its lane, so lane q takes the line whose b is v[4 + q], which has i = 4 + (q + 3) % 4. For example, lane 1 takes
// G(r,4,v[ 0],v[ 5],v[10],v[15]). That line's a, c and d belong to columns q + 3, q + 1 and q + 2 modulo 4, so lane q fetches them from
// f.before, f.after and f.across. Afterwards each word returns to its column's lane: lane q fetches its a, c and d from f.after, f.before and
// f.across. The words move in the order a, d, c, in which G finishes them and the next G first reads them.
#define BLK_LANES_ROUND(r)                       \
  do {                                           \
    BLK_LANES_G(r,column,v[0],v[1],v[2],v[3]);   \
    v[0] = __shfl_sync(f.lanes, v[0], f.before); \
    v[3] = __shfl_sync(f.lanes, v[3], f.across); \
    v[2] = __shfl_sync(f.lanes, v[2], f.after);  \
    BLK_LANES_G(r,diagonal,v[0],v[1],v[2],v[3]); \
    v[0] = __shfl_sync(f.lanes, v[0], f.after);  \
    v[3] = __shfl_sync(f.lanes, v[3], f.across); \
    v[2] = __shfl_sync(f.lanes, v[2], f.before); \
  } while(0)

// Fn's compression function computed by four lanes, and how they hold its values.
template <class Fn> struct LaneCall;

template <> struct LaneCall<Blake2bF> {
  typedef u64 W; // word(i) below gives the block's word i of 8 bytes, i < 16
  struct Cv {
    W lo, hi; // in lane q: words q and 4 + q of the eight
  };
  // The first chaining value of a hash whose start lies in the lanes, word q of its four in lane q. Adapted from blake2b_init_param and
  // blake2b_init0 of blake2b-ref.c, as Blake2bF::begin64 is: start is the parameter block's word 4 + q (salt and personalization).
  __device__ __forceinline__ static Cv begin(const Four &f, W start, u32 digest_bytes) {
    return Cv{f.pick<W>(blake2b_IV0 ^ Blake2bF::param_word0(digest_bytes), blake2b_IV1, blake2b_IV2, blake2b_IV3),
              f.pick<W>(blake2b_IV4, blake2b_IV5, blake2b_IV6, blake2b_IV7) ^ start};
  }
  // Adapted from blake2b_compress of blake2b-ref.c. The four lanes of f compute one call. The caller sees to it that word, t and the mask
  // f.lanes are the same in the four lanes and that word changes nothing. word is called with an i that is known only at run time. No branch,
  // index or source lane depends on cv or on what word returns.
  template <class Word> __device__ __forceinline__ static Cv call(const Four &f, const Cv &cv, Word word, const Tweak &t) {
    W v[4];
    const int column = f.q, diagonal = 4 + ((f.q + 3) & 3); // G's second argument
    const W f0 = t.last ? (W)-1 : 0;                        // S->f[0]

    v[0] = cv.lo; // v[q] = S->h[q]
    v[1] = cv.hi; // v[4 + q] = S->h[4 + q]

    v[2] = f.pick<W>(blake2b_IV0,                                // v[ 8]
                     blake2b_IV1,                                // v[ 9]
                     blake2b_IV2,                                // v[10]
                     blake2b_IV3);                               // v[11]
    v[3] = f.pick<W>(blake2b_IV4 ^ (t.c0 | (u64)t.c1 << 32),     // v[12], S->t[0]
                     blake2b_IV5 ^ (t.num | (u64)t.l << 32),     // v[13], S->t[1]
                     blake2b_IV6 ^ f0,                           // v[14]
                     blake2b_IV7);                               // v[15], S->f[1] = 0

    BLK_LANES_ROUND( 0 );
    BLK_LANES_ROUND( 1 );
    BLK_LANES_ROUND( 2 );
    BLK_LANES_ROUND( 3 );
    BLK_LANES_ROUND( 4 );
    BLK_LANES_ROUND( 5 );
    BLK_LANES_ROUND( 6 );
    BLK_LANES_ROUND( 7 );
    BLK_LANES_ROUND( 8 );
    BLK_LANES_ROUND( 9 );
    BLK_LANES_ROUND( 10 );
    BLK_LANES_ROUND( 11 );

    return Cv{cv.lo ^ v[0] ^ v[2],  // S->h[q] = S->h[q] ^ v[q] ^ v[q + 8]
              cv.hi ^ v[1] ^ v[3]}; // the same for 4 + q
  }
  // Word j of o's eight, in every lane. j: the same in the four lanes, known at compile time.
  // Warp broadcast: __shfl_sync() with one source lane hands that lane's word to every lane. The idea is public: Listing 4 of "Using CUDA
  // Warp-Level Primitives" (at the file's head) shows such a broadcast with lane 0 as the source lane; here the source is the lane of each four
  // that holds the word.
  __device__ __forceinline__ static u64 word(const Four &f, const Cv &o, int j) { return __shfl_sync(f.lanes, j < 4 ? o.lo : o.hi, f.first | (j & 3)); }
};
#undef BLK_LANES_ROUND
#undef BLK_LANES_G

// ---- a call of a round ----
// A digest as four lanes hold it: word q of its four words of 8 bytes in lane q.
struct LaneDigest {
  u64 w;
  __device__ __forceinline__ static LaneDigest from(const Four &f, const Digest &r) { // from a value that every lane knows
    return LaneDigest{f.pick<u64>(r.w[0] | (u64)r.w[1] << 32, r.w[2] | (u64)r.w[3] << 32, r.w[4] | (u64)r.w[5] << 32, r.w[6] | (u64)r.w[7] << 32)};
  }
  __device__ __forceinline__ Digest in_every_lane(const Four &f) const {
    Digest d;
    BLK_UNROLL
    for (int j = 0; j < 4; j++) {
      const u64 x = __shfl_sync(f.lanes, w, f.first | j);
      d.w[2 * j] = (u32)x, d.w[2 * j + 1] = (u32)(x >> 32);
    }
    return d;
  }
};
// chain_call_cells() by four lanes: hash call j of a round, on a message of CELLS words of 8 bytes.
// The answer's first 32 bytes are the four lanes' .lo (a LaneDigest's w), its second 32 their .hi.
// room: LANE_ROOM<Fn, CELLS> words in memory that the four lanes share and no other lane touches.
// On entry room[2] .. room[CELLS] contain the message's first CELLS - 1 words, written by any of
// the four. Its last word, which takes j, is the argument last_word, the same in all four lanes.
// The function writes the rest of the hashed string around them (tag and round, last word with j,
// nonce, zeros), so that a lane reads each word of a block by its index alone. It synchronizes the
// lanes in f.lanes before its first access to room and after its last. A message of up to 13 words
// is one compression. Unlike chain_call_cells(), it does not check that round is below 2^63.
// Public sources for the techniques of this function. Exchange through memory within a warp: lanes
// fill an array, and after a warp barrier each lane reads it at indices of its own. Here the four
// lanes between them write the words of room[] around the message, and a lane reads a word that
// another lane wrote only after the __syncwarp() that follows the writes. This order of write,
// barrier and read is public:
// "Using CUDA Warp-Level Primitives" (at the file's head), "Warp Synchronization", Listing 6;
// according to that post, __syncwarp() is both a barrier for the threads in its mask and a memory
// fence. The compiler hint is public: BLK_UNROLL (#pragma unroll) on a loop of constant trip count:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <class Fn, int CELLS> constexpr int LANE_ROOM = (Fn::BLKW / 2) * (((int)LABEL_WORDS + CELLS + 1 + Fn::BLKW / 2 - 1) / (Fn::BLKW / 2));
template <class Fn, int CELLS>
__device__ __forceinline__ typename LaneCall<Fn>::Cv chain_call_lanes(const Four &f, const LaneDigest &rho_prev, u64 *room, RoundWord last_word, u64 round,
                                                                      u64 nonce, u32 j, u64 chain = 0) {
  typedef LaneCall<Fn> Call;
  constexpr int PER = Fn::BLKW / 2, ALL = (int)LABEL_WORDS + CELLS + 1, STEPS = (ALL - 1) / PER;
  typename Call::Cv cv = Call::begin(f, rho_prev.w, 2 * DIGEST_BYTES);
  __syncwarp(f.lanes);
  if (f.q == 0) room[0] = TagT<Fn>(Domain::CHAIN_QUERY, chain).word, room[1] = round;
  if (f.q == 1) room[ALL - 2] = last_word.v | (u64)j << 16;
  if (f.q == 2) room[ALL - 1] = nonce;
  if (f.q == 3)
    for (int i = ALL; i < LANE_ROOM<Fn, CELLS>; i++) room[i] = 0;
  __syncwarp(f.lanes);
  BLK_UNROLL
  for (int s = 0; s <= STEPS; s++)
    cv = Call::call(f, cv, [&](int i) -> u64 { return room[s * PER + i]; }, s < STEPS ? Fn::step((u32)s) : Fn::last(8 * ALL));
  __syncwarp(f.lanes);
  return cv;
}

} // namespace blk
