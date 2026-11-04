// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// F, the compression function of BLAKE2b, for host code and for GPU kernels.
//
// The text below of F and of its parts (the rotation in host code, G, a round, the compression, the
// first state) comes from the BLAKE2 team's reference implementation in C. The head of its file
// blake2b-ref.c, which its two headers have too:
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
// Our changes. F takes plain arguments where blake2b_compress reads a blake2b_state S and loads the
// block's bytes: h is S->h, m[i] is load64 of the block's i-th eight bytes, t_low is S->t[0] and
// t_high is S->t[1]. last takes the place of S->f[0]: true gives the word of 64 one bits that
// blake2b_set_lastblock stores, false gives 0. Nothing takes the place of S->f[1], which only
// blake2b_set_lastnode sets: it counts as 0. In the reference S->t[1] is the upper half of a count
// of bytes; here t_high is any word, and C (below) puts num + 2^32 l there.
// The macros G and ROUND are named F2B_G and F2B_ROUND, because files that include this header use
// the short names, and F2B_ROUND stays defined, so that a later header can test with #ifdef that
// this one came first. blake2b_IV and blake2b_sigma are constexpr and local to F, and F's two loops
// have F2B_UNROLL, so that in GPU code every index is a constant and no table is read at run time.
// The loops count in int, not in size_t. right<c>(w) takes the place of rotr64(w, c). first_state
// has as constants the state that blake2b_init(S, 64) computes.
// RFC 7693 (M-J. Saarinen, J-P. Aumasson: "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)",
// November 2015; https://www.rfc-editor.org/rfc/rfc7693) specifies the function: section 2.1 (w = 64 bits in a word,
// r = 12 rounds, the rotations 32, 24, 16, 63), section 2.6 (IV), section 2.7 (SIGMA), section 3.1 (G) and section 3.2
// (F).
// Source for modelling F itself as a random function, with all its inputs free: Luykx, Mennink, Neves, "Security
// Analysis of BLAKE2's Modes of Operation", IACR Transactions on Symmetric Cryptology 2016(1), Cryptology ePrint
// Archive 2016/827 (version of August 31, 2016), section 2.2 and Theorem 1. Theorem 1 says that F is indifferentiable
// from a random function, up to about 2^(n/2) queries, if the block cipher inside F is a weakly ideal cipher (the
// paper's section 3.1). There n is the number of bits of h. Here h has 512 bits, so this is about 2^256 queries. The
// theorem idealizes the cipher: it supports the model and does not prove it. Calls of F whose counter is not BLAKE2b's
// count of bytes, such as the call in C below, rest on this model. A hash that is BLAKE2b as specified on a whole
// string (blocks/hash/hash.h) is supported more directly by the same paper's Corollary 1 (section 5.1): BLAKE2's
// sequential hashing mode is indifferentiable from a random function, with about the same bound, under the same
// idealization of the cipher.
// In GPU code the four rotations are written on the two 32-bit halves of a 64-bit word:
//   by 32: the halves are swapped;
//   by 24 and by 16: whole bytes move, by __byte_perm(x, y, s), which picks four of the eight
//   bytes of x (0 to 3) and y (4 to 7), one for each hexadecimal digit of s, the lowest digit for the lowest byte;
//   by 63: this is a rotation to the left by 1, by __funnelshift_l(lo, hi, shift), which gives the
//   upper 32 bits of hi:lo shifted to the left.
// Both functions: NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics"
// (https://docs.nvidia.com/cuda/cuda-math-api/).
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define F2B_BOTH __host__ __device__ __forceinline__
#else
#define F2B_BOTH inline
#endif
// Namespace f2b: BLAKE2b's F, its rotation, C (F on 32-bit words) and BLAKE2b-512's first state.
namespace f2b {
// x rotated to the right by N bits, N one of 32, 24, 16, 63.
// Public source: that a rotation of a 64-bit word by 32 bits is only a swap of its two 32-bit words on a 32-bit machine
// is stated in Aumasson and Bernstein, "SipHash: a fast short-input PRF" (2012, https://eprint.iacr.org/2012/351,
// section 4, "Choice of rotation counts"); here it serves BLAKE2b's rotation by 32 on the GPU's 32-bit registers.
template <int N> F2B_BOTH uint64_t right(uint64_t x) {
  static_assert(N == 32 || N == 24 || N == 16 || N == 63,
                "N must be one of the four rotation counts of RFC 7693, section 2.1");
#ifdef __CUDA_ARCH__
  const uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
  uint32_t l, h;
  if (N == 32)
    l = hi, h = lo;
  else if (N == 24)
    l = __byte_perm(lo, hi, 0x6543), h = __byte_perm(lo, hi, 0x2107);
  else if (N == 16)
    l = __byte_perm(lo, hi, 0x5432), h = __byte_perm(lo, hi, 0x1076);
  else
    l = __funnelshift_l(hi, lo, 1), h = __funnelshift_l(lo, hi, 1);
  return (uint64_t)h << 32 | l;
#else
  return (x >> N) | (x << (64 - N)); // rotr64 of blake2-impl.h
#endif
}
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++ Language Extensions",
// "#pragma unroll", https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html; for
// loops over small per-thread arrays: M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical
// Blog, 2015, "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
#ifdef __CUDA_ARCH__
#define F2B_UNROLL _Pragma("unroll")
#else
#define F2B_UNROLL
#endif
// The mixing function G, adapted from the macro G of blake2b-ref.c. It is expanded only inside F,
// which has m and blake2b_sigma in scope. r and i are literals, so each index into m is a constant.
#define F2B_G(r,i,a,b,c,d)                  \
  do {                                      \
    a = a + b + m[blake2b_sigma[r][2*i+0]]; \
    d = right<32>(d ^ a);                   \
    c = c + d;                              \
    b = right<24>(b ^ c);                   \
    a = a + b + m[blake2b_sigma[r][2*i+1]]; \
    d = right<16>(d ^ a);                   \
    c = c + d;                              \
    b = right<63>(b ^ c);                   \
  } while(0)
// One round, adapted from the macro ROUND of blake2b-ref.c: G eight times on the work vector v. It
// is expanded only inside F, with a literal r from 0 to 11. After F only the name is of use, since
// F2B_G is undefined there.
#define F2B_ROUND(r)                    \
  do {                                  \
    F2B_G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
    F2B_G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
    F2B_G(r,2,v[ 2],v[ 6],v[10],v[14]); \
    F2B_G(r,3,v[ 3],v[ 7],v[11],v[15]); \
    F2B_G(r,4,v[ 0],v[ 5],v[10],v[15]); \
    F2B_G(r,5,v[ 1],v[ 6],v[11],v[12]); \
    F2B_G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
    F2B_G(r,7,v[ 3],v[ 4],v[ 9],v[14]); \
  } while(0)
// h := F(h, m, t, f) with t = t_low + 2^64 t_high and f = last.
// Adapted from blake2b_compress of blake2b-ref.c. F replaces the chaining value h by the
// compression of h and the message block m under the 128-bit counter t_low + 2^64 t_high; last is
// true for a last block. Every value of h, m, t_low and t_high is allowed. No branch and no address
// depends on them, so they may be secret; last is treated as public.
F2B_BOTH void F(uint64_t h[8], const uint64_t m[16], uint64_t t_low, uint64_t t_high, bool last) {
  constexpr uint64_t blake2b_IV[8] =
  {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
  };
  constexpr uint8_t blake2b_sigma[12][16] =
  {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
  };
  uint64_t v[16];
  int i;

  F2B_UNROLL
  for (i = 0; i < 8; ++i) {
    v[i] = h[i];
  }

  v[ 8] = blake2b_IV[0];
  v[ 9] = blake2b_IV[1];
  v[10] = blake2b_IV[2];
  v[11] = blake2b_IV[3];
  v[12] = blake2b_IV[4] ^ t_low;
  v[13] = blake2b_IV[5] ^ t_high;
  v[14] = blake2b_IV[6] ^ (last ? (uint64_t)-1 : (uint64_t)0);
  v[15] = blake2b_IV[7];

  F2B_ROUND(0);
  F2B_ROUND(1);
  F2B_ROUND(2);
  F2B_ROUND(3);
  F2B_ROUND(4);
  F2B_ROUND(5);
  F2B_ROUND(6);
  F2B_ROUND(7);
  F2B_ROUND(8);
  F2B_ROUND(9);
  F2B_ROUND(10);
  F2B_ROUND(11);

  F2B_UNROLL
  for (i = 0; i < 8; ++i) {
    h[i] = h[i] ^ v[i] ^ v[i + 8];
  }
}
#undef F2B_G
} // namespace f2b

// C is F on 32-bit words: two words make a 64-bit word of F, the low word first.
//   cv: SW words (16), replaced by the output; block: BLKW words (32);
//   t mod 2^64 = c0 + 2^32 c1; t >> 64 = num + 2^32 l; f = last.
// The same function is defined in verifier/HASHES.md, "F and C", where f stands for num, not last.
// earlier_hash_c() in prover/common/hash.h calls C. That file's hash is not the proof format's own; num tells
// its uses apart. For earlier_hash_h_state(), a list's hash, c0 + 2^32 c1 counts the bytes fed so far, l is
// the list's length in words and last marks the last block. For a bare call (earlier_hash_bare_c()) the caller
// gives c0 and c1; l = 0 and last = false.
namespace f2b {
F2B_BOTH void C(uint32_t *cv, const uint32_t *block, uint32_t c0, uint32_t c1, uint32_t l, uint32_t num, bool last) {
  uint64_t h[8], m[16];
  F2B_UNROLL
  for (int i = 0; i < 8; i++) h[i] = cv[2 * i] | (uint64_t)cv[2 * i + 1] << 32;
  F2B_UNROLL
  for (int i = 0; i < 16; i++) m[i] = block[2 * i] | (uint64_t)block[2 * i + 1] << 32;
  F(h, m, c0 | (uint64_t)c1 << 32, num | (uint64_t)l << 32, last);
  F2B_UNROLL
  for (int i = 0; i < 8; i++) cv[2 * i] = (uint32_t)h[i], cv[2 * i + 1] = (uint32_t)(h[i] >> 32);
}
// IV': the first state of BLAKE2b-512 (RFC 7693, section 3.3: IV with h[0] xor 0x01010040), as sixteen words.
F2B_BOTH void first_state(uint32_t *cv) {
  // What blake2b_init(S, 64) of blake2b-ref.c leaves in S->h, adapted: blake2b_init_param takes the parameter block
  // into blake2b_IV by xor, one 64-bit word after the other. The block begins with the bytes digest_length = 64,
  // key_length = 0, fanout = 1 and depth = 1, and all its other bytes are 0. So its first word (load64: the lowest byte
  // first) is 0x01010040, and its other seven words are 0.
  const uint64_t iv[8] = {0x6a09e667f3bcc908ULL ^ 0x01010040, 0xbb67ae8584caa73bULL,
                          0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                          0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                          0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
  F2B_UNROLL
  for (int i = 0; i < 8; i++) cv[2 * i] = (uint32_t)iv[i], cv[2 * i + 1] = (uint32_t)(iv[i] >> 32);
}
} // namespace f2b
