// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives private random values ("coins" in identifiers), such as mask rows and salts, from a run's
// key (`process_key`) and a label, with the ChaCha20 block function f (chacha_block.h).
// The rule is defined in docs/format/COIN-KEYS.md, "2a. From a key to a random value", which
// does not describe the streams numbered from a serial number (kSerialStreams).
// This is a second implementation of the rule: the programs cp_commit and cp_prove derive some
// values again with it and compare the results.
// It leaves copies of keys on the stack: see BasicCoins.
//
// A label has sixteen 32-bit words, w0 to w15. Words 4 to 15 (`Context`) name the blinding context,
// the proof or message that the values are for. One call of f under the context's key, on words
// 0 to 3, gives a block of sixteen words: four or five values.
//   w0, w3   the number of the call within its stream, low and high word
//   w1       the stream, a numbered sequence of values within a context
//   w2       the retry word, 0 at the first try
#pragma once

#if defined(BLK_FIELD) && BLK_FIELD != 2
#error "BLK_FIELD, if it is given, must be 2 (p = 2^64 - 2^32 + 1)"
#endif

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "chacha_block.h"

namespace coins {

// A cell is a field element, in 0 .. p - 1 (p is `kP`). Pair q of a block o is
// o[2q] + 2^32 o[2q + 1]; lane q is the pair's low bits, as many as p has (the mask `kLaneBits`):
// all 64. A lane that is not below p is rejected (accept()): about one lane in 2^32.
// Rejection and retry, the steps that depend on the field, are also implemented by coin_admit() in
// blocks/coins/coins.h. This file includes nothing of that one, but the two implementations follow
// the same understanding of the rule, so an error in that understanding would be in both, and the
// programs' comparison would not show it.
// No one has looked into whether the rest of this file is separate from blocks/coins/coins.h, or
// whether any of it is separate from any other file of cp_commit, cp_prove or cp_verify (it
// includes chacha_block.h). Nothing is known of that, either way, and no separation is claimed,
// which limits what a passing comparison shows.
constexpr std::uint64_t kP = 0xFFFFFFFF00000001;  // 2^64 - 2^32 + 1
constexpr std::uint64_t kLaneBits = ~std::uint64_t(0);
static_assert(kP <= kLaneBits && kLaneBits / 2 < kP, "a lane has as many bits as p");

// The smallest stream number: check() throws below it.
constexpr Word kFirstStream = 16384;

// Streams from 2^30 up are numbered by serial_stream() in coin_streams.h, from a serial number, a
// kind and a row.
constexpr Word kSerialStreams = Word(1) << 30;

// Words 4 to 15 of a label, low word of `counter` first; words 6 to 9 and 13 to 15 are always 0.
// `version` is a file format's version word, `object` a number for the kind of proof or message,
// `member` a request's index in its batch, or 0.
struct Context {
  std::uint64_t counter = 0;  // words 4 and 5
  Word version = 0;           // word 10
  Word object = 0;            // word 11
  Word member = 0;            // word 12
};

// Cells per call of f: five in a stream from 2^30 up whose kind, bits 14 and 15 of its number, is 2
// (tree leaves' salts, of five cells each), four in every other stream.
constexpr unsigned cells_per_call(Word stream) {
  return (stream >= kSerialStreams && ((stream >> 14) & 3) == 2) ? 5 : 4;
}

// The block function as a type, so that a test can substitute another one and reach the branch for
// a rejected lane (accept()). A substitute needs operator() and, for cells(), wide(), which gives W
// blocks as chacha_blocks() does (chacha_block.h).
struct ChaChaBlock {
  Block operator()(const Key& k, Word x0, Word x1, Word x2, Word x3) const {
    return chacha_block(k, x0, x1, x2, x3);
  }
  template <std::size_t W>
  void wide(const Key& k, const Word (&x0)[W], Word x1, Word x2, const Word (&x3)[W],
            Word (&out)[16][W]) const {
    chacha_blocks<W>(k, x0, x1, x2, x3, out);
  }
};

// K_ctx, the context's key: three chained calls of f absorb the label's words 4 to 15, four words
// per call. K_ctx is as secret as `process_key`.
template <class F = ChaChaBlock>
Key context_key(const Key& process_key, const Context& c, F f = F()) {
  const auto first_eight = [](const Block& o) {
    Key k;
    for (int i = 0; i < 8; ++i) k[i] = o[i];
    return k;
  };
  Key k1 = first_eight(f(process_key, Word(c.counter), Word(c.counter >> 32), 0, 0));
  Key k2 = first_eight(f(k1, 0, 0, c.version, c.object));
  Key k_ctx = first_eight(f(k2, c.member, 0, 0, 0));
  wipe(k1);
  wipe(k2);
  return k_ctx;
}

// The values of one blinding context. None depends on the order of the calls.
// wipe() zeroes only k1, k2 and k_ctx_. Copies of them and of `process_key` stay on the stack, in
// the states and output blocks of f. A caller with a secret key must use scrub_the_stack_below():
// prover/common/stack_scrub.h gives the conditions, and what is not erased even then.
template <class F = ChaChaBlock>
class BasicCoins {
 public:
  BasicCoins(const Key& process_key, const Context& context, F f = F())
      : f_(f), k_ctx_(context_key(process_key, context, f)) {}
  ~BasicCoins() { wipe(k_ctx_); }
  BasicCoins(const BasicCoins&) = delete;
  BasicCoins& operator=(const BasicCoins&) = delete;

  std::uint64_t cell(Word stream, std::uint64_t index) const {
    check(stream);
    const unsigned per = cells_per_call(stream);
    const std::uint64_t call = index / per;
    const unsigned lane = unsigned(index % per);
    const Block o = f_(k_ctx_, Word(call), stream, 0, Word(call >> 32));
    return accept(pair(o, lane), stream, call, lane);
  }

  // Word `index` of a stream that is read as bytes (the salts and random bytes that a proof takes
  // from blocks/coins/coins.h): all 64 bits of pair index mod 4 of call index / 4. No value is
  // rejected.
  std::uint64_t word(Word stream, std::uint64_t index) const {
    check(stream);
    if (stream >= kSerialStreams)
      throw std::invalid_argument("coins: the streams from 2^30 up give cells, not bytes");
    const std::uint64_t call = index / 4;
    const unsigned lane = unsigned(index % 4);
    const Block o = f_(k_ctx_, Word(call), stream, 0, Word(call >> 32));
    return o[2 * lane] | std::uint64_t(o[2 * lane + 1]) << 32;
  }

  // Cells first .. first + count - 1 of `stream`, into out[0 .. count - 1].
  void cells(Word stream, std::uint64_t first, std::uint64_t count,
             std::uint64_t* out) const {
    check(stream);
    if (count != 0 && count - 1 > ~std::uint64_t(0) - first)
      throw std::invalid_argument("coins: the range ends beyond cell 2^64 - 1");
    const unsigned per = cells_per_call(stream);
    std::uint64_t i = first, left = count;

    // Single cells, up to the start of a call.
    for (; left != 0 && i % per != 0; ++i, --left) *out++ = cell(stream, i);

    // Whole calls, kWide at a time.
    for (; left >= std::uint64_t(per) * kWide; i += per * kWide, left -= per * kWide) {
      const std::uint64_t call = i / per;
      Word x0[kWide], x3[kWide], o[16][kWide];
      for (std::size_t b = 0; b < kWide; ++b) {
        x0[b] = Word(call + b);
        x3[b] = Word((call + b) >> 32);
      }
      f_.template wide<kWide>(k_ctx_, x0, stream, 0, x3, o);
      for (std::size_t b = 0; b < kWide; ++b)
        for (unsigned q = 0; q < per; ++q) {
          const std::uint64_t v = (o[2 * q][b] | std::uint64_t(o[2 * q + 1][b]) << 32) & kLaneBits;
          *out++ = v < kP ? v : accept(v, stream, call + b, q);
        }
    }

    // The remaining calls; the last one may be partial.
    while (left != 0) {
      const std::uint64_t call = i / per;
      const Block o = f_(k_ctx_, Word(call), stream, 0, Word(call >> 32));
      for (unsigned q = 0; q < per && left != 0; ++q, ++i, --left)
        *out++ = accept(pair(o, q), stream, call, q);
    }
  }

 private:
  static constexpr std::size_t kWide = 16;

  static void check(Word stream) {
    if (stream < kFirstStream)
      throw std::invalid_argument("coins: the stream's number is below 16,384, and such a stream gives no coin");
  }

  // Lane q of block o.
  static std::uint64_t pair(const Block& o, unsigned q) {
    return (o[2 * q] | std::uint64_t(o[2 * q + 1]) << 32) & kLaneBits;
  }

  // Returns v if it is below p. Otherwise derives the rejected lane again, by calls in which only
  // the retry word changes, to 8 k + lane for k = 1, 2, .. (`lane` is its number in its block),
  // until lane 0 of the call's block is below p: that is the cell.
  std::uint64_t accept(std::uint64_t v, Word stream, std::uint64_t call,
                       unsigned lane) const {
    for (Word k = 1; v >= kP; ++k) {
      // The rule sets no bound on k. The retry word has 32 bits, which 8 k + lane fits for k below
      // 2^29, and two implementations need to agree only there. This one throws at k = 2^29.
      if (k >> 29) throw std::runtime_error("coins: no retry word is left");
      v = pair(f_(k_ctx_, Word(call), stream, 8 * k + lane, Word(call >> 32)), 0);
    }
    return v;
  }

  F f_;
  Key k_ctx_;
};

using Coins = BasicCoins<>;

// One cell. Every call derives the context's key again.
inline std::uint64_t coin(const Key& process_key, const Context& context,
                          Word stream, std::uint64_t index) {
  return Coins(process_key, context).cell(stream, index);
}

}  // namespace coins
