// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// One round of a Fiat-Shamir hash chain (blocks/chain/chain.h), without proof of work, computed by
// four neighbouring lanes of a warp. A sumcheck round waits for its hash, and four lanes that
// share it (blocks/hash/lanes.cuh) finish sooner than one thread.
//
// The query (the round's hash input), the challenge and the next chain value are those of
// ChainT::round_and_draw() with one challenge (D = 1), so a chain may mix rounds run here, by one
// thread and on the host. As there, the chain allows no further draw before its next round.
#pragma once
#include "chain/chain.h"
#include "hash/lanes.cuh"

namespace blk {

// The size of the buffer `room`, in 8-byte words, for a message of N such words, and the message's
// offset. chain_call_lanes() assembles the entire hashed string there.
template <class Fn, int N> constexpr int CHAIN_LANES_ROOM = LANE_ROOM<Fn, N + 1>;
constexpr int CHAIN_LANES_MESSAGE = (int)LABEL_WORDS;

// The words from which a challenge is drawn, as draw_E_from() reads them: those after the first
// four (the next chain value) of the round's first hash call (call 0), then all eight words of
// calls 1, 2 and so on. Call ask(0) first.
// The four lanes make every hash call together, since all see the same words.
// `f`: the lane's place in its group of four lanes (f.q) and the mask of all lanes that make the
// call together (f.lanes). `label`: which protocol the chain belongs to; it is part of every query.
// E: the extension field, from which challenges are drawn.
template <class Fn, int N> struct LanesRow {
  const Four &f;
  LaneDigest before;
  u64 *room;
  RoundWord last_word; // follows the message: the round's kind; no proof of work
  u64 round, label;
  u32 next_call = 1;
  u64 w[8];
  unsigned at = 4;
  __device__ __forceinline__ void ask(u32 j) {
    const typename LaneCall<Fn>::Cv o = chain_call_lanes<Fn, N + 1>(f, before, room, last_word, round, 0, j, label);
    BLK_UNROLL
    for (int i = 0; i < 8; i++) w[i] = LaneCall<Fn>::word(f, o, i);
  }
  __device__ __forceinline__ u64 operator()() {
    if (at == 8) ask(next_call++), at = 0;
    return w[at++];
  }
};

// Draws a round's challenge by the general rules, which skip rejected words, when the fast path,
// draw_E_straight(), fails. That is rare: a word is rejected with probability just under 2^-32.
// Inlined, this function would be most of a kernel's code, since every read of a word
// includes the code of a hash call. It repeats hash call 0, so that the caller keeps no LanesRow
// and the fast path can stay in registers.
template <class Fn, int N>
__device__ __noinline__ E chain_lanes_draw_by_the_rules(Four f, LaneDigest before, u64 *room, RoundWord last_word, u64 round, u64 label, bool nonzero) {
  LanesRow<Fn, N> row{f, before, room, last_word, round, label};
  row.ask(0);
  return nonzero ? draw_E_nonzero_from(row) : draw_E_from(row);
}
// Hash call 0 of a round: its first `WORDS` words, in all four lanes.
template <class Fn, int N, int WORDS>
__device__ __forceinline__ void chain_lanes_call_0(const Four &f, const LaneDigest &before, u64 *room, RoundWord last_word, u64 round, u64 label, u64 (&w)[WORDS]) {
  const typename LaneCall<Fn>::Cv o = chain_call_lanes<Fn, N + 1>(f, before, room, last_word, round, 0, 0, label);
  BLK_UNROLL
  for (int i = 0; i < WORDS; i++) w[i] = LaneCall<Fn>::word(f, o, i);
}
__device__ __forceinline__ Digest chain_lanes_value(const u64 *w) { // a call 0's first four words
  Digest d;
  BLK_UNROLL
  for (int i = 0; i < 4; i++) d.w[2 * i] = (u32)w[i], d.w[2 * i + 1] = (u32)(w[i] >> 32);
  return d;
}

// Runs one round and returns its challenge in all four lanes (never 0 if `nonzero` is set). `chain`
// is in memory that the four lanes share. `room` is CHAIN_LANES_ROOM<Fn, N> words of such memory
// that no other lane touches. On entry the N words from room[CHAIN_LANES_MESSAGE] on are the
// round's entire prover message, written by any of the four lanes, which must pass equal `kind`
// and `nonzero`. The call begins and ends with a barrier of the calling lanes (f.lanes).
// Not checked: that the caller has no open message from ChainT::message(); words pushed into one
// are left out of the round's query.
// f.lanes may cover several groups of four lanes (`Four`), and the barriers and shuffles of every
// hash call name all of f.lanes. Not checked: that all these groups make equally many hash calls.
// With groups on different chains, if the fast path fails in only some, the behaviour is undefined.
// `STRAIGHT` = false, for tests, skips the fast path.
template <class Fn, int N, bool STRAIGHT = true>
__device__ __forceinline__ E chain_round_by_lanes(const Four &f, ChainT<Fn> *chain, u64 *room, ChainKind kind, bool nonzero = false) {
  __syncwarp(f.lanes);
  const LaneDigest before = LaneDigest::from(f, chain->value());
  const RoundWord last_word((u8)kind, 0);
  const u64 round = chain->rounds() + 1, label = chain->label();
  u64 w[7];
  chain_lanes_call_0<Fn, N>(f, before, room, last_word, round, label, w);
  E drawn;
  if (!STRAIGHT || !draw_E_straight(w[4], w[5], w[6], nonzero, drawn)) // same branch in all 4 lanes
    drawn = chain_lanes_draw_by_the_rules<Fn, N>(f, before, room, last_word, round, label, nonzero);
  if (f.q == 0) chain->take_shut(chain_lanes_value(w));
  __syncwarp(f.lanes);
  return drawn;
}
// The same for a round in which the verifier's message is empty.
template <class Fn, int N> __device__ __forceinline__ void chain_empty_round_by_lanes(const Four &f, ChainT<Fn> *chain, u64 *room) {
  __syncwarp(f.lanes);
  u64 w[4];
  chain_lanes_call_0<Fn, N>(f, LaneDigest::from(f, chain->value()), room, RoundWord((u8)ChainKind::EMPTY, 0), chain->rounds() + 1, chain->label(), w);
  if (f.q == 0) chain->take_shut(chain_lanes_value(w));
  __syncwarp(f.lanes);
}

} // namespace blk
