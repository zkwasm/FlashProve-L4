// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The GPU side of a Fiat-Shamir transcript, which is a hash chain (blocks/chain/chain.h). The host
// runs the chain through blk::ProverRounds (blocks/chain/rounds.h). Rounds of PSC, the sumcheck
// with a hiding polynomial (blocks/iop/psc.h), come too fast for a trip to the host in each, so the
// host lends the chain to the GPU for a stretch of rounds and takes it back (`Stretch`).
//
// In a stretch the prover sends only scalars, elements of the degree-EDEG extension field E (T
// here, blk::E in blocks/), and the verifier answers with challenges from E or an empty message.
// Merkle roots, out-of-domain samples, codeword positions and proof of work belong to host rounds.
//
// The kernels log their rounds and challenges. ProverRounds::take_back() runs the same rounds again
// on the host from the scalars sent, checks each by name against the plan, the list of the proof's
// rounds (blocks/numbers/plan.h), and ends the process unless it draws the same challenges and
// reaches the GPU's chain value.
//
// prover/hiding/transcript.cuh includes this file after defining T and Round.
#pragma once
#include <vector>

#include "chain/chain_lanes.cuh"
#include "chain/rounds.h"
#include "numbers/round_names.h"

// The round's name in the plan.
static inline const char *round_name(Round r) {
  switch (r) {
  case Round::PSC_EPSILON: return blk::R_PSC_EPSILON;
  case Round::PSC_ROUND: return blk::R_PSC_ROUND;
  case Round::PSC_Y: return blk::R_PSC_Y;
  case Round::LINE: return blk::R_LINE;
  case Round::ZERO: return blk::R_ZERO;
  case Round::BAT_SIGMA: return blk::R_BAT_SIGMA;
  case Round::LKP: return blk::R_LKP;
  case Round::LKP_SHIFT: return blk::R_LKP_SHIFT;
  case Round::LKP_PLANES: return blk::R_LKP_PLANES;
  case Round::FOREST_POINT: return blk::R_FOREST_POINT;
  case Round::JOIN: return blk::R_JOIN;
  case Round::PSC_VALUES: return blk::R_PSC_VALUES;
  }
  return "";
}

// The lent chain as the kernels see it. Every scalar sent is stored in a message slot, an element
// of an array in global memory, and a stretch's slots are consecutive, in the order of sending:
// tr_open() traps otherwise. So what was sent is an address and a count: the last `sent` slots
// before `next`. The last `open` of them are the open message, which no round has closed yet. The
// slots must keep their values until Stretch::end() reads them. A round beyond log_room or a
// challenge beyond drew_room traps.
struct Tr {
  blk::Chain chain;
  const blk::ChainAbsorb *lent; // the host's open message; null once a round has closed it
  const T *next;                // where the next scalar sent must be stored; null: anywhere
  unsigned sent, open;          // scalars sent in this stretch; of these, in the open message
  unsigned short *log;          // per round: Round + 128 * (drawn nonzero) + 256 * (number drawn)
  unsigned rounds, log_room;
  T *drew;                      // every challenge drawn, in order
  unsigned drawn, drew_room;
  psc::State<T> p;              // the sumcheck running on this transcript (sumcheck_door.cuh)
};

// True for the one `Round` whose challenge is drawn from the nonzero elements: PSC's round of the
// extra variable Y. The plan fixes this for each round, so tr_answer(), tr_point() and
// tr_answer_not_zero() trap on a round of the other sort.
__device__ __forceinline__ constexpr bool tr_not_zero(Round round) {
  return round == Round::PSC_Y;
}
// blk::E as a T; written for EDEG = 3.
__device__ __forceinline__ T tr_element(const blk::E &x) {
  const u64 c[EDEG] = {x.c[0].v, x.c[1].v, x.c[2].v};
  return ext_from(c);
}
__device__ __forceinline__ T tr_drew(Tr &t, const blk::E &x) {
  if (t.drawn >= t.drew_room) __trap();
  return t.drew[t.drawn++] = tr_element(x);
}
__device__ __forceinline__ void tr_logged(Tr &t, Round round, unsigned draws, bool not_zero) {
  if (t.rounds >= t.log_room || draws > 255 || (unsigned)round > 127) __trap();
  t.log[t.rounds++] = (unsigned short)((unsigned)round | (not_zero ? 128u : 0u) | draws << 8);
}
// n scalars, already stored in slots[0 .. n), join the open message. Nothing is hashed yet.
__device__ __forceinline__ void tr_open(Tr &t, const T *slots, int n) {
  if (t.next && t.next != slots) __trap(); // a stretch's slots must be consecutive
  t.next = slots + n;
  t.sent += (unsigned)n, t.open += (unsigned)n;
}
__device__ __forceinline__ constexpr blk::ChainKind tr_kind(Round round) {
  return round == Round::PSC_EPSILON ? blk::ChainKind::EPSILON : round == Round::PSC_VALUES ? blk::ChainKind::EMPTY : blk::ChainKind::ALGEBRAIC;
}
// The functions that run the hash are declared __noinline__. A step of the chain is long code,
// and the kernels send or draw in hundreds of places: with every step inlined, ptxas took more than
// seven minutes and 18 GB on one test that includes product_test.cuh, and had not finished.
//
// Closes the open message: hashes it into the chain as one round, after the host's open message if
// that is not closed yet, and logs the round. `draws`: how many challenges the caller then draws; 0
// if and only if the verifier's message is empty (ChainKind::EMPTY).
static __device__ __noinline__ void tr_close(Tr &t, Round round, unsigned draws, bool not_zero = false) {
  if ((draws == 0) != (tr_kind(round) == blk::ChainKind::EMPTY)) __trap();
  blk::ChainAbsorb m = t.lent ? *t.lent : t.chain.message();
  for (const T *q = t.next - t.open; q != t.next; q++)
    for (int k = 0; k < EDEG; k++) m.push_u64(coord(*q, k));
  t.chain.close(m, tr_kind(round));
  t.lent = nullptr, t.open = 0;
  tr_logged(t, round, draws, not_zero);
}
// Draws the next challenge of the round just closed.
static __device__ __noinline__ T tr_draw(Tr &t, bool not_zero) {
  return tr_drew(t, not_zero ? t.chain.draw_E_nonzero() : t.chain.draw_E());
}
// Closes the open message as a round without challenges; it is hashed like any other.
__device__ __forceinline__ void tr_empty(Tr &t, Round round) {
  tr_close(t, round, 0);
}
// Closes the open message, empty or not, and draws one challenge.
__device__ __forceinline__ T tr_answer(Tr &t, Round round) {
  if (tr_not_zero(round)) __trap();
  tr_close(t, round, 1);
  return tr_draw(t, false);
}
// The same, with the challenge drawn from the nonzero elements.
__device__ __forceinline__ T tr_answer_not_zero(Tr &t, Round round) {
  if (!tr_not_zero(round)) __trap();
  tr_close(t, round, 1, true);
  return tr_draw(t, true);
}
// As tr_answer(), for n challenges, 1 to 255, written to `out`.
__device__ __forceinline__ void tr_point(Tr &t, Round round, T *out, int n) {
  if (tr_not_zero(round)) __trap();
  tr_close(t, round, (unsigned)n);
  for (int i = 0; i < n; i++) out[i] = tr_draw(t, false);
}
// tr_round<N>() without the drawing: the caller then calls tr_draw() once.
// Public sources for the techniques of this routine.
// The idea is public: the loop count N (scalars sent) as a template parameter: M. Harris,
// "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology, slides "Complete
// Unrolling", "Unrolling with Templates" and "Reduction #6: Completely Unrolled",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. There the parameter is the
// block size.
// The compiler hint is public: #pragma unroll on loops over small per-thread arrays: M. Milakov,
// "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static
// indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int N> static __device__ __noinline__ void tr_round_closed(Tr &t, Round round, const T *sent, const T *slots) {
  if (t.lent || t.open) {
    tr_open(t, slots, N);
    tr_close(t, round, 1, tr_not_zero(round));
    return;
  }
  tr_open(t, slots, N);
  t.open = 0;
  u64 words[EDEG * N];
#pragma unroll
  for (int i = 0; i < N; i++)
#pragma unroll
    for (int k = 0; k < EDEG; k++) words[EDEG * i + k] = coord(sent[i], k);
  t.chain.round(words, tr_kind(round));
  tr_logged(t, round, 1, tr_not_zero(round));
}
// tr_open() and tr_answer() in one (tr_answer_not_zero() if the round asks for it). sent[i] must be
// the value already stored in slots[i]. With nothing else open the chain hashes `sent`
// (ChainT::round_and_draw()), otherwise the slots. Stretch::end() reads the slots: if `sent` was
// hashed and differs, take_back() ends the process.
// Public sources for the techniques of this routine: see tr_round_closed().
template <int N> static __device__ __noinline__ T tr_round(Tr &t, Round round, const T *sent, const T *slots) {
  if (t.lent || t.open) {
    tr_round_closed<N>(t, round, sent, slots);
    return tr_draw(t, tr_not_zero(round));
  }
  tr_open(t, slots, N);
  t.open = 0;
  u64 words[EDEG * N];
#pragma unroll
  for (int i = 0; i < N; i++)
#pragma unroll
    for (int k = 0; k < EDEG; k++) words[EDEG * i + k] = coord(sent[i], k);
  blk::E answer[1];
  t.chain.round_and_draw(words, tr_kind(round), answer, tr_not_zero(round));
  tr_logged(t, round, 1, tr_not_zero(round));
  return tr_drew(t, answer[0]);
}
// tr_round<N>() for the first SEND2_N lanes of a warp, which call it together; N = 0 sends nothing
// (prover/tests/psc_shell_test.cu runs N = 0 to 3). Only lane 0 reads `sent` and `slots`, *t is in
// memory that the lanes share, and every lane gets the challenge. With nothing else open the four
// lanes share the hash (blocks/chain/chain_lanes.cuh); otherwise lane 0 runs the round. Not
// checked: only one warp of a block may be in the call at a time, since the message passes through
// one shared array; no other lane of that warp may call; threadIdx.x & 31 must be the lane number.
// Public sources for the techniques of this routine.
// The idea is public: write, __syncwarp() (in blk::chain_round_by_lanes()), then read in shared
// memory: shown in NVIDIA Technical Blog, "Using CUDA Warp-Level Primitives"
// (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), "Warp Synchronization",
// Listing 6.
// The idea is public: a __shfl_sync() broadcast from lane 0, here where only lane 0 runs the round:
// shown in Listing 4 of the same post.
// A leader lane does what must happen once:
// see sc_door::round_by_lanes() (prover/hiding/sumcheck_door.cuh).
// N is a template parameter: see tr_round_closed().
constexpr int SEND2_N = 4;
template <int N> static __device__ __noinline__ T tr_round_by_lanes(Tr *t, Round round, const T *sent, const T *slots) {
  constexpr unsigned lanes = (1u << SEND2_N) - 1;
  __shared__ u64 room[blk::CHAIN_LANES_ROOM<blk::HashFn, N * EDEG>]; // the entire hashed string
  u64 *const cells = room + blk::CHAIN_LANES_MESSAGE;                // the round's message in it
  const bool leader = (threadIdx.x & 31) == 0, fresh = !t->lent && !t->open;
  __syncwarp(lanes); // every lane has read *t before lane 0 changes it
  if (fresh) {
    if (leader) {
      if (N) tr_open(*t, slots, N);
      t->open = 0;
      tr_logged(*t, round, 1, tr_not_zero(round));
      for (int i = 0; i < N; i++)
        for (int k = 0; k < EDEG; k++) cells[EDEG * i + k] = coord(sent[i], k);
    }
    const blk::E answer = blk::chain_round_by_lanes<blk::HashFn, N * EDEG>(
        blk::Four(lanes), &t->chain, room, tr_kind(round), tr_not_zero(round));
    return leader ? tr_drew(*t, answer) : tr_element(answer);
  }
  u64 c[EDEG] = {0, 0, 0};
  if (leader) {
    T r;
    if constexpr (N > 0)
      r = tr_round<N>(*t, round, sent, slots);
    else
      r = tr_not_zero(round) ? tr_answer_not_zero(*t, round) : tr_answer(*t, round);
    for (int k = 0; k < EDEG; k++) c[k] = coord(r, k);
  }
  for (int k = 0; k < EDEG; k++) c[k] = __shfl_sync(lanes, c[k], 0);
  __syncwarp(lanes); // after it every lane of the call may read *t
  return ext_from(c);
}
__device__ __forceinline__ T tr_round2_by_lanes(Tr *t, Round round, T a, T b, const T *slots) {
  const T sent[2] = {a, b};
  return tr_round_by_lanes<2>(t, round, sent, slots);
}
// tr_open() of N scalars and then tr_empty(), by the same lanes and under the same conditions.
// Public sources for the techniques of this routine: see tr_round_by_lanes().
template <int N> static __device__ __noinline__ void tr_empty_by_lanes(Tr *t, Round round, const T *sent, const T *slots) {
  static_assert(N >= 1, "an empty round closes a message");
  constexpr unsigned lanes = (1u << SEND2_N) - 1;
  __shared__ u64 room[blk::CHAIN_LANES_ROOM<blk::HashFn, N * EDEG>];
  u64 *const cells = room + blk::CHAIN_LANES_MESSAGE;
  const bool leader = (threadIdx.x & 31) == 0, fresh = !t->lent && !t->open;
  if (tr_kind(round) != blk::ChainKind::EMPTY) __trap();
  __syncwarp(lanes); // every lane has read *t before lane 0 changes it
  if (fresh) {
    if (leader) {
      tr_open(*t, slots, N);
      t->open = 0;
      tr_logged(*t, round, 0, false);
      for (int i = 0; i < N; i++)
        for (int k = 0; k < EDEG; k++) cells[EDEG * i + k] = coord(sent[i], k);
    }
    blk::chain_empty_round_by_lanes<blk::HashFn, N * EDEG>(blk::Four(lanes), &t->chain, room);
    return;
  }
  if (leader) {
    tr_open(*t, slots, N);
    tr_empty(*t, round);
  }
  __syncwarp(lanes); // after it every lane of the call may read *t
}

// The host's side of a stretch. init() allocates four small GPU buffers, which this struct never
// frees, for stretches of at most most_rounds rounds and most_draws challenges; the object can be
// reused. A sumcheck must not cross the end of a stretch: begin() starts with none running, and
// end() exits with code 2 if one is.
struct Stretch {
  Tr *tr = nullptr; // the transcript object that the kernels get
  blk::ChainAbsorb *lent = nullptr;
  unsigned short *log = nullptr;
  T *drew = nullptr;
  unsigned log_room = 0, drew_room = 0;
  // Set by end(): where the stretch's next scalar would have been stored (null: it sent none and
  // begin() got null), and how many it sent. A caller that runs one sub-proof (the argument for one
  // part of the forward pass) in several stretches passes next_slot to the next begin(): tr_open()
  // then traps unless the sub-proof's slots are consecutive over all its stretches.
  const T *next_slot = nullptr;
  unsigned sent_scalars = 0;
  void init(unsigned most_rounds, unsigned most_draws) {
    log_room = most_rounds, drew_room = most_draws;
    CK(cudaMalloc(&drew, most_draws * sizeof(T)));
    CK(cudaMalloc(&tr, sizeof(Tr)));
    CK(cudaMalloc(&lent, sizeof(blk::ChainAbsorb)));
    CK(cudaMalloc(&log, most_rounds * sizeof(unsigned short)));
  }
  // Lends ch's chain to *tr. Until end(), a send or a close on ch ends the process.
  // goes_on_at: null, or where the stretch's first scalar must be stored.
  void begin(blk::ProverRounds &ch, cudaStream_t stream, const T *goes_on_at = nullptr) {
    const blk::ProverRounds::Lent l = ch.lend();
    Tr t{};
    t.next = goes_on_at;
    t.chain = l.chain, t.lent = lent, t.log = log, t.log_room = log_room, t.drew = drew, t.drew_room = drew_room;
    CK(cudaMemcpyAsync(lent, &l.message, sizeof l.message, cudaMemcpyHostToDevice, stream));
    CK(cudaMemcpyAsync(tr, &t, sizeof t, cudaMemcpyHostToDevice, stream));
    CK(cudaStreamSynchronize(stream)); // l and t are about to go out of scope
  }
  // Waits for the work on `stream` only, then takes the chain back; the stretch's scalars enter the
  // proof here. Returns the challenges drawn, in order. Scalars sent after the last closed round
  // stay open in ch, as after a send.
  std::vector<blk::E> end(blk::ProverRounds &ch, cudaStream_t stream) {
    Tr t;
    CK(cudaMemcpyAsync(&t, tr, sizeof t, cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    if (!psc::through(t.p)) {
      printf("internal error: a stretch of rounds ends inside a sum-check\n");
      exit(2);
    }
    next_slot = t.next, sent_scalars = t.sent;
    std::vector<unsigned short> ran(t.rounds);
    std::vector<T> sent(t.sent), there(t.drawn);
    if (t.drawn) CK(cudaMemcpy(there.data(), drew, t.drawn * sizeof(T), cudaMemcpyDeviceToHost));
    if (t.rounds) CK(cudaMemcpy(ran.data(), log, t.rounds * sizeof(unsigned short), cudaMemcpyDeviceToHost));
    if (t.sent) CK(cudaMemcpy(sent.data(), t.next - t.sent, t.sent * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<blk::u8> bytes(sent.size() * blk::E_BYTES);
    for (size_t i = 0; i < sent.size(); i++) {
      blk::E x;
      for (int k = 0; k < EDEG; k++) x.c[k].v = coord(sent[i], k);
      blk::E_to_bytes(x, bytes.data() + i * blk::E_BYTES);
    }
    std::vector<blk::E> drawn_there(there.size());
    for (size_t i = 0; i < there.size(); i++)
      for (int k = 0; k < EDEG; k++) drawn_there[i].c[k].v = coord(there[i], k);
    size_t draws = 0;
    std::vector<blk::ProverRounds::LentRound> rounds;
    for (unsigned short r : ran) {
      const Round round = (Round)(r & 127);
      rounds.push_back({round_name(round),
                        round == Round::PSC_EPSILON  ? blk::RoundKind::EPSILON
                        : round == Round::PSC_VALUES ? blk::RoundKind::EMPTY
                                                     : blk::RoundKind::ALGEBRAIC,
                        (r & 128) != 0, (unsigned)(r >> 8)});
      draws += r >> 8;
    }
    if (draws != drawn_there.size()) {
      printf("internal error: a stretch's two logs differ: %zu elements by the rounds, %zu written\n", draws, drawn_there.size());
      exit(2);
    }
    return ch.take_back(rounds, bytes.data(), bytes.size(), drawn_there.data(), t.chain);
  }
};
