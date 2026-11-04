// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The prover's private random values (`coins` in identifiers; blocks/coins/coins.h), generated
// directly in GPU memory (`card` in identifiers). A draw is one call of fill(), bytes(),
// fill_card() or bytes_card(); each draw opens its own numbered stream of values. On the host or
// on the GPU, a draw gets the same stream number, the same values and the same entry in
// `ledger()`, the list of draws.
//
// No verifier, and no honest run, can see a generator that goes wrong: a proof whose masks, pads or
// salts are zeros, or another stream's, verifies like any other and reveals what it should hide. A
// known-answer test at start-up cannot see a generator that goes wrong in the middle of a run. So
// this file offers two run-time checks by value: samples of the draws' values, for the caller to
// compare with a second implementation (sample(), coin_samples_differ()), and a comparison of whole
// draws right after their Merkle tree is built (held_card()). Both compare exact values and keep no
// state beyond a run: a run without a fault cannot fail them by chance. Neither ends the process at
// a difference: the caller must read samples() and bound() and act, as EngineCoins::hold() does
// (prover/hiding/engine_coins.cuh).
#pragma once
#include <sys/random.h>

#include <cerrno>
#include <vector>

#include "coins/coins.h"
#include "engine/card_coins.cuh"
#include "engine/device.cuh"

namespace blk {

// 8 bytes from the operating system's generator, read once per process, that vary which values
// between a draw's ends are sampled (pick()). Fixed places might never meet a fault that spares
// both ends of a draw and hits the same places in every run of one shape. The bytes depend on no
// secret, go into no file or printed line, and no byte of a proof depends on them: they only
// select samples. If the call fails they are 0 and the places are fixed: a weaker check, no other
// harm.
inline u64 coin_pick_turn() {
  static const u64 turn = [] {
    u64 x = 0;
    ssize_t given;
    do given = getrandom(&x, sizeof x, 0);
    while (given < 0 && errno == EINTR);
    return given == (ssize_t)sizeof x ? x : 0;
  }();
  return turn;
}

// The kernels that generate a draw's values take the context's ChaCha20 key (context_key()) as an
// argument; a context is one proof or message with its own random values. The key is secret: it is
// passed in the launch's argument memory, and no kernel here writes it to global or shared memory.
// Whether the compiler copies it into local memory cannot be seen in the source.
//
// In this kernel and the next, each thread computes ChaCha20 block number `call`, which gives
// values 4 call .. 4 call + 3 of the stream, and writes those numbered first .. first + n - 1.
// coin_admit() computes further blocks when a 64-bit word is not a field element (rejection
// sampling).
static __global__ void k_stream_coins_elements(CoinKey key, u32 stream, u64 first, F *out, size_t n) {
  const u64 call = first / COIN_LANES + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (call * COIN_LANES >= first + n) return;
  u32 o[16];
  chacha20_block(o, key.w, (u32)call, stream, 0, (u32)(call >> 32));
  BLK_UNROLL
  for (unsigned lane = 0; lane < COIN_LANES; lane++) {
    const u64 i = call * COIN_LANES + lane;
    if (i >= first && i < first + n) out[i - first] = coin_admit(o[2 * lane] | (u64)o[2 * lane + 1] << 32, key, stream, i);
  }
}
static __global__ void k_stream_coins_words(CoinKey key, u32 stream, u64 first, u64 *out, size_t n) {
  const u64 call = first / COIN_LANES + (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (call * COIN_LANES >= first + n) return;
  u32 o[16];
  chacha20_block(o, key.w, (u32)call, stream, 0, (u32)(call >> 32));
  BLK_UNROLL
  for (unsigned lane = 0; lane < COIN_LANES; lane++) {
    const u64 i = call * COIN_LANES + lane;
    if (i >= first && i < first + n) out[i - first] = o[2 * lane] | (u64)o[2 * lane + 1] << 32;
  }
}
// The kernel of again_words_at_card(): words width at[j] .. width at[j] + width - 1 of the stream
// go to out[width j ..].
static __global__ void k_stream_coins_words_at(CoinKey key, u32 stream, const u64 *at, size_t places, unsigned width, u64 *out) {
  const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e < places * width) out[e] = coin_word(key, stream, at[e / width] * width + e % width);
}

// The kernel of held_card(): each thread takes one ChaCha20 block's values, as in the first two
// kernels. held[0] += the number that differ from `laid`'s; held[1] += the number compared, added
// once per thread block: no thread returns before that sum.
// chacha20_block() and coin_admit() are shared
// with those kernels, so this check cannot catch
// an error in them: the samples are for that.
template <bool ELEMENTS> static __global__ void k_stream_coins_held(CoinKey key, u32 stream, const u64 *laid, size_t n, unsigned long long *held) {
  __shared__ unsigned compared;
  if (threadIdx.x == 0) compared = 0;
  __syncthreads();
  const u64 call = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned differ = 0, mine = 0;
  if (call * COIN_LANES < n) {
    u32 o[16];
    chacha20_block(o, key.w, (u32)call, stream, 0, (u32)(call >> 32));
    BLK_UNROLL
    for (unsigned lane = 0; lane < COIN_LANES; lane++) {
      const u64 i = call * COIN_LANES + lane, word = o[2 * lane] | (u64)o[2 * lane + 1] << 32;
      if (i >= n) continue;
      mine++, differ += laid[i] != (ELEMENTS ? coin_admit(word, key, stream, i).v : word);
    }
  }
  if (differ) atomicAdd(held, (unsigned long long)differ);
  if (mine) atomicAdd(&compared, mine);
  __syncthreads();
  if (threadIdx.x == 0 && compared) atomicAdd(held + 1, (unsigned long long)compared);
}

// coins_own_test_card(): a program's self-test (`own_test` in identifiers) of the generator on the
// GPU, to be run at its start. It first calls coins_own_test(), which compares the host's block
// with RFC 8439's block (2.3.2). This kernel then computes the RFC's block and, under the same
// published key, the first 32 elements of a stream; the host compares both with its own. Ends the
// process at a difference. The kernels above are not run by it.
static __global__ void k_stream_coins_own_test(CoinKey key, u32 *block, F *elements, unsigned n) {
  if (threadIdx.x == 0) chacha20_block(block, key.w, 1, 0x09000000u, 0x4a000000u, 0);
  if (threadIdx.x < n) elements[threadIdx.x] = coin_element(key, COIN_FIRST_STREAM, threadIdx.x);
}
inline void coins_own_test_card(cudaStream_t stream = 0) {
  coins_own_test();
  constexpr unsigned N = 32;
  CoinKey key;
  for (u32 i = 0; i < 8; i++) key.w[i] = 0x03020100u + 0x04040404u * i;
  Card<u32> block(16);
  Card<F> elements(N);
  k_stream_coins_own_test<<<1, N, 0, stream>>>(key, block.get(), elements.get(), N);
  ENGINE_CK(cudaGetLastError());
  ENGINE_CK(cudaStreamSynchronize(stream));
  u32 got[16], want[16];
  F e[N];
  block.to_host(got), elements.to_host(e);
  chacha20_block(want, key.w, 1, 0x09000000u, 0x4a000000u, 0); // matches the RFC (`coins_own_test`)
  bool same = !std::memcmp(got, want, sizeof want);
  for (unsigned i = 0; i < N; i++) same = same && e[i] == coin_element(key, COIN_FIRST_STREAM, i);
  BLK_REQUIRE_NAMED(same, "own-test-coins", "coins: the generator's own test fails on the GPU: its block or its elements are not the host's");
}

// One recorded value of a draw. index: its number in its stream; for bytes, the number of the
// word of 8 bytes.
struct CoinSample {
  u32 stream;
  bool element; // a field element, or a word of a stream of bytes
  u64 index, value;
};
constexpr unsigned COIN_SAMPLES_MOST = 8; // of one draw
struct CoinPicks {
  u64 at[COIN_SAMPLES_MOST];
};
static __global__ void k_stream_coins_pick(const u64 *laid, CoinPicks picks, u64 *to) { to[threadIdx.x] = laid[picks.at[threadIdx.x]]; }
// How many samples differ from a second implementation of the generator, which the caller gives
// as second(stream, element, index) -> the value.
template <class Second> inline size_t coin_samples_differ(const std::vector<CoinSample> &samples, Second second) {
  size_t differ = 0;
  for (const CoinSample &s : samples) differ += second(s.stream, s.element, s.index) != s.value;
  return differ;
}

// Running totals of held_card()'s check, in values (an element, or a word of 8 bytes). All but
// `differ` follow from public shapes.
struct CoinsBound {
  u64 differ, compared; // by the kernels, in GPU memory
  u64 handed, left, laid; // by the host: passed to `held_card`; exempted; drawn into GPU memory
  // These are sums, not a record of each draw; they are meant for one caller, EngineGpu::commit()
  // (prover/engine/engine_gpu.cuh). Two errors would still balance: a draw passed to held_card()
  // twice while another of its length is never passed; a draw that is never passed while its
  // length is reported as exempt.
  bool good() const { return differ == 0 && compared == handed && handed + left == laid; }
};

// Every cudaStream_t passed to this class or to coins_own_test_card() must keep order with the
// default stream, on which Card<> allocates and frees (prover/engine/device.cuh): never one
// created with cudaStreamNonBlocking; under nvcc's --default-stream per-thread, only the calling
// thread's default stream. Not checked here; EngineGpu's constructor checks its own stream.
class CardStreamCoins : public StreamCoinsT<CardCoins> {
public:
  using StreamCoinsT<CardCoins>::StreamCoinsT;
  // From now on every draw records up to `places` of its values (all of them if it has fewer; 0:
  // none), read back from the buffer that it has just written; pick() chooses which.
  // Not caught: a value that is written correctly and then not used, used twice or overwritten.
  // Of the public functions declared here, only sample(), samples(), bound() and
  // left_to_the_path_check() do not check that the calling thread holds the generator
  // (blocks/coins/coins.h): call them from that thread only.
  void sample(unsigned places) {
    BLK_REQUIRE(places <= COIN_SAMPLES_MOST, "coins: at most 8 samples a draw");
    places_ = places;
  }
  // The values recorded since the last call. Waits for the CUDA streams of the draws concerned.
  // The values are as secret as the draws themselves: the caller must wipe them.
  std::vector<CoinSample> samples() {
    std::vector<std::vector<u64>> home(chunks_.size());
    for (size_t c = 0; c < chunks_.size(); c++) {
      home[c].resize(chunks_[c].used);
      ENGINE_CK(cudaMemcpyAsync(home[c].data(), chunks_[c].room.get(), 8 * chunks_[c].used, cudaMemcpyDeviceToHost, chunks_[c].stream));
      ENGINE_CK(cudaStreamSynchronize(chunks_[c].stream));
    }
    std::vector<CoinSample> out;
    for (const Noted &x : noted_) {
      out.push_back(x.sample);
      if (x.chunk != ON_THE_HOST) out.back().value = home[x.chunk][x.offset];
    }
    for (std::vector<u64> &h : home) wipe(h.data(), 8 * h.size());
    wipe(noted_.data(), noted_.size() * sizeof(Noted));
    noted_.clear(), chunks_.clear();
    return out;
  }
  void fill(F *out, size_t n) override {
    StreamCoinsT<CardCoins>::fill(out, n);
    note_host(true, reinterpret_cast<const u8 *>(out), n);
  }
  void bytes(u8 *out, size_t n) override {
    StreamCoinsT<CardCoins>::bytes(out, n);
    note_host(false, out, n / 8); // whole words only
  }
  void fill_card(F *out, size_t n, cudaStream_t stream) override { laid_ += n, elements_card(open(true, n), 0, out, n, stream); }
  void bytes_card(u8 *out, size_t n, cudaStream_t stream) override {
    BLK_REQUIRE(n % 8 == 0, "coins: bytes for the GPU come in words of 8");
    laid_ += n / 8, words_card(open(false, n), 0, reinterpret_cast<u64 *>(out), n / 8, stream);
  }
  // Compares a whole draw, where it is in GPU memory (`laid`), with the generator's values,
  // computed again. Why: a draw that exists in GPU memory only (a main row's random tail: its
  // random coefficients; the salts of tree leaves) can be lost, cut short or overwritten after
  // its samples are taken and before its Merkle tree is built. What reads that memory later then
  // agrees with the tree, the proof verifies (with salts that are not kept: unless it opens such
  // a leaf) and hides less than it should. (A kept draw lost later makes the proof invalid.)
  // s: a stream that fill_card() (elements) or bytes_card() has drawn. n: the number of values by
  // the caller's own shape; it must be the whole draw. The kernel is queued on `stream`: the CUDA
  // stream on which the values were read to build the tree. Only the first call with n > 0 waits
  // for that stream, so that the two counters are zeroed before a kernel on any stream adds to
  // them. bound() gives the result.
  void held_card(u32 s, bool elements, const u64 *laid, size_t n, cudaStream_t stream) {
    drawn(s, elements, 0, elements ? n : 8 * n);
    BLK_REQUIRE(ledger()[s - COIN_FIRST_STREAM].count == (elements ? n : 8 * n), "coins: held_card(): the given count is not the length of that draw");
    if (!n) return;
    if (!held_.size()) { // new GPU memory is not zeroed (device.cuh)
      held_.resize(2);
      ENGINE_CK(cudaMemsetAsync(held_.get(), 0, 16, stream));
      ENGINE_CK(cudaStreamSynchronize(stream));
    }
    handed_ += n;
    if (elements)
      k_stream_coins_held<true><<<blocks_for(calls(0, n), 256), 256, 0, stream>>>(key(), s, laid, n, held_.get());
    else
      k_stream_coins_held<false><<<blocks_for(calls(0, n), 256), 256, 0, stream>>>(key(), s, laid, n, held_.get());
    ENGINE_CK(cudaGetLastError());
  }
  // Reports n values drawn into GPU memory as exempt from held_card(): left, as the name says, to
  // the verifier's Merkle path check, which tests opened leaves' salts only. Only tests call it:
  // EngineGpu::commit() (prover/engine/engine_gpu.cuh) passes each of its draws to held_card().
  void left_to_the_path_check(size_t n) { left_ += n; }
  // The totals so far. The caller must have waited for the CUDA streams of the queued checks:
  // samples() waits for the streams of the sampled draws, EngineGpu::let_go() for the engine's.
  // A check that has not finished shows as compared < handed, which stops a caller that asks too
  // early; but nothing guarantees that a copy taken while a kernel runs, once it shows a thread
  // block's addition to `compared`, also shows that block's additions to `differ`.
  CoinsBound bound() const {
    unsigned long long home[2] = {0, 0};
    if (held_.size()) held_.to_host(home);
    return CoinsBound{home[0], home[1], handed_, left_, laid_};
  }
  // Generates values of an earlier draw again, into GPU memory: for what is recomputed rather than
  // kept. This call and the next are sampled like a draw.
  void again_card(u32 s, u64 first, F *out, size_t n, cudaStream_t stream) const { elements_card(drawn(s, true, first, n), first, out, n, stream); }
  // first, n: in words of 8 bytes of a stream of bytes.
  void again_words_card(u32 s, u64 first, u64 *out, size_t n, cudaStream_t stream) const {
    words_card(drawn(s, false, 8 * first, 8 * n), first, out, n, stream);
  }
  // The salts of listed leaves: `width` words for each place at[j] (an array in GPU memory). Every
  // at[j] must be below `leaves`: not checked. Not sampled.
  void again_words_at_card(u32 s, const u64 *at, size_t places, unsigned width, u64 leaves, u64 *out, cudaStream_t stream) const {
    drawn(s, false, 0, 8 * leaves * width);
    if (!places) return;
    k_stream_coins_words_at<<<blocks_for(places * width, 128), 128, 0, stream>>>(key(), s, at, places, width, out);
    ENGINE_CK(cudaGetLastError());
  }


private:
  static u64 calls(u64 first, size_t n) { return (first + n + COIN_LANES - 1) / COIN_LANES - first / COIN_LANES; }
  void elements_card(u32 s, u64 first, F *out, size_t n, cudaStream_t stream) const {
    if (!n) return;
    k_stream_coins_elements<<<blocks_for(calls(first, n), 256), 256, 0, stream>>>(key(), s, first, out, n);
    ENGINE_CK(cudaGetLastError());
    static_assert(sizeof(F) == 8, "an element lies in memory as one word");
    note_card(s, true, first, reinterpret_cast<const u64 *>(out), n, stream);
  }
  void words_card(u32 s, u64 first, u64 *out, size_t n, cudaStream_t stream) const {
    if (!n) return;
    k_stream_coins_words<<<blocks_for(calls(first, n), 256), 256, 0, stream>>>(key(), s, first, out, n);
    ENGINE_CK(cudaGetLastError());
    note_card(s, false, first, out, n, stream);
  }
  // The places of a draw of n values that are recorded: all if n <= places_; otherwise the first
  // places_ of 0, n - 1, then places that mix the stream's number, the sample's number k and
  // coin_pick_turn(), which may repeat.
  // The first multiplier is the constant that SplitMix64 adds at each step (splitmix64.c by
  // Sebastiano Vigna, public domain); the mix as a whole is not a step of that generator.
  unsigned pick(u32 s, size_t n, CoinPicks &p) const {
    const unsigned count = (unsigned)(n < places_ ? n : places_);
    for (unsigned k = 0; k < count; k++) {
      u64 x = (((u64)s << 8 | k) * 0x9E3779B97F4A7C15ull ^ coin_pick_turn()) * 0xD6E8FEB86659FD93ull;
      x ^= x >> 29;
      p.at[k] = n <= places_ ? k : k == 0 ? 0 : k == 1 ? n - 1 : x % n;
    }
    return count;
  }
  void note_card(u32 s, bool element, u64 first, const u64 *laid, size_t n, cudaStream_t stream) const {
    CoinPicks p{};
    const unsigned count = pick(s, n, p);
    if (!count) return;
    if (chunks_.empty() || chunks_.back().stream != stream || chunks_.back().used + count > CHUNK) {
      chunks_.emplace_back();
      chunks_.back().room.resize(CHUNK), chunks_.back().stream = stream;
    }
    Chunk &c = chunks_.back();
    k_stream_coins_pick<<<1, count, 0, stream>>>(laid, p, c.room.get() + c.used);
    ENGINE_CK(cudaGetLastError());
    for (unsigned k = 0; k < count; k++) noted_.push_back(Noted{CoinSample{s, element, first + p.at[k], 0}, chunks_.size() - 1, c.used + k});
    c.used += count;
  }
  void note_host(bool element, const u8 *laid, size_t n) {
    if (!places_ || !n) return;
    CoinPicks p{};
    const u32 s = last_stream();
    const unsigned count = pick(s, n, p);
    for (unsigned k = 0; k < count; k++) {
      u64 v;
      std::memcpy(&v, laid + 8 * p.at[k], 8);
      noted_.push_back(Noted{CoinSample{s, element, p.at[k], v}, ON_THE_HOST, 0});
    }
  }
  static constexpr size_t CHUNK = 1024, ON_THE_HOST = ~(size_t)0;
  struct Chunk { // GPU memory into which the draws on one CUDA stream copy their samples
    Card<u64> room;
    cudaStream_t stream = 0;
    size_t used = 0;
  };
  struct Noted {
    CoinSample sample;
    size_t chunk, offset; // where in chunks_ its value is; chunk ON_THE_HOST: already in `sample`
  };
  unsigned places_ = 0;
  // Allocated from the calling thread's arena, if any (prover/engine/arena.h): only tests name one.
  Card<unsigned long long> held_; // `held_card`'s two counters, allocated at its first call
  u64 handed_ = 0, left_ = 0, laid_ = 0;
  mutable std::vector<Chunk> chunks_;
  mutable std::vector<Noted> noted_;
};

} // namespace blk
