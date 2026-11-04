// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests CardStreamCoins (prover/engine/stream_coins.cuh), which writes private random values
// (`coins` in identifiers) into GPU memory (`card`). Under one key it must give the same values and
// list of draws (ledger()) as StreamCoins (blocks/coins/coins.h) on the host, also where
// again_card() and its variants generate part of an earlier draw again. Its samples (some values of
// each draw, recorded after sample()) must match a second implementation (`second_writing` in
// paths, two words in messages), and two processes must sample different middle indices
// (`places`). held_card() and bound() are not tested here.
// Exit code 0: all as expected; 1: something differs; 3: no GPU.
// The only fault that it tries is its own: it changes one recorded sample on the host and expects
// one difference more. No run of this test shows that the samples catch a fault injected into
// the values in GPU memory.
// Build outside the source tree:
//   nvcc -std=c++17 -O2 -arch=sm_90 -I blocks -I prover -DBLK_FIELD=2 -DBLK_HASH=1 prover/engine/tests/stream_coins_test.cu -o <somewhere>/stream_coins_test
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <vector>

#include "engine/stream_coins.cuh"
using namespace blk;

static int failures = 0;
static void hold(bool ok, const char *what) {
  if (!ok) failures++, std::printf("DIFFERS: %s\n", what);
}
static __global__ void k_elements_at(CoinKey key, u32 stream, const u64 *at, size_t n, F *out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = coin_element(key, stream, at[i]);
}
// Here the second implementation is only `twin`, which makes the same draws on the host and shares
// blocks/coins/coins.h with CardStreamCoins; the programs `cp_commit` and `cp_prove` use
// prover/second_writing/coin_generator.h, which does not. Stream `COIN_FIRST_STREAM + 20` is the
// fill_card() of 65,536 elements. Returns how many samples differ.
static size_t samples_case(const CoinKey &process) {
  const CoinContext context{6, 0x81902, 1, 3};
  CardStreamCoins card(process, context);
  StreamCoins twin(process, context);
  card.sample(5);
  size_t expected = 0;
  std::vector<F> f(1 << 16);
  std::vector<u8> b(8 << 16);
  Card<F> d(1 << 16);
  Card<u64> w(1 << 16);
  for (size_t n : {1, 3, 5, 6, 4000, 1 << 16}) {
    card.fill_card(d.get(), n, 0), twin.fill(f.data(), n), expected += n < 5 ? n : 5;
    card.bytes_card(reinterpret_cast<u8 *>(w.get()), 8 * n, 0), twin.bytes(b.data(), 8 * n), expected += n < 5 ? n : 5;
    card.fill(f.data(), n), twin.fill(f.data(), n), expected += n < 5 ? n : 5; // on the host, through the same object
    card.bytes(b.data(), 8 * n + 3), twin.bytes(b.data(), 8 * n + 3), expected += n < 5 ? n : 5;
  }
  card.again_card(COIN_FIRST_STREAM + 20, 100, d.get(), 3000, 0), expected += 5; // of the draw of 65,536 elements
  std::vector<CoinSample> got = card.samples();
  hold(got.size() == expected, "the number of samples");
  bool first = false, last = false, again = false;
  for (const CoinSample &s : got) {
    first |= s.stream == COIN_FIRST_STREAM + 20 && s.index == 0, last |= s.stream == COIN_FIRST_STREAM + 20 && s.index == (1 << 16) - 1;
    again |= s.stream == COIN_FIRST_STREAM + 20 && s.index == 100 + 2999;
  }
  hold(first && last && again, "a draw's first and last value are among its samples");
  hold(card.samples().empty(), "samples are handed out once");
  auto second = [&](u32 stream, bool element, u64 index) {
    u64 v = 0;
    if (element) {
      F x;
      twin.again(stream, index, &x, 1), v = x.v;
    } else {
      u8 word[8];
      twin.again(stream, 8 * index, word, 8), std::memcpy(&v, word, 8);
    }
    return v;
  };
  const size_t differ = coin_samples_differ(got, second);
  got[got.size() / 2].value ^= 1;
  hold(coin_samples_differ(got, second) == differ + 1, "one changed value is one difference more");
  card.sample(0);
  card.fill_card(d.get(), 10, 0);
  hold(card.samples().empty(), "no samples when none are asked for");
  return differ;
}

// Returns the four indices that a child process samples in one draw of 2^20 elements on the host,
// under a fixed key and context; empty if pipe() or the child fails.
// Call it before this process has called coin_pick_turn(), so that each child draws its own 8
// random bytes, on which the middle indices depend, and before it has initialised CUDA, which a
// forked child cannot use.
static std::vector<u64> places_of_a_process() {
  int ends[2];
  if (pipe(ends)) return {};
  const pid_t child = fork();
  if (child < 0) { std::perror("fork"); std::abort(); }
  if (!child) {
    CoinKey process;
    for (u32 i = 0; i < 8; i++) process.w[i] = 0x9E3779B9u * (i + 1);
    CardStreamCoins card(process, CoinContext{6, 0x81902, 1, 3});
    card.sample(4);
    std::vector<F> f((size_t)1 << 20);
    card.fill(f.data(), f.size());
    u64 at[4] = {0, 0, 0, 0};
    const std::vector<CoinSample> got = card.samples();
    for (size_t k = 0; k < got.size() && k < 4; k++) at[k] = got[k].index;
    _exit(got.size() == 4 && write(ends[1], at, sizeof at) == (ssize_t)sizeof at ? 0 : 1);
  }
  close(ends[1]);
  std::vector<u64> at(4);
  const bool whole = read(ends[0], at.data(), 32) == 32;
  close(ends[0]);
  int status = -1;
  waitpid(child, &status, 0);
  return whole && WIFEXITED(status) && !WEXITSTATUS(status) ? at : std::vector<u64>();
}

int main() {
  { // before the GPU is touched
    const std::vector<u64> one = places_of_a_process(), two = places_of_a_process();
    hold(one.size() == 4 && two.size() == 4, "two processes hand their four places over");
    if (one.size() == 4 && two.size() == 4) {
      hold(one[0] == 0 && two[0] == 0 && one[1] == ((u64)1 << 20) - 1 && two[1] == one[1], "a draw's first and last place are among the four in every process");
      hold(one[2] != two[2] && one[3] != two[3], "two processes choose other middle places for one stream (equal by chance once in 2^20 for each)");
      hold(one[2] < ((u64)1 << 20) && one[3] < ((u64)1 << 20) && two[2] < ((u64)1 << 20) && two[3] < ((u64)1 << 20), "the middle places lie in the draw");
    }
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
    std::printf("no GPU: nothing was compared\n");
    return 3;
  }
  coins_own_test_card();
  CoinKey process;
  for (u32 i = 0; i < 8; i++) process.w[i] = 0x9E3779B9u * (i + 1);
  hold(samples_case(process) == 0, "every sample is what the second writing gives");
  const CoinContext context{5, 0x81902, 1, 3};
  StreamCoins host(process, context);
  CardStreamCoins card(process, context);
  for (size_t n : {1, 3, 4, 5, 1000, 65537, 1 << 20}) {
    std::vector<F> a(n), b(n);
    Card<F> d(n);
    host.fill(a.data(), n), card.fill_card(d.get(), n, 0), d.to_host(b.data());
    hold(!memcmp(a.data(), b.data(), n * sizeof(F)), "elements");
    std::vector<u8> x(8 * n), y(8 * n);
    Card<u64> w(n);
    host.bytes(x.data(), 8 * n), card.bytes_card(reinterpret_cast<u8 *>(w.get()), 8 * n, 0), w.to_host(reinterpret_cast<u64 *>(y.data()));
    hold(x == y, "bytes");
    if (n > 10) {
      const u32 bytes_stream = card.last_stream(), elements_stream = bytes_stream - 1;
      const size_t first = n / 3 + 1, some = n / 2 - 1;
      Card<F> e(some);
      std::vector<F> got(some);
      card.again_card(elements_stream, first, e.get(), some, 0), e.to_host(got.data());
      hold(!memcmp(got.data(), a.data() + first, some * sizeof(F)), "elements once more, from the middle");
      Card<u64> v(some);
      std::vector<u64> words(some);
      card.again_words_card(bytes_stream, first, v.get(), some, 0), v.to_host(words.data());
      hold(!memcmp(words.data(), x.data() + 8 * first, 8 * some), "words once more, from the middle");
      const std::vector<u64> at = {0, 2, n / 2 - 1, 1};
      Card<u64> places, salts(2 * at.size());
      places.from_host(at.data(), at.size());
      std::vector<u64> two(2 * at.size());
      card.again_words_at_card(bytes_stream, places.get(), at.size(), 2, n / 2, salts.get(), 0), salts.to_host(two.data());
      bool same = true;
      for (size_t j = 0; j < at.size(); j++) same = same && !memcmp(&two[2 * j], x.data() + 16 * at[j], 16);
      hold(same, "the salts of listed leaves");
    }
  }
  hold(host.ledger().size() == card.ledger().size(), "the two ledgers");
  for (size_t i = 0; i < host.ledger().size(); i++)
    hold(host.ledger()[i].stream == card.ledger()[i].stream && host.ledger()[i].elements == card.ledger()[i].elements &&
             host.ledger()[i].count == card.ledger()[i].count, "a line of the ledgers");
  // With p = 2^64 - 2^32 + 1 about one word in 2^32 is rejected (`refused`) by F_from_word(), and
  // coin_admit() derives another. GPU and host must agree on coin_element() at four indices with a
  // rejected first word and at index 5, an ordinary one.
  { // indices from blocks/coins/tests/coin_values.py's list REFUSED, under that file's key
    CoinKey seed;
    u8 fresh[32];
    Digest inputs;
    for (u32 i = 0; i < 8; i++) {
      seed.w[i] = 0x01010101u * (i + 1), inputs.w[i] = 0x0A0B0C0Du * (i + 1);
      for (int b = 0; b < 4; b++) fresh[4 * i + b] = (u8)((0x10203040u + i) >> 8 * b);
    }
    const CoinKey K = context_key(process_key(seed_child(seed, 6, 0), fresh, inputs), context);
    const std::vector<u64> at = {362584566ull, 1327292691ull, 6312099836ull, 16942013412ull, 5};
    Card<u64> places;
    places.from_host(at.data(), at.size());
    Card<F> e(at.size());
    std::vector<F> got(at.size());
    k_elements_at<<<1, 32>>>(K, 16400, places.get(), at.size(), e.get());
    e.to_host(got.data());
    unsigned refused = 0;
    for (size_t i = 0; i < at.size(); i++) {
      F x;
      refused += !F_from_word(coin_word(K, 16400, at[i]), x);
      hold(got[i] == coin_element(K, 16400, at[i]), "an element at a place that may be refused");
    }
    hold(refused == 4u, "the number of really refused places");
  }
  std::printf("stream_coins_test: field %d: %s\n", BLK_FIELD, failures ? "NOT as expected" : "host and GPU give the same coins: all as expected");
  return failures != 0;
}
