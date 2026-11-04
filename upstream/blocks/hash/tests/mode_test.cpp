// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/hash/hash.h. Run from blocks/. Parts 1 and 2 compare its hashes with plain() below.
//  1. A `Hasher` fed in pieces of several sizes.
//  2. vertex(), run_value(), test_coins(), leaf() and its form for kernels, a hash chain's calls.
//  3. Known answers, each one call of Python's hashlib.blake2b: blocks/hash/tests/mode_vectors.txt.
//  4. Inputs that differ in length, tag, index, start or call number give different digests.
//  5. work_test().  6. Invalid calls abort the process.
#include "hash/hash.h"
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
using namespace blk;

static int bad = 0, checks = 0;
static void check(bool ok, const char *what) {
  checks++;
  if (!ok) bad++, printf("\nFAILED: %s", what);
}
static u64 rng_state = 0x243F6A8885A308D3ull;
static u64 rnd() { // splitmix64 (S. Vigna, https://prng.di.unimi.it/splitmix64.c): test data only
  u64 z = (rng_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ z >> 27) * 0x94D049BB133111EBull;
  return z ^ z >> 31;
}
static std::vector<u8> bytes_of(size_t n) {
  std::vector<u8> b(n);
  for (auto &x : b) x = (u8)rnd();
  return b;
}
static Digest some_digest() {
  Digest d;
  for (auto &w : d.w) w = (u32)rnd();
  return d;
}
static std::string hex(const Digest &d) {
  u8 b[32];
  digest_to_bytes(d, b);
  std::string s;
  char t[3];
  for (u8 x : b) snprintf(t, sizeof t, "%02x", x), s += t;
  return s;
}

static std::string hex(const std::vector<u8> &b) {
  std::string s;
  char t[3];
  for (u8 x : b) snprintf(t, sizeof t, "%02x", x), s += t;
  return s;
}
// BLAKE2b once more: the whole string at hand, the parameter block written out byte by byte, F called directly. F and the initialization vector are the
// library's (hash/blake2b_f.h): only the known answers of part 3 can show a fault in either.
// tree: all of the label's first word above the number: a tree's number, a chain's label, and for a leaf + 2^16 symbols + 2^40 (salt bytes / 8).
static std::vector<u8> plain(u32 number, u64 tree, u64 index, const Digest *start, const std::vector<u8> &body, unsigned length) {
  u8 block[64] = {(u8)length, 0, 1, 1};
  if (start) digest_to_bytes(*start, block + 32); // salt, personalization
  std::vector<u8> in;
  for (int i = 0; i < 8; i++) in.push_back((u8)((number | tree << 16) >> 8 * i));
  for (int i = 0; i < 8; i++) in.push_back((u8)(index >> 8 * i));
  in.insert(in.end(), body.begin(), body.end());
  const u64 IV[8] = {blake2b_IV0, blake2b_IV1, blake2b_IV2, blake2b_IV3, blake2b_IV4, blake2b_IV5, blake2b_IV6, blake2b_IV7};
  u64 h[8], m[16];
  for (int i = 0; i < 8; i++) {
    h[i] = IV[i];
    for (int k = 0; k < 8; k++) h[i] ^= (u64)block[8 * i + k] << 8 * k;
  }
  const size_t n = in.size(), steps = n ? (n + 127) / 128 - 1 : 0;
  for (size_t s = 0; s <= steps; s++) {
    for (int i = 0; i < 16; i++) {
      m[i] = 0;
      for (int k = 0; k < 8; k++)
        if (128 * s + 8 * i + k < n) m[i] |= (u64)in[128 * s + 8 * i + k] << 8 * k;
    }
    Blake2bF::F(h, m, s < steps ? 128 * (s + 1) : n, 0, s == steps);
  }
  std::vector<u8> out(length);
  for (unsigned i = 0; i < length; i++) out[i] = (u8)(h[i / 8] >> 8 * (i % 8));
  return out;
}
static Digest plain32(u32 number, u64 tree, u64 index, const Digest *start, const std::vector<u8> &in) {
  Digest d;
  digest_from_bytes(plain(number, tree, index, start, in, 32).data(), d);
  return d;
}
static std::vector<u8> bytes_of(const ChainCall &c) {
  std::vector<u8> b(64);
  for (int i = 0; i < 64; i++) b[i] = (u8)(c.w[i / 8] >> 8 * (i % 8));
  return b;
}
static void put(std::vector<u8> &b, u64 x) {
  for (int i = 0; i < 8; i++) b.push_back((u8)(x >> 8 * i));
}
static void put(std::vector<u8> &b, const Digest &d) {
  u8 x[32];
  digest_to_bytes(d, x);
  b.insert(b.end(), x, x + 32);
}

static void pieces() {
  for (size_t n : {0, 1, 3, 4, 7, 8, 9, 31, 32, 63, 64, 65, 100, 111, 112, 113, 127, 128, 129, 130, 191, 192, 239, 240, 241, 255, 256, 257, 367, 368, 369, 383, 384, 385, 1000}) {
    const std::vector<u8> in = bytes_of(n);
    const u64 index = rnd() >> 1;
    const u32 tree = (u32)rnd() & 0xFFFF;
    const Digest start = some_digest();
    const Digest want = plain32(166, 0, index, &start, in), bare = plain32(168, tree, index, nullptr, in), own = plain32(171, 0, index, &start, in);
    for (int way = 0; way < 4; way++) {
      Hasher a(Tag(Domain::TREE_NODE, 0, start), index), b(Tag(Domain::STATEMENT, tree), index), c(Domain::TEST_COINS, index, start);
      for (Hasher *h : {&a, &b, &c}) {
        size_t at = 0;
        while (at < n) {
          size_t take = way == 0 ? n : way == 1 ? 1 : way == 2 ? 1 + rnd() % 9 : 4 * (1 + rnd() % 40);
          if (take > n - at) take = n - at;
          if (way == 3 && take == 4)
            h->push_u32(in[at] | (u32)in[at + 1] << 8 | (u32)in[at + 2] << 16 | (u32)in[at + 3] << 24);
          else
            h->push(in.data() + at, take);
          at += take;
        }
        check(h->bytes() == n, "a Hasher counts its bytes");
      }
      check(a.finish() == want && b.finish() == bare && c.finish() == own, "a Hasher fed in pieces is BLAKE2b on the whole string");
    }
  }
}

static void two_digests_and_leaves() {
  for (int i = 0; i < 60; i++) {
    const Digest l = some_digest(), r = some_digest(), run = some_digest();
    const u64 index = rnd() >> 1;
    std::vector<u8> lr;
    put(lr, l), put(lr, r);
    check(vertex(Tag(Domain::TREE_NODE, 0, run), l, r) == plain32(166, 0, 0, &run, lr) && vertex(l, r, run) == plain32(166, 0, 0, &run, lr) &&
              vertex(Tag(Domain::BINDING_NODE, 0, run), l, r) == plain32(178, 0, 0, &run, lr),
          "a vertex is the hash of its two children from the parameters' digest");
    check(run_value(index, l, r) == plain32(172, 0, index, nullptr, lr), "a run's value is the hash of two digests");
    check(test_coins(run, index) == plain32(171, 0, index, &run, {}), "made-up coins are the hash of nothing from a seed");
  }
  for (size_t width = 1; width <= 70; width++)
    for (size_t salt_bytes : {40, 48, 56, 64}) {
      const Digest run = some_digest();
      const Tag tag(Domain::LEAF_MAIN, 7, run, width, salt_bytes);
      check(tag.word == 164 + ((u64)7 << 16) + ((u64)width << 32) + ((u64)(salt_bytes / 8) << 56) && tag.symbols() == width && tag.salt_bytes() == salt_bytes,
            "a leaf's tag holds number, tree, symbols and the salt's words");
      const u64 index = ((u64)1 << 23) + rnd() % (1 << 23);
      std::vector<F> symbols(width);
      for (F &x : symbols) x = F{rnd() % P};
      const std::vector<u8> salt = bytes_of(salt_bytes);
      std::vector<u8> in; // the whole string, the label in front
      put(in, tag.word), put(in, index);
      for (const F &x : symbols) put(in, x.v);
      if (in.size() % 128 + salt_bytes > 128)
        while (in.size() % 128) in.push_back(0);
      check(in.size() == leaf_salt_at(width, salt_bytes), "where a leaf's salt begins");
      in.insert(in.end(), salt.begin(), salt.end());
      check(in.size() == leaf_bytes(width, salt_bytes) && (in.size() - 1) / 128 == (in.size() - salt_bytes) / 128, "the salt lies whole in the last block");
      const Digest want = plain32(164, tag.word >> 16, index, &run, std::vector<u8>(in.begin() + 16, in.end()));
      check(leaf(tag, index, symbols.data(), width, salt.data(), salt_bytes) == want, "a leaf is the hash of symbols, fill and salt");
      u64 cv[8], block[16];
      leaf_begin(cv, tag);
      const size_t steps = (in.size() - 1) / 128;
      for (size_t s = 0; s <= steps; s++) {
        for (int i = 0; i < 16; i++) {
          block[i] = 0;
          for (int k = 0; k < 8; k++)
            if (128 * s + 8 * i + k < in.size()) block[i] |= (u64)in[128 * s + 8 * i + k] << 8 * k;
        }
        if (s < steps) leaf_step(cv, block, (u32)s);
      }
      check(leaf_last(cv, block, (u32)in.size()) == want, "the three calls for kernels give the leaf");
    }
  { // a leaf whose content is a digest
    const Digest from = some_digest(), content = some_digest();
    const std::vector<u8> salt = bytes_of(40);
    const Tag tag(Domain::LEAF_BINDING, 0, from, 4, 40);
    std::vector<u8> in;
    put(in, content);
    in.insert(in.end(), salt.begin(), salt.end());
    check(leaf(tag, 37, content, salt.data(), 40) == plain32(173, tag.word >> 16, 37, &from, in), "a leaf of one digest is the hash of digest and salt");
  }
  { // leaves that would hash the same string if the label lacked their shape (symbols, salt bytes)
    const Digest run = some_digest();
    const std::vector<u8> salt = bytes_of(48);
    std::vector<F> symbols(14, F::zero());
    for (int i = 0; i < 10; i++) symbols[i] = F{rnd() % P};
    std::set<std::string> seen;
    for (size_t width = 10; width <= 14; width++) seen.insert(hex(leaf(Tag(Domain::LEAF_MAIN, 1, run, width, 40), 5, symbols.data(), width, salt.data(), 40)));
    check(seen.size() == 5, "leaves of 10 to 14 symbols that differ in zeros alone differ");
    u64 first = 0;
    for (int k = 0; k < 8; k++) first |= (u64)salt[k] << 8 * k;
    symbols[3] = F{first % P};
    if (symbols[3].v == first)
      check(leaf(Tag(Domain::LEAF_MAIN, 1, run, 3, 48), 5, symbols.data(), 3, salt.data(), 48) != leaf(Tag(Domain::LEAF_MAIN, 1, run, 4, 40), 5, symbols.data(), 4, salt.data() + 8, 40),
            "3 symbols with a salt of 48 bytes and 4 with one of 40 on the same bytes differ");
  }
}

template <int CELLS> static void cells_case() {
  u64 cells[CELLS];
  for (u64 &c : cells) c = rnd();
  cells[CELLS - 1] &= 0xFFFF;
  const Digest rho = some_digest();
  const u64 round = rnd() >> 1, nonce = rnd();
  for (u64 j : {(u64)0, (u64)1, (u64)77, ((u64)1 << 48) - 1}) {
    std::vector<u8> in;
    for (int i = 0; i + 1 < CELLS; i++) put(in, cells[i]);
    put(in, cells[CELLS - 1] | j << 16), put(in, nonce);
    check(bytes_of(chain_call_cells(rho, round, cells, nonce, j)) == plain(162, 0, round, &rho, in, 64) &&
              bytes_of(chain_call_cells(rho, round, cells, nonce, j, ((u64)1 << 32) + 513)) == plain(162, ((u64)1 << 32) + 513, round, &rho, in, 64),
          "a call on cells is the hash of the string");
  }
}
static void rounds() {
  for (size_t words = 1; words <= 50; words++) { // the message's words of 8 bytes, the last of which takes the call's number
    const Digest rho = some_digest();
    const u64 round = rnd() >> 1, nonce = rnd(), kind = rnd() & 0xFFFF;
    std::vector<u64> message(words);
    for (u64 &x : message) x = rnd();
    message.back() = kind;
    const u64 chain = rnd() >> 16;
    ChainAbsorb a(rho, round, chain);
    for (u64 x : message) a.push_u64(x);
    const ChainPrefix p = a.prefix();
    for (u64 j : {(u64)0, (u64)1, (u64)2, (u64)300, (u64)1 << 40}) {
      std::vector<u8> in;
      for (size_t i = 0; i + 1 < words; i++) put(in, message[i]);
      put(in, kind | j << 16), put(in, nonce);
      const ChainCall c = chain_call(p, nonce, j);
      check(bytes_of(c) == plain(162, chain, round, &rho, in, 64), "a round's call is the hash of message, number and nonce, 64 bytes");
      u8 half[32];
      digest_to_bytes(chain_block(c, 1), half);
      check(!memcmp(half, bytes_of(c).data() + 32, 32), "a call's second block is its bytes 32 to 63");
      if (j == 0)
        for (unsigned bits : {0u, 1u, 3u}) check(chain_work_passes(p, nonce, bits) == work_test(c.w[0], bits), "a round's work is tested on the first word of call 0");
    }
  }
  cells_case<1>(), cells_case<2>(), cells_case<7>(), cells_case<14>(), cells_case<15>(), cells_case<16>(), cells_case<17>(), cells_case<31>(), cells_case<33>();
}

static void known_answers() {
  std::ifstream file("hash/tests/mode_vectors.txt");
  int lines = 0;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    u32 number;
    u64 tree, index;
    size_t width, salt_bytes, n;
    unsigned length;
    std::string start, value;
    s >> number >> tree >> index >> start >> width >> salt_bytes >> n >> length >> value;
    std::vector<u8> in(n);
    for (size_t i = 0; i < n; i++) in[i] = (u8)((7 * i + 3) % 251);
    if (salt_bytes) // a leaf: symbols, zeros, salt
      for (size_t i = 8 * width; i < n; i++) in[i] = i < n - salt_bytes ? 0 : (u8)((5 * (i - (n - salt_bytes)) + 1) % 251);
    Digest from{};
    if (start != "-") {
      u8 b[32];
      for (int i = 0; i < 32; i++) b[i] = (u8)std::stoi(start.substr(2 * i, 2), nullptr, 16);
      digest_from_bytes(b, from);
    }
    lines++;
    check(hex(plain(number, tree + ((u64)width << 16) + ((u64)(salt_bytes / 8) << 40), index, start == "-" ? nullptr : &from, in, length)) == value,
          "the second writing meets the library's value");
    const Domain d = (Domain)number;
    if (d == Domain::CHAIN_QUERY) { // the input's last sixteen bytes as a word with the call's number in it, and the nonce
      ChainAbsorb a(from, index, tree);
      a.push(in.data(), n - 16);
      u64 word = 0, nonce = 0;
      for (int k = 0; k < 8; k++) word |= (u64)in[n - 16 + k] << 8 * k, nonce |= (u64)in[n - 8 + k] << 8 * k;
      a.push_u64(word & 0xFFFF);
      check(hex(bytes_of(chain_call(a.prefix(), nonce, word >> 16))) == value, "a round's call meets the library's value");
      continue;
    }
    Hasher h = start == "-"               ? Hasher(Tag(d, tree), index)
               : domain_is_leaf(d)      ? Hasher(Tag(d, (u32)tree, from, width, salt_bytes), index)
               : domain_start_in_tag(d) ? Hasher(Tag(d, (u32)tree, from), index)
                                        : Hasher(Tag(d, tree), index, from);
    h.push(in.data(), n);
    check(hex(h.finish()) == value, "a Hasher meets the library's value");
  }
  check(lines == 58, "mode_vectors.txt has 58 lines of values (run the test from blocks/)");
}

static void differ() {
  std::set<std::string> seen;
  size_t made = 0;
  auto add = [&](const Digest &d) { seen.insert(hex(d)), made++; };
  const Digest s = some_digest(), t = some_digest();
  for (size_t n : {0, 1, 2, 127, 128, 129, 255, 256}) { // all zeros: BLAKE2b's padding is zeros too
    Hasher h(Domain::STATEMENT, 1);
    const std::vector<u8> zeros(n, 0);
    h.push(zeros.data(), n);
    add(h.finish());
  }
  const std::vector<u8> in = bytes_of(200);
  auto one = [&](const Tag &tag, u64 index) {
    Hasher h(tag, index);
    h.push(in.data(), in.size());
    add(h.finish());
  };
  for (Domain d : {Domain::LEAF_MAIN, Domain::LEAF_BINDING, Domain::LEAF_STORED_MAIN})
    for (u32 tree : {0u, 1u, 256u, 65535u})
      for (u64 index : {(u64)0, (u64)1, (u64)1 << 32, ((u64)1 << 63) - 1})
        for (const Digest *start : {&s, &t}) one(Tag(d, tree, *start, 20, 40), index); // 160 + 40 bytes, no fill
  for (Domain d : {Domain::CHAIN_START, Domain::STATEMENT, Domain::PARAMETERS, Domain::RUN, Domain::COIN_INPUTS, Domain::PUBLISHED_ID})
    for (u64 index : {(u64)0, (u64)2, (u64)1 << 40}) one(Tag(d), index);
  for (Domain d : {Domain::TEST_COINS})
    for (u64 index : {(u64)0, (u64)3})
      for (const Digest *start : {&s, &t}) {
        Hasher h(d, index, *start);
        h.push(in.data(), in.size());
        add(h.finish());
      }
  ChainAbsorb a(s, 5);
  a.push(in.data(), in.size()), a.push_u64(9);
  const ChainPrefix p = a.prefix();
  for (u64 j : {0, 1, 2})
    for (u64 nonce : {0, 1}) add(chain_block(chain_call(p, nonce, j), 0)), add(chain_block(chain_call(p, nonce, j), 1));
  { // 32 bytes of a call of 64 are not the hash of 32 bytes on the same string: the length read is in the parameter block
    std::vector<u8> whole = in;
    put(whole, 9), put(whole, 0);
    add(plain32(162, 0, 5, &s, whole));
  }
  check(seen.size() == made, "hashes that must differ do");
}

static void work() {
  check(work_test(~(u64)0, 0) && work_test(0, 64) && !work_test(1, 1) && work_test(2, 1) && !work_test((u64)1 << 63, 64) && work_test((u64)1 << 63, 63) &&
            work_test(0xFF00, 8) && !work_test(0xFF00, 9),
        "the work test reads the low bits");
}

// Runs f in a child process with stderr discarded; returns true if SIGABRT ended the child.
template <class Call> static bool aborts(Call f) {
  fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    if (!freopen("/dev/null", "w", stderr)) _exit(3);
    f();
    _exit(0);
  }
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
static void out_of_bounds() {
  const Digest s = some_digest();
  check(aborts([&] { Hasher h(Domain::STATEMENT, (u64)1 << 63); }), "an index of 2^63 is refused");
  check(aborts([&] { Tag t(Domain::STATEMENT, 1u << 16); }) && aborts([&] { Tag t(Domain::CHAIN_QUERY, (u64)1 << 48); }), "a tree's number of 2^16 and a chain's label of 2^48 are refused");
  check(aborts([&] { Tag t((Domain)159); }) && aborts([&] { Tag t((Domain)161); }) && aborts([&] { Tag t((Domain)163); }) && aborts([&] { Tag t((Domain)165, 0, s); }) &&
            aborts([&] { Tag t((Domain)167, 0, s); }) && aborts([&] { Tag t((Domain)170); }) && aborts([&] { Tag t((Domain)175, 0, s); }) && aborts([&] { Tag t(Domain::RESERVED); }) &&
            aborts([&] { Tag t((Domain)180); }) && aborts([&] { Tag t((Domain)(164 + 256), 0, s); }),
        "a number that is not given out is refused");
  check(aborts([&] { Tag t(Domain::LEAF_MAIN, 1); }) && aborts([&] { Tag t(Domain::LEAF_BINDING, 1); }) && aborts([&] { Tag t(Domain::LEAF_STORED_MAIN, 1); }) &&
            aborts([&] { Tag t(Domain::TREE_NODE); }) && aborts([&] { Tag t(Domain::BINDING_NODE); }),
        "a tree's hashes have no tag without their start");
  check(aborts([&] { Tag t(Domain::TREE_NODE, 1, s); }) && aborts([&] { Tag t(Domain::BINDING_NODE, 1, s); }) && aborts([&] { node(Tag(Domain::TREE_NODE, 0, s), 0, s, s); }) &&
            aborts([&] { node(Tag(Domain::BINDING_NODE, 0, s), 0, s, s); }) && aborts([&] { vertex(Tag(Domain::LEAF_MAIN, 0, s, 8, 40), s, s); }),
        "a vertex's hash takes no tree's number, is not made by node(), and has the vertices' number");
  check(aborts([&] { Tag t(Domain::LEAF_MAIN, 1, Digest{}, 1, 40); }) && aborts([&] { Tag t(Domain::TREE_NODE, 0, Digest{}); }), "a start of zeros is refused");
  check(aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s); }) && aborts([&] { Tag t(Domain::LEAF_BINDING, 0, s); }) && aborts([&] { Tag t(Domain::LEAF_STORED_MAIN, 1, s); }) &&
            aborts([&] { Tag t(Domain::TREE_NODE, 0, s, 1, 40); }) && aborts([&] { Tag t(Domain::STATEMENT, 0, s, 1, 40); }),
        "a leaf's tag is made with the leaves' shape, and no other tag is");
  check(aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s, (u64)1 << 24, 40); }) && aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s, 1, 32); }) && aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s, 1, 44); }) &&
            aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s, 1, 72); }) && aborts([&] { Tag t(Domain::LEAF_MAIN, 1, s, 1, 128); }),
        "2^24 symbols, and a salt below 40 bytes, above 64 or not of whole words, are refused");
  check(aborts([&] { Tag t(Domain::STATEMENT, 0, s); }) && aborts([&] { Tag t(Domain::CHAIN_QUERY, 0, s); }), "only a tree's tags hold a start");
  check(aborts([&] { Hasher h(Domain::TEST_COINS, 0); }) && aborts([&] { Hasher h(Domain::CHAIN_QUERY, 0); }), "a hash that has a start of its own is not made without it");
  check(aborts([&] { Hasher h(Domain::STATEMENT, 0, s); }) && aborts([&] { Hasher h(Tag(Domain::LEAF_MAIN, 1, s, 1, 40), 0, s); }), "a start is refused where none belongs");
  check(aborts([&] { node(Domain::TEST_COINS, 0, s, s); }), "two digests are not hashed under a number with a start of its own");
  check(aborts([&] {
          Hasher h(Domain::STATEMENT, 0);
          h.finish(), h.finish();
        }),
        "a Hasher is finished once");
  check(aborts([&] {
          Hasher h(Domain::STATEMENT, 0);
          h.finish();
          h.push_u64(1);
        }) &&
            aborts([&] {
              const u8 b = 1;
              Hasher h(Domain::STATEMENT, 0);
              h.push(&b, 1);
              h.finish();
              h.push(&b, 1);
            }),
        "a finished Hasher takes no more input");
  check(aborts([&] {
          ChainAbsorb a(s, 1);
          a.prefix();
        }) &&
            aborts([&] {
              ChainAbsorb a(s, 1);
              a.push_u32(1);
              a.prefix();
            }),
        "a round's message has at least a word and whole words");
  check(aborts([&] {
          ChainAbsorb a(s, 1);
          a.push_u64(1);
          chain_call(a.prefix(), 0, (u64)1 << 48);
        }),
        "a call's number of 2^48 is refused");
  const F x = F::zero();
  const u8 salt[136] = {0};
  const Tag tag(Domain::LEAF_MAIN, 1, s, 1, 40);
  check(!aborts([&] { leaf(tag, 1, &x, 1, salt, 40); }) && aborts([&] { leaf(tag, 1, &x, 1, salt, 48); }) && aborts([&] { leaf(tag, 1, &x, 0, salt, 40); }) &&
            aborts([&] { leaf(tag, 1, s, salt, 40); }) && aborts([&] { leaf(Tag(Domain::TREE_NODE, 0, s), 1, &x, 1, salt, 40); }),
        "a leaf of another shape than its tag's is refused");
  check(aborts([&] {
          Hasher h(tag, 1);
          h.push(salt, 40);
          h.finish();
        }) &&
            !aborts([&] {
              Hasher h(tag, 1);
              h.push(salt, 48);
              h.finish();
            }),
        "a Hasher under a leaf's tag must be fed what the tag says");
  check(aborts([&] {
          ChainAbsorb a(s, 1);
          a.push_u64((u64)1 << 16);
          a.prefix();
        }) &&
            aborts([&] {
              ChainAbsorb a(s, 1);
              a.push_u64((u64)1 << 63);
              a.prefix();
            }) &&
            aborts([&] {
              const u64 cells[2] = {5, (u64)1 << 16};
              chain_call_cells(s, 1, cells, 0, 0);
            }),
        "a message whose last word does not leave the call's number its bits is refused");
  check(aborts([&] { work_test(0, 65); }), "a work test of 65 bits is refused");
}

int main() {
  pieces(), two_digests_and_leaves(), rounds(), known_answers(), differ(), work(), out_of_bounds();
  printf("%sthe hash of hash.h (%s), %d checks: %s\n", bad ? "\n" : "", HashFn::NAME, checks, bad ? "SOME FAILED" : "all passed");
  return bad != 0;
}
