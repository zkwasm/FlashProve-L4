// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Known-answer tests of the hash modes of blocks/hash/older.h. Run from blocks/.
//  1. BLAKE2b-512 of "abc" (RFC 7693, Appendix A) through the format's call, and the bare call printed in verifier/HASHES.md.
//  2. verifier/tests/hash_vectors.txt (made in Python): 44 list hashes H as states, 13 tree hashes.
//  3. prover/tests/longrow/tables/leaf_vectors.txt (made in Python): 8 leaves, with the state after
//     every slab, and 5 nodes of a request's row commitments. They follow the weight table's rules,
//     but with hash numbers 103 to 105, plus 2^16 times the tree's number, and a one-word index.
//     Older::leaf() writes two words: the same only because every index in the file is below 2^32.
//  4. tests/s8_known_answers.tsv (made by tests/s8_model.py), digests of 8 words only: the list
//     (digest, nonce) under 127 and under 3 on 9 lines, and the 11 leaves that are not folded.
//  5. blocks/hash/tests/table_tree_vectors.txt: leaves and nodes of the weight table's tree under
//     these hash modes, computed by host code that is not in this repository.
//  6. Calls with a node of level 0, a node's place of 2^32 or hash number 160 or 166 must abort.
#include "hash/older.h"
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>
using namespace blk;

static int bad = 0, checks = 0;
static void check(bool ok, const std::string &what) {
  checks++;
  if (!ok) bad++, printf("\nFAILED: %s", what.c_str());
}
static std::string words_hex(const u32 *w, size_t n) {
  std::string s;
  char b[9];
  for (size_t i = 0; i < n; i++) snprintf(b, sizeof b, "%08x", w[i]), s += b;
  return s;
}
static std::string bytes_hex(const u32 *w, size_t n) {
  std::string s;
  char b[9];
  for (size_t i = 0; i < n; i++)
    snprintf(b, sizeof b, "%02x%02x%02x%02x", w[i] & 255, w[i] >> 8 & 255, w[i] >> 16 & 255, w[i] >> 24), s += b;
  return s;
}
static Digest digest_of_words(const std::string &s) {
  Digest d;
  for (int i = 0; i < 8; i++) d.w[i] = (u32)strtoul(s.substr(8 * i, 8).c_str(), nullptr, 16);
  return d;
}
static std::vector<u64> cells(const std::string &s) {
  std::vector<u64> c;
  for (size_t i = 0; i + 16 <= s.size(); i += 16) c.push_back(strtoull(s.substr(i, 16).c_str(), nullptr, 16));
  return c;
}
static std::string joined(std::istringstream &s, int n) { // n words separated by blanks, as one string
  std::string all, w;
  for (int i = 0; i < n && s >> w; i++) all += w;
  return all;
}

static void single_calls() {
  u32 s[16];
  const u32 abc[32] = {0x00636261u};
  Blake2bF::first(s);
  Blake2bF::compress(s, abc, Tweak{3, 0, 0, 0, true});
  check(bytes_hex(s, 16) == "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
                            "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
        "BLAKE2b-512 of abc");
  u32 m[32];
  for (u32 i = 0; i < 32; i++) m[i] = i;
  Blake2bF::first(s);
  Older::bare(s, m, 1, 2, 3);
  check(words_hex(s, 16) == "f5727365c6b1178ebb489e21a739d5edde6eb7157ce3c623ac9f86185c77d0d7"
                            "24e40b1b5c884aceaea66bf5a7d65570f06c1b009e7a6b62ada026b4cbb7ec9d",
        "the bare call of verifier/HASHES.md");
}

// The test list of L words: word k is ((k + 1) * 0x89ABCDEF) mod 2^32.
static std::vector<u32> list_of(size_t L) {
  std::vector<u32> w(L);
  for (size_t k = 0; k < L; k++) w[k] = (u32)(k + 1) * 0x89ABCDEFu;
  return w;
}
static void lists() {
  std::ifstream in("../verifier/tests/hash_vectors.txt");
  int states = 0, trees = 0;
  for (std::string kind; in >> kind;) {
    if (kind == "H") {
      u32 num;
      size_t L;
      std::string want;
      in >> num >> L >> want;
      const std::vector<u32> w = list_of(L);
      u32 s[16];
      Older::list_state(s, num, w.data(), L);
      check(words_hex(s, 16) == want && words_hex(Older::list_digest(num, w.data(), L).w, 8) == want.substr(0, 64), "H of a list");
      states++;
    } else if (kind == "TH") {
      size_t L;
      std::string want;
      in >> L >> want;
      const std::vector<u32> w = list_of(L);
      check(words_hex(Older::tree_hash(w.data(), L).w, 8) == want, "the tree hash of a list");
      trees++;
    }
  }
  check(states == 44 && trees == 13, "hash_vectors.txt has 44 states and 13 tree hashes");
}

static void row_commitments() {
  std::ifstream in("../prover/tests/longrow/tables/leaf_vectors.txt");
  int leaves = 0, nodes = 0;
  for (std::string l; std::getline(in, l);) {
    std::istringstream s(l);
    std::string kind;
    s >> kind;
    std::map<std::string, std::string> f;
    for (std::string t; s >> t;)
      if (t.find('=') != std::string::npos) f[t.substr(0, t.find('='))] = t.substr(t.find('=') + 1);
    if (kind == "LEAF") {
      const u32 R = (u32)stoul(f["R"]), real = (u32)stoul(f["Rr"]), ln = (u32)stoul(f["ln"]), NUM = (u32)stoul(f["NUM"]);
      const u64 I = stoull(f["I"]);
      std::vector<u64> sym = cells(f["sym"]);
      const std::vector<u64> salt = cells(f["salt"]);
      check(sym.size() == real && salt.size() == 5, "a LEAF line has its cells");
      sym.resize(R, 0);
      std::string states;
      for (u32 b = 1; b <= Older::slabs(R); b++) {
        u32 S[16];
        Older::chain(S, sym.data(), R < 16 * b ? R : 16 * b, 103 + (NUM << 16));
        states += (b > 1 ? "," : "") + words_hex(S, 16);
      }
      check(states == f["states"], "the states of a leaf's chain");
      check(words_hex(Older::leaf(sym.data(), R, salt.data(), ln, NUM, I, 103 + (NUM << 16), 104 + (NUM << 16)).w, 8) == f["leaf"],
            "a leaf");
      leaves++;
    } else if (kind == "NODE") {
      check(words_hex(Older::node(digest_of_words(f["left"]), digest_of_words(f["right"]), (u32)stoul(f["level"]),
                                  stoull(f["place"]), 105 + ((u32)stoul(f["NUM"]) << 16)).w, 8) == f["node"], "a node");
      nodes++;
    }
  }
  check(leaves == 8 && nodes == 5, "leaf_vectors.txt has 8 leaves and 5 nodes");
}

static void lists_of_nonces_and_plain_leaves() {
  std::ifstream in("../tests/s8_known_answers.tsv");
  const u64 p = 0x1FFFFFFFFFFFFFFFull;
  int nonces = 0, leaves = 0, folded_leaves = 0;
  for (std::string l; std::getline(in, l);) {
    std::istringstream s(l);
    std::string kind;
    int DW;
    if (!(s >> kind >> DW) || DW != 8) continue;
    if (kind == "work" || kind == "nonce") { // d = H(127; digest, nonce), s1 = H(3; digest, nonce) as a state
      u64 g, a, b;
      s >> g >> a >> b;
      const u64 nonce = kind == "work" ? a : a | b << 32;
      u32 list[10], state[16];
      for (u32 i = 0; i < 8; i++) list[i] = 0x01010101u * (i + 1);
      list[8] = (u32)nonce, list[9] = (u32)(nonce >> 32);
      check(words_hex(Older::list_digest(127, list, 10).w, 8) == joined(s, 8), "the list of a digest and a nonce under 127");
      Older::list_state(state, 3, list, 10);
      check(words_hex(state, 16) == joined(s, 16), "the same list under 3, as a state");
      nonces++;
    } else if (kind == "leaf") {
      u32 lsalt, real, R, folded, calls;
      s >> lsalt >> real >> R >> folded >> calls;
      if (folded) {
        folded_leaves++; // a form of a request's leaves that the weight table does not have
        continue;
      }
      std::vector<u64> sym(R, 0), salt(5);
      for (u32 r = 0; r < real; r++) sym[r] = (u64)((u128)(r + 1) * 0x0123456789ABCDFull % p);
      for (u32 j = 0; j < 5; j++) salt[j] = (u64)((u128)(j + 1) * 0x0FEDCBA98765432ull % p);
      check(calls == Older::slabs(R) + 1, "the calls of a leaf");
      check(words_hex(Older::leaf(sym.data(), R, salt.data(), 20, 7, 5, 103 + (7u << 16), 104 + (7u << 16)).w, 8) == joined(s, 8), "a leaf");
      leaves++;
    }
  }
  check(nonces == 9 && leaves + folded_leaves == 16 && folded_leaves == 5, "s8_known_answers.tsv has 9 and 16 lines for digests of 8 words");
}

static void weight_table() {
  std::ifstream in("hash/tests/table_tree_vectors.txt");
  const u64 p = 0x1FFFFFFFFFFFFFFFull;
  u64 salt[5], column[200];
  for (u32 j = 0; j < 5; j++) salt[j] = (u64)((u128)(j + 1) * 0x0FEDCBA98765432ull % p);
  for (u32 r = 0; r < 200; r++) column[r] = (u64)((u128)(r + 1) * 0x0123456789ABCDFull % p);
  Digest left, right;
  for (u32 i = 0; i < 8; i++) left.w[i] = 0x01010101u * (i + 1), right.w[i] = 0x10101010u + i;
  int leaves = 0, nodes = 0;
  for (std::string l; std::getline(in, l);) {
    std::istringstream s(l);
    std::string kind, want;
    u32 a;
    u64 b, c;
    s >> kind;
    if (kind == "LEAF" && s >> a >> b >> c >> want)
      check(words_hex(Older::table_leaf(column, a, salt, (u32)b, 16384, c).w, 8) == want, "a leaf of the weight table"), leaves++;
    if (kind == "NODE" && s >> a >> b >> want)
      check(words_hex(Older::table_node(left, right, a, b).w, 8) == want, "a node of the weight table"), nodes++;
  }
  check(leaves == 8 && nodes == 5, "table_tree_vectors.txt has 8 leaves and 5 nodes");
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
static void bounds() {
  const Digest d{};
  check(aborts([&] { Older::table_node(d, d, 0, 1); }), "a node of level 0 is refused");
  check(aborts([&] { Older::node(d, d, 1, 1, 166); }) && aborts([&] { Older::list_digest(160 + (1u << 16), d.w, 8); }), "hash numbers from 160 on are refused (tried: 166, and 160 + 2^16)");
  check(aborts([&] { Older::table_node(d, d, 1, 1ull << 32); }), "a node's place of 2^32 is refused");
  check(!aborts([&] { Older::table_node(d, d, 1, (1ull << 32) - 1); }), "a node's place of 2^32 - 1 is taken");
}

int main() {
  bounds();
  single_calls();
  lists();
  row_commitments();
  lists_of_nonces_and_plain_leaves();
  weight_table();
  printf("%sthe hash modes of older.h, %d checks: %s\n", bad ? "\n" : "", checks, bad ? "SOME FAILED" : "all passed");
  return bad != 0;
}
