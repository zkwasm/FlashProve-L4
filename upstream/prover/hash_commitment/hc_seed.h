// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The commitment's secret 32-byte seed, `isk/seed.bin` (`isk` is the secret key directory): how it
// is drawn, read, compared and zeroed. The committer program cp_commit writes it, and draws it
// unless -public-weights or (test builds) -test-seed gives it. Each
// weight table's making key (a ChaCha20 key, which generates its masks, salts and pads) and the keys
// of the committer's and prover's private randomness are derived from it
// (prover/common/coin_keys.h). The functions here do not print, write a file or exit. Tags in
// square brackets are printed error names.
//
// A making key must serve one table only: two tables under the same key differ by an unmasked
// encoding of the difference of their data. So the keys of the anchor (the first weight table) and
// of each leaf table (a further weight table of the same weights) are derived under different
// labels, and a leaf table's key also absorbs 32 bytes that are drawn each time the table is
// generated (coin_keys::copy_table_key()): a leaf table that is generated again has other masks.
// These bytes stay zero only under a test build's `-test-seed` or a production build's
// `-public-weights` (prover/common/public_weights.h), and then nothing is hidden.
//
// prover/hash_commitment/hf_seed.h re-exports all functions but seed_is() in namespace hf.
#pragma once
#include "hash_commitment/hc_chacha.cuh" // hc_seed32, hc_seed_from_bytes()
#include "hash_commitment/hc_format.h"
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace hc {
typedef ssize_t (*SeedSource)(void *bytes, size_t n); // returns how many bytes it gave, or a negative number
// The operating system's generator. It repeats getrandom() only after EINTR, which is not the
// silent second call that seed_draw() rules out.
static inline ssize_t seed_machine_source(void *bytes, size_t n) {
  ssize_t given;
  do given = ::getrandom(bytes, n, 0);
  while (given < 0 && errno == EINTR); // a signal interrupted the call before it gave a byte
  return given;
}
// Unless seed_draw() returns `TAKEN`, it leaves `seed` all zero. The fixed
// test seeds are the values SEED_TEST_n, n < HC_TEST_SEEDS, which hc_test_seed_word() (hc_format.h)
// defines: anyone who has the source code knows them.
enum class SeedEnd {
  TAKEN,
  NO_RANDOMNESS, // `source` did not return 32, the number of bytes requested
  IN_THE_SOURCE, // it gave 32 zero bytes, as a dead generator does, or a fixed test seed
  NO_NUMBER,     // a test's text is no n < HC_TEST_SEEDS in one to three decimal digits; nothing in this repository returns it
};
// The error's name. The caller prints it in parentheses, then the message from seed_says().
static inline const char *seed_stop(SeedEnd end) {
  return end == SeedEnd::NO_RANDOMNESS   ? "no-randomness"
         : end == SeedEnd::IN_THE_SOURCE ? HC_STOP_KEY_TEST
         : end == SeedEnd::NO_NUMBER     ? "usage"
                                         : "";
}
static inline const char *seed_says(SeedEnd end) {
  return end == SeedEnd::NO_RANDOMNESS   ? "the operating system's generator gave no seed"
         : end == SeedEnd::IN_THE_SOURCE ? "the generator gave 32 zero bytes or a seed that is written in the source"
#ifndef SERVING_BUILD // production builds omit the test variable's name, which no program reads; the fixed test seeds stay
         : end == SeedEnd::NO_NUMBER ? "H_HC_SEED is not the number of a seed of the tests, 0 to 255"
#endif
                                     : "";
}
static inline void seed_wipe(hc_seed32 &seed) {
  volatile uint32_t *v = seed.w;
  for (int i = 0; i < HC_SEED_WORDS; i++) v[i] = 0;
}
// In the source code of the three comparisons below, no branch, loop count or address depends on a
// word of a seed, and none exits early: their running time is meant not to depend on the seeds.
//
// True if `seed` is SEED_TEST_n or, for n < 0, all zero.
static inline bool seed_is(const hc_seed32 &seed, int n) {
  uint32_t joined = 0;
  for (int i = 0; i < HC_SEED_WORDS; i++) joined |= seed.w[i] ^ (n < 0 ? 0u : hc_test_seed_word(n, i));
  return !joined;
}
static inline bool seed_is_zero(const hc_seed32 &seed) {
  return seed_is(seed, -1);
}
static inline bool seed_is_of_the_tests(const hc_seed32 &seed) {
  bool found = false;
  for (int n = 0; n < HC_TEST_SEEDS; n++) found |= seed_is(seed, n);
  return found;
}

// Draws a new seed: 32 bytes from `source`, read as eight little-endian words, as RFC 8439 reads
// a key. cp_commit passes seed_machine_source(). `source` is called once, with no fallback: a
// silent second call would hide a faulty generator. 32 zero bytes and the fixed test seeds are
// rejected [hc-key-test]. `source` is a parameter, and this file needs no CUDA, so that a test on a
// CPU can pass faulty sources.
static inline SeedEnd seed_draw(SeedSource source, hc_seed32 &seed) {
  seed_wipe(seed);
  uint8_t bytes[HC_SEED_BYTES] = {};
  const bool gave = source(bytes, sizeof bytes) == (ssize_t)sizeof bytes;
  if (gave) hc_seed_from_bytes(bytes, seed);
  explicit_bzero(bytes, sizeof bytes);
  if (!gave) return SeedEnd::NO_RANDOMNESS;
  const bool known = seed_is_zero(seed) | seed_is_of_the_tests(seed);
  if (!known) return SeedEnd::TAKEN;
  seed_wipe(seed);
  return SeedEnd::IN_THE_SOURCE;
}
// Reads a stored seed [hc-key-file]: a regular file, not a symbolic link, of exactly 32 bytes, with
// one hard link and permissions for its owner only, owned by the effective user (checked because
// root can open anyone's file). The checks are made on the open file, not on the name, so the file
// that is checked is the file that is read; 33 bytes are requested, to reject a file that grew
// after fstat(). The bytes pass through no buffer of the C library, which would be freed without
// being zeroed. On failure returns false, with `seed` all zero.
//
// The value is not checked, and the file has no checksum. An all-zero seed (seed_is_zero()) or a
// fixed test seed is for the caller to reject. Any other wrong seed is caught later [hc-own-node]:
// when a table is opened, show_columns() (prover/hash_commitment/hf_tree.cuh) recomputes the Merkle
// node of the lowest stored level above each opened column, under the table's making key, and
// compares it with the stored node.
//
// cp_commit derives each making key a second time from the seed file (TableKey::derive(),
// prover/hash_commitment/hf_commit_keys.h), so its self-checks
// (prover/hash_commitment/hf_mask_own.cuh) catch a making key that was not derived from the seed in
// the file. The prover program cp_prove reads the file with a function of its own,
// Prover::read_the_seed() (prover/prover/isk.cuh), and derives the key once: there these
// self-checks miss such a key, and the comparison of nodes [hc-own-node] catches it.
static inline bool seed_from_file(const char *path, hc_seed32 &seed) {
  seed_wipe(seed);
  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK); // O_NONBLOCK: a fifo does not hold us up
  if (fd < 0) return false;
  struct stat is;
  uint8_t bytes[HC_SEED_BYTES + 1] = {};
  size_t have = 0;
  bool good = !fstat(fd, &is) && S_ISREG(is.st_mode) && is.st_uid == geteuid() && is.st_nlink == 1 && !(is.st_mode & 077) && is.st_size == (off_t)HC_SEED_BYTES;
  while (good && have < sizeof bytes) {
    const ssize_t got = read(fd, bytes + have, sizeof bytes - have);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) {
      good = got == 0;
      break;
    }
    have += (size_t)got;
  }
  close(fd);
  good = good && have == HC_SEED_BYTES;
  if (good) hc_seed_from_bytes(bytes, seed);
  explicit_bzero(bytes, sizeof bytes);
  return good;
}
} // namespace hc
