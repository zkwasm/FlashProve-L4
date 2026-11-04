// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The code that cp_commit and cp_prove share for the option -public-weights <file of 32 bytes>, for
// weights that are public anyway. cp_commit derives the commitment's seed from the file and puts no
// random byte from the operating system into its output files. So the same call on the same weights
// directory (model files, wsums.txt, fold.txt), configuration and 32-byte file gives the same key
// file (`ivk/key.bin`), leaf files (`ivk/leaf/<k>.bin`) and commitment ID, which anyone can compare
// with published ones. Calls on different machines have been compared for Maverick's 4-bit files alone: what the equal
// values show and do not show is in examples/maverick/README.md. The cost: anyone who has the 32-byte file can compute
// every mask, so such a commitment hides nothing. The key file, the leaf files and the ID bear no mark of that.
//
// No single mistake should create or serve such a commitment. For model files, cp_commit needs the
// option and, next to each, a marker file (prover/hash_commitment/public_weights_markers.h) with
// that file's SHA-256. With the option, `isk.txt` ends with the line "seed public". cp_prove needs
// that line, the option, and the same 32-byte file (prover/prover/isk.cuh). The option switches off
// one rejection: cp_prove accepts the 32 zero bytes that cp_commit then writes to each leaf table's
// `isk/leaf/<k>/fresh.bin`. With neither the option nor the line "seed public", nothing in either
// program changes. The option is not a test option: only a
// production build (`serving`, the default of prover/build.sh) has it; every other build's cp_prove
// rejects a commitment with the line "seed public".
#pragma once
#include "common/coin_chacha.h" // cp_key()
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace public_weights {
// The seed is cp_key(the file's 32 bytes, "CPPW", 0, 0, 0). cp_commit accepts any 32-byte file. If
// the bytes were the seed, a call under the same configuration that named another commitment's
// `seed.bin` would repeat the masks of that commitment's anchor (its first weight table). On the
// same weights it would rebuild the anchor and open more of its columns without raising that
// commitment's column count (`count.bin`). All of a commitment's keys come from its seed under the
// first word "CPSD" (blk::seed_child() in blocks/coins/coins.h), so the derived seed shares none.
// Callers must zero both arrays.
constexpr uint32_t SEED_WORD = 0x57505043u; // "CPPW"
static inline void derive_the_seed(uint8_t seed[32], const uint8_t file_bytes[32]) {
  cp_key(seed, file_bytes, SEED_WORD, 0, 0, 0);
}
// Reads the option's file and derives the seed. Returns false, and leaves `seed` unchanged, unless
// the file is a readable regular file of exactly 32 bytes. At most 33 bytes are read, and they are
// zeroed before the return, because a file that is named by mistake may be a secret one.
// A symbolic link is followed; owner and permissions are not examined.
static inline bool seed_of_the_file(const char *path, uint8_t seed[32]) {
  uint8_t bytes[33] = {0};
  const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK); // O_NONBLOCK: does not block on a FIFO
  struct stat is;
  bool good = fd >= 0 && !fstat(fd, &is) && S_ISREG(is.st_mode);
  size_t have = 0;
  while (good && have < sizeof bytes) {
    const ssize_t got = read(fd, bytes + have, sizeof bytes - have);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) {
      good = got == 0;
      break;
    }
    have += (size_t)got;
  }
  if (fd >= 0) close(fd);
  good = good && have == 32;
  if (good) derive_the_seed(seed, bytes);
  explicit_bzero(bytes, sizeof bytes);
  return good;
}
} // namespace public_weights
