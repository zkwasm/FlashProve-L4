// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Splits work on a long byte range into pieces that several host threads share. in_pieces() is the
// general form; read_in_pieces() reads part of a file with it, and zero_in_pieces() zeroes memory.
// ends_at() checks that a file ends where the caller expects. No function here prints or exits:
// failure is reported by returning false. Host code only (no CUDA). Tested by
// prover/tests/in_pieces_test.cpp.
#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// The default length of a piece: 64 MiB.
static constexpr size_t PIECE_BYTES = (size_t)64 << 20;

// Calls piece(first, bytes) once for each piece of `length` bytes of [0, n) (the last may be
// shorter), in no particular order, on the calling thread and up to `helpers` more threads. Returns
// after the last call has returned: true if every call returned true. A call that returns false
// clears `good`; each thread then begins at most one more call, and the rest are skipped. `length`
// must not be 0 (not checked). `piece` runs on several threads at once and must not print, throw or
// exit: it reports an error only by its return value, so only the calling thread prints or exits.
// If a helper thread cannot be started, the threads that exist do all the work (the test named at
// the top makes a memory allocation fail while the helpers are started; it never makes the
// operating system fail to create a thread).
// Public source: fewer threads than asked for if the input is too small: XZ Utils,
// xz(1) manual page (dated 2025-03-08), option -T threads, --threads=threads,
// https://tukaani.org/xz/man/xz.1.html (there: for compression in independent blocks).
// Public source: the first error cancels a group of tasks: The Go Authors, documentation of
// package golang.org/x/sync/errgroup, func (*Group) Go,
// https://pkg.go.dev/golang.org/x/sync/errgroup.
// Public source: fewer threads if resources are lacking, in Sun's OpenMP: Oracle's archive of
// Sun's compiler documentation, document 819-0501, chapter 4, Implementation-Defined Behaviors,
// https://docs.oracle.com/cd/E19059-01/stud.10/819-0501/4_implement.html.
template <class Piece> static bool in_pieces(size_t n, Piece piece, int helpers = 15, size_t length = PIECE_BYTES) {
  std::atomic<size_t> next{0};
  std::atomic<bool> good{true};
  auto work = [&] {
    for (size_t first; good && (first = next.fetch_add(length)) < n;)
      if (!piece(first, std::min(length, n - first))) good = false;
  };
  std::vector<std::thread> threads;
  try { // a helper is started only if a piece is left for it besides the calling thread's
    for (int i = 0; i < helpers && (size_t)(i + 1) * length < n; i++) threads.emplace_back(work);
  } catch (...) { // no further thread, or no memory for one, is available: the vector holds exactly those that run
  }
  work();
  for (std::thread &t : threads) t.join();
  return good;
}

// Reads the bytes [from, from + n) of the open file directly into `to` with pread(). Returns false
// if a read fails or the file ends before from + n.
static bool read_in_pieces(int file, size_t from, size_t n, char *to, int helpers = 15, size_t length = PIECE_BYTES) {
  return in_pieces(
      n,
      [=](size_t first, size_t bytes) {
        for (size_t done = 0; done < bytes;) {
          const ssize_t r = pread(file, to + first + done, bytes - done, (off_t)(from + first + done));
          if (r == 0 || (r < 0 && errno != EINTR)) return false; // 0: the file ends here, and errno is not set
          if (r > 0) done += (size_t)r;
        }
        return true;
      },
      helpers, length);
}

// Zeroes [to, to + n) with memset(), using the default `helpers` and `length` of in_pieces().
static void zero_in_pieces(char *to, size_t n) {
  in_pieces(n, [=](size_t first, size_t bytes) {
    memset(to + first, 0, bytes);
    return true;
  });
}

// Returns true if the file ends exactly at offset `place`. A one-byte read there must find the end
// of the file; the byte read (`behind`), if any, is wiped. The size that fstat() reports must equal
// `place`: only this detects a shorter file, for example one truncated after read_in_pieces() had
// read the bytes that are now missing. On a regular file the size would be enough, but the read
// also detects a read error. An interrupted read is repeated; any other failure of pread() or
// fstat() gives false.
static bool ends_at(int file, size_t place) {
  char behind = 0;
  ssize_t r;
  do r = pread(file, &behind, 1, (off_t)place);
  while (r < 0 && errno == EINTR);
  explicit_bzero(&behind, 1);
  struct stat is;
  return r == 0 && !fstat(file, &is) && (size_t)is.st_size == place;
}
