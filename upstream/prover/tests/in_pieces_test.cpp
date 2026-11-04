// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A CPU test of prover/weights_commitment/in_pieces.h, which cuts a range of bytes into pieces that
// several threads share. in_pieces() must put every byte in exactly one piece, report a failing
// piece and stop, and finish the work even if an allocation fails while the helper threads are
// started. read_in_pieces() must read a file unchanged and report one that ends too early,
// ends_at() must find where a file ends, and zero_in_pieces() must zero exactly the given range.
// Build, from prover/: ./build.sh in_pieces_test. Prints one line per check. Exit code 0 if all
// pass, 1 if not, 2 if mkstemp() fails.
#include "weights_commitment/in_pieces.h"
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <set>
#include <string>

// Replaces the global operator new, to inject one failure: while `refusals_in` is not negative,
// each call decrements it, and the call that finds 0 throws std::bad_alloc.
static std::atomic<long> refusals_in{-1};
void *operator new(size_t n) {
  if (refusals_in >= 0 && refusals_in-- == 0) throw std::bad_alloc();
  if (void *p = malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
  free(p);
}
void operator delete(void *p, size_t) noexcept {
  free(p);
}

static int failed = 0;
static void check(bool good, const std::string &what) {
  printf("%s %s\n", good ? "as wanted" : "DIFFERS  ", what.c_str());
  failed += !good;
}
// True if in_pieces() succeeds, on at most `helpers` + 1 threads, and cuts [0, n) into pieces of
// `length` bytes (only the last may be shorter), each run once.
static bool tiles(size_t n, int helpers, size_t length, size_t *threads_seen = nullptr) {
  std::mutex lock;
  std::vector<std::pair<size_t, size_t>> got;
  std::set<std::thread::id> who;
  const bool good = in_pieces(
      n,
      [&](size_t first, size_t bytes) {
        std::lock_guard<std::mutex> held(lock);
        got.push_back({first, bytes});
        who.insert(std::this_thread::get_id());
        return true;
      },
      helpers, length);
  if (threads_seen) *threads_seen = who.size();
  std::sort(got.begin(), got.end());
  size_t at = 0;
  for (auto &g : got) {
    if (g.first != at || g.second == 0 || g.second > length || (g.second < length && g.first + g.second != n))
      return false;
    at += g.second;
  }
  return good && at == n && who.size() <= (size_t)helpers + 1;
}

int main() {
  for (size_t length : {(size_t)1, (size_t)7, (size_t)4096})
    for (int helpers : {0, 1, 3, 15})
      for (size_t n : {(size_t)0, (size_t)1, length - 1, length, length + 1, 5 * length + 3, 64 * length})
        if (!tiles(n, helpers, length))
          check(false, "pieces of " + std::to_string(length) + ", " + std::to_string(helpers) + " helpers, " +
                           std::to_string(n) + " bytes");
  check(!failed, "every byte in exactly one piece: 3 lengths x 4 numbers of helpers x 7 lengths of the whole");
  size_t seen = 0;
  check(tiles(4096, 15, 4096, &seen) && seen == 1, "one piece: only the calling thread works on it");
  check(tiles(1000, 0, 10, &seen) && seen == 1, "no helpers: the calling thread does all 100 pieces by itself");
  check(tiles((size_t)3 * PIECE_BYTES + 5, 15, PIECE_BYTES), "the piece length that the library uses");

  { // a failing piece is reported, and the thread that ran it begins no further piece
    std::atomic<int> calls{0};
    const bool good = in_pieces(1000, [&](size_t first, size_t) { return calls++, first != 0; }, 0, 10);
    check(!good && calls == 1, "a false from the first piece, no helpers: false, and no second piece");
    // The first piece fails at once and each other piece takes a millisecond. All 10,000 would take
    // 16 threads more than half a second, so this check fails wrongly only if the thread that takes
    // the first piece is delayed that long before it records the failure.
    std::atomic<int> many{0};
    check(!in_pieces(
              100000, [&](size_t first, size_t) { return many++, first ? usleep(1000), true : false; }, 15, 10) &&
              many < 10000,
          "a false from the first piece, 15 helpers: false, and the work stops");
  }

  { // At most one allocation fails: call k (from 0) of operator new while the helpers are started, when the vector
    // of threads grows and a thread allocates its state. k runs past the number of such calls; from
    // there on none fails. No exception may leave in_pieces() and no thread may be left unjoined,
    // which would end this program; the threads that exist do the work. The piece's function
    // allocates nothing. Other failures to start a thread (std::system_error) are not injected.
    bool all = true;
    static std::atomic<int> given[160];
    for (long k = 0; k < 40; k++) {
      for (auto &g : given) g = 0;
      bool good = false, came_out = false;
      refusals_in = k;
      try {
        good = in_pieces(
            1600, [](size_t first, size_t bytes) { return usleep(200), given[first / 10] += (int)bytes, true; }, 15,
            10);
      } catch (...) {
        came_out = true;
      }
      refusals_in = -1;
      all = all && good && !came_out &&
            std::all_of(given, given + 160, [](const std::atomic<int> &g) { return g == 10; });
    }
    check(all,
          "an allocation fails at the k-th call while 15 helpers are started, k = 0 to 39: true, and every piece done once");
  }

  const char *tmp = getenv("TMPDIR"); // where the caller keeps temporary files, or /tmp
  std::string name = std::string(tmp && *tmp ? tmp : "/tmp") + "/in_pieces_test.XXXXXX";
  const int file = mkstemp(&name[0]);
  if (file < 0) return printf("in_pieces_test: no file for the test's bytes\n"), 2;
  unlink(name.c_str());
  const size_t n = 100003; // a prime, so the file's length is no multiple of a piece's
  std::vector<char> bytes(n), back(n + 64, 'x');
  for (size_t i = 0; i < n; i++) bytes[i] = (char)(i * 2654435761u >> 13);
  check(file >= 0 && write(file, bytes.data(), n) == (ssize_t)n, "a file of 100,003 bytes is made");
  for (int helpers : {0, 15}) {
    const std::string with = ", " + std::to_string(helpers) + " helpers";
    std::fill(back.begin(), back.end(), 'x');
    check(read_in_pieces(file, 0, n, back.data(), helpers, 4096) && !memcmp(back.data(), bytes.data(), n) &&
              back[n] == 'x',
          "the whole file, in pieces of 4096: its bytes are read, and no byte after them is written" + with);
    std::fill(back.begin(), back.end(), 'x');
    check(read_in_pieces(file, 64, n - 64, back.data(), helpers, 4096) &&
              !memcmp(back.data(), bytes.data() + 64, n - 64) && back[n - 64] == 'x',
          "from byte 64 to the end, as after a header" + with);
    check(!read_in_pieces(file, 0, n + 1, back.data(), helpers, 4096), "one byte more than the file has: false" + with);
    check(!read_in_pieces(file, 64, n, back.data(), helpers, 4096), "the file's length from byte 64: false" + with);
    check(read_in_pieces(file, n, 0, back.data(), helpers, 4096), "no byte at the file's end: true" + with);
  }
  check(!read_in_pieces(-1, 0, 10, back.data()), "a descriptor that is not open: false");
  check(ends_at(file, n), "the file ends at its length");
  check(!ends_at(file, n + 1), "it does not end after its length");
  check(!ends_at(file, n - 1) && !ends_at(file, 0), "it does not end at its last byte, nor at its first");
  check(!ends_at(-1, 0), "a descriptor that is not open does not end anywhere");
  check(read_in_pieces(file, 0, n, back.data()) && !ftruncate(file, (off_t)(n - 1)) && !ends_at(file, n) &&
            ends_at(file, n - 1),
        "read whole and then cut by one byte, it no longer ends where it did, but one byte earlier");

  std::vector<char> dirty(100003, 'x');
  zero_in_pieces(dirty.data() + 1, dirty.size() - 2);
  check(dirty.front() == 'x' && dirty.back() == 'x' &&
            std::all_of(dirty.begin() + 1, dirty.end() - 1, [](char c) { return c == 0; }),
        "zeros over the given range, and no byte outside it is changed");
  printf("in_pieces_test: %d checks differ\n", failed);
  return failed != 0;
}
