// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The committer program cp_commit creates every file of its public and secret key directories
// (`ivk`, `isk`) with write_a_new_file(), through making::write_new() in hf_commit_making.h, so
// that these rules are enforced in one place: an existing path is rejected. A symbolic link as the
// path's last component is not followed; links in earlier components are. Only the owner may read
// or write the file. On success the bytes are on disk: fsync() on the file, not on its directory,
// which write_new() syncs. A build of cp_commit's own main() with the test hook below stops at an
// #error (hf_commit.cu): the hook is only for test programs. Host code only.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace hc {
// Overwrites the open file with zeros, from the descriptor's offset, which must be 0, to the file's
// end, and syncs it. The size is read again before each write(): another thread may still be adding
// to the file. What this cannot reach: on flash, on a copy-on-write or log-structured file system,
// or where snapshots are taken, a changed block is written elsewhere and the old one can live on.
// Returns false with errno set. System calls only, all of which a signal handler may make.
static inline bool zeros_over(int fd) {
  static const char zeros[1 << 16] = {};
  struct stat is;
  for (off_t done = 0;;) {
    if (fstat(fd, &is)) return false;
    if (done >= is.st_size) return !done || !fsync(fd);
    const ssize_t w = write(fd, zeros, (size_t)std::min<off_t>(is.st_size - done, (off_t)sizeof zeros));
    if (!w) errno = EIO;
    if (w > 0)
      done += w;
    else if (!w || errno != EINTR)
      return false;
  }
}
// `path` is looked up from the open directory `directory`. opened: if not null, it is called with
// the new file's descriptor before anything is written, and the call fails if it returns false.
// Returns false on failure. If this call created the file, it has then overwritten what it wrote
// (zeros_over()) and tried to remove the file, again relative to `directory`. The file stays if it
// could not be overwritten, or if only close() failed. errno is then that of the last failed call
// before the overwriting, or close()'s, if any; a write() that returns 0 leaves it unchanged.
static inline bool write_a_new_file(int directory, const std::string &path, const void *bytes, size_t n, bool (*opened)(int) = nullptr) {
  const int fd = openat(directory, path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  bool good = fd >= 0 && (!opened || opened(fd));
  // Test hook: starts every file with one 128-byte line that begins NO PROOF, which marks it as a
  // test's; failing that, it clears `good` and the call fails. The verifier cp_verify rejects a
  // `key.bin` (the commitment's public key file, in `ivk`) that starts with that line, because its
  // first four bytes are not the key file's tag (rejection name `cp-key-head`; cpKeyHead() in
  // verifier/src/vf_input_files.h).
#ifdef TEST_HOOKS
  test_hooks::write_this_builds_file_head(fd, path, good); // a test program's; may abandon the file
#endif
  for (const char *at = (const char *)bytes; good && n;) {
    const ssize_t w = write(fd, at, std::min<size_t>(n, (size_t)1 << 30));
    if (w <= 0)
      good = false;
    else
      at += w, n -= (size_t)w;
  }
  if (good && fsync(fd)) good = false;
  bool overwritten = false;
  if (!good && fd >= 0) {
    const int why = errno;
    overwritten = !lseek(fd, 0, SEEK_SET) && zeros_over(fd);
    errno = why;
  }
  if (fd >= 0 && close(fd)) good = false;
  if (overwritten) {
    const int why = errno;
    unlinkat(directory, path.c_str(), 0);
    errno = why;
  }
  return good;
}
// The same, with the directory taken from `path` and opened first: the removal after a failure
// then acts in the directory where the file was created, even if `path` leads elsewhere by then.
static inline bool write_a_new_file(const std::string &path, const void *bytes, size_t n) {
  const size_t name = path.rfind('/') + 1; // 0 if the path has no '/'
  const int directory = open(name ? path.substr(0, name).c_str() : ".", O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) return false;
  const bool good = write_a_new_file(directory, path.substr(name), bytes, n);
  const int why = errno;
  close(directory);
  errno = why;
  return good;
}
} // namespace hc
