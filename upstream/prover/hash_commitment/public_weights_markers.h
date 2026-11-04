// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The marker files that cp_commit requires under -public-weights (prover/common/public_weights.h).
//
// With -public-weights nothing is hidden, and while a commitment is created the command line is the
// only record of that choice. So the operator must declare a model's weights public a second time,
// file by file: next to each model file <name> (`shard` in identifiers) there must be a marker file
// <name>.weights-are-public with that file's SHA-256, so that a marker left behind does not cover
// other weights that get the same name. A marker that is a symbolic link or has a second name (a
// hard link) is rejected (printed: refused), so a link to a checksum file elsewhere is no marker.
// Its owner and permissions must pass too: marker_owner_counts() has the rule and its gaps.
// Only a production build (`SERVING_BUILD`) has the option. cp_prove does not read the markers.
// The markers are checked twice: owner, permissions and form at the start of the run
// (check_the_markers()), and these again, with the digests, when the loader (`gguf::Model`) has
// opened the model files (check_the_digests()).
// Not detected, besides those gaps: a marker created for weights that must stay hidden; a model
// file modified in place after the loader parses its header and before the last tensor is read.
#pragma once
#include "hash_commitment/hf_commit_making.h" // refuse()
#include "weights_commitment/gguf_file.h"
#include "weights_commitment/sha256.h"
#include <chrono>

namespace hf {
namespace making {
// Whether a marker that belongs to `owner` counts. `caller` is the account that runs cp_commit,
// `file_owner` the owner of the model file that the marker declares public.
// A SHA-256 is no secret, and several accounts may be able to write to the directory. So a marker
// counts only if it belongs to `caller`, to `file_owner` or to root, and if neither its group nor
// others can write it (read_public_weights_marker() checks both). Another account that can create
// files next to weights that must stay hidden can create only markers of its own, which do not
// count. `file_owner` counts so that root, or a colleague, can use the markers of whoever provided
// the model files; that account has the weights anyway. There is one marker for each model file, so
// where several accounts use one directory, the model files' owner or root creates the markers,
// which count for all of them. The directory's own permissions are not examined: public weights
// are often in a shared directory. On a file system that keeps no permissions and shows each file
// as writable by all, every marker is rejected until a mount option masks that permission.
// Not detected:
// - A file that is renamed to the marker's name. Whoever can write to the directory can give that
//   name to any file that begins with the right 64 digits, belongs to one of the three accounts and
//   is in a directory of the same file system that they can also write, such as a checksum file
//   that the operator made (not if that file's directory has the sticky bit and is not theirs).
// - Owners that are not what they seem: a mount or a container that shows several accounts as one,
//   or a file system that lets an account give its files away.
// - What root or the model file's owner does.
// - A symbolic link that another account puts under the marker's name in the seconds between the
//   operator's look at it (ls -l) and the shell's redirection, which follows it. cp_commit and
//   cp_prove write no marker themselves.
static bool marker_owner_counts(uid_t owner, uid_t caller, uid_t file_owner) { return owner == caller || owner == file_owner || owner == 0; }
// Reads the SHA-256 from a model file's marker. `fd` is the opened marker, which is closed here, or
// is negative; `error` then says why, as an errno (ELOOP: a symbolic link; EMLINK: a second name;
// EISDIR: not a regular file). An opened marker is checked for the last two here too. `path` is the
// model file's path, shown escaped in the messages. For `caller` and `file_owner` see marker_owner_counts().
// A marker whose owner does not count, or that its group or others can write, is rejected. At most
// 65 bytes are read. They must be 64 lower-case hexadecimal digits, then a space, a newline or the
// end of the file, as sha256sum prints them; the file name after them is not compared. Returns ""
// and sets `declared`, or a message for refuse().
static std::string read_public_weights_marker(int fd, int error, const std::string &path, uid_t caller, uid_t file_owner, std::string &declared) {
  const std::string file = ::gguf::shown(path), marker = file + ".weights-are-public"; // a file name in the weights directory can hold any bytes
  const std::string but = "with -public-weights nothing is hidden, so each weight file must be declared public, but ";
  const std::string hidden = ". If the weights in " + file + " must stay hidden, omit -public-weights. ";
  const std::string write = "create " + marker + ", which must hold what sha256sum prints for " + file + ", with the first command in docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\", and run this command again. Do not create it with a shell's redirection, which follows a symbolic link and can leave a file that the group can write";
  const std::string create = "If they are public, " + write;
  // What has the marker's name may be somebody else's: no message tells the reader to remove it.
  const std::string look = "Nothing was touched. Remove nothing on this program's word: find out what it is. If the weights are public, " + marker +
                           " must be a regular file with no second name that holds what sha256sum prints for " + file;
  struct stat is;
  if (fd >= 0) error = fstat(fd, &is) ? errno : 0;
  else if (!error) error = EIO;
  if (!error && !S_ISREG(is.st_mode)) error = EISDIR;
  if (!error && is.st_nlink != 1) error = EMLINK;
  const bool foreign = !error && !marker_owner_counts(is.st_uid, caller, file_owner), writable = !error && (is.st_mode & 022) != 0;
  char start[65];
  size_t length = 0;
  while (!error && length < sizeof start) {
    const ssize_t got = read(fd, start + length, sizeof start - length);
    if (!got) break;
    if (got > 0)
      length += (size_t)got;
    else if (errno != EINTR)
      error = errno;
  }
  if (fd >= 0) close(fd);
  if (error == ENOENT) return but + "there is no regular file " + marker + hidden + create;
  if (error == EISDIR) return but + marker + " is not a regular file" + hidden + look;
  if (error == ELOOP) return but + marker + " is a symbolic link, which is not accepted" + hidden + look;
  if (error == EMLINK) return but + marker + " has a second name (a hard link), which is not accepted" + hidden + look;
  if (error) return but + marker + " cannot be read: " + strerror(error) + hidden + "Otherwise fix the cause and run this command again";
  if (foreign)
    return but + marker + " belongs to another account (user ID " + std::to_string(is.st_uid) + "), and counts only if it belongs to the account that runs cp_commit, to the owner of " + file +
           " or to root. Nothing was touched. Remove nothing on this program's word. If the weights in " + file + " must stay hidden, omit -public-weights, and find out who else writes into that " +
           "directory. If they are public, ask that account to take its file away, and the owner of " + file + ", or root, to create it: their file counts for every account that uses that directory. It must hold what sha256sum prints for " + file +
           ", and neither its group nor others may be able to write it";
  if (writable)
    return but + marker + " can be written by its group or by everybody, so another account may have written it" + hidden + "If they are public, narrowing the permissions of " + marker +
           " is not enough: an account that opened the file before can still write to it. See docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\" for what to do instead; if the file is not yours, ask its owner to do it. Then run cp_commit again. A further call names the next such file, if there is one. If the group or others can also write " +
           "a file that the first command there has just created, the file system keeps no permissions: keep the model files and their .weights-are-public files on one that does";
  size_t digits = 0;
  while (digits < length && digits < 64 && ((start[digits] >= '0' && start[digits] <= '9') || (start[digits] >= 'a' && start[digits] <= 'f'))) digits++;
  if (digits < 64 || (length > 64 && start[64] != ' ' && start[64] != '\n'))
    return but + marker + " is empty or does not start with 64 lower-case hexadecimal digits followed by a space or a newline" + hidden + look;
  declared.assign(start, 64);
  return std::string();
}
// Call it before any random value is drawn, anything is written or a weight is read. `model_files`
// is the command-line parser's list. No model file is opened: its owner comes from stat() on the
// path, and if stat() fails only the caller's and root's markers count. Returns 0, or 2 after the
// message for the first listed file whose marker is rejected. This check only saves the operator a
// wait: the one that decides is check_the_digests(), on the loader's own list and descriptors.
// check_the_markers() passes geteuid() as `caller`; a test passes another account.
static int check_the_markers_as(uid_t caller, const std::vector<std::string> &model_files) {
  for (const std::string &file : model_files) {
    std::string declared;
    struct stat of_file;
    const uid_t file_owner = stat(file.c_str(), &of_file) ? caller : of_file.st_uid;
    const int fd = open((file + ".weights-are-public").c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW); // O_NONBLOCK: does not block on a FIFO
    const std::string why = read_public_weights_marker(fd, errno, file, caller, file_owner, declared);
    if (!why.empty()) return refuse("cp-public-weights-marker", why);
  }
  return 0;
}
static int check_the_markers(const std::vector<std::string> &model_files) { return check_the_markers_as(geteuid(), model_files); }
// Compares the SHA-256 of an open file, from its first byte to its end, with `declared` (64
// hexadecimal digits). It reads with pread(), so the descriptor's file offset does not matter and
// does not move. Returns 0 if they are equal, -1 if they differ, or the errno of a failed read. The
// file may hold weights that must stay hidden, so the read buffer and the hash state are zeroed.
// The temporary string that Sha256::hex() returns, with the digest, is not zeroed.
// At the file's end it asks the system to drop the file from its page cache, to free memory.
static int compare_sha256_of_open_file(int fd, const std::string &declared) {
  std::vector<uint8_t> buffer((size_t)4 << 20);
  ::Sha256 hash;
  int error = 0;
  for (off_t at = 0;;) {
    const ssize_t got = pread(fd, buffer.data(), buffer.size(), at);
    if (!got) {
      // The whole file has been read, and no tensor yet: the system is asked to drop the file's
      // cached pages now, before memory is locked for the weights. The reads of the tensors may
      // have to fetch them from the disk a second time.
      ::gguf::Model::drop_from_page_cache(fd, 0, (uint64_t)at);
      break;
    }
    if (got > 0) {
      hash.update(buffer.data(), (size_t)got), at += got;
    } else if (errno != EINTR) {
      error = errno;
      break;
    }
  }
  const bool equal = !error && hash.hex() == declared;
  explicit_bzero(buffer.data(), buffer.size());
  explicit_bzero(&hash, sizeof hash);
  return error ? error : equal ? 0 : -1;
}
// Call it when the loader has opened the model files and before a tensor is read. For each file on
// the loader's list it reads the marker again, from the directory that the loader holds open
// (file_in_the_folder()), and computes the file's SHA-256 through the descriptor from which the
// tensors are then read. So the bytes that are compared belong to the file that is loaded, even if
// a name in the directory, or the directory's path, now leads elsewhere. `directory` is that path,
// for the messages. The model file's owner is read from the loader's descriptor; if that fails,
// only the caller's and root's markers count. A bad marker is reported before any file is hashed.
// Returns 0 or 2. check_the_digests() passes geteuid() as `caller`; a test passes another account.
static int check_the_digests_as(uid_t caller, const ::gguf::Model &model, const std::string &directory) {
  const size_t files = model.shard_count();
  const auto path = [&](size_t s) { return directory + "/" + model.shard_name(s); };
  std::vector<std::string> declared(files);
  uint64_t bytes = 0;
  for (size_t s = 0; s < files; s++) {
    struct stat is, of_file;
    const bool examined = !fstat(model.shard_descriptor(s), &of_file);
    const int fd = model.file_in_the_folder(model.shard_name(s) + ".weights-are-public", is);
    const int error = fd == ::gguf::Model::A_LINK ? ELOOP : fd == ::gguf::Model::A_SECOND_NAME ? EMLINK : fd == ::gguf::Model::NOT_REGULAR ? EISDIR : errno;
    const std::string why = read_public_weights_marker(fd, error, path(s), caller, examined ? of_file.st_uid : caller, declared[s]);
    if (!why.empty()) return refuse("cp-public-weights-marker", why);
    if (examined) bytes += (uint64_t)of_file.st_size;
  }
  printf("cp_commit: next: computing each model file's SHA-256 (files: %zu; %.1f GB in all) to compare with its .weights-are-public file\n", files, bytes / 1e9);
  const auto begin = std::chrono::steady_clock::now();
  for (size_t s = 0; s < files; s++) {
    const int result = compare_sha256_of_open_file(model.shard_descriptor(s), declared[s]);
    if (!result) continue;
    const std::string file = ::gguf::shown(path(s)), marker = file + ".weights-are-public";
    const std::string unused = "No weight was used, and nothing of a commitment was written. ";
    if (result > 0)
      return refuse("cp-public-weights-marker", "the SHA-256 of " + file + ", which is compared with the one in " + marker + ", cannot be computed, because a read failed: " +
                                                    strerror(result) + ". " + unused + "Check the disk and the file, then run this command again");
    return refuse("cp-public-weights-marker", "the SHA-256 of " + file + " differs from the one in " + marker + ", so " + file +
                                                  " is not the file that was declared public: it was replaced, changed or copied incompletely. " + unused +
                                                  "With -public-weights nothing is hidden: if the weights in " + file + " must stay hidden, omit -public-weights. Find out first why it differs. If " + file +
                                                  " is complete, is the file that you mean, and its weights are public, " + marker + " is out of date: see docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\" for what to do," +
                                                  " then run cp_commit again; if " + marker + " is not yours, ask its owner to do it. Do not write to it with a shell's redirection, which follows a symbolic link");
  }
  printf("cp_commit: each model file has the SHA-256 that was declared public: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
  return 0;
}
static int check_the_digests(const ::gguf::Model &model, const std::string &directory) { return check_the_digests_as(geteuid(), model, directory); }
} // namespace making
} // namespace hf
