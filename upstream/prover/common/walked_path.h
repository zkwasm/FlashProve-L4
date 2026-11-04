// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the prover program cp_prove and the committer program cp_commit open a directory that the
// command line names and in which they go on to create, write or remove something: part by part,
// so that no other account can lead them to another place (WalkedPath::open_by_parts()). One text
// for both programs, so that cp_commit accepts no place for a secret key directory that cp_prove
// would reject. The one place that reads SUDO_UID is here too (take_who_typed_sudo()). Messages say
// "folder" for directory. Host code.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

struct WalkedPath {
  // A path for a printed line, with each byte that is not printable ASCII as \x and two hex digits,
  // so that a name with a line feed in it cannot begin a printed line.
  static std::string shown(const std::string &path) {
    std::string all;
    for (const unsigned char c : path) {
      char hex[5];
      snprintf(hex, sizeof hex, "\\x%02x", c);
      if (c >= 0x20 && c < 0x7f)
        all += (char)c;
      else
        all += hex;
    }
    return all;
  }
  // The account that typed sudo, or 0 for none. A program sets it at its start
  // (prover/prover/cp_prove.cu, prover/hash_commitment/hf_commit.cu) from take_who_typed_sudo(),
  // below, which returns 0 unless the caller is root. Nothing else here reads the environment.
  // With it, "sudo cp_prove ..." and "sudo cp_commit ..." work below that account's own
  // directories: open_by_parts() trusts them along the path as it trusts root's and the caller's
  // (trusted_owners() in run_suite.py does the same), and the directory given to Spool::open() or
  // Spool::clear() may itself belong to that account.
  // The number widens only tests of an owner: no test of a mode, none below the given directory.
  // Limit: an account that is granted sudo for these programs alone can lead them elsewhere through
  // a directory of its own on the path. Grant that to no account that is not trusted as root is.
  inline static uid_t typed_sudo = 0;
  // For Spool::open() and Spool::clear() (prover/prover/spool.h), which remove entries,
  // Prover::read_isk() (prover/prover/isk.cuh) and cp_commit's making::reach_by_parts()
  // (prover/hash_commitment/hf_commit_making.h): opens the directory `path` part by part, from /
  // (no program passes a relative path). Each part is opened relative to the descriptor of the
  // directory before it, without following a link, so no link in the path, whoever created it,
  // leads the call elsewhere. Nor may another account be able to swap a part: each directory in
  // which a part is looked up must belong to root, the caller or `typed_sudo` and give group and
  // others no write permission. One exception: a directory of one of these three with the sticky
  // bit, such as /tmp, if the part found there is also one of theirs. In such a directory only an
  // entry's owner, the directory's owner and root can rename or remove the entry.
  // Returns the descriptor; or -1 with errno set; or -1 with `refused` set to text that names the
  // part and says what is wrong with it.
  static int open_by_parts(const std::string &path, std::string &refused) {
    auto trusted = [](const struct stat &is) { return is.st_uid == 0 || is.st_uid == geteuid() || is.st_uid == typed_sudo; }; // (typed_sudo = 0 adds nothing)
    std::string here = path[0] == '/' ? "/" : ".";
    int at = ::open(here.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
    for (size_t from = 0, to; at >= 0 && (from = path.find_first_not_of('/', from)) != std::string::npos; from = to) {
      to = std::min(path.find('/', from), path.size());
      const std::string part = path.substr(from, to - from), whole = here == "/" ? "/" + part : here == "." ? part : here + "/" + part,
                        where = here == "." ? std::string("the working directory") : shown(here);
      const int next = openat(at, part.c_str(), O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      const int error = errno;
      struct stat above, found;
      const bool looked = !fstat(at, &above) && (next < 0 || !fstat(next, &found));
      close(at);
      if (!looked || !trusted(above))
        refused = where + ", in which " + shown(part) + " is looked up, belongs to an account other than root and you (ls -ld shows its owner), which "
                  "could put another folder there. If the folder that you mean is that user's, run this as the user who ran cp_prove";
      else if ((above.st_mode & 022) && !(above.st_mode & S_ISVTX))
        refused = where + ", in which " + shown(part) + " is looked up, can be written by its group or by others (ls -ld shows its mode), so that "
                  "somebody else could put another folder there. " +
                  // Root is never told to chmod: a directory of root's may be one that others
                  // share, such as a shared disk (cure() in run_suite.py does the same).
                  (geteuid() && above.st_uid == geteuid() ? "If the folder that you mean does lie there, first take that permission from " + where + " (chmod go-w)"
                                                          : "Use a folder below a directory in which only you and root can write, and leave the mode of " + where + " as it is: others may share it");
      else if (next < 0 ? error == ELOOP || error == ENOTDIR : !S_ISDIR(found.st_mode))
        refused = shown(whole) + " is a link or a file. Name the folder itself that was named with -spool or -out, by its real path";
      else if (next >= 0 && (above.st_mode & 022) && !trusted(found)) // the sticky bit guards only an entry of a `trusted` account's
        refused = shown(whole) + " belongs to an account other than root and you (ls -ld shows its owner), and any account can make a folder in " + where +
                  ". If another folder was named with -spool or -out, name that one; if it was this one, run this as the user who ran cp_prove";
      if (next >= 0 && !refused.empty()) close(next);
      if (next < 0 || !refused.empty()) return errno = error, -1;
      at = next, here = whole;
    }
    if (at < 0) return -1;
    const int opened = openat(at, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); // the same directory, for reading and locking
    const int error = errno;
    close(at);
    return errno = error, opened;
  }
};

// Parses a user ID: returns it only if `said` is exactly its decimal form (no sign, blank, leading
// zero or further character) and it is below 2^32 - 1. Otherwise, and for "0" (root), returns 0.
static inline uid_t account_in(const char *said) {
  if (!said) return 0;
  const unsigned long number = strtoul(said, nullptr, 10);
  char printed[24];
  snprintf(printed, sizeof printed, "%lu", number);
  return number < 0xffffffffUL && !strcmp(printed, said) ? (uid_t)number : 0;
}
// Returns the user ID of the account that ran sudo, or 0, and removes SUDO_UID from the
// environment, whoever runs the program. SUDO_UID counts only if the real and the effective user
// are both root, as sudo sets them: there the program assumes that sudo or root has set the
// variable. A set-user-ID program file runs with root as its effective user only, in its caller's
// own environment: there SUDO_UID is ignored.
static inline uid_t take_who_typed_sudo() {
  const uid_t who = getuid() || geteuid() ? 0 : account_in(getenv("SUDO_UID"));
  unsetenv("SUDO_UID");
  return who;
}
