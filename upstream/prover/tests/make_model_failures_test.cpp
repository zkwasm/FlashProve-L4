// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A test of make_model (prover/tests/weights_blocks/make_model.cpp), the CPU tool that writes the
// tests' small synthetic models: what it removes after a failure, and where it writes, if the
// directory that it has made is taken from its path while it runs.
// The test compiles the tool's source into itself, under another name for main(), and replaces one
// call: openat(), through which the tool creates its four files. The replacement passes every call
// on. Right after one of the tool's creates it can move the tool's directory to <path>.moved and
// make another directory at the path, as a second party could who may write to the directory above;
// and it can make the tool's second create fail. So each scene happens at a fixed point of the run,
// without a signal and without a race. Each scene runs the tool in a child process.
// Six scenes:
//   plain    nothing is replaced: status 0, the four files are in the directory;
//   fails    the second create fails: status 2, and the directory is gone with what was in it;
//   files    the path is taken by a directory with four files of the tool's names, then the second
//            create fails: status 2; the four files are there with their bytes; the moved directory
//            is there and empty;
//   empty    the same with an empty directory at the path: status 2; that directory is there; the
//            moved one is there and empty;
//   no-fault the path is taken by an empty directory and nothing fails: status 2; nothing was
//            written into that directory; the moved one is there and empty;
//   late     the path is taken right after the fourth and last create, by a directory with the
//            four files, and nothing fails: status 2; the four files are there with their bytes;
//            the moved directory is there and empty.
// In "files", "empty" and "no-fault" the path is taken right after the first create.
// A tool that removed by stored paths would fail the four scenes in which the path is taken; one
// that did not look at the path once more before its last line, "no-fault" and "late"; one that
// created its files by their paths would never reach the replaced call: all scenes but "plain".
// Everything happens below <new directory>, which the test makes and which must not exist. The test
// removes nothing itself: what the scenes leave stays there for whoever made the directory above.
// Not tested: a directory put at the path between the tool's mkdir() and open(), which the tool's
// own comments name as not noticed; what main() checks when it opens the directory (owner, mode, no
// link); that an existing directory is refused; O_EXCL and O_NOFOLLOW in create(); a symbolic link
// at the path; another user's account (both parties are this process); a write that fails half way;
// the tool's messages; the bytes of the model.
// Build and run, from the repository's top (SUITE.md, prover.make_model_failures_test):
//   g++ -O2 -fno-ipa-modref -std=c++17 -Wall -Wextra -pthread -I prover -o <program>
//     prover/tests/make_model_failures_test.cpp
//   <program> <new directory>
// Built so, without -msha, the tool hashes in plain C++ (prover/weights_commitment/sha256.h), so
// the test runs on any CPU.
// Last line: "make_model_failures_test: 6 scenes: passes" (exit status 0) or
// "make_model_failures_test: <k> of 6 scenes differ" (1). Exit status 2: wrong usage, <new
// directory> cannot be made, or a child process cannot be started or is lost.
// All system headers that the tool's source includes are included here first, before main and
// openat get other names.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

static const char *const NAMES[4] = {"made-00001-of-00002.gguf", "made-00002-of-00002.gguf", "wsums.txt", "fold.txt"};
static const char OTHERS[] = "somebody else's\n";

// What the replaced call does in the scene that runs (set in the child, before the tool starts).
static std::string path;        // the directory that the tool is told to make
static int take_after;          // after this create (0: never): move the directory, make another at the path
static bool lay_files;          // ... and lay four files of the tool's names in it
static bool second_fails;       // the second create fails, as on a full disk
static int creates;

static void lay(const std::string &file, const char *text) {
  FILE *f = fopen(file.c_str(), "wx");
  if (!f || fputs(text, f) < 0 || fclose(f)) _exit(9);
}

static int test_openat(int directory, const char *name, int flags, ...) {
  va_list more;
  va_start(more, flags);
  const mode_t mode = (flags & O_CREAT) ? (mode_t)va_arg(more, int) : 0;
  va_end(more);
  if (!(flags & O_CREAT)) return openat(directory, name, flags);
  if (++creates == 2 && second_fails) {
    errno = ENOSPC;
    return -1;
  }
  const int made_now = openat(directory, name, flags, mode);
  if (creates == take_after && made_now >= 0) {
    if (rename(path.c_str(), (path + ".moved").c_str()) || mkdir(path.c_str(), 0700)) _exit(9);
    if (lay_files)
      for (const char *n : NAMES) lay(path + "/" + n, OTHERS);
  }
  return made_now;
}

#define main make_model_main
#define openat test_openat
#include "tests/weights_blocks/make_model.cpp"
#undef openat
#undef main

// How many entries the directory has; -1: it is not there, or is no directory.
static int entries(const std::string &directory) {
  DIR *d = opendir(directory.c_str());
  if (!d) return -1;
  int n = 0;
  while (const dirent *e = readdir(d))
    if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) n++;
  closedir(d);
  return n;
}

static bool holds(const std::string &file, const char *text) {
  FILE *f = fopen(file.c_str(), "r");
  if (!f) return false;
  char read_[64] = {0};
  const size_t n = fread(read_, 1, sizeof read_ - 1, f);
  fclose(f);
  return n == strlen(text) && !memcmp(read_, text, n);
}

struct Scene {
  const char *name;
  int take;             // after which create the path is taken; 0: never
  bool files, fails;
  int status;           // expected: the tool's exit status
  int at_path, moved;   // expected: entries of the directory at the path and of <path>.moved; -1: not there
};

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: make_model_failures_test <new directory>\n");
    return 2;
  }
  const std::string work = argv[1];
  if (mkdir(work.c_str(), 0700)) {
    printf("make_model_failures_test: %s cannot be made: %s\n", work.c_str(), strerror(errno));
    return 2;
  }
  static const Scene scenes[6] = {{"plain", 0, false, false, 0, 4, -1},
                                  {"fails", 0, false, true, 2, -1, -1},
                                  {"files", 1, true, true, 2, 4, 0},
                                  {"empty", 1, false, true, 2, 0, 0},
                                  {"no-fault", 1, false, false, 2, 0, 0},
                                  {"late", 4, true, false, 2, 4, 0}};
  int differ = 0;
  for (const Scene &s : scenes) {
    const std::string at = work + "/" + s.name, said = at + ".said";
    fflush(stdout);
    const pid_t child = fork();
    if (child < 0) {
      printf("make_model_failures_test: no child process: %s\n", strerror(errno));
      return 2;
    }
    if (child == 0) {
      if (!freopen(said.c_str(), "wx", stdout)) _exit(9); // what the tool prints
      path = at, take_after = s.take, lay_files = s.files, second_fails = s.fails;
      // The smallest model that the tool makes: layers, H, G, f, fe, experts, V, seed.
      const char *call[] = {"make_model", at.c_str(), "1", "2", "1", "256", "256", "0", "2", "0", nullptr};
      exit(make_model_main(10, (char **)call));
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) {
      printf("make_model_failures_test: a child process is lost\n");
      return 2;
    }
    const int ended = WIFEXITED(status) ? WEXITSTATUS(status) : -1, here = entries(at), there = entries(at + ".moved");
    bool theirs = true;
    if (s.files)
      for (const char *n : NAMES) theirs = theirs && holds(at + "/" + n, OTHERS);
    if (ended != s.status || here != s.at_path || there != s.moved || !theirs) {
      differ++;
      printf("make_model_failures_test: %s: status %d (expected %d); entries at the path %d (%d), "
             "in the moved directory %d (%d)%s\n",
             s.name, ended, s.status, here, s.at_path, there, s.moved,
             theirs ? "" : "; a file of the other party is not there or has other bytes");
    }
  }
  if (differ) printf("make_model_failures_test: %d of 6 scenes differ\n", differ);
  else printf("make_model_failures_test: 6 scenes: passes\n");
  return differ ? 1 : 0;
}
