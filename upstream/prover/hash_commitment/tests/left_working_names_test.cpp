// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests refuse_what_is_left() (prover/hash_commitment/hf_commit_making.h). `cp_commit` builds the
// directories named by -isk (secret) and -ivk (public) under the working names <isk>.making and
// <ivk>.making; if either exists already, the function rejects the call (printed: "refused"). It
// must change nothing: listings before and after are compared. Its message may say that directories
// can be removed, but never tells the reader to remove them. It may say so only of what looks like
// the remains of a dead call (killed or crashed), and only after the advice to give two new names
// and a caution that names each parent directory. It must say something else of a symbolic link, a
// file, or a directory that is open to others, holds what `cp_commit` does not create, is locked or
// cannot be locked. Usage: left_working_names_test <a new directory>. The test creates that
// directory, which must not exist, and works below it. It removes nothing and sends no signal.
// Not covered: another user's directory (the test runs under one account); a working name that
// cannot be inspected; an existing <isk>; a change between the function's two looks, 0.3 s apart.
// A complete run ends with the line "left_working_names_test: <n> checks, 0 differ: passes" (exit
// status 0) or "... <k> differ: fails" (1). Status 2: wrong usage, or the test could not go on and
// printed "left_working_names_test: cannot ...".
#include "common/cp_files.h"
#include "hash_commitment/hf_format.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// Case 17 makes flock() fail with ENOLCK, to stand for a file system without locks. Other headers
// that name flock() are included above, so the macro replaces only the tested header's calls.
static bool no_locks = false;
static int flock_or_not(int fd, int how) { return no_locks ? (errno = ENOLCK, -1) : flock(fd, how); }
#define flock flock_or_not
#include "hash_commitment/hf_commit_making.h"
#undef flock

using namespace hf::making;

static int checks = 0, differ = 0;
static std::string label, top, base, isk, ivk;
static void check(bool good, const std::string &what) {
  checks++;
  if (good) return;
  differ++;
  printf("DIFFERS: %s%s\n", label.c_str(), what.c_str());
}
[[noreturn]] static void stop(const std::string &what) {
  fprintf(stderr, "left_working_names_test: cannot %s: %s\n", what.c_str(), strerror(errno));
  exit(2);
}
static void make(const std::string &directory, mode_t mode = 0700) {
  if (mkdir(directory.c_str(), 0700) || chmod(directory.c_str(), mode)) stop("create " + directory);
}
static void put(const std::string &file) {
  const int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0 || write(fd, "x", 1) != 1 || close(fd)) stop("write " + file);
}
// Every name below a directory, with st_mode and a regular file's size; follows no symbolic link.
static void list(const std::string &directory, const std::string &shown, std::vector<std::string> &out) {
  DIR *const names = opendir(directory.c_str());
  if (!names) return;
  while (const struct dirent *const entry = readdir(names)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    struct stat is;
    if (lstat((directory + "/" + name).c_str(), &is)) continue;
    out.push_back(shown + name + " " + std::to_string(is.st_mode) + " " + std::to_string(S_ISREG(is.st_mode) ? (long long)is.st_size : 0));
    if (S_ISDIR(is.st_mode)) list(directory + "/" + name, shown + name + "/", out);
  }
  closedir(names);
}
static std::vector<std::string> listing() {
  std::vector<std::string> out;
  list(base, "", out);
  std::sort(out.begin(), out.end());
  return out;
}
static void set_up(int number, const char *what) {
  label = std::to_string(number) + ", " + what + ": ";
  base = top + "/" + std::to_string(number);
  make(base);
  isk = base + "/isk", ivk = base + "/ivk";
}
// Imitates what a dead call leaves: names that foreign_name() accepts, with and without ".making".
static void a_dead_calls_secret(const std::string &d) {
  make(d), make(d + "/config"), make(d + "/anchor.making"), make(d + "/leaf"), make(d + "/leaf/0"), make(d + "/leaf/1.making");
  for (const char *f : {"/seed.bin", "/key.bin.making", "/isk.txt.making", "/config/policy.txt", "/config/KL.bin", "/config/layout.txt", "/anchor.making/hcnodes.bin", "/anchor.making/count.bin",
                        "/leaf/0/fresh.bin", "/leaf/0/entry.bin", "/leaf/0/count.bin", "/leaf/0/hcnodes.bin", "/leaf/1.making/fresh.bin"})
    put(d + f);
}
static void a_dead_calls_public(const std::string &d) {
  make(d), make(d + "/leaf");
  for (const char *f : {"/key.bin", "/leaf/0.bin", "/leaf/1.bin.making"}) put(d + f);
}
// Calls the function in a child process. Returns its status; `said` gets what it wrote to stderr.
static int call(std::string &said) {
  const std::string screen = base + ".screen";
  fflush(stdout);
  const pid_t child = fork();
  if (child < 0) stop("fork");
  if (!child) {
    const int fd = open(screen.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0 || dup2(fd, 2) < 0) _exit(99);
    _exit(refuse_what_is_left(isk, ivk));
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) stop("wait for the child");
  said.clear();
  ::cp::read_file(screen, said);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
static bool says(const std::string &said, const char *words) { return said.find(words) != std::string::npos; }
static const char *const ORDER = "these two directories, they can be removed", *const ONE = "this directory, it can be removed", *const MAY = "it holds nothing secret and can be removed", *const RUNNING = "a running cp_commit holds", *const NOT_MADE = "No call of cp_commit by this user has made this";
// One rejected case. Of the five texts above only `wanted`, compared by address, may be printed.
static void refused_with(const char *wanted, const char *more = nullptr) {
  const std::vector<std::string> before = listing();
  std::string said;
  const int status = call(said);
  check(status == 2 && says(said, "cp_commit: refused (file-there-already): ") && says(said, ".making exists already"), "refused by its name, with status 2");
  check(says(said, wanted) && (!more || says(said, more)), std::string("the refusal says: ") + wanted + (more ? std::string(" .. ") + more : std::string()));
  for (const char *other : {ORDER, ONE, MAY, RUNNING, NOT_MADE})
    if (other != wanted) check(!says(said, other), std::string("it does not say: ") + other);
  check(!says(said, "remove th") && !says(said, "Remove th") && !says(said, "erase"), "no sentence bids a removal");
  if (wanted == ORDER || wanted == ONE || wanted == MAY) { // the three that say what can be removed
    const std::string above_isk = isk.substr(0, isk.rfind('/')), above_ivk = ivk.substr(0, ivk.rfind('/'));
    const std::string parents = wanted == MAY ? above_ivk : wanted == ONE || above_isk == above_ivk ? above_isk : above_isk + " or " + above_ivk;
    const size_t way_on = said.find(". Give -isk and -ivk two new names: nothing has to be removed first. "), caution = said.find("Before anything is removed: if anybody else can write " + parents + ", look at what has ");
    check(way_on != std::string::npos && way_on < said.find(wanted), "the way on without a removal comes first");
    check(caution != std::string::npos && way_on < caution && caution < said.find(wanted), "the caution names every parent of what can be removed, " + parents + ", and comes before it");
  }
  check(listing() == before, "nothing is changed");
}

int main(int argc, char **argv) {
  if (argc != 2) return fprintf(stderr, "usage: left_working_names_test <a new directory>\n"), 2;
  if (mkdir(argv[1], 0700)) stop(std::string("create ") + argv[1]);
  char *const real = realpath(argv[1], nullptr);
  if (!real) stop("find the directory's real path");
  top = real;
  std::string said;

  set_up(1, "neither name exists");
  check(call(said) == 0 && said.empty(), "status 0, and nothing is printed");
  set_up(2, "the parent does not exist");
  isk = base + "/none/isk", ivk = base + "/none/ivk";
  check(call(said) == 0 && said.empty(), "status 0, and nothing is printed: the caller's next look refuses the parent");

  set_up(3, "a dead call's two directories");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk));
  refused_with(ORDER, "no cp_commit on this machine is running in");
  set_up(4, "a dead call's public directory alone");
  a_dead_calls_public(beside(ivk));
  refused_with(MAY);
  set_up(5, "a dead call's secret directory alone");
  a_dead_calls_secret(beside(isk));
  refused_with(ONE, "/isk.making (it holds the seed)\n"); // the list ends here: no <ivk>.making

  set_up(6, "a link to a dead call's directory");
  a_dead_calls_secret(base + "/elsewhere"), a_dead_calls_public(beside(ivk));
  if (symlink((base + "/elsewhere").c_str(), beside(isk).c_str())) stop("link " + beside(isk));
  refused_with(NOT_MADE, "is a symbolic link");
  set_up(7, "a file");
  put(beside(isk)), a_dead_calls_public(beside(ivk));
  refused_with(NOT_MADE, "is no directory");
  set_up(8, "a directory that is open to others");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk));
  if (chmod(beside(isk).c_str(), 0750)) stop("change a mode");
  refused_with(NOT_MADE, "open to group or others");
  set_up(9, "another name in the secret directory");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk)), put(beside(isk) + "/mine.txt");
  refused_with(NOT_MADE, "/isk.making/mine.txt");
  set_up(10, "another name three directories down");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk)), put(beside(isk) + "/leaf/0/mine.txt");
  refused_with(NOT_MADE, "/leaf/0/mine.txt");
  set_up(11, "a link under an inner directory's name");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk)), make(base + "/elsewhere");
  if (symlink((base + "/elsewhere").c_str(), (beside(isk) + "/anchor").c_str())) stop("link");
  refused_with(NOT_MADE, "/isk.making/anchor");
  set_up(12, "a directory under a file's name");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk)), make(beside(isk) + "/isk.txt");
  refused_with(NOT_MADE, "/isk.making/isk.txt");
  set_up(13, "another name in the public directory");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk)), put(beside(ivk) + "/leaf/x.bin");
  refused_with(NOT_MADE, "/ivk.making/leaf/x.bin");
  set_up(14, "another name in the public directory, which stands alone");
  a_dead_calls_public(beside(ivk)), put(beside(ivk) + "/mine.txt");
  refused_with(NOT_MADE, "/ivk.making/mine.txt");

  for (int which = 0; which < 2; which++) {
    set_up(15 + which, which ? "this process holds the public directory's lock" : "this process holds the secret directory's lock");
    a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk));
    const int held = open(beside(which ? ivk : isk).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); // another open file description than the child's
    if (held < 0 || flock(held, LOCK_EX | LOCK_NB)) stop("lock a directory");
    refused_with(RUNNING, which ? "/ivk.making (its lock is taken)" : "/isk.making (its lock is taken)");
    close(held);
  }
  set_up(17, "a file system without locks");
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk));
  no_locks = true;
  refused_with("cannot be told", "cannot be locked");
  no_locks = false;

  set_up(18, "two empty closed directories of the user's"); // called removable too: a known limit
  make(beside(isk)), make(beside(ivk));
  refused_with(ORDER);
  set_up(19, "a dead call's two directories under two parents");
  make(base + "/a"), make(base + "/b");
  isk = base + "/a/isk", ivk = base + "/b/ivk";
  a_dead_calls_secret(beside(isk)), a_dead_calls_public(beside(ivk));
  refused_with(ORDER);

  printf("left_working_names_test: %d checks, %d differ: %s\n", checks, differ, differ ? "fails" : "passes");
  return differ ? 1 : 0;
}
