// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the two routines that cp_commit calls at its start on the directories in which it is to
// create <isk>, <ivk> and -worst-case-table's file (prover/hash_commitment/hf_commit_making.h).
// reach_by_parts() must open such a directory part by part and reject a symbolic link on the way, a
// directory on the way, or the directory itself, in which group or others can write without the
// sticky bit, and a directory that is absent or cannot be written. try_the_file_system() must pass
// on a file system that can do what a run needs later, reject one on which a step fails, name that
// step, and leave nothing behind but a directory that it cannot tell to be its own (case 11).
//   walked_parents_test <a new directory>
// It creates that directory, which must not exist, and works below it. It removes nothing but
// through the tested routines and sends no signal. Each case calls the routines in a child process,
// because the list of kept directories only grows.
// Under sudo it acts as root on names below that directory. Where a directory above it belongs to
// the account that ran sudo, that account can swap a part of the path and so lead root's calls
// elsewhere, much as the owner of a checkout can change any program that sudo runs from it.
// For this test program alone, the #define lines below redirect four system calls to wrappers while
// the header is compiled, so that a case can make one of them fail; no build of cp_commit does this.
// Not covered: another account's directory or link (the test runs under one account; case 12 shows
// the account that ran sudo, for root alone); that cp_commit calls the two routines, and where; a
// signal during the try (case 8 shows only that three are blocked); the try's ends "opened" and
// "removed again", its wait for a lock that another holds, and "/" as its parent; a real file system
// without locks or without RENAME_NOREPLACE.
// Last line: "walked_parents_test: <n> checks, 0 differ: passes", status 0; or "... <k> differ:
// fails", status 1. Status 2 and a line with "not run": the directories above the new directory are
// themselves open to another account, so that every case would be rejected.
// Build and run, from prover/: tests/unit.sh both hash_commitment/tests/walked_parents_test.cpp <directory> <a new directory>
#include "common/cp_files.h"
#include "common/setting.h"
#include "common/walked_path.h"
#include "hash_commitment/hf_format.h"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
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

// What a case makes fail in the tested header: the call's name, and the errno that it then sets.
static const char *failing = "";
static int failing_with = 0;
static bool fails(const char *call) { return !strcmp(failing, call) && (errno = failing_with, true); }
// Whether SIGINT, SIGTERM and SIGHUP are all blocked in the calling thread.
static bool three_are_blocked() {
  sigset_t now;
  return !pthread_sigmask(SIG_BLOCK, nullptr, &now) && sigismember(&now, SIGINT) == 1 && sigismember(&now, SIGTERM) == 1 && sigismember(&now, SIGHUP) == 1;
}
static int calls_seen = 0, calls_blocked = 0; // the tested header's calls through the four wrappers, and those made with the three blocked
static void seen() { calls_seen++, calls_blocked += three_are_blocked(); }
static mode_t made_with = 0; // not 0: mkdirat() gives the new directory this mode, as a file system that keeps no modes would
static int test_mkdirat(int directory, const char *name, mode_t mode) {
  seen();
  if (fails("mkdirat")) return -1;
  const int failed = mkdirat(directory, name, mode);
  return failed || !made_with ? failed : fchmodat(directory, name, made_with, 0);
}
static int test_flock(int fd, int how) { return seen(), fails("flock") ? -1 : flock(fd, how); }
static int test_renameat2(int a, const char *from, int b, const char *to, unsigned how) { return seen(), fails("renameat2") ? -1 : renameat2(a, from, b, to, how); }
static int test_fsync(int fd) { return seen(), fails("fsync") ? -1 : fsync(fd); }
// The headers that declare these functions are included above, so the macros change only the tested header.
#define mkdirat test_mkdirat
#define flock test_flock
#define renameat2 test_renameat2
#define fsync test_fsync
#include "hash_commitment/hf_commit_making.h"
#undef mkdirat
#undef flock
#undef renameat2
#undef fsync
using namespace hf::making;

static int checks = 0, differ = 0;
static std::string label, top, base;
static void check(bool good, const std::string &what) {
  checks++;
  if (good) return;
  differ++;
  printf("DIFFERS: %s%s\n", label.c_str(), what.c_str());
}
[[noreturn]] static void stop(const std::string &what) {
  fprintf(stderr, "walked_parents_test: cannot %s: %s\n", what.c_str(), strerror(errno));
  exit(1);
}
static void make(const std::string &directory, mode_t mode = 0700) {
  if (mkdir(directory.c_str(), 0700) || chmod(directory.c_str(), mode)) stop("create " + directory);
}
static void link_to(const std::string &target, const std::string &name) {
  if (symlink(target.c_str(), name.c_str())) stop("link " + name);
}
static void set_up(int number, const char *what) {
  label = std::to_string(number) + ", " + what + ": ";
  base = top + "/" + std::to_string(number);
  make(base);
}
// The names in a directory, sorted, joined by blanks.
static std::string names_in(const std::string &directory) {
  std::vector<std::string> names;
  if (DIR *const list = opendir(directory.c_str())) {
    while (const struct dirent *const entry = readdir(list))
      if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) names.push_back(entry->d_name);
    closedir(list);
  }
  std::sort(names.begin(), names.end());
  std::string all;
  for (const std::string &name : names) all += (all.empty() ? "" : " ") + name;
  return all;
}
static int open_descriptors() {
  const std::string all = names_in("/proc/self/fd");
  return (int)std::count(all.begin(), all.end(), ' ');
}
// Runs `steps` in a child process. Returns its result; `said` gets what it wrote to stderr, through a pipe.
static std::string said;
static int call(const std::function<int()> &steps) {
  int ends[2];
  fflush(stdout);
  if (pipe(ends)) stop("make a pipe");
  const pid_t child = fork();
  if (child < 0) stop("fork");
  if (!child) {
    if (dup2(ends[1], 2) < 0) _exit(99);
    close(ends[0]), close(ends[1]);
    const int status = steps();
    fflush(stderr);
    _exit(status);
  }
  close(ends[1]);
  said.clear();
  char piece[4096];
  for (ssize_t n; (n = read(ends[0], piece, sizeof piece)) > 0;) said.append(piece, (size_t)n);
  close(ends[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child) stop("wait for the child");
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
static bool says(const std::string &words) { return said.find(words) != std::string::npos; }
static int walk(const std::string &parent, const char *option = "-isk", const std::string &leaf = "isk") {
  return call([&] { return reach_by_parts(parent, parent + "/" + leaf, option); });
}
// The walk's rejection: status 2, its name, the option, and none of the words that the routine has for cp_prove.
// `top` is cut out of the text first: the messages print that path, and its name may contain one of those words.
static bool refused_on_the_way(int status, const char *option = "-isk") {
  std::string words = said;
  for (size_t at; (at = words.find(top)) != std::string::npos;) words.erase(at, top.size());
  const auto has = [&](const char *word) { return words.find(word) != std::string::npos; };
  return status == 2 && says("refused (cp-commit-path): the path named with " + std::string(option) + " is refused") && !has("-spool") && !has("cp_prove") && !has(" . is looked up");
}
// reach_by_parts() and then try_the_file_system(), as cp_commit calls them. 90: the walk has failed.
static int try_in(const std::string &parent) {
  return call([&] { return reach_by_parts(parent, parent + "/isk", "-isk") ? 90 : try_the_file_system(parent + "/isk"); });
}
static bool refused_at(int status, const char *step) {
  return status == 2 && says("refused (cp-commit-file-system): a new folder, ") && says(std::string(", cannot be ") + step + " in ") && says("no weight was loaded");
}

int main(int argc, char **argv) {
  if (argc != 2) return fprintf(stderr, "usage: walked_parents_test <a new directory>\n"), 1;
  if (mkdir(argv[1], 0700)) return fprintf(stderr, "walked_parents_test: cannot create %s: %s\n", argv[1], strerror(errno)), 1;
  char *const real = realpath(argv[1], nullptr);
  if (!real) return 1;
  top = real;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  WalkedPath::typed_sudo = take_who_typed_sudo(); // as cp_commit does: under sudo, directories of the account that ran sudo are trusted on the way
  if (walk(top)) return printf("not run: a directory on the way to %s can be written by its group or by others and is not sticky, or is another account's: name a new directory elsewhere\n", top.c_str()), 2;

  set_up(1, "the caller's own closed directory");
  check(call([&] {
          if (reach_by_parts(base, base + "/isk", "-isk")) return 1;
          const Road::Open *const kept = kept_open(base);
          const size_t held = road().kept.size();
          struct stat is, named;
          if (!kept || kept->created || fstat(kept->fd, &is) || stat(base.c_str(), &named) || is.st_ino != named.st_ino || is.st_dev != named.st_dev) return 2;
          if (via(base + "/isk").fd != kept->fd || strcmp(via(base + "/isk").name, "isk") || there(base + "/isk")) return 3;
          return reach_by_parts(base, base + "/ivk", "-ivk") || !reach(base) || road().kept.size() != held ? 4 : 0; // the second call adds nothing
        }) == 0 && said.empty(), "it is opened once and kept, and a name in it is looked up from its descriptor");
  check(call([&] { return reach_by_parts(".", "isk", "-isk") || road().kept.size(); }) == 0, "\".\" is not opened");
  if (geteuid()) check(call([&] { return reach_by_parts("/", "/isk", "-isk"); }) == 2 && says("refused ("), "\"/\" is judged as any other directory: a caller who cannot write in it is refused here");
  else printf("1: \"/\" as the directory: not shown: root can write in it\n");

  set_up(2, "a symbolic link at the directory's own name");
  make(base + "/real"), link_to(base + "/real", base + "/link");
  check(call([&] { return reach(base + "/link") ? 0 : 1; }) == 0, "control: reach(), which opens by the whole path, follows it");
  check(refused_on_the_way(walk(base + "/link")) && says(base + "/link is a link or a file. Give -isk a path without links: the real path"), "refused on the way, and named, with advice in cp_commit's words");

  set_up(3, "a symbolic link above the directory");
  make(base + "/real"), make(base + "/real/keys"), link_to(base + "/real", base + "/link");
  check(call([&] { return reach(base + "/link/keys") ? 0 : 1; }) == 0, "control: reach() follows it");
  check(refused_on_the_way(walk(base + "/link/keys")) && says(base + "/link is a link or a file"), "refused on the way, and named");
  check(walk(base + "/real/keys") == 0, "by its real path the same directory is taken");

  set_up(4, "the directory itself is open");
  int number = 0;
  for (const mode_t open_to : {0720, 0702, 0770, 0777, 02770}) {
    const std::string wide = base + "/" + std::to_string(number++);
    make(wide, open_to);
    check(call([&] { return reach(wide) ? 0 : 1; }) == 0, "control: reach() asks nothing of it");
    check(refused_on_the_way(walk(wide)) && says(wide + ", in which " + wide + "/isk is to be made, can be written by its group or by others"), "group or others can write in it, and it is not sticky: refused, with the name that was to be made");
    check(names_in(wide).empty(), "nothing is made in it");
  }
  make(base + "/sticky", 01777), make(base + "/read", 0755);
  check(walk(base + "/sticky") == 0 && walk(base + "/read") == 0, "with the sticky bit, or open for reading alone, it is taken");

  set_up(5, "a directory above is open");
  make(base + "/closed"), make(base + "/closed/keys"), make(base + "/wide", 0777), make(base + "/wide/keys");
  check(walk(base + "/closed/keys") == 0, "control: below closed directories it is taken");
  check(refused_on_the_way(walk(base + "/wide/keys")) && says(base + "/wide, in which keys is looked up, can be written"), "refused on the way, and named");

  set_up(6, "no such directory, a file, a directory that cannot be written");
  check(walk(base + "/none") == 2 && says("refused (no-folder-to-write): " + base + "/none, in which " + base + "/none/isk is to be made, is not a folder"), "an absent one: the refusal that names no fault of the path");
  if (close(open((base + "/file").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600))) stop("create " + base + "/file");
  check(refused_on_the_way(walk(base + "/file")), "a file: refused on the way");
  make(base + "/closed", 0500);
  if (geteuid()) check(walk(base + "/closed") == 2 && says("refused (no-folder-to-write): "), "one without write permission");
  else printf("6: a directory without write permission: not shown: root can write in it\n");

  set_up(7, "the words of the rejection");
  make(base + "/wide", 0777);
  for (const char *option : {"-ivk", "-worst-case-table"}) check(refused_on_the_way(walk(base + "/wide", option), option), "the option that was given is named");
  check(walk(base + "/wide", "-isk", ", in which . is looked up, x") == 2 && says("refused (cp-commit-path)"), "a name that holds the words that are exchanged: the call ends");
  make(base + "/wide/keys");
  check(walk(base + "/wide/./keys") == 2 && says(base + "/wide, in which . is looked up, can be written"), "a \".\" in the middle of the path: the refusal keeps the walk's own words for it");

  set_up(8, "the try on a file system that can do all");
  check(call([&] {
          if (reach_by_parts(base, base + "/isk", "-isk")) return 90;
          const int before = open_descriptors();
          if (three_are_blocked()) return 93;
          calls_seen = calls_blocked = 0;
          const int status = try_the_file_system(base + "/isk");
          if (calls_seen < 4 || calls_blocked != calls_seen || three_are_blocked()) return 94;
          return status ? status : open_descriptors() != before ? 91 : road().kept.size() != 1 || !road().made.empty() ? 92 : 0;
        }) == 0 && said.empty(), "it passes without a line, keeps no descriptor and lists nothing; SIGINT, SIGTERM and SIGHUP are blocked at each of its calls and not before or afterwards");
  check(names_in(base).empty(), "nothing is left");
  make(base + "/sticky", 01777);
  check(try_in(base + "/sticky") == 0 && names_in(base + "/sticky").empty(), "the same in a sticky directory");

  set_up(9, "a name of the try is taken");
  make(base + "/a"), make(base + "/a/isk.making"), make(base + "/b"), make(base + "/b/isk.making.making");
  check(try_in(base + "/a") == 2 && says(std::string("refused (") + HF_STOP_FILE_THERE + "): " + base + "/a/isk.making exists already") && says("nothing has to be removed first") && names_in(base + "/a") == "isk.making", "the working name: refused by that name; what has it stays");
  check(try_in(base + "/b") == 2 && says(std::string("refused (") + HF_STOP_FILE_THERE + "): " + base + "/b/isk.making.making exists already") && says("which rmdir removes (it removes nothing that is not empty)") && says("two new names") && !says("is left as it is") && names_in(base + "/b") == "isk.making.making", "the second name: refused by that name, with both roads out; what has it stays, and the try's own directory is gone");

  set_up(10, "a step fails");
  const struct {
    const char *call;
    int error;
    const char *step;
  } steps[] = {{"mkdirat", EROFS, "created"}, {"flock", ENOLCK, "locked (flock())"}, {"renameat2", EINVAL, "renamed to a free name (RENAME_NOREPLACE)"}, {"fsync", EIO, "synced to the disk under its new name"}};
  for (const auto &step : steps) {
    const std::string parent = base + "/" + step.call;
    make(parent);
    failing = step.call, failing_with = step.error;
    const int status = try_in(parent);
    failing = "";
    check(refused_at(status, step.step) && says(strerror(step.error)) && says("is on a shared or network file system, give -isk and -ivk names on a local one"), std::string(step.call) + "(): refused, with the step and the system's reason");
    check(names_in(parent).empty() && !says("is left as it is"), std::string(step.call) + "(): nothing is left");
  }

  set_up(11, "the new directory does not keep its mode");
  made_with = 0750;
  const int status = try_in(base);
  made_with = 0;
  check(refused_at(status, "kept for the calling user alone") && says("right after its creation it belongs to another user or has permissions for group or others") && !says("Operation not permitted") && says("What has the name " + base + "/isk.making now is left as it is") && names_in(base) == "isk.making", "refused; the directory, which the run cannot tell to be its own, is left, and the line says so");

  set_up(12, "the account that ran sudo");
  if (geteuid()) {
    printf("12: not shown: it takes root\n");
  } else {
    // Under sudo, typed_sudo keeps the account that ran sudo (directories above may belong to it),
    // and the second check gives home to that account. Without sudo, it sets typed_sudo to `third`.
    const uid_t ran_sudo = WalkedPath::typed_sudo, third = ran_sudo == 12345 ? 54321 : 12345;
    const auto give_home_to = [&](uid_t account) {
      if (chown((base + "/home").c_str(), account, (gid_t)-1)) stop("give away " + base + "/home");
    };
    make(base + "/home"), make(base + "/home/keys");
    give_home_to(third);
    check(refused_on_the_way(walk(base + "/home/keys")), "a directory on the way that belongs to a third account: refused");
    if (ran_sudo) give_home_to(ran_sudo);
    else WalkedPath::typed_sudo = third;
    check(walk(base + "/home/keys") == 0 && walk(base + "/home") == 0, "a directory that belongs to the account that ran sudo: taken, on the way and as the directory itself");
    give_home_to(0); // nothing that the test leaves behind belongs to another account
    WalkedPath::typed_sudo = ran_sudo;
  }

  printf("walked_parents_test: %d checks, %d differ: %s\n", checks, differ, differ ? "fails" : "passes");
  return differ ? 1 : 0;
}
