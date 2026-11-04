// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests cp_commit's file routines (hf_commit_making.h, hc_create_file.h in prover/hash_commitment/)
// when a directory (`folder` in identifiers) is exchanged while a run works in it. Whatever the
// run then writes or removes must be in the directory that it created or opened. Below what has
// taken that directory's name, the same names must remain, each with its type, mode, modification
// time and bytes (unchanged(), cases 4 to 19; cases 2, 3 and 21 count the names, and 21 compares
// its one file's bytes). The test does not compare the mode and time of that directory itself, nor,
// where a symbolic link has taken the name, the link: only its target. No GPU is needed, and
// nothing depends on timing: a case makes its exchange itself, between two calls or in a wrapper
// around a system call. An exchange moves the run's directory to <name>.away and puts a decoy
// (DecoyKind) under its name.
// For this test program alone, the #define lines below redirect nine system calls to wrappers
// while the two headers are compiled; no build of cp_commit does this.
// Not covered: an exchange, between a directory's creation and its opening, for an empty directory
// of the same user that is closed to others (cases 2, 3 and 21 cover a symbolic link, a directory
// open to others, a closed one with a file); an exchange inside rmdir_if_same() or rename_to(),
// between the look at a name and the removal or rename; what cp_commit only reads, which it looks
// up by whole paths; the file operations that prover/hash_commitment/hf_commit.cu and the sources
// that it includes make themselves (case 4 raises the column count of the first table, the anchor,
// as they do).
//   exchanged_directories_test <a new directory>
// It creates that directory and works below it. It removes nothing but through the tested routines,
// except one setup file (set_up_existing()). It sends no signal to another process; case 6's child
// sets its limit on a core file's size to 0 and calls abort().
// Last line: "exchanged_directories_test: <n> checks, 0 differ: passes", status 0; or "... <k> differ: fails", status 1.
// Build and run, from prover/: tests/unit.sh both hash_commitment/tests/exchanged_directories_test.cpp <directory> <a new directory>
#include "common/setting.h"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "hash_commitment/hf_format.h"
#include "common/cp_files.h"

// The routines' system calls at which a case can make its exchange.
enum CallKind { CREATION, OPENING, REMOVAL };
static std::function<void()> exchange;
static CallKind exchange_kind;
static int exchange_number;
// Makes `function` run once: right after the routines' successful creation of a directory, or
// opening of anything, number `number` (from 0), or right before their removal number `number`.
static void exchange_at(CallKind kind, int number, const std::function<void()> &function) {
  exchange_kind = kind;
  exchange_number = number;
  exchange = function;
}
static void reached(CallKind kind) {
  if (!exchange || kind != exchange_kind || exchange_number-- > 0) return;
  const std::function<void()> now = exchange;
  exchange = nullptr;
  now();
}
static int test_mkdir(const char *name, mode_t mode) {
  const int failed = mkdir(name, mode);
  if (!failed) reached(CREATION);
  return failed;
}
static int test_mkdirat(int directory, const char *name, mode_t mode) {
  const int failed = mkdirat(directory, name, mode);
  if (!failed) reached(CREATION);
  return failed;
}
static int test_open(const char *name, int flags, mode_t mode = 0) {
  const int fd = open(name, flags, mode);
  if (fd >= 0) reached(OPENING);
  return fd;
}
static int test_openat(int directory, const char *name, int flags, mode_t mode = 0) {
  const int fd = openat(directory, name, flags, mode);
  if (fd >= 0) reached(OPENING);
  return fd;
}
static DIR *test_opendir(const char *name) {
  DIR *const list = opendir(name);
  if (list) reached(OPENING);
  return list;
}
static int syncs_fail = 0; // set by a case: how many of the routines' next fsync() fail
static int test_fsync(int fd) {
  if (syncs_fail <= 0) return fsync(fd);
  syncs_fail--;
  errno = EIO;
  return -1;
}
static int test_unlinkat(int directory, const char *name, int flags) {
  reached(REMOVAL);
  return unlinkat(directory, name, flags);
}
static int test_rmdir(const char *name) {
  reached(REMOVAL);
  return rmdir(name);
}
static int test_unlink(const char *name) {
  reached(REMOVAL);
  return unlink(name);
}
// Eight of the nine wrappers above make the real call and can run a case's exchange right before or
// after it; test_fsync() makes the real call unless `syncs_fail` is above 0. hf_commit_making.h
// includes hc_create_file.h.
// The headers that declare these functions and the program's other headers are included above, so
// the macros change only the two headers under test. Neither calls mkdir(), opendir(), rmdir() or
// unlink(): if a routine came to call one, by a whole path, the exchange would still happen there,
// and the case would report the result.
#define mkdir test_mkdir
#define mkdirat test_mkdirat
#define open test_open
#define openat test_openat
#define opendir test_opendir
#define fsync test_fsync
#define unlinkat test_unlinkat
#define rmdir test_rmdir
#define unlink test_unlink
#include "hash_commitment/hf_commit_making.h"
#undef mkdir
#undef mkdirat
#undef open
#undef openat
#undef opendir
#undef fsync
#undef unlinkat
#undef rmdir
#undef unlink
using namespace hf::making;

static std::string top; // the directory that main() creates; every path of the test is below it
// File contents: 32 bytes, the size of the seed's file. The cleanup overwrites a file that the run
// has created with zeros before it removes the name. So a decoy's file that is overwritten, and
// not removed, differs too.
static const std::string OURS(32, 'o'), DECOY(32, 'd');
static int checks = 0, differ = 0;

// The current case, set by set_up(). DecoyKind: what an exchange puts in the place of the run's
// directory; LINK is a symbolic link to a directory. For the first two, set_up() fills the
// directory: make_secret(), make_public().
enum DecoyKind { LINK, DIRECTORY, EMPTY_DIRECTORY };
static const DecoyKind DECOY_KINDS[] = {LINK, DIRECTORY, EMPTY_DIRECTORY};
static DecoyKind decoy_kind;
static std::string label;                    // begins each check's name: the case's number and the decoy's kind
static std::string base, parent, isk, ivk;   // the case's directory; in it the parent of the two key directories; their final names
static std::string secret_decoy, public_decoy;

static void check(bool good, const std::string &what) {
  checks++;
  if (!good) differ++;
  printf("%s: %s%s\n", good ? "as expected" : "DIFFERS", label.c_str(), what.c_str());
}
[[noreturn]] static void stop(const std::string &what) {
  printf("the test cannot %s: %s\n", what.c_str(), strerror(errno));
  exit(1);
}

// The test's own file operations, by whole paths: its setup and the exchanges.
static void put(const std::string &path, const std::string &bytes) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0 || write(fd, bytes.data(), bytes.size()) != (ssize_t)bytes.size() || close(fd)) stop("write " + path);
}
static void make(const std::string &path) {
  if (mkdir(path.c_str(), 0700)) stop("create " + path);
}
static void move(const std::string &from, const std::string &to) {
  if (renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE)) stop("rename " + from);
}
// Puts `other` under the unused name `name`.
static void put_under(const std::string &name, const std::string &other) {
  if (decoy_kind != LINK)
    move(other, name);
  else if (symlink(other.c_str(), name.c_str()))
    stop("link " + name);
}
// Moves the run's directory to <name>.away and puts `other` under its name.
static void exchange_for(const std::string &name, const std::string &other) {
  move(name, name + ".away");
  put_under(name, other);
}
static bool exists(const std::string &path) {
  struct stat result;
  return !lstat(path.c_str(), &result);
}
static std::string text_of(const std::string &path) {
  std::string bytes;
  ::cp::read_file(path, bytes);
  return bytes;
}
// Every name below a directory with its type, mode, modification time and bytes, sorted. Empty also if there is no such directory.
static std::vector<std::string> snapshot(const std::string &directory) {
  std::vector<std::string> lines;
  std::error_code error;
  for (const auto &item : std::filesystem::recursive_directory_iterator(directory, error)) {
    const std::string path = item.path().string();
    struct stat result;
    if (lstat(path.c_str(), &result)) stop("examine " + path);
    lines.push_back(path.substr(directory.size()) + " " + std::to_string(result.st_mode) + " " + std::to_string(result.st_mtim.tv_sec) + "." +
                    std::to_string(result.st_mtim.tv_nsec) + " " + (S_ISREG(result.st_mode) ? text_of(path) : ""));
  }
  std::sort(lines.begin(), lines.end());
  return lines;
}
// After `other` was put under `name`: whether it is still a directory with this snapshot.
static bool unchanged(const std::string &name, const std::string &other, const std::vector<std::string> &before) {
  const std::string now = decoy_kind == LINK ? other : name;
  return std::filesystem::is_directory(std::filesystem::symlink_status(now)) && snapshot(now) == before;
}
// After exchange_for(name, ...): whether the run's directory is still there, and empty.
static bool emptied(const std::string &name) {
  return exists(name + ".away") && snapshot(name + ".away").empty();
}

static void make_with(const std::string &path, const std::vector<std::string> &files, const std::string &content) {
  make(path);
  for (const std::string &name : files) put(path + "/" + name, content);
}
static const std::vector<std::string> CONFIGURATION = {"policy.txt", "KL.bin", "layout.txt"}, LEAF_FILES = {"count.bin", "hcnodes.bin", "fresh.bin", "entry.bin", "wcard.bin"};
static ::cp::IskText isk_txt_with(uint32_t leaves) {
  ::cp::IskText text;
  text.version = 1, text.M = leaves, text.keep = 3, text.weights = "generated";
  return text;
}
// A secret key directory (`isk`) with every name that all_steps() creates in one, unfinished
// (.making) and final, and the names for cases 16 and 17. So a cleanup by whole paths would change
// it.
static void make_secret(const std::string &path, const std::string &content, uint32_t leaves = 4) {
  make_with(path, {"seed.bin", "seed.bin.making", "key.bin", "key.bin.making", "isk.txt.making", "isk.txt.making.123", "isk.txt.making.old"}, content);
  put(path + "/isk.txt", ::cp::isk_text(isk_txt_with(leaves)));
  make_with(path + "/config", CONFIGURATION, content);
  make_with(path + "/config.making", CONFIGURATION, content);
  make_with(path + "/anchor", {"count.bin", "hcnodes.bin"}, content);
  make(path + "/leaf");
  for (const char *leaf : {"/leaf/0", "/leaf/0.making", "/leaf/3.making"}) make_with(path + leaf, LEAF_FILES, content);
}
// The same for a public key directory (`ivk`).
static void make_public(const std::string &path, const std::string &content) {
  make_with(path, {"key.bin", "key.bin.making"}, content);
  make_with(path + "/leaf", {"0.bin", "0.bin.making"}, content);
}
// Starts a case: a new directory with the parent of the two key directories and two decoys in it.
static void set_up(int number, DecoyKind kind) {
  static const char *const NAMES[] = {"a link", "a directory", "an empty directory"};
  decoy_kind = kind;
  exchange = nullptr; // a child has used its copy of the case before; this process has not
  label = std::to_string(number) + ", exchanged for " + NAMES[kind] + ": ";
  base = top + "/" + std::to_string(number) + "." + std::to_string(kind);
  parent = base + "/parent", isk = parent + "/isk", ivk = parent + "/ivk";
  secret_decoy = base + "/secret.decoy", public_decoy = base + "/public.decoy";
  make(base), make(parent);
  if (kind == EMPTY_DIRECTORY)
    make(secret_decoy), make(public_decoy);
  else
    make_secret(secret_decoy, DECOY), make_public(public_decoy, DECOY);
}

// Runs cp_commit's side of a case in a child process, which prints to the file top/output. Returns the child's exit status, or
// 1000 plus the number of the signal that ended it.
static int run(const std::function<int()> &body) {
  const std::string log = top + "/output";
  fflush(stdout);
  const pid_t child = fork();
  if (child < 0) stop("start a process");
  if (!child) {
    const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (dup2(fd, 1) < 0 || dup2(fd, 2) < 0) _exit(99);
    exit(body()); // runs the cleanup, as cp_commit's exits do
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) stop("wait for its process");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1000 + WTERMSIG(status);
}
// A run that creates a commitment, in parts, in hf_commit.cu's order. It works in <isk>.making and
// <ivk>.making: beside(isk), beside(ivk).
static bool both_directories() {
  return !begin(nullptr, true, 1) && make_folder(beside(isk)) && make_folder(beside(ivk));
}
static bool configuration_and_seed() {
  const std::string configuration = beside(::cp::isk_config(beside(isk)));
  if (!make_folder(configuration)) return false;
  for (const std::string &name : CONFIGURATION)
    if (!write_new(configuration + "/" + name, OURS.data(), OURS.size())) return false;
  return write_new(beside(::cp::isk_seed(beside(isk))), OURS.data(), OURS.size());
}
static bool key_step() {
  const std::string secret = beside(isk), key = ::cp::ivk_key(beside(ivk));
  if (!make_folder(::cp::isk_anchor_dir(secret)) || !write_new(::cp::isk_anchor_count(secret), "", 0) || !write_new(beside(key), OURS.data(), OURS.size()) ||
      !write_new(::cp::isk_anchor_nodes(secret), OURS.data(), OURS.size()) || !write_new(beside(::cp::isk_key(secret)), OURS.data(), OURS.size()) || !write_new(beside(::cp::isk_txt(secret)), OURS.data(), OURS.size()))
    return false;
  for (const std::string &goal : {::cp::isk_seed(secret), ::cp::isk_config(secret), ::cp::isk_key(secret), ::cp::isk_txt(secret)})
    if (!takes_its_name(goal)) return false;
  if (!public_file_takes_its_name(key)) return false;
  step_done();
  return true;
}
static bool leaf_step() {
  const std::string secret = beside(isk), open = beside(ivk), folder = ::cp::isk_leaf_dir(secret, 0);
  if (!make_folder(::cp::isk_leaves(secret)) || !make_folder(::cp::ivk_leaves(open)) || !make_folder(beside(folder)) || hold_leaf(beside(folder)) < 0) return false;
  for (const std::string &name : LEAF_FILES)
    if (!write_new(beside(folder) + "/" + name, OURS.data(), OURS.size())) return false;
  if (!sync_dir(beside(folder)) || !takes_its_name(folder)) return false;
  step_done();
  if (!zero_times(::cp::ivk_key(open)) || !publish_entry(::cp::ivk_entry(open, 0), OURS)) return false;
  step_done();
  return true;
}
static bool all_steps() { return both_directories() && configuration_and_seed() && key_step() && leaf_step(); }
static const std::vector<std::string> WHOLE_SECRET = {"/seed.bin", "/key.bin", "/isk.txt", "/config/policy.txt", "/anchor/count.bin", "/leaf/0/entry.bin"};
static bool has_all(const std::string &directory, const std::vector<std::string> &names) {
  return std::all_of(names.begin(), names.end(), [&](const std::string &name) { return exists(directory + name); });
}

// 1. No exchange: a run that fails after its last step. The control for the cases below.
static void test_cleanup_without_an_exchange() {
  set_up(1, DIRECTORY);
  label = "1, no exchange: ";
  check(run([] { return all_steps() ? 4 : 90; }) == 4, "the run comes to its last step and exits with status 4");
  check(snapshot(parent).empty(), "nothing of the run is left");
}
// 2. The exchange: for a link to an empty directory, between the creation of <isk>.making and its
// opening. If the run followed the link, it would write its first files and end without a cleanup.
static void test_a_link_before_the_opening() {
  set_up(2, LINK);
  const std::string target = base + "/target";
  make(target);
  exchange_at(CREATION, 0, [&] { exchange_for(beside(isk), target); });
  const int status = run([] {
    if (!both_directories()) return 4;
    configuration_and_seed();
    _exit(9);
  });
  check(status == 4, "the link is not followed: the run stops with status 4");
  check(snapshot(target).empty(), "nothing is written where the link leads");
}
// 3. As case 2, for a directory of the same user with mode 0755. keep_open() rejects a new
// directory with any permission for group or others: with write permission another account could
// put links under the inner names.
static void test_an_open_directory_before_the_opening() {
  set_up(3, DIRECTORY);
  const std::string open_to_others = base + "/open";
  make(open_to_others);
  if (chmod(open_to_others.c_str(), 0755)) stop("change the mode of " + open_to_others);
  exchange_at(CREATION, 0, [&] { exchange_for(beside(isk), open_to_others); });
  const int status = run([] {
    if (!both_directories()) return 4;
    configuration_and_seed();
    _exit(9);
  });
  check(status == 4, "the directory is not taken: the run stops with status 4");
  check(exists(beside(isk)) && snapshot(beside(isk)).empty(), "nothing is written into it, and it is not removed");
}
// 21. As case 2, for a directory of the same user that is closed to others and has a file in it.
// By owner and mode it looks like the run's own; keep_open() rejects it because a directory that
// was just created is empty.
static void test_a_closed_directory_with_a_file_before_the_opening() {
  set_up(21, DIRECTORY);
  const std::string closed = base + "/closed";
  make(closed);
  put(closed + "/mine.txt", DECOY);
  exchange_at(CREATION, 0, [&] { exchange_for(beside(isk), closed); });
  const int status = run([] {
    if (!both_directories()) return 4;
    configuration_and_seed();
    _exit(9);
  });
  check(status == 4, "the directory is not taken: the run stops with status 4");
  check(snapshot(beside(isk)).size() == 1 && text_of(beside(isk) + "/mine.txt") == DECOY, "nothing is written into it, its file keeps its bytes, and it is not removed");
}
// 4. The exchange: right after both directories are made. The run writes its first files, raises
// the first table's column count (the length of anchor/count.bin) and ends without a cleanup.
static void test_what_is_written(DecoyKind kind) {
  set_up(4, kind);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([] {
    if (!both_directories()) return 90;
    exchange_for(beside(isk), secret_decoy);
    if (!configuration_and_seed() || !key_step()) _exit(91);
    const std::string anchor_folder = ::cp::isk_anchor_dir(beside(isk)); // as once_make() and entry_make() raise the first table's count
    _exit(::cp::count_first_table_columns(via(anchor_folder).fd, via(anchor_folder).name, 3, 100).say == ::cp::Counts::ROOM ? 9 : 92);
  });
  check(status == 9, "the run goes on, and raises the first table's count");
  check(text_of(beside(isk) + ".away/anchor/count.bin").size() == 3, "the count that has grown is the run's own");
  check(unchanged(beside(isk), secret_decoy, before), "the decoy is unchanged");
  check(has_all(beside(isk) + ".away", {"/seed.bin", "/key.bin", "/isk.txt", "/config/policy.txt", "/anchor/count.bin"}), "the run's files are in the directory that it made");
}
// 5 to 9. The exchange: before the cleanup of a run that fails. `steps`: how far the run has come.
// `by_abort`: it ends by abort(), and the signal handler cleans up. `public_directory`:
// <ivk>.making is exchanged, not <isk>.making.
static void test_cleanup(int number, DecoyKind kind, bool (*steps)(), bool by_abort, bool public_directory) {
  set_up(number, kind);
  const std::string name = beside(public_directory ? ivk : isk), decoy = public_directory ? public_decoy : secret_decoy, other = beside(public_directory ? isk : ivk);
  const std::vector<std::string> before = snapshot(decoy);
  const int status = run([&] {
    if (!steps()) return 90;
    exchange_for(name, decoy);
    if (by_abort) {
      const struct rlimit none = {0, 0};
      setrlimit(RLIMIT_CORE, &none); // the abort is wanted: at 0 the kernel writes no core file
      abort();
    }
    return 4;
  });
  check(status == (by_abort ? 1000 + SIGABRT : 4), by_abort ? "the run ends by SIGABRT" : "the run exits with status 4");
  check(unchanged(name, decoy, before), "the decoy is unchanged");
  check(emptied(name), "the run's directory is emptied");
  check(!exists(other), "the run's other directory is removed");
}
// A decoy for the parent: it has directories under the run's two working names.
static std::string make_parent_decoy() {
  const std::string path = base + "/parent.decoy";
  make(path);
  move(secret_decoy, beside(path + "/isk")), move(public_decoy, beside(path + "/ivk"));
  return path;
}
// 10. The exchange: the parent of both directories, before the cleanup of a run that fails.
static void test_cleanup_below_another_parent(DecoyKind kind) {
  set_up(10, kind);
  const std::string decoy = make_parent_decoy();
  const std::vector<std::string> before = snapshot(decoy);
  const int status = run([&] {
    if (!all_steps()) return 90;
    exchange_for(parent, decoy);
    return 4;
  });
  check(status == 4, "the run exits with status 4");
  check(unchanged(parent, decoy, before), "the decoy is unchanged");
  check(emptied(parent), "both of the run's directories are removed from the parent that it opened");
}
// 11. The exchange: the parent, before the two last renames of a run that ends well.
static void test_the_last_renames_below_another_parent(DecoyKind kind) {
  set_up(11, kind);
  const std::string decoy = make_parent_decoy();
  const std::vector<std::string> before = snapshot(decoy);
  const int status = run([&] {
    if (!all_steps()) return 90;
    exchange_for(parent, decoy);
    if (!top_folders_take_their_names(isk, ivk)) return 4;
    run_done();
    return 0;
  });
  check(status == 0, "the run ends well");
  check(unchanged(parent, decoy, before), "the decoy is unchanged: nothing in it is renamed");
  check(has_all(parent + ".away/isk", WHOLE_SECRET) && exists(parent + ".away/ivk/leaf/0.bin"), "the commitment has its final names in the parent that the run opened");
}
// 12. ivk's final name is taken, so the last renames fail and isk is renamed back. Then a decoy is
// put under isk's final name, which is on the cleanup list (`Road::made`) by then.
static void test_a_decoy_under_a_final_name(DecoyKind kind) {
  set_up(12, kind);
  const std::vector<std::string> before = snapshot(secret_decoy);
  make_with(ivk, {"key.bin"}, "taken");
  const int status = run([&] {
    if (!all_steps()) return 90;
    if (top_folders_take_their_names(isk, ivk)) return 91;
    put_under(isk, secret_decoy);
    return 4;
  });
  check(status == 4, "the last renames fail, and the run exits with status 4");
  check(unchanged(isk, secret_decoy, before), "the decoy is unchanged");
  check(text_of(ivk + "/key.bin") == "taken", "the directory that had ivk's name is unchanged");
  check(!exists(beside(isk)) && !exists(beside(ivk)), "both of the run's directories are removed");
}
// 13. write_a_new_file() with a whole path cannot sync its file, overwrites it and removes it. The exchange: the file's directory, before that removal.
static void test_a_file_that_cannot_be_synced(DecoyKind kind) {
  set_up(13, kind);
  const std::string directory = base + "/records";
  make(directory);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([&] {
    syncs_fail = 1; // the sync after the overwriting succeeds
    exchange_at(REMOVAL, 0, [&] { exchange_for(directory, secret_decoy); });
    return ::hc::write_a_new_file(directory + "/seed.bin", OURS.data(), OURS.size()) ? 91 : 4;
  });
  check(status == 4, "the call fails");
  check(unchanged(directory, secret_decoy, before), "the decoy is unchanged");
  check(emptied(directory), "the file is removed from the directory in which it was created");
}
// 14. write_new() cannot sync the seed's file. The exchange: <isk>.making, before the file's removal.
static void test_a_seed_that_cannot_be_synced(DecoyKind kind) {
  set_up(14, kind);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([&] {
    if (!both_directories()) return 90;
    syncs_fail = 1; // the sync after the overwriting succeeds
    exchange_at(REMOVAL, 0, [&] { exchange_for(beside(isk), secret_decoy); });
    return write_new(beside(::cp::isk_seed(beside(isk))), OURS.data(), OURS.size()) ? 91 : 4;
  });
  check(status == 4, "the call fails, and the run exits with status 4");
  check(unchanged(beside(isk), secret_decoy, before), "the decoy is unchanged");
  check(emptied(beside(isk)), "the run's directory is emptied");
}

// 15. The exchange: <isk>.making, before the two last renames of a run that has made everything.
static void test_the_last_renames_after_an_exchange(DecoyKind kind) {
  set_up(15, kind);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([&] {
    if (!all_steps()) return 90;
    exchange_for(beside(isk), secret_decoy);
    if (top_folders_take_their_names(isk, ivk)) return 91;
    return 4;
  });
  check(status == 4, "the last renames fail, and the run exits with status 4");
  check(unchanged(beside(isk), secret_decoy, before), "the decoy is unchanged, and keeps the name under which it was put");
  check(!exists(isk) && !exists(ivk), "no final name appears");
  check(emptied(beside(isk)) && !exists(beside(ivk)), "the run's directories are emptied; the one whose name still leads to it is removed");
}

// ---- cases 16 to 20: routines as a run on an existing commitment calls them ----
// cp_commit rejects such a run (`cp-commit-first-form-alone`).
static void set_up_existing(int number, DecoyKind kind, uint32_t decoy_leaves = 4) {
  set_up(number, kind);
  if (kind != EMPTY_DIRECTORY && decoy_leaves != 4) {
    if (unlink((secret_decoy + "/isk.txt").c_str())) stop("replace the decoy's isk.txt");
    put(secret_decoy + "/isk.txt", ::cp::isk_text(isk_txt_with(decoy_leaves)));
  }
  make_secret(isk, OURS), make_public(ivk, OURS);
}
// 16. sweep_texts(). The exchange: isk, right after the routine has opened it.
static void test_sweep_texts(DecoyKind kind) {
  set_up_existing(16, kind);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([&] {
    if (begin(nullptr)) return 90;
    exchange_at(OPENING, 0, [&] { exchange_for(isk, secret_decoy); });
    return sweep_texts(isk) ? 0 : 4;
  });
  check(status == 0, "the routine succeeds");
  check(unchanged(isk, secret_decoy, before), "the decoy is unchanged");
  check(!exists(isk + ".away/isk.txt.making.123"), "isk.txt.making.123 is removed from the directory that was opened");
  check(exists(isk + ".away/isk.txt.making.old") && exists(isk + ".away/isk.txt.making") && exists(isk + ".away/seed.bin"), "a name of another form stays");
}
// 17. sweep_leaf(). The exchange: isk/leaf/3.making, before the routine's first removal.
static void test_sweep_leaf(DecoyKind kind) {
  set_up_existing(17, kind);
  const std::string folder = isk + "/leaf/3.making", decoy = base + "/leaf.decoy";
  make_with(decoy, kind == EMPTY_DIRECTORY ? std::vector<std::string>() : LEAF_FILES, DECOY);
  const std::vector<std::string> before = snapshot(decoy);
  const int status = run([&] {
    if (begin(nullptr)) return 90;
    exchange_at(REMOVAL, 0, [&] { exchange_for(folder, decoy); });
    return sweep_leaf(isk, 3);
  });
  check(status == 0, "the routine succeeds");
  check(unchanged(folder, decoy, before), "the decoy is unchanged");
  check(emptied(folder), "the directory that was locked is emptied");
}
// 18. text_moves_on(). The exchange: isk, right after the routine's first opening. With 9 leaf
// tables (`leaves`) in the decoy's isk.txt the routine would reject the new text, which has 5, if
// it read that file.
static void test_text_moves_on(DecoyKind kind) {
  set_up_existing(18, kind, 9);
  const std::vector<std::string> before = snapshot(secret_decoy);
  const int status = run([&] {
    if (begin(nullptr)) return 90;
    exchange_at(OPENING, 0, [&] { exchange_for(isk, secret_decoy); });
    return text_moves_on(isk, isk_txt_with(5));
  });
  check(status == 0, "the routine succeeds");
  check(unchanged(isk, secret_decoy, before), "the decoy is unchanged");
  check(text_of(isk + ".away/isk.txt") == ::cp::isk_text(isk_txt_with(5)), "the new text is in the directory that was opened");
}
// 19. publish_entry() where a killed run has left the leaf file (`entry` in identifiers and
// messages) under its unfinished name. The exchange: ivk/leaf, before the routine's first removal.
static void test_publish_entry(DecoyKind kind) {
  set_up_existing(19, kind);
  const std::string folder = ivk + "/leaf", decoy = base + "/leaf.decoy";
  make_with(decoy, kind == EMPTY_DIRECTORY ? std::vector<std::string>() : std::vector<std::string>{"1.bin.making"}, DECOY);
  put(folder + "/1.bin.making", "a part");
  const std::vector<std::string> before = snapshot(decoy);
  const int status = run([&] {
    if (begin(nullptr)) return 90;
    exchange_at(REMOVAL, 0, [&] { exchange_for(folder, decoy); });
    if (!publish_entry(::cp::ivk_entry(ivk, 1), OURS)) return 4;
    step_done();
    return 0;
  });
  check(status == 0, "the routine succeeds");
  check(unchanged(folder, decoy, before), "the decoy is unchanged");
  check(text_of(folder + ".away/1.bin") == OURS && !exists(folder + ".away/1.bin.making"), "the entry is published in the directory that was opened");
}

// 20. shared_folder(), no exchange: a name that exists is taken if it is a directory, and not if it is a link to one.
static void test_shared_folder() {
  set_up_existing(20, DIRECTORY);
  label = "20, no exchange: ";
  if (symlink(secret_decoy.c_str(), (isk + "/link").c_str())) stop("link " + isk + "/link");
  check(run([&] { return begin(nullptr) ? 90 : shared_folder(isk + "/leaf") ? 0 : 4; }) == 0, "a directory under the name is taken");
  check(run([&] { return begin(nullptr) ? 90 : shared_folder(isk + "/link") ? 0 : 4; }) == 4, "a link to a directory under the name is not");
  check(run([&] { return begin(nullptr) ? 90 : shared_folder(isk + "/new") ? 0 : 4; }) == 0 && exists(isk + "/new"), "a free name is created");
}

int main(int argc, char **argv) {
  if (argc != 2 || mkdir(argv[1], 0700)) {
    fprintf(stderr, "usage: exchanged_directories_test <a new directory>\n");
    return 2;
  }
  char *const real = realpath(argv[1], nullptr);
  if (!real) return 2;
  top = real;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  test_cleanup_without_an_exchange();
  test_a_link_before_the_opening();
  test_an_open_directory_before_the_opening();
  test_a_closed_directory_with_a_file_before_the_opening();
  for (const DecoyKind kind : DECOY_KINDS) {
    test_what_is_written(kind);
    test_cleanup(5, kind, [] { return both_directories() && configuration_and_seed(); }, false, false);
    test_cleanup(6, kind, [] { return both_directories() && configuration_and_seed(); }, true, false);
    test_cleanup(7, kind, [] { return both_directories() && configuration_and_seed() && key_step(); }, false, false);
    test_cleanup(8, kind, all_steps, false, false);
    test_cleanup(9, kind, all_steps, false, true);
    test_cleanup_below_another_parent(kind);
    test_the_last_renames_below_another_parent(kind);
    test_a_decoy_under_a_final_name(kind);
    test_a_file_that_cannot_be_synced(kind);
    test_a_seed_that_cannot_be_synced(kind);
    test_the_last_renames_after_an_exchange(kind);
    test_sweep_texts(kind);
    test_sweep_leaf(kind);
    test_text_moves_on(kind);
    test_publish_entry(kind);
  }
  test_shared_folder();
  printf("exchanged_directories_test: %d checks, %d differ: %s\n", checks, differ, differ ? "fails" : "passes");
  return differ ? 1 : 0;
}
