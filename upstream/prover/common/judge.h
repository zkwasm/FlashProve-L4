// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Finds the verifier program cp_verify (`judge` in identifiers) and runs it: as a child process or,
// if main() begins with supervise_if_asked(), under a supervisor process.
// Only the committer program cp_commit uses this: the prover program cp_prove runs no other
// program. cp_commit uses it to plan, and to check its
// key file and leaf files: judge_says() in prover/hash_commitment/hf_commit_making.h calls the
// function of that name below. Host code, Linux only.
#pragma once
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <spawn.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
namespace judge {
// Returns the path of cp_verify in this program's directory, or "" if no executable file is there.
static std::string find_the_judge() {
  char self[4096];
  const ssize_t length = readlink("/proc/self/exe", self, sizeof self - 1);
  const std::string program(self, length > 0 ? (size_t)length : 0);
  const size_t cut = program.rfind('/');
  const std::string judge = cut == std::string::npos ? "" : program.substr(0, cut) + "/cp_verify";
  struct stat is;
  return judge.empty() || stat(judge.c_str(), &is) || !S_ISREG(is.st_mode) || access(judge.c_str(), X_OK) ? "" : judge;
}
// The list of running judge_says() calls, for a process that may have to exit while some run. Each
// call on it runs its child in a separate process group, because cp_verify forks a child for each
// of its stages (key, leaf, request), and killing the group ends these too. A call holds the lock
// `guard` while it starts its child and while it kills, reaps and removes it, so a process ID in
// `children` still belongs to its child when end_all() uses it. Threads may share one list.
struct Running {
  std::mutex guard;
  std::vector<pid_t> children;
  bool ended = false;
  // Time limit of one call, in milliseconds; 0: no limit. Set it while no call runs: it is read
  // without `guard`. A call waits beyond it, holding `guard`, for a child that is stuck in a system
  // call despite SIGKILL. With a supervisor the child is the supervisor, which is never stuck: a
  // stuck cp_verify is not waited for and is left behind.
  long long limit_ms = 0;
  // Kills the children's groups; from then on a call with this list starts no child and returns -1.
  // Gives up, without effect, after about two seconds without `guard`, so as not to delay an exit.
  void end_all() {
    std::unique_lock<std::mutex> lock(guard, std::defer_lock);
    for (int i = 0; i < 200 && !lock.try_lock(); i++) usleep(10000);
    if (!lock) return;
    ended = true;
    for (const pid_t child : children) kill(-child, SIGKILL);
  }
  // True once end_all() has taken effect: a call's -1 may then be end_all()'s doing.
  bool has_ended() {
    std::lock_guard<std::mutex> lock(guard);
    return ended;
  }
};
// The supervisor of one run of cp_verify. A caller of judge_says() that is killed outright (SIGKILL,
// the kernel's out-of-memory killer) cannot end the run that it has started, which would go on
// without a time limit, with its inputs in files in memory. So judge_says() starts
// the caller's program again, from /proc/self/exe, as the leader of a new process group, and that
// process, the supervisor, starts cp_verify in the group. It kills the group when the caller dies
// or when its own time limit passes. When cp_verify exits, the supervisor exits with the same exit
// code. A parent-death signal in cp_verify alone would not be enough: the kernel clears it in the
// children that cp_verify forks.
//
// True once main() has called supervise_if_asked(). Started again, a program without that call
// would run its ordinary main() a second time, so in that program judge_says() starts cp_verify
// directly.
inline bool main_can_supervise = false;
static constexpr const char *SUPERVISE = "--supervise-cp_verify";
// Signal handler of the supervisor: kills its own process group, which includes itself.
static void kill_own_group(int) { kill(-getpid(), SIGKILL); }
// main() calls this first: before it reads, prints or starts anything, and before any CUDA call.
// Returns -1 at once unless argv[1] is SUPERVISE; otherwise runs the supervisor and returns the
// exit code for main() to return. argv[2], ...: the caller's process ID; the time limit in seconds,
// 0 for none; the lowest descriptor that cp_verify does not inherit; cp_verify's path; its argument
// vector. If cp_verify cannot be started or does not exit normally, the supervisor dies by SIGKILL,
// so that judge_says() returns -1.
// The path must be the one that find_the_judge() returns in this process. So SUPERVISE, typed by
// hand, starts no program but the cp_verify in this program's directory.
[[maybe_unused]] static int supervise_if_asked(int argc, char **argv) {
  main_can_supervise = true;
  if (argc < 2 || strcmp(argv[1], SUPERVISE)) return -1;
  // kill_own_group() needs a group that this process leads; in process 1 it would signal all other
  // processes. If this process does not lead its group, lacks an argument or names a path other
  // than the one that find_the_judge() returns, it starts nothing and signals nobody. 2 is also
  // cp_verify's exit code for a wrong command line: it is never a verdict.
  if (argc < 7 || getpid() <= 1 || getpgrp() != getpid() || find_the_judge() != argv[5]) return 2;
  struct sigaction act {};
  act.sa_handler = kill_own_group;
  sigset_t both, inherited;
  sigemptyset(&both), sigaddset(&both, SIGTERM), sigaddset(&both, SIGALRM);
  // Public source: prctl(2), PR_SET_PDEATHSIG, https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html
  // The signal is sent when the thread that started this process ends; that thread stays in
  // judge_says() until the run is over. A caller that died before the prctl() call is no longer the
  // parent: hence getppid(). The caller's process ID, which arrives as text, is only compared: no
  // signal is sent to it. SIGTERM and SIGALRM stay blocked until cp_verify has been started, so the
  // group is killed only once it holds a child.
  // PR_SET_DUMPABLE: as in cp_verify, from this call on, other processes of the user cannot open
  // the inputs through /proc.
  bool fine = !sigprocmask(SIG_BLOCK, &both, &inherited) && !sigaction(SIGTERM, &act, nullptr) && !sigaction(SIGALRM, &act, nullptr) &&
              !prctl(PR_SET_DUMPABLE, 0) && !prctl(PR_SET_PDEATHSIG, SIGTERM) && getppid() == (pid_t)strtol(argv[2], nullptr, 10);
  if (fine) alarm((unsigned)strtoul(argv[3], nullptr, 10));
  // cp_verify inherits the descriptors and the signal mask that judge_says() gave this process,
  // nothing that this program may have opened before main(), and an empty environment.
  char *const no_environment[] = {nullptr};
  posix_spawn_file_actions_t acts;
  posix_spawnattr_t manner;
  pid_t child = -1;
  int status = 0;
  fine = fine && !posix_spawn_file_actions_init(&acts) && !posix_spawn_file_actions_addclosefrom_np(&acts, atoi(argv[4])) &&
         !posix_spawnattr_init(&manner) && !posix_spawnattr_setflags(&manner, POSIX_SPAWN_SETSIGMASK) &&
         !posix_spawnattr_setsigmask(&manner, &inherited) && !posix_spawn(&child, argv[5], &acts, &manner, argv + 6, no_environment);
  if (!fine) raise(SIGKILL); // nothing was started: only this process ends
  fine = !sigprocmask(SIG_UNBLOCK, &both, nullptr);
  while (fine && waitpid(child, &status, 0) < 0) fine = errno == EINTR;
  if (fine && WIFEXITED(status)) return WEXITSTATUS(status);
  kill_own_group(0);
  return 2; // not reached
}
// Runs `judge` with the argument vector `words` (program name first) and waits for it. Returns its
// exit code; LATE if it was killed, with its process group, at the time limit among->limit_ms; -1
// if it could not be started, did not exit normally, or its output could not be read. Neither -1
// nor LATE is a verdict on the inputs. Sets SIGCHLD to SIG_DFL for the whole process.
// inputs: up to five byte strings. `judge` can open them as INPUT[0] (its standard input),
// INPUT[1], ...; the caller puts these paths into `words`. So cp_verify checks exactly the bytes
// that this process has in memory. heard: null, or a string that receives all of cp_verify's
// standard output, which is not printed. among: the call's `Running` list, or null. With null there
// is no time limit, a separate process group only under a supervisor, and nobody to end the
// cp_verify that a supervisor leaves running if it is itself killed.
// Under a supervisor, `judge` must be the path that find_the_judge() returns in the supervisor.
// Otherwise (another path, or cp_verify removed or renamed since the caller found it) cp_verify is
// not started and the call returns 2.
static const char *const INPUT[5] = {"/dev/stdin", "/dev/fd/3", "/dev/fd/4", "/dev/fd/5", "/dev/fd/6"};
static constexpr int LATE = -3;
static int judge_says(const std::string &judge, std::vector<const char *> words, const std::vector<const std::string *> &inputs,
                      std::string *heard = nullptr, Running *among = nullptr) {
  // If SIGCHLD were ignored (inherited, or set by a library), the kernel would reap each child
  // itself: waitid() and waitpid() would find none, and end_all() could kill an unrelated process
  // group that has got the child's process ID. Each call resets it rather than rely on its caller.
  signal(SIGCHLD, SIG_DFL);
  words.push_back(nullptr);
  if (inputs.size() > 5) return -1;
  // A file in memory that contains `bytes`, or -1. Its descriptor is 10 or above: no dup2() in the
  // child overwrites an input not yet duplicated.
  // Public source: memfd_create(2), https://man7.org/linux/man-pages/man2/memfd_create.2.html
  auto hold = [](const std::string &bytes) {
    const int low = memfd_create("input", MFD_CLOEXEC), file = low < 0 ? -1 : fcntl(low, F_DUPFD_CLOEXEC, 10);
    if (low >= 0) close(low);
    size_t at = 0;
    while (file >= 0 && at < bytes.size()) {
      const ssize_t w = write(file, bytes.data() + at, bytes.size() - at);
      if (w <= 0) break;
      at += (size_t)w;
    }
    if (file < 0 || (at == bytes.size() && !lseek(file, 0, SEEK_SET))) return file;
    close(file);
    return -1;
  };
  std::vector<int> held;
  bool all_held = true;
  for (const std::string *bytes : inputs) held.push_back(hold(*bytes)), all_held = all_held && held.back() >= 0;
  const int kept = heard ? hold(std::string()) : -1;
  auto close_the_inputs = [&held] {
    for (const int file : held)
      if (file >= 0) close(file);
  };
  if (!all_held || (heard && kept < 0)) {
    close_the_inputs();
    if (kept >= 0) close(kept);
    return -1;
  }
  // The supervisor's limit is about ten seconds above this call's, so while this process lives the
  // loop below acts first. If the supervisor does act first, the call returns -1 and not LATE.
  const bool supervised = main_can_supervise;
  const std::string caller = std::to_string(getpid()), seconds = std::to_string(among && among->limit_ms ? among->limit_ms / 1000 + 10 : 0),
                    closed_from = std::to_string(std::max(3, (int)held.size() + 2));
  if (supervised) words.insert(words.begin(), {"/proc/self/exe", SUPERVISE, caller.c_str(), seconds.c_str(), closed_from.c_str(), judge.c_str()});
  char *const no_environment[] = {nullptr};
  posix_spawn_file_actions_t acts;
  posix_spawnattr_t manner;
  pid_t child = -1;
  int status = 0;
  fflush(stdout);
  fflush(stderr);
  // The child gets no open files except stdout, stderr and the inputs, as descriptors 0, 3, 4, ...
  // (without inputs, 0 is /dev/null). With `heard` its stdout is `kept`. cp_verify gets an empty
  // environment, here or from its supervisor. A supervisor gets this process's environment, in
  // which the loader has found this program's libraries.
  bool started = !posix_spawn_file_actions_init(&acts);
  const bool acts_made = started;
  if (inputs.empty()) started = started && !posix_spawn_file_actions_addopen(&acts, 0, "/dev/null", O_RDONLY, 0);
  for (size_t i = 0; i < held.size(); i++) started = started && !posix_spawn_file_actions_adddup2(&acts, held[i], i ? (int)i + 2 : 0);
  started = started && !(heard && posix_spawn_file_actions_adddup2(&acts, kept, 1)) &&
            !posix_spawn_file_actions_addclosefrom_np(&acts, std::max(3, (int)held.size() + 2));
  // Public source: posix_spawnattr_setpgroup(3p): with the flag POSIX_SPAWN_SETPGROUP and the
  // group 0 the child becomes the leader of a new process group.
  const bool manner_made = !posix_spawnattr_init(&manner);
  started = started && manner_made &&
            !((among || supervised) && (posix_spawnattr_setflags(&manner, POSIX_SPAWN_SETPGROUP) || posix_spawnattr_setpgroup(&manner, 0)));
  {
    std::unique_lock<std::mutex> lock;
    if (among) lock = std::unique_lock<std::mutex>(among->guard);
    started = started && !(among && among->ended) &&
              !posix_spawn(&child, supervised ? words[0] : judge.c_str(), &acts, &manner, (char *const *)words.data(),
                           supervised ? environ : no_environment);
    if (started && among) among->children.push_back(child);
  }
  close_the_inputs();
  bool ended = started;
  if (acts_made) posix_spawn_file_actions_destroy(&acts);
  if (manner_made) posix_spawnattr_destroy(&manner);
  bool late = false;
  if (among && started) { // waits without reaping (WNOWAIT); reaps the child only under `guard`
    // Public source: waitid(2), on WNOHANG: si_pid, set to 0, changes only if the child has ended.
    auto now_ms = [] {
      struct timespec t;
      clock_gettime(CLOCK_MONOTONIC, &t);
      return (long long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
    };
    const long long limit = among->limit_ms, begun = now_ms();
    for (siginfo_t how; ended && !late;) {
      how.si_pid = 0;
      if (waitid(P_PID, (id_t)child, &how, WEXITED | WNOWAIT | (limit ? WNOHANG : 0)) < 0)
        ended = errno == EINTR;
      else if (how.si_pid)
        break;
      else if (!(late = now_ms() - begun >= limit))
        usleep(20000);
    }
    std::lock_guard<std::mutex> lock(among->guard);
    // Kills the group: a late child, or whatever a child that was killed from outside left running.
    if (ended) kill(-child, SIGKILL);
    while (ended && waitpid(child, &status, 0) < 0) ended = errno == EINTR;
    among->children.erase(std::find(among->children.begin(), among->children.end(), child));
  }
  while (!among && ended && waitpid(child, &status, 0) < 0) ended = errno == EINTR;
  if (heard) {
    char piece[4096];
    ssize_t n = 0;
    heard->clear();
    while (ended && (n = pread(kept, piece, sizeof piece, (off_t)heard->size())) > 0) heard->append(piece, (size_t)n);
    close(kept);
    if (n < 0 && !late) return -1;
  }
  return late ? LATE : ended && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
} // namespace judge
