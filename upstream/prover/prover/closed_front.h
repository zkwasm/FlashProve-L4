// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Restricted output for the prover program cp_prove (the "closed front" in identifiers and
// documents), in a production build (`SERVING_BUILD`: prover/build.sh without an argument). A test
// build prints everything, unless CLOSED_KEEPS_THE_TEST_WORD is set.
// enter(), at the start of main(), restarts the program with a reduced environment and disables
// core files. Until close(), called just before the first secret is read, the program prints
// ordinary messages: the usage text, each rejection (`refuse()`; printed "refused") with its
// reason, the set-up lines. close() redirects stdout and stderr to /dev/null; from then on the
// program prints only a fixed list of lines, to its original stdout (prover/docs/closed_front.md
// lists them). The last, "p258 end <code>" from leave(), may add a name after the exit code. A name
// must be a string literal (prover/common/end_name.h; nothing checks it), not a number or a
// sentence, which could depend on hidden memory or on a request. Many lines begin with "p258", a
// prefix without meaning.
#pragma once
#include "common/end_name.h"
#include "common/reduced_env_start.h"
#include "common/refusal_test.h"
#include "common/setting.h"
#include "common/walked_path.h" // take_who_typed_sudo()
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// CLOSED_KEEPS_THE_TEST_WORD = 1 gives a test build restricted output. No target of
// prover/build.sh sets it. A production build has no test option, so no fault can be injected into
// it; this build shows what restricted output lets through when a self-check on secret random
// values fails. The restart in enter() then also keeps three test variables, which make the
// program lose random values, copies of them or kernel launches
// (prover/hiding/lost_coins_door.cuh, prover/hiding/lost_copies_door.cuh). This build runs only
// with -test-coins (random values, "coins" in names, from a test file). Its first line of output
// says that it is a test build, except where main() returns before enter(): if --setting, -h or
// --help is the only argument, or if the first is -clear.
#ifndef CLOSED_KEEPS_THE_TEST_WORD
#define CLOSED_KEEPS_THE_TEST_WORD 0
#endif
// The variables that the restart keeps, besides CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER.
#if CLOSED_KEEPS_THE_TEST_WORD
#define CLOSED_KEEPS "HOME", "TMPDIR", "ROWS_LOSE_COINS", "ROWS_LOSE_COPY", "ROWS_LOSE_LAUNCH"
#else
#define CLOSED_KEEPS "HOME", "TMPDIR"
#endif
// For root the restart keeps one more variable: SUDO_UID, in which sudo gives the user ID of the
// account that ran it. SUDO_UID is kept across the restart only. After the restart
// second_start_unless_nothing_is_hidden() (prover/prover/cp_prove.cu) reads it once through
// take_who_typed_sudo(), which removes it from the environment, so that from then on no library
// and no child process finds it. The ID stays in WalkedPath::typed_sudo, which Spool inherits
// (prover/common/walked_path.h), for the ownership checks on the paths given with -spool, -isk,
// -out and -clear (for -clear main() calls take_who_typed_sudo() itself, with no restart).
#define CLOSED_KEEPS_FOR_ROOT CLOSED_KEEPS, "SUDO_UID"
// Nothing in these sources is compiled under REFUSAL_TEST_WORDS (refusal_test.h). Set alone, it does not compile:
static_assert(
    !REFUSAL_TEST_WORDS || CLOSED_KEEPS_THE_TEST_WORD,
    "REFUSAL_TEST_WORDS=1 needs CLOSED_KEEPS_THE_TEST_WORD=1: test words may be compiled "
    "only into the build that runs on a test's coins alone");

// Calls `call` on each signal that asks the program to exit: on Linux, every signal that can be
// caught and whose default action terminates the process, except the seven that enter() treats as
// faults. SIGXCPU and SIGXFSZ are in the list: a run meets them under a supervisor's limits on CPU
// time and file size. With restricted output none may end the process without the last line or,
// after arm_ends() (ends.h), without the wipe of its secrets: enter() and arm_ends() both use this
// list. SIGSYS and SIGTRAP are among the faults (on_fault(): the last line at once, no wipe): as a
// rule the thread's own system call or instruction raises them, and after a handler that only sets
// a flag, such as arm_ends() installs, the thread would continue past a call that was never made.
template <class Call> static void each_signal_that_asks(Call call) {
  for (int s : {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGALRM, SIGUSR1, SIGUSR2, SIGPIPE, SIGXCPU, SIGXFSZ, SIGVTALRM, SIGPROF, SIGIO, SIGPWR, SIGSTKFLT}) call(s);
  for (int s = SIGRTMIN; s <= SIGRTMAX; s++) call(s);
}

// The line of say_that_bytes_are_kept(). A macro because leave() and stop() (ends.h) write the
// same text again.
#define CLOSED_FRONT_KEPT_LINE "p258 -keep-for-the-audit: this batch's 32 random bytes are kept in the secret folder. Until the audit has erased them, hiding is void for this batch and no service may start on this folder; hand out none of its responses unless the audit has said YES. Run tools/audit/audit_batch.sh.\n"
// The line that leave() writes while `wipe_is_due` is set.
#define CLOSED_FRONT_WIPE_LINE "p258 not every step of the wipe ran: GPU memory, host memory or staged response files may still hold secrets (prover/docs/ends-in-short.md)\n"

// The state of restricted output (the object `closed_front`, below). No field may depend on the
// outcome of a request: `hb` is printed after close(). `wipe_is_due` depends on where an exit began
// and on whether the wipe's steps succeeded.
struct ClosedFront {
  bool on = false;     // this build restricts its output
  bool closed = false; // close() has been called: the program holds, or is about to hold, a secret
  // The target in bits: the value of the line `sec` in the parameter file (policy.txt).
  int hb = -1;

  // Must be the first thing that main() does (before it cp_prove only answers the arguments listed
  // at CLOSED_KEEPS_THE_TEST_WORD, and returns): code before it runs twice, the first time in the
  // caller's environment. With restricted output it restarts the program with only CLOSED_KEEPS
  // (for root CLOSED_KEEPS_FOR_ROOT) and the two variables that select the GPU in the environment
  // (prover/common/reduced_env_start.h), saves stdout in `kept` for the lines written after close(),
  // disables core files, sets the umask to 077 and installs the handlers that print the last line.
  void enter(char **argv) {
    on = SERVING_OR(true, false) || CLOSED_KEEPS_THE_TEST_WORD;
    if (!on) return;
    // The restart comes first: the restarted program inherits descriptors 0, 1 and 2 as they are
    // here. A failed restart is reported below, once stdout is saved.
    const char *const no_second_start = geteuid() ? reduced_env_start(argv, {CLOSED_KEEPS}) : reduced_env_start(argv, {CLOSED_KEEPS_FOR_ROOT});
    kept = fcntl(1, F_DUPFD_CLOEXEC, 3);
    if (kept < 0) _exit(usage_end); // started without a stdout: there is nobody to tell
    // If the caller has closed descriptor 0 or 2, it is opened on /dev/null. Otherwise the next
    // file that the program opens would get that number, and a line for stderr would go into it.
    for (int fd : {0, 2})
      if (fcntl(fd, F_GETFD) < 0 && open("/dev/null", O_RDWR) != fd) _exit(usage_end);
    // Line buffering: a message is written before the next step begins, and so keeps its order
    // among the lines written to `kept`.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (CLOSED_KEEPS_THE_TEST_WORD && !say_closed("p258 THIS IS A TEST BUILD: ITS FILES ARE NOT PROOFS\n")) leave(usage_end);
    if (no_second_start) refuse("no-fresh-start", "the program cannot start a second time with a reduced environment", no_second_start);
    for (const char *name : {"HOME", "TMPDIR"})
      if (!names_a_folder(name))
        refuse("cp-prove-environment", name,
               "must be set and name a folder that exists, by an absolute path of at most 400 characters out of letters, digits and / . _ -. Of its caller's environment the program keeps HOME, TMPDIR, "
               "CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER, and drops every other variable (as root it reads SUDO_UID once, for the paths named with -spool, -isk, -out and -clear, and then drops it too)");
    struct rlimit none = {0, 0};
    if (setrlimit(RLIMIT_CORE, &none) || prctl(PR_SET_DUMPABLE, 0)) refuse("no-core-limit", "core files cannot be disabled", strerror(errno));
    umask(077);
    for (int s : {SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGSYS, SIGTRAP}) signal(s, on_fault);
    // Until arm_ends() (ends.h) takes these signals over, the program prints the last line and
    // exits on each that its parent did not ignore. No key is chosen and no weight is read before
    // that call.
    each_signal_that_asks([](int s) {
      struct sigaction old {};
      sigaction(s, nullptr, &old);
      if (old.sa_handler != SIG_IGN) signal(s, on_early_signal);
    });
    // Exit handlers registered later run first: the last line is the last thing the process does.
    on_exit(on_any_exit, nullptr);
  }

  // Redirects stdout and stderr to /dev/null, for this process and for children that inherit them.
  // Prover::run() (prover/prover/service.cuh) calls it once, just before it reads its first secret,
  // the seed. Until then the program has read no weight and, of the secret key directory (`isk`),
  // only isk.txt and the public files kept there. Of clients' requests it has printed nothing
  // except, with -in, a file name from the requests' directory in one rejection (cp-in-name). Two
  // more rejections depend on the names or the number of clients' files: cp-batch (in that
  // directory) and cp-spool (in the spool directory). A cp-spool rejection can also print up to 20
  // names from the top of the spool directory, the name of one entry of its in/ that is no request,
  // or the kind in its mark.txt. So what is left in the stdio buffers is
  // public and is flushed first. Returns false if a step fails; the caller then exits, holding no
  // secret yet. A copy of descriptor 1 or 2 that a library made before this call (CUDA is
  // initialized by then) still reaches the caller's output; whether a library keeps one has not
  // been examined.
  bool close() {
    if (!on || closed) return true;
    if (!say_closed("p258 closed: the program reads its secrets now. From here on it prints only the fixed lines listed in prover/docs/closed_front.md; prover/docs/ends-in-short.md explains the name in the last line\n")) return false;
    fflush(stderr);
    const int null = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (null < 0 || dup2(null, 1) < 0 || dup2(null, 2) < 0) return false;
    ::close(null); // above 2: enter() has seen to it that 0, 1 and 2 are open
    closed = true;
    wipe_is_due = true;
    return true;
  }

  // Prints one line: to the saved stdout after close(), to stdout before it and in a build without
  // restricted output. Returns false if the line does not fit the buffer or cannot be written: the
  // process may hold secrets, so the caller must then exit through stop() (prover/prover/ends.h),
  // which first runs the wipe (`g_end_wipe` there: it does not reach every secret). Pass only
  // constant text and public values; lines that a production build does not print ("measure:", the
  // long form of "batch") may have more, such as times. Nothing here checks the arguments.
  bool say_closed(const char *format, ...) {
    char b[768]; // the longest line, a batch's "measure:" line, has about 500 characters
    va_list a;
    va_start(a, format);
    const int n = vsnprintf(b, sizeof b, format, a);
    va_end(a);
    if (n <= 0 || n >= (int)sizeof b) return false;
    if (closed) return write(kept, b, (size_t)n) == n;
    return fwrite(b, 1, (size_t)n, stdout) == (size_t)n && !fflush(stdout);
  }
  // [keep-for-the-audit] Prints the constant line which says that the batch's 32 random bytes are
  // kept (option -keep-for-the-audit). After this call the line is printed once more when the
  // process exits. With restricted output leave() writes it, directly before the last line, so at
  // every exit through leave(), a crash (on_fault()) included. A build without restricted output
  // (a test build) prints it at every exit that runs stop() (prover/prover/ends.h) to its end, not
  // at a crash. At a SIGKILL or a loss of power nothing is printed.
  std::atomic<bool> bytes_are_kept{false};
  static_assert(std::atomic<bool>::is_always_lock_free, "leave() reads the flag in a signal handler");
  bool say_that_bytes_are_kept() {
    bytes_are_kept = true;
    return say_closed(CLOSED_FRONT_KEPT_LINE);
  }
  // Set by close(): the process is about to hold secrets, so the wipe (`g_end_wipe`,
  // prover/prover/ends.h) is due. stop() there clears the flag once every step of the wipe has run
  // and none has failed. While it is set, leave() writes CLOSED_FRONT_WIPE_LINE: at an exit whose
  // wipe skipped a step, failed in one or was given up at its time limit, and at a crash
  // (on_fault()), which runs no wipe. Where the last line is printed without that line (restricted
  // output, after close()), the program has counted every step as run; secrets can still be in
  // memory: see g_end_wipe. At a SIGKILL or a loss of power nothing is printed.
  std::atomic<bool> wipe_is_due{false};
  // Prints the line "p258 setting": five constants (prover/common/setting.h), then `hb`. Called
  // once, after the first batch's release (Prover::close_the_books()).
  bool print_lines() { return say_closed("p258 setting %d %d %d %d %d %d\n", EDEG, ETAU, ENJ, ETAUF, ENJW, hb); }

  // Prints the last line and exits with the code in it (`status`). Before it come, if due,
  // CLOSED_FRONT_WIPE_LINE (`wipe_is_due`) and then the line of say_that_bytes_are_kept(). Safe in a
  // signal handler: of the system's and the library's routines it calls only memcpy(), write() and
  // _exit().
  //   "p258 end 0"         all that was requested is done
  //   "p258 end 1"         anything else, if no name was set
  //   "p258 end 1 <name>"  anything else; <name> is the reason (prover/common/end_name.h)
  //   "p258 end 9 <name>"  a failed self-check on secret random values or keys
  // Of the argument `status` only 0 and "other" count: any other value gives 9 if the name begins
  // "own-test-" (names_a_self_check()) and 1 if not.
  // If "p258 end 0" cannot be written, the exit code is 1. Without restricted output (`kept` is -1)
  // leave() writes nothing and only exits. stop() (prover/prover/ends.h) passes its own name, as a
  // rule the one that it took when the exit began; every other caller uses the name set when it
  // calls.
  [[noreturn]] void leave(int status) { leave(status, g_end_name.load()); }
  [[noreturn]] void leave(int status, const char *name) {
    if (!status || (name && !*name)) name = nullptr; // "": an exit has begun with no name set
    if (status != 0) status = names_a_self_check(name) ? 9 : 1; // for every caller, also the handler of a fault that comes after a self-check has failed
    char line[sizeof CLOSED_FRONT_WIPE_LINE + sizeof CLOSED_FRONT_KEPT_LINE + 128];
    size_t n = 0;
    if (wipe_is_due) memcpy(line, CLOSED_FRONT_WIPE_LINE, n = sizeof CLOSED_FRONT_WIPE_LINE - 1);
    if (bytes_are_kept) memcpy(line + n, CLOSED_FRONT_KEPT_LINE, sizeof CLOSED_FRONT_KEPT_LINE - 1), n += sizeof CLOSED_FRONT_KEPT_LINE - 1;
    memcpy(line + n, "p258 end ", 9), n += 9;
    line[n++] = (char)('0' + status);
    if (name) line[n++] = ' ';
    for (const char *c = name; c && *c && n + 1 < sizeof line; c++) line[n++] = *c;
    line[n++] = '\n';
    const bool said = kept < 0 || write(kept, line, n) == (ssize_t)n; // one write, the lines before the last with it: no other thread's line falls into it or between them
    _exit(status || said ? status : 1);
  }
  // Public because arm_ends() (ends.h) recognizes this handler, to replace it with its own.
  static void on_early_signal(int);

private:
  static constexpr int usage_end = 1;
  int kept = -1;
  // A rejection in enter(): prints the reason, then exits with code 1 and `name` in the last line.
  [[noreturn]] void refuse(const char *name, const char *what, const char *why) {
    printf("cp_prove: refused (%s): %s: %s\n", name, what, why);
    fflush(stdout);
    name_the_end(name);
    leave(usage_end);
  }
  // Whether the environment variable names an existing directory ("folder" in identifiers and
  // messages) by an absolute path of at most 400 characters, all of them letters, digits or "/._-".
  static bool names_a_folder(const char *name) {
    const char *v = getenv(name);
    if (!v || v[0] != '/' || strlen(v) > 400) return false;
    for (const char *c = v; *c; c++)
      if (!isalnum((unsigned char)*c) && !strchr("/._-", *c)) return false;
    struct stat s;
    return !stat(v, &s) && S_ISDIR(s.st_mode);
  }
  static void on_fault(int);
  static void on_any_exit(int status, void *);
};
static ClosedFront closed_front;
inline void ClosedFront::on_fault(int) {
  closed_front.leave(1);
}
inline void ClosedFront::on_early_signal(int) {
  name_the_end("asked-to-end");
  closed_front.leave(1);
}
// Runs at an exit() or a return from main() before arm_ends() (ends.h). A successful run exits
// through stop(), so this exit reports failure. Pending output is flushed first: leave() calls
// _exit(), which flushes nothing.
inline void ClosedFront::on_any_exit(int, void *) {
  fflush(stdout);
  fflush(stderr);
  closed_front.leave(1);
}
