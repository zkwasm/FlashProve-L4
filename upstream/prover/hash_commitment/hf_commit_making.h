// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the committer program cp_commit creates a commitment's files: temporary names
// (<name>.making), cleanup after a failed run, renames, locks, runs of the verifier program
// cp_verify. Host code; include it after prover/weights_commitment/common.cuh.
// Most identifiers and messages say `folder` for directory, `refuse` for reject, "call" for a run.
// A commitment is a public key directory (`ivk`) and a secret one (`isk`). Its first weight table
// is the anchor. Leaf table k is a further table of the same weights with its own masks; its leaf
// file, ivk/leaf/<k>.bin (`entry` in identifiers), ties it to the anchor by a LINK proof.
// - A run works in <isk>.making and <ivk>.making, which get their final names by its last two
//   renames: top_folders_take_their_names(). Below, isk and ivk mean the two under either name.
// - Nothing of a table is written to isk or ivk before its key file (ivk/key.bin, the anchor's) or
//   leaf file, read from this process's memory, has passed cp_verify (judge_passes()). Written
//   earlier: directories, the seed, a copy of the configuration, -keep-for-the-audit's files, and
//   count.bin (counts a table's opened columns; created empty before the first opening raises it).
// - A run that fails, or gets one of five signals, removes both directories, name by name (`Road`;
//   what that can miss: stand_still(), make_folder()). After another signal, a crash or a power
//   failure both stay, with the seed; what the operator does with them: docs/faults.md, "A killed
//   or failed cp_commit" (see `Road`).
// cp_commit rejects a run on an existing commitment (`cp-commit-first-form-alone`), so the code
// here for such runs is not reached.
#pragma once
#include "common/cp_files.h"
#include "common/judge.h"
#include "common/walked_path.h"
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hc_create_file.h"
#include <atomic>
#include <dirent.h>
#include <exception>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
namespace hf {
namespace making {
// ---- cleanup ----
// `Road` is the cleanup's state. Its list, `made`, has the names that this run has created and may
// still have to remove, in the order of creation.
// - A name is added before the file or directory is created (make_folder(), write_new()), and the
//   creation fails if the name exists. So a listed name is this run's, unless another process can
//   write in the same directory. -worst-case-table's file is added after its creation: see below.
// - An entry below the list's count is never rewritten or moved: the signal handler can run at any
//   instruction, on any thread. A rename adds the new names before it happens and keeps the old
//   ones (rename_to()). One of the two is then absent, and an absent name is not an error.
// - The cleanup removes only the listed names, each by one unlinkat() (remove_name()): no
//   recursion, no pattern, no directory listing, no name formed at cleanup time. It uses them as
//   they were passed in, so a caller that passes relative names must not change its current
//   directory. cp_commit passes absolute paths (Opt::parse() in
//   prover/weights_commitment/arguments.h).
// - A file that the run has created is overwritten with zeros before its name is removed
//   (zeros_first()). `created` has the device and inode of each, read from the descriptor that
//   created it before anything is written to it, and the cleanup writes to no other file. A file
//   that cannot be overwritten keeps its name, and the run's last lines say so.
// - No listed name is created, renamed or removed by its whole path (two exceptions: see reach()).
//   The run keeps open, in `kept`, each directory that it has not created and in which it creates a
//   name (the parents of <isk>.making, of <ivk>.making and of -worst-case-table's file; cp_commit
//   opens the three at its start, part by part: reach_by_parts()), and each directory that it
//   creates in such a one (the two temporary directories; not try_the_file_system()'s trial
//   directory, which it closes again), opened right after mkdirat() without
//   following a link. via() turns a path into the deepest of these descriptors
//   and the rest of the path, for the ...at() system calls. So if someone who can write in a parent
//   renames one of the two, or puts a link or another directory under its name, the run still
//   writes into the directory that it created, and its cleanup still removes from that one. Below
//   the two, the rest of the path has several parts: they are looked up inside a directory of mode
//   0700 that the run has created, where only its owner and root can change a name. Each of the two
//   is itself removed only while its name still refers to the directory that was opened
//   (rmdir_if_same()). The descriptors stay open until the process ends, so no descriptor number is
//   reused for another directory.
// - A run that creates a commitment (`all_or_nothing`) drops no name before run_done(), but for a
//   name whose creation or rename failed (there at most the run's own empty directory stays). So
//   every end that the process makes itself (a return, exit(), abort(), an uncaught exception) and
//   every signal that leave_at_a_signal() catches, in any step (the anchor's, a leaf table's),
//   removes all that the run has created, also ivk/key.bin and leaf files that have their final
//   names.
//   Otherwise a failure in a leaf table's step would leave key.bin, earlier leaf files and isk with
//   the seed: a commitment that no run can complete. One file is listed only after its creation:
//   the one named with -worst-case-table, computed from the weights.
//   prover/hash_commitment/hf_commit.cu creates it as a new file (O_EXCL) before begin(), removes
//   it at every failed return until then, and lists it right after begin(), so a run that fails
//   later, or gets a caught signal, removes it too, overwritten first like the others. It stays
//   after a successful run and after exit status 7 (the worst-case check failed). An exit() or a
//   signal before it is listed also leaves it. Its directory is the caller's choice, and it is
//   removed by name alone: a failed run removes whatever file another process has put under that
//   name, without writing to it.
// - At any other signal (SIGKILL, SIGSEGV, SIGBUS, SIGQUIT ...), a crash or a power failure nothing
//   is removed, overwritten or printed: the two temporary directories (mode 0700) stay, with the
//   seed and, under -keep-for-the-audit, isk/audit/; -worst-case-table's file stays too. Between
//   the run's last two renames isk already has its final name, so <isk> and <ivk>.making stay
//   (limit (b) at top_folders_take_their_names()). No program removes them. They hold the seed and
//   should not be kept indefinitely. What the operator does is told in docs/faults.md, "A killed or
//   failed cp_commit": the two directories whose names end in .making can be removed, and nothing
//   else can; what is under <isk> is kept as a retired commitment is.
//   cp_commit creates no commitment while <isk>, <ivk> or a temporary directory exists. If only
//   temporary ones exist, refuse_what_is_left() rejects the run, and names what can be removed only
//   if it looks like a dead run's. A plain removal, unlike the cleanup (zeros_first()), does not
//   overwrite the seed's file first.
// - A run on an existing commitment drops a step's names at step_done().
struct Road {
  struct Made {
    std::string name;
    bool folder;
  };
  struct Open { // a directory that is kept open. Device and inode identify it. created: this run has made it
    std::string name;
    int fd;
    dev_t device;
    ino_t inode;
    bool created;
  };
  struct File { // a file that this run has created: see note_created()
    dev_t device;
    ino_t inode;
  };
  // The type of the three lists. Its storage is allocated before any handler is set (lay_out()),
  // and the count is stored, with release order, only after an entry is complete: a handler on any
  // thread reads only complete entries. Only the main thread may write (not checked). Not covered:
  // a main thread that goes on while a handler on another thread reads `made` can drop names and
  // overwrite their elements (see stand_still()).
  template <class Entry> class List {
    std::vector<Entry> room_;
    std::atomic<size_t> n_{0};

  public:
    void lay_out(size_t names) { room_.resize(names); }
    size_t size() const { return n_.load(std::memory_order_acquire); }
    bool empty() const { return !size(); }
    const Entry &operator[](size_t i) const { return room_[i]; }
    void push_back(Entry m) {
      const size_t n = n_.load(std::memory_order_relaxed);
      if (n == room_.size()) { // growing it would move entries that a handler may be reading
        fprintf(stderr, "cp_commit: internal error: the cleanup list is full\n");
        abort(); // the handler takes away what is listed
      }
      room_[n] = std::move(m);
      n_.store(n + 1, std::memory_order_release);
    }
    void cut_to(size_t n) { n_.store(n, std::memory_order_release); }
    void pop_back() { cut_to(n_.load(std::memory_order_relaxed) - 1); }
    void clear() { cut_to(0); }
  };
  List<Made> made;
  List<Open> kept; // a renamed directory is added again, under its new name (`rename_to`)
  List<File> created; // the only files to which the cleanup writes (zeros_first())
  // Allocated here and not in begin(), because cp_commit calls reach_by_parts(), which adds the
  // parents, before begin(). A run that creates a commitment adds at most seven:
  // the two parents, the parent of -worst-case-table's file, and the two temporary directories
  // under two names each.
  Road() { kept.lay_out(64), created.lay_out(1); } // created: -worst-case-table's file comes before begin(), which makes room for the others
  // `cleaner`: who cleans up. 0: not armed (before begin(), after run_done()); ARMED; t > 0: thread
  // t is cleaning up; done_by(t): thread t has finished, and the process is about to end. One
  // atomic word, so that claiming the cleanup and recording the claimant is one step. "Finished" is
  // not 0: a main thread that returns from long work after another thread's cleanup would read 0 as
  // "go on" and create files that nothing removes.
  static constexpr long ARMED = -1;
  static constexpr long done_by(long thread) { return -1 - thread; } // <= -2
  std::atomic<long> cleaner{0};
  std::atomic<bool> removing{false}; // take_away() is at work: see whose_cleanup()
  bool all_or_nothing = false; // the run creates a commitment: see above
  void (*wipe)() = nullptr; // `leave` calls it after the removal, to wipe the host's seed and keys
};
static_assert(std::atomic<long>::is_always_lock_free && std::atomic<size_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free, "the signals' handler reads Road::cleaner and the list's count: an atomic with a lock would not be allowed there");
static Road &road() {
  static Road &r = *new Road; // never destroyed: `leave` reads it after static destructors
  return r;
}
// Adds a file to the cleanup list.
static void made(const std::string &name) {
  road().made.push_back({name, false});
}
// Adds the open file `fd`, which this run has just created, to `created`. Call it before anything
// is written to the file. Returns false with errno set.
static bool note_created(int fd) {
  struct stat is;
  if (fstat(fd, &is)) return false;
  road().created.push_back({is.st_dev, is.st_ino});
  return true;
}
// Whether `is` has the device and inode of a file in `created`. The signal handler calls it.
static bool created_here(const struct stat &is) {
  const Road::List<Road::File> &created = road().created;
  for (size_t i = created.size(); i--;)
    if (created[i].device == is.st_dev && created[i].inode == is.st_ino) return true;
  return false;
}
// The parent directory of `path`.
static std::string folder_of(const std::string &path) {
  const size_t cut = path.rfind('/');
  return cut == std::string::npos ? std::string(".") : cut ? path.substr(0, cut) : std::string("/");
}
// The arguments of an ...at() system call for `path`: the descriptor of the deepest directory in
// `kept` above it, and the rest of the path; or AT_FDCWD and the whole path if none is above it.
// `name` points into `path`, which must outlive it. via() allocates no memory and takes no lock:
// the signal handler calls it.
struct Via {
  int fd;
  const char *name;
  const Road::Open *folder; // the kept directory, or null
};
static Via via(const std::string &path) {
  const Road::List<Road::Open> &kept = road().kept;
  Via deepest = {AT_FDCWD, path.c_str(), nullptr};
  for (size_t i = kept.size(), longest = 0; i--;) {
    const size_t n = kept[i].name.size();
    if (n > longest && path.size() > n + 1 && path[n] == '/' && !path.compare(0, n, kept[i].name)) deepest = {kept[i].fd, path.c_str() + n + 1, &kept[i]}, longest = n;
  }
  return deepest;
}
// The entry of `kept` with exactly the name `folder`, or null.
static const Road::Open *kept_open(const std::string &folder) {
  const Road::List<Road::Open> &kept = road().kept;
  for (size_t i = kept.size(); i--;)
    if (kept[i].name == folder) return &kept[i];
  return nullptr;
}
// Whether the open directory is empty. False also if it cannot be listed.
static bool holds_nothing(int fd) {
  const int twin = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  DIR *const list = twin < 0 ? nullptr : fdopendir(twin);
  if (!list) {
    if (twin >= 0) close(twin);
    return false;
  }
  bool nothing = true;
  while (const struct dirent *const entry = nothing ? (errno = 0, readdir(list)) : nullptr)
    if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) nothing = false;
  if (errno) nothing = false; // the listing broke off
  closedir(list);
  return nothing;
}
// Opens a directory and adds it to `kept`. created: the run has just created it. In that case a
// link under its name is not followed, and the directory must belong to the calling user, have no
// permissions for group or others and be empty (else EPERM): a directory of another account, or one
// with anything in it, that was put under the name since mkdirat() is not accepted. Returns false
// with errno set.
static bool keep_open(const std::string &folder, bool created) {
  const Via v = via(folder);
  const int fd = created ? openat(v.fd, v.name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW) : openat(v.fd, v.name, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  struct stat is;
  const bool seen = fd >= 0 && !fstat(fd, &is);
  if (seen && !(created && (is.st_uid != geteuid() || (is.st_mode & 077) || !holds_nothing(fd)))) {
    road().kept.push_back({folder, fd, is.st_dev, is.st_ino, created});
    return true;
  }
  const int why = seen ? EPERM : errno;
  if (fd >= 0) close(fd);
  errno = why;
  return false;
}
// Call it before a name is created in `folder`, which must exist. Afterwards via() looks such a
// name up from a descriptor: that of a directory that this run has created, at or above `folder`,
// or else that of `folder` itself, which is opened here at its first use. The process's current
// directory and "/" need none: AT_FDCWD stands for the first, which this program never changes,
// and nobody can rename the second. Returns false with errno set.
// It opens `folder` by its whole path and follows links. cp_commit has opened its three directories
// by reach_by_parts() before; a caller that brings another one here passes that walk by.
static bool reach(const std::string &folder) {
  if (folder == "." || folder == "/" || kept_open(folder)) return true;
  const Via v = via(folder);
  return (v.folder && v.folder->created) || keep_open(folder, false);
}
// Removes an empty directory, but only while its name still refers to the directory with this
// device and inode. A gap remains between that check and the removal. Returns 0, or -1 with errno
// set: ENOENT also if the name refers to something else by now. System calls only.
static int rmdir_if_same(const std::string &folder, dev_t device, ino_t inode) {
  const Via v = via(folder);
  struct stat is;
  if (fstatat(v.fd, v.name, &is, AT_SYMLINK_NOFOLLOW)) return -1;
  if (is.st_dev != device || is.st_ino != inode) {
    errno = ENOENT;
    return -1;
  }
  return unlinkat(v.fd, v.name, AT_REMOVEDIR);
}
// Removes one listed name. Returns 0, or a negative number with errno set. System calls only.
// A file is overwritten first (zeros_first()); if that fails it keeps its name, and the result is
// NOT_OVERWRITTEN. A directory in `kept` is removed by rmdir_if_same(). A listed directory that is
// not in `kept` is inside a directory that the run has created; or it is one of the two temporary
// directories between its listing and its opening (make_folder()), or its opening has failed. What
// is removed in these two cases is at most an empty directory, under a name that was free when it
// was listed.
enum { NOT_OVERWRITTEN = -2 };
static bool zeros_first(int at, const char *name, bool small_only);
static int remove_name(const Road::Made &m) {
  const Road::Open *const own = m.folder ? kept_open(m.name) : nullptr;
  if (own) return rmdir_if_same(m.name, own->device, own->inode);
  const Via v = via(m.name);
  if (!m.folder && !zeros_first(v.fd, v.name, false)) return NOT_OVERWRITTEN;
  return unlinkat(v.fd, v.name, m.folder ? AT_REMOVEDIR : 0);
}
// Prints why a listed name stays. failed: remove_name()'s result, with errno as that call left it.
static void say_why_it_stays(const std::string &name, int failed) {
  if (failed == NOT_OVERWRITTEN)
    fprintf(stderr, "cp_commit: %s could not be overwritten with zeros: %s. So it was not removed, and it may still have what this call wrote to it. "
                    "What to do with it: docs/faults.md, \"A killed or failed cp_commit\", the case of a killed call\n", name.c_str(), strerror(errno));
  else
    fprintf(stderr, "cp_commit: %s could not be removed: %s\n", name.c_str(), strerror(errno));
}
// whose_cleanup() lets only one thread clean up, and makes the others wait (how long: below). A
// thread that arrives while another cleans up (a second abort() or signal, an exception or an
// exit() on another thread) waits until the cleanup is finished, and only then goes on to end the
// process. Without the wait it would kill the process during the removal: seen in a test harness,
// where nearly all names stayed, the seed's file among them, and no message was printed.
// The wait ends five seconds after the cleaner's removals (take_away()). Those make only system
// calls, so they wait for no lock, and they can take longer than that: each gigabyte that
// zeros_first() overwrites takes seconds.
// If one of them never returns (a dead network disk), only a signal that is not caught, such as
// SIGKILL, ends the process.
// After its removals a cleaner in leave() takes locks (the heap's, stdio's) that the waiting thread
// may have held since before its signal, and without a limit neither would ever finish. Removing
// some thousand names takes milliseconds; each sync of a small file in zeros_first() takes a third
// of a millisecond on a local disk.
// The signal handler calls it. Besides the atomic it calls only pthread_self(), which is on POSIX's
// list of async-signal-safe functions (2.4.3), and nanosleep(), which Linux's signal-safety(7) does
// not have on that list.
// Returns MINE: this thread has now claimed the cleanup; AGAIN: it had claimed it before (an
// abort() inside its own cleanup, or leave() entered again through exit()); NOT: nothing to do (not
// armed, already done, or another thread's, after the wait).
static long this_thread() { return (long)pthread_self(); } // on Linux an address: above 0
enum class Turn { MINE, AGAIN, NOT };
static Turn whose_cleanup() {
  Road &r = road();
  const long me = this_thread();
  long was = Road::ARMED;
  if (r.cleaner.compare_exchange_strong(was, me)) return Turn::MINE;
  if (was == me) return Turn::AGAIN;
  const struct timespec a_while = {0, 10 * 1000 * 1000};
  for (int i = 0; i < 500 && r.cleaner.load() > 0; i += !r.removing.load()) nanosleep(&a_while, nullptr);
  return Turn::NOT;
}
// The main thread's side. Once another thread has claimed the cleanup, the main thread must create,
// rename and drop nothing more, during that cleanup and after it (an exit() on another thread runs
// further exit handlers after leave(), which can take long). So step_done(), make_folder(),
// write_new() and rename_to() begin with stand_still_if_another_cleans(), holding no lock, and the
// main thread then waits there until the process ends. After one minute it aborts, so that nothing
// hangs for ever (an exit handler might wait for this very thread).
// Narrowed, not closed: if another thread starts the cleanup (its own abort(), or a signal from
// outside that the system delivers to it) while the main thread is between that check and its
// system call, a file or a new name can appear after the first pass over the list. take_away()
// therefore goes over the list twice, which still misses a thread that is kept off the CPU at that
// point for longer than both passes take. The main thread does at most that one operation more.
[[noreturn]] static void stand_still() {
  const struct timespec a_while = {0, 100 * 1000 * 1000};
  for (int i = 0; i < 600; i++) nanosleep(&a_while, nullptr);
  abort(); // the handler finds the cleanup claimed or finished, and the process dies by SIGABRT
}
static void stand_still_if_another_cleans() {
  const long who = road().cleaner.load(), me = this_thread();
  if ((who > 0 && who != me) || (who < Road::ARMED && who != Road::done_by(me))) stand_still();
}
// Overwrites a file that this run has created with zeros before its name is removed: unlink() alone
// leaves the bytes in the file system's free blocks. Among these files are the seed, a leaf table's
// fresh.bin (system random bytes that are combined with the seed) and, under -keep-for-the-audit,
// the files of isk/audit/, of 32 bytes each; hcnodes.bin (a large secret file: a table's stored
// tree nodes, digests over salted leaves, some of which each opening publishes); and the file named
// with -worst-case-table (computed from the weights). On storage that writes a changed block
// elsewhere, as flash does, the old block can outlive the overwrite, and this is then worth little
// (hc::zeros_over() in prover/hash_commitment/hc_create_file.h).
// It writes only to a regular file with one hard link whose device and inode are in `created`: this
// is checked on the open file, which is opened without following a link.
// small_only: a file of more than 32 bytes is skipped.
// Returns false, with errno set, only if a file in `created` could not be opened, overwritten or
// synced. Only calls that a signal handler may make (see leave_at_a_signal()).
// With -sweep, remove_flat_folder() unlinks what a killed run has left of a leaf table, fresh.bin
// among it, without overwriting anything. See also docs/limits-all-rows.tsv, row R3.4.
static bool zeros_first(int at, const char *name, bool small_only) { // at: AT_FDCWD or a directory's descriptor
  struct stat is;
  if (fstatat(at, name, &is, AT_SYMLINK_NOFOLLOW) || !created_here(is) || (small_only && is.st_size > 32)) return true;
  const int file = openat(at, name, O_WRONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (file < 0) return false;
  const bool done = !fstat(file, &is) && (!S_ISREG(is.st_mode) || is.st_nlink != 1 || !created_here(is) || ::hc::zeros_over(file));
  const int why = errno;
  close(file);
  errno = why;
  return done;
}
// Removes what is listed. First one forward pass with zeros_first() over the files of at most 32
// bytes, so that the seed, created early, is overwritten early, and no such file waits for a large
// one, should the process be killed before the removal is finished. Then, last created first, one
// remove_name() per name, which overwrites the larger files, twice over the list (see
// stand_still()).
// System calls only: the handler needs that, and in leave() the cleaner then waits for no lock
// before the removal. Returns how many names could not be removed; an absent name does not count.
static size_t take_away() {
  Road &r = road();
  size_t stay = 0;
  r.removing = true;
  for (size_t i = 0, n = r.made.size(); i < n; i++)
    if (!r.made[i].folder) {
      const Via v = via(r.made[i].name); // one call: `kept` may grow between two, on another thread
      (void)zeros_first(v.fd, v.name, true); // a failure shows again below
    }
  for (int walk = 0; walk < 2; walk++)
    for (size_t i = r.made.size(); i--;)
      if (remove_name(r.made[i]) && errno != ENOENT && walk) stay++;
  r.removing = false;
  return stay;
}
// The cleanup when the process ends by itself. It is the exit handler, for an exit() anywhere;
// cp_commit also calls it after its run returns, because the process then ends by _exit(), which
// runs no handler. The removal comes before wipe() and before any message: those take the heap's
// and stdio's locks, and a thread that has held one since before its signal waits five seconds for
// this one and then ends the process (whose_cleanup()). `cleaner` stays this thread's until the
// end, so that after an abort() in here (wipe() frees heap memory) the handler goes on with the
// cleanup.
static void leave() {
  Road &r = road();
  if (whose_cleanup() != Turn::MINE) return; // AGAIN: its own recursion
  const size_t stay = take_away();
  if (r.wipe) r.wipe();
  if (!r.made.empty()) {
    if (stay)
      for (size_t i = r.made.size(); i--;) // again, to print why each one failed
        if (const int failed = remove_name(r.made[i]); failed && errno != ENOENT) say_why_it_stays(r.made[i].name, failed);
    fprintf(stderr, "cp_commit: the run has ended in the middle of a step: what that step had written is removed%s%s\n", r.all_or_nothing ? ", and all that the call had made before it" : "", stay ? ", but for the names above" : "");
  }
  r.cleaner = Road::done_by(this_thread()); // after the message: waiters may now end the process
}
// The cleanup at five signals. SIGABRT, whatever raises it: a failed guard (blocks/base.h) or CUDA
// call (prover/engine/device.cuh; out of GPU memory is one), a library's abort() on any thread,
// also inside the heap's functions, or a signal from outside. And the four by which a run is
// usually stopped from outside: SIGINT (Ctrl-C), SIGTERM (a scheduler), SIGHUP (the terminal
// closes) and SIGPIPE (`cp_commit | tee` whose reader dies). Neither abort() nor a signal runs an
// exit handler. Without a signal handler the seed's file stays on the disk: this was seen in a run
// that ended by abort(). No other signal is caught.
// As a handler must, it allocates and frees no memory, takes no lock and uses no stdio. unlinkat(),
// fstatat(), openat(), fstat(), write(), fsync(), close(), strlen(), signal() and raise() are on
// POSIX's list of async-signal-safe functions (2.4.3); on nanosleep() see whose_cleanup(). The keys
// in memory are not wiped (wipe_the_keys() in prover/hash_commitment/hf_commit.cu empties
// containers, which frees heap memory): the process dies at once. The message is one write(), so
// that no line of cp_verify or of another thread splits it, and is printed only if the list has a
// name, as in leave(). SIGPIPE is ignored first, unless it is the signal that came: the message
// may go into a pipe that has lost its reader; its write() then fails, and the process still dies
// by the signal that came.
// The process dies by the signal's default action, so its exit status is the signal's usual one (in
// a shell 134 for abort(); 130, 143, 129, 141 for the four); of two different signals, one inside
// the other's handler, the inner one's. What the handler can miss: see stand_still().
static void leave_at_a_signal(int which) {
  Road &r = road();
  if (which != SIGPIPE) signal(SIGPIPE, SIG_IGN); // for the message: see above
  if (whose_cleanup() != Turn::NOT) { // AGAIN: it was inside `leave` or this handler
    const size_t stay = take_away();
#define HF_SAID " in the middle of a step: what that step had written is removed by name\n"
#define HF_SAID_ALL " in the middle of a step: what that step had written is removed by name, and all that the call had made before it\n"
#define HF_NOT_ALL " in the middle of a step: NOT ALL that that step had written could be removed: look under the two folders' names, and under the same names with .making behind them\n"
#define HF_BY_ABORT "cp_commit: the run has ended by abort()"
#define HF_FROM_OUTSIDE "cp_commit: the run was ended by a signal (INT, TERM, HUP or PIPE)"
    static const char *const lines[2][3] = {{HF_BY_ABORT HF_SAID, HF_BY_ABORT HF_SAID_ALL, HF_BY_ABORT HF_NOT_ALL}, {HF_FROM_OUTSIDE HF_SAID, HF_FROM_OUTSIDE HF_SAID_ALL, HF_FROM_OUTSIDE HF_NOT_ALL}};
#undef HF_SAID
#undef HF_SAID_ALL
#undef HF_NOT_ALL
#undef HF_BY_ABORT
#undef HF_FROM_OUTSIDE
    const char *const line = lines[which != SIGABRT][stay ? 2 : r.all_or_nothing ? 1 : 0];
    if (!r.made.empty()) (void)!write(2, line, strlen(line)); // a single `write`: see above
    r.cleaner = Road::done_by(this_thread()); // after the message: waiters may now end the process
  }
  signal(which, SIG_DFL);
  raise(which);
}
// Arms the cleanup. Call it once, before the first name is listed. all_or_nothing: the run creates
// a commitment (see `Road`); leaves: how many leaf tables it creates at most. Returns 0 or 2.
// One of SIGINT, SIGTERM, SIGHUP and SIGPIPE that was ignored when the program started, as under
// nohup, stays ignored; sigaction() asks first, so that it is not caught even for a moment.
// Why the list's size suffices: the anchor's step adds 21 names (13 files and directories, eight
// of them again at their renames), isk/leaf and ivk/leaf 2, a leaf table's step 12;
// -keep-for-the-audit adds two, and one per leaf table; the last two renames add each name once
// more. The file named with -worst-case-table adds one more. At most 51, and 26 per leaf table.
// `created` gets the same size: the run creates fewer files than it lists names.
static int begin(void (*wipe)(), bool all_or_nothing = false, size_t leaves = 0) {
  if (atexit(leave)) return 2;
  const size_t names = 1024 + (all_or_nothing ? 32 * leaves : 0); // see above
  road().made.lay_out(names), road().created.lay_out(names); // no handler is set yet, so `created`'s one entry may move
  if (signal(SIGABRT, leave_at_a_signal) == SIG_ERR) return 2;
  for (int which : {SIGINT, SIGTERM, SIGHUP, SIGPIPE}) { // an ignored one stays ignored: see above
    struct sigaction was;
    if (sigaction(which, nullptr, &was) || (was.sa_handler != SIG_IGN && signal(which, leave_at_a_signal) == SIG_ERR)) return 2;
  }
  // An uncaught exception (out of memory) ends the process without running its exit handlers, so
  // the cleanup runs here.
  std::set_terminate([] {
    leave();
    _exit(4);
  });
  road().all_or_nothing = all_or_nothing, road().wipe = wipe, road().cleaner = Road::ARMED;
  return 0;
}
// A step is finished. A run on an existing commitment drops the step's names: they are final, and
// no later end removes them. (An abort() on another thread between the check and the clear() still
// removes them.) A run that creates a commitment keeps them until run_done().
static void step_done() {
  stand_still_if_another_cleans();
  if (!road().all_or_nothing) road().made.clear();
}
// The run has succeeded: all names are final, and the cleanup is disarmed, by one
// compare-and-exchange on `cleaner`. If another thread has claimed the cleanup before, the run has
// not succeeded: this thread prints no success message and waits in stand_still(). (With a separate
// check and clear(), an abort() on another thread between the two could remove everything while the
// main thread printed "done" and exited with status 0.)
// After this call nothing is removed: a SIGABRT then ends the process with status 134 and a
// complete commitment. Nor does leave() then call wipe(). cp_commit wipes its keys before this
// call and again after leave() (hf_commit.cu); an exit() elsewhere after this call would skip that
// second wipe.
static void run_done() {
  long was = Road::ARMED;
  if (!road().cleaner.compare_exchange_strong(was, 0) && was) stand_still();
  road().made.clear();
}

// ---- names, directories, files, renames ----
// Prints a rejection; `name` is its fixed identifier, such as cp-commit-ivk-holds. Returns 2,
// cp_commit's exit status for a rejected run.
static int refuse(const char *name, const std::string &why) {
  fprintf(stderr, "cp_commit: refused (%s): %s\n", name, why.c_str());
  return 2;
}
// The temporary name of `goal`.
static std::string beside(const std::string &goal) {
  return goal + ".making";
}
// Where the leaf file is stored once it has passed cp_verify: in the leaf table's directory in isk.
// ivk's file is a copy. A run on an existing commitment that finds isk/leaf/<k> with this file, and
// no leaf file in ivk, publishes these bytes and does not rebuild the table. So no kill, failed run
// or lost leaf file leads to a second published LINK proof for one k. (After a kill before
// isk/leaf/<k> has its final name, a run with -sweep makes a new one; the first was never in ivk:
// sweep_leaf().) Restoring isk to an earlier state, or using two copies of one isk, can lead to a
// second published proof, and no program detects that.
static std::string kept_entry(const std::string &leaf_folder) {
  return leaf_folder + "/entry.bin";
}
static bool there(const std::string &name) { // also a name that cannot be examined
  struct stat is;
  return !fstatat(via(name).fd, via(name).name, &is, AT_SYMLINK_NOFOLLOW) || errno != ENOENT;
}
// Whether something is seen under `name`: the check before a name goes on the cleanup list. A name
// that cannot be examined counts as absent, so that the system call that follows fails with the
// true reason; with there() the message would be "File exists".
static bool stands(const std::string &name) {
  struct stat is;
  return !fstatat(via(name).fd, via(name).name, &is, AT_SYMLINK_NOFOLLOW);
}
// Opens a directory for reading. Returns a new descriptor, which the caller closes, or -1 with
// errno set. A directory in `kept` is opened from its own descriptor, whatever its name refers to
// by now.
static int open_for_reading(const std::string &folder) {
  const Road::Open *const own = kept_open(folder);
  return openat(own ? own->fd : via(folder).fd, own ? "." : via(folder).name, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
}
// Syncs a directory, so that the names in it are on the disk.
static bool sync_dir(const std::string &folder) {
  const int fd = open_for_reading(folder);
  const bool good = fd >= 0 && !fsync(fd);
  if (fd >= 0) close(fd);
  return good;
}
// Creates a directory, for its owner only. The name is listed before mkdirat() (see `Road`). Were
// it added afterwards, a SIGABRT between the two would leave an empty directory that the list does
// not name, and with it the directories above it: seen once in 400 signals at random moments, in a
// test harness. The cost: a signal between the adding and a mkdirat() that fails removes what is
// under the name if that is an empty directory, and nothing else.
// make_folder(), write_new() and, unless how = 0, rename_to() first check that nothing is seen
// under the new name, so the cleanup can hit only a foreign name that appears between that check
// and the system call.
// A directory whose parent this run has not created (<isk>.making, <ivk>.making) is opened at once
// and kept open: see `Road`. If keep_open() does not accept it (EPERM), its name is dropped from
// the list, and the failed run leaves what is under that name, perhaps its own empty directory.
static bool make_folder(const std::string &name) {
  stand_still_if_another_cleans();
  if (!reach(folder_of(name))) {
    fprintf(stderr, "cp_commit: cannot create the folder %s: %s cannot be opened: %s\n", name.c_str(), folder_of(name).c_str(), strerror(errno));
    return false;
  }
  const Via v = via(name);
  if (stands(name)) { // what is seen there already is not this run's and is not listed
    fprintf(stderr, "cp_commit: cannot create the folder %s: %s\n", name.c_str(), strerror(EEXIST));
    return false;
  }
  road().made.push_back({name, true});
  if (mkdirat(v.fd, v.name, 0700)) {
    const int why = errno;
    road().made.pop_back();
    fprintf(stderr, "cp_commit: cannot create the folder %s: %s\n", name.c_str(), strerror(why));
    return false;
  }
  if (!(v.folder && v.folder->created) && !keep_open(name, true)) {
    if (errno == EPERM) {
      road().made.pop_back(); // the run cannot tell whether what has the name is its own: the cleanup leaves it alone
      fprintf(stderr, "cp_commit: the folder %s is made, but what has that name now does not belong to the calling user, or is open to group or others, or is not empty or cannot be listed: the file system keeps no owners or modes, or maps root to another user, or "
                      "somebody has put another folder there. Nothing was written into it, and it is left as it is. Use a local file system\n", name.c_str());
    } else // the name stays on the list: the cleanup removes the directory, which is empty
      fprintf(stderr, "cp_commit: the folder %s is made, but cannot be opened: %s\n", name.c_str(), strerror(errno));
    return false;
  }
  // The new name is synced at once, in its parent directory. Otherwise, by POSIX's wording, a power
  // failure later on could undo a directory's name (isk itself, isk/leaf) although files in it have
  // been synced. The name stays listed: a run that ends here removes the directory.
  if (!sync_dir(folder_of(name))) {
    fprintf(stderr, "cp_commit: the folder %s is made, but its name could not be brought to the disk: %s\n", name.c_str(), strerror(errno));
    return false;
  }
  return true;
}
// Creates a directory that several runs on an existing commitment share (isk/leaf, ivk/leaf),
// unless it exists. It is never on the cleanup list.
static bool shared_folder(const std::string &name) {
  struct stat is;
  if (reach(folder_of(name))) {
    if (!mkdirat(via(name).fd, via(name).name, 0700)) return true;
    if (errno == EEXIST && !fstatat(via(name).fd, via(name).name, &is, AT_SYMLINK_NOFOLLOW)) {
      if (S_ISDIR(is.st_mode)) return true;
      errno = ENOTDIR; // a link under the name is not taken: a leaf's files would go where it leads
    }
  }
  fprintf(stderr, "cp_commit: cannot create the folder %s: %s\n", name.c_str(), strerror(errno));
  return false;
}
// Writes a new file, for its owner only, and syncs it and its name. The name is listed before the
// file is created (see `Road`): the writing and the syncs come between the creation and the return,
// and a signal there would otherwise find a file that the list does not name (seen: the anchor's
// empty count.bin stayed, and its directories with it; the seed's file has the same gap). The cost:
// a signal between the adding and the creation removes whatever is under that name. cp_commit
// passes only names in a directory that this run has created, for its owner only, and temporary
// names (ending in ".making", or in ".making." and the process ID), which it takes to be this
// run's; write_new() does not check that. If the call fails after note_created() has noted the
// file, the name stays listed, and the cleanup tries again to overwrite and remove what is left of
// it.
static bool write_new(const std::string &name, const void *bytes, size_t n) {
  stand_still_if_another_cleans();
  if (!reach(folder_of(name))) {
    fprintf(stderr, "cp_commit: %s cannot be written as a new file: %s cannot be opened: %s\n", name.c_str(), folder_of(name).c_str(), strerror(errno));
    return false;
  }
  if (stands(name)) { // checked first: see `make_folder`
    fprintf(stderr, "cp_commit: %s cannot be written as a new file: %s\n", name.c_str(), strerror(EEXIST));
    return false;
  }
  const size_t files = road().created.size();
  made(name);
  if (!::hc::write_a_new_file(via(name).fd, via(name).name, bytes, n, note_created)) { // it has overwritten and removed what it had begun
    const int why = errno;
    if (road().created.size() == files) road().made.pop_back(); // nothing was noted as created: a file that was there before is not ours and stays
    fprintf(stderr, "cp_commit: %s cannot be written as a new file: %s\n", name.c_str(), strerror(why));
    return false;
  }
  if (!sync_dir(folder_of(name))) { // the anchor's count.bin is used right after its creation
    fprintf(stderr, "cp_commit: %s is written, but its name could not be brought to the disk: %s\n", name.c_str(), strerror(errno));
    return false;
  }
  return true;
}
// Renames `from` to `to`, which must not exist, and syncs the parent directory of `to`. how = 0:
// `to` may exist and is replaced; only text_takes_the_place() asks for that.
// In a run that creates a commitment a failed sync fails the rename: the run ends, everything is
// removed, and "done" is printed only of names that have reached the disk. In a run on an existing
// commitment a rename that has happened counts, whatever the sync returns, so the caller never
// takes back a step's last rename, which gives ivk's file its name. (A signal between that rename
// and step_done() does: the handler removes the step's names, the final ones too. The leaf file is
// then still in isk: kept_entry().)
static bool rename_to(const std::string &from, const std::string &to, unsigned how = RENAME_NOREPLACE) {
  // The new names go on the list first, after all others and in the old names' order: a directory
  // before its contents, so that the cleanup, which goes backwards, empties it first. (With the
  // list updated after the rename, a SIGABRT between the two found only the old names, in a test
  // harness: a leaf table's directory stayed under its final name, and the message said "removed by
  // name".) If the rename fails, the new names are dropped again; a signal before that removes what
  // is under them, as in write_new(): inside a directory that the run has created. Under the new
  // name of a directory in `kept` the cleanup removes only that directory itself
  // (rmdir_if_same()). With how = 0 the caller makes sure that the list has no name that the
  // rename touches.
  stand_still_if_another_cleans();
  if (how == RENAME_NOREPLACE && stands(to)) { // checked first: see `make_folder`
    fprintf(stderr, "cp_commit: cannot rename %s to %s: %s\n", from.c_str(), to.c_str(), strerror(EEXIST));
    return false;
  }
  Road::List<Road::Made> &list = road().made;
  Road::List<Road::Open> &kept = road().kept;
  const size_t before = list.size(), kept_before = kept.size();
  const Via old_name = via(from), new_name = via(to);
  if (const Road::Open *const own = kept_open(from)) { // a directory that is kept open gets its new name only while the old one still leads to it; a gap stays between this look and the rename
    struct stat is;
    if (fstatat(old_name.fd, old_name.name, &is, AT_SYMLINK_NOFOLLOW) || is.st_dev != own->device || is.st_ino != own->inode) {
      fprintf(stderr, "cp_commit: cannot rename %s to %s: the first name no longer leads to the folder that this call has made: somebody has renamed that folder or put something else in its place. What is there now was not touched. "
                      "Give both folders a parent in which only you can write, and run the call again\n", from.c_str(), to.c_str());
      return false;
    }
  }
  // A directory in `kept` is added under its new name first of all, so that every new name on the
  // list is looked up from that directory's own descriptor from the moment at which it is listed.
  // The entry stays if the rename fails: a handler on another thread may still be using the new
  // names, and must not look them up from the parent, where something else has the name. A surplus
  // entry does no harm: it refers to the run's own directory.
  for (size_t i = 0; i < kept_before; i++)
    if ((kept[i].name == from || kept[i].name.compare(0, from.size() + 1, from + "/") == 0) && !kept_open(to + kept[i].name.substr(from.size())))
      kept.push_back({to + kept[i].name.substr(from.size()), kept[i].fd, kept[i].device, kept[i].inode, kept[i].created});
  for (size_t i = 0; i < before; i++)
    if (list[i].name == from || list[i].name.compare(0, from.size() + 1, from + "/") == 0) list.push_back({to + list[i].name.substr(from.size()), list[i].folder});
  const std::string parent = folder_of(to); // before the rename: nothing from there to the sync can throw
  if (renameat2(old_name.fd, old_name.name, new_name.fd, new_name.name, how)) {
    const int why = errno;
    list.cut_to(before);
    fprintf(stderr, "cp_commit: cannot rename %s to %s: %s\n", from.c_str(), to.c_str(), strerror(why));
    return false;
  }
  if (!sync_dir(parent)) {
    fprintf(stderr, "cp_commit: %s%s has its name, but its folder could not be synced to the disk: %s\n", road().all_or_nothing ? "" : "warning: ", to.c_str(), strerror(errno));
    if (road().all_or_nothing) return false; // no retry: a second `fsync` could hide the error
  }
  return true;
}
// Renames `goal` from its temporary name to its final one.
static bool takes_its_name(const std::string &goal) {
  return rename_to(beside(goal), goal);
}
// The last two renames of a run that creates a commitment: <isk>.making to <isk>, then <ivk>.making
// to <ivk>, each followed by a sync of the parent directory that must succeed. So a directory named
// <ivk> appears only by the run's last rename, after everything that the run was asked to create
// has been created, passed by cp_verify and synced. A kill, any signal, a crash or a power failure
// before these two renames leaves of the commitment at most the two directories whose names end
// in ".making", and of try_the_file_system()'s trial an empty one that ends in ".making.making".
// isk goes first, so that a public directory does not appear without its secret one.
// Limits:
//   (a) In a test build, a run with -leaves `none`, or with a part of the range, succeeds with an
//       incomplete ivk. A production build rejects -leaves.
//   (b) The sync between the two renames can take seconds. A SIGKILL, a crash or a power failure in
//       that time leaves <isk> and <ivk>.making, which does not show that both are complete
//       (docs/faults.md, "A killed or failed cp_commit"). Wait for exit status 0, never for
//       <isk> to appear.
//   (c) If a rename or its sync fails, whatever has its final name by then is renamed back (as far
//       as that succeeds; not synced), and the cleanup removes everything. So <ivk> can appear for
//       a moment and vanish again. A caught signal, an exit() or an exception between a rename and
//       run_done() removes the directories under their final names, one name at a time: a kill or a
//       power failure during that removal leaves a part of them under final names. cp_commit
//       calls run_done() directly after this function returns, and in a run that succeeds no
//       statement between the last rename and that call allocates memory or can throw.
//   (d) To survive a power failure it relies on file systems that honour fsync() under both parent
//       directories.
//   (e) The parent directories are not only this run's. reach_by_parts() rejects a parent in which
//       group or others can write, unless it is sticky, as /tmp is. In a sticky parent another
//       account can still put something under a name that is free, a final name too, but cannot
//       rename or remove the run's directories. Root, the caller's other processes and, under
//       sudo, the account that ran sudo can do both. Neither changes what the run writes and
//       removes, which goes through the kept descriptors (see `Road`). Each of the two
//       renames happens in the parent that the run opened at its start, and only if the temporary
//       name still refers to the run's own directory; a gap remains between that check and the
//       rename. What the run only reads is still looked up by its whole path: the seed's file when
//       it is read again (for a comparison with the seed in memory, from which the keys are
//       derived), and the configuration that cp_verify reads. Give both directories a parent that
//       only you can write. See also docs/limits-all-rows.tsv, rows R2 and R2.5.
// between: only for tests: it is called between the two renames, so that a test can end a run
// there. cp_commit passes none.
static bool top_folders_take_their_names(const std::string &isk, const std::string &ivk, void (*between)() = nullptr) {
  enum { NOTHING_HAPPENED, RENAMED_AND_NOT_SYNCED, LASTING };
  auto named = [](const std::string &goal) { // the list has grown only if the rename happened
    const size_t before = road().made.size();
    return takes_its_name(goal) ? LASTING : road().made.size() > before ? RENAMED_AND_NOT_SYNCED : NOTHING_HAPPENED;
  };
  auto back = [](const std::string &goal) {
    const std::string working = beside(goal);
    (void)renameat2(via(goal).fd, via(goal).name, via(working).fd, via(working).name, RENAME_NOREPLACE);
  }; // the list has both names of `goal` and its contents: the cleanup finds them under either
  const int secret = named(isk);
  if (secret == RENAMED_AND_NOT_SYNCED) back(isk);
  if (secret != LASTING) return false;
  if (between) between();
  const int open = named(ivk);
  if (open == LASTING) return true;
  if (open == RENAMED_AND_NOT_SYNCED) back(ivk);
  back(isk);
  return false;
}

// ---- the files of ivk ----
// [cp-commit-times] Sets a file's access and modification times to 0, so that these two times of a
// file of ivk (key.bin or a leaf file) do not reveal when it was written. The status-change time
// cannot be set and stays.
static bool zero_times(const std::string &file) {
  const struct timespec zero[2] = {{0, 0}, {0, 0}};
  if (!utimensat(via(file).fd, via(file).name, zero, AT_SYMLINK_NOFOLLOW)) return true;
  fprintf(stderr, "cp_commit: refused (cp-commit-times): the times of %s cannot be set: %s\n", file.c_str(), strerror(errno));
  return false;
}
// takes_its_name() for a file of ivk, whose times are zeroed directly before the rename.
static bool public_file_takes_its_name(const std::string &goal) {
  return zero_times(beside(goal)) && takes_its_name(goal);
}
// [cp-commit-ivk-holds] ivk is published as it is, so it may contain only key.bin, the directory
// leaf and in it the leaf files <k>.bin, and these files' temporary names; each file must be a
// regular one. <k> is any run of digits: whether it is below the number of leaf tables is not
// checked. Returns the path of the first other name, or of one that cannot be examined; or the
// directory's own path if it cannot be listed to its end; or "". A name that is gone when it is
// examined is skipped: a concurrent run may have renamed it. In odd_in(), `entries` means that
// `folder` is ivk/leaf.
static std::string holds_besides(const std::string &ivk) {
  auto odd_in = [](const std::string &folder, bool entries) -> std::string {
    const int handle = open_for_reading(folder);
    DIR *const list = handle < 0 ? nullptr : fdopendir(handle);
    if (!list) {
      const bool absent = handle < 0 && errno == ENOENT;
      if (handle >= 0) close(handle);
      return absent && entries ? std::string() : folder; // no ivk/leaf yet; or it cannot be listed
    }
    std::string odd;
    while (odd.empty()) {
      errno = 0;
      const struct dirent *const item = readdir(list);
      if (!item && errno) odd = folder;
      if (!item) break;
      std::string name = item->d_name;
      if (name == "." || name == "..") continue;
      const std::string path = folder + "/" + name;
      struct stat is;
      const bool seen = !fstatat(handle, name.c_str(), &is, AT_SYMLINK_NOFOLLOW);
      if (!seen && errno == ENOENT) continue;
      if (!entries && name == "leaf" && seen && S_ISDIR(is.st_mode)) continue;
      if (name.size() > 7 && !name.compare(name.size() - 7, 7, ".making")) name.resize(name.size() - 7);
      const size_t digits = name.find_first_not_of("0123456789");
      if (!seen || !S_ISREG(is.st_mode) || (entries ? !digits || digits == std::string::npos || name.compare(digits, std::string::npos, ".bin") : name != "key.bin"))
        odd = path;
    }
    closedir(list);
    return odd;
  };
  const std::string odd = odd_in(ivk, false);
  return odd.empty() ? odd_in(::cp::ivk_leaves(ivk), true) : odd;
}
// Call it before a file of ivk gets its final name. Returns 0, or 2 after a rejection.
static int nothing_else_in(const std::string &ivk) {
  const std::string odd = holds_besides(ivk);
  return odd.empty() ? 0
                     : refuse("cp-commit-ivk-holds", odd + " is no file of a commitment's public folder, or the folder cannot be listed. The folder is handed out as it "
                                                     "lies, so nothing is published while that name is there. cp_commit has not created it. Remove nothing on this program's word: find out what it is and who has put it there");
}
// Publishes `bytes` as `goal`, ivk/leaf/<k>.bin: a leaf file that has passed "cp_verify -leaf" and
// is stored in isk (kept_entry()). The caller must hold the leaf table's lock (hold_leaf()); this
// is not checked. A temporary file that a killed run has left contains some or all of the same
// bytes and is replaced.
static bool publish_entry(const std::string &goal, const std::string &bytes) {
  const std::string making = beside(goal);
  if ((!reach(folder_of(goal)) || unlinkat(via(making).fd, via(making).name, 0)) && errno != ENOENT) {
    fprintf(stderr, "cp_commit: %s cannot be removed, or its folder cannot be opened: %s\n", making.c_str(), strerror(errno));
    return false;
  }
  return write_new(making, bytes.data(), bytes.size()) && public_file_takes_its_name(goal);
}

// ---- a new isk.txt: only in a run on an existing commitment, so not reached ----
// The temporary name of isk.txt's new contents (the commitment's settings: cp::IskText in
// prover/common/cp_files.h). It includes the process ID, so that concurrent runs on one machine
// write different files.
static std::string text_in_the_making(const std::string &isk) {
  return beside(::cp::isk_txt(isk)) + "." + std::to_string((long long)getpid());
}
// Renames that file over isk.txt, so that a reader finds the old contents or the new, complete.
// step_done() empties the cleanup list first, as it does in a run on an existing commitment:
// otherwise rename_to() would list isk.txt itself, and leave_at_a_signal(), which can run at any
// instruction, would remove it, the old contents and the new. After a failed rename only the
// temporary file is listed again, so a caller in the middle of a step would silently lose its
// cleanup; cp_commit's only caller, text_moves_on(), runs after the last leaf table's step. A
// signal between step_done() and the rename leaves the temporary file, which nothing reads and
// which sweep_texts() removes.
static bool text_takes_the_place(const std::string &isk) {
  const std::string from = text_in_the_making(isk);
  step_done();
  if (rename_to(from, ::cp::isk_txt(isk), 0)) return true;
  stand_still_if_another_cleans(); // checked before every addition to the list
  made(from); // the rename failed: the cleanup removes the temporary file
  return false;
}
// Takes the lock that guards updates of isk.txt: an exclusive flock() on isk/seed.bin, which is
// never replaced (isk.txt itself is, by the rename) and which only its owner and root can open
// (checked here, on the open file). Waits while another run holds the lock; the system drops it
// when a process exits. Returns the lock's descriptor, which the caller closes, or -1 after an
// error message.
static int hold_the_text(const std::string &isk) {
  const std::string file = ::cp::isk_seed(isk);
  const int held = reach(isk) ? openat(via(file).fd, via(file).name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK) : -1; // O_NONBLOCK: a fifo does not hold us up
  struct stat is;
  std::string why;
  const char *then = "Mend the cause and run the same call again: it gives isk.txt its new text"; // replaced in two cases below
  if (held < 0 || fstat(held, &is))
    why = strerror(errno);
  else if (!S_ISREG(is.st_mode))
    why = "it is not a regular file";
  else if (is.st_mode & 077) {
    why = "it has permissions for group or others";
    then = "A seed that another account could read counts as read, and chmod or chown alone repairs nothing: switch off whatever restarts cp_prove, then stop every "
           "cp_prove on this isk and wait until each has ended, make no further call of cp_commit or cp_prove on it, and go by docs/operators-duties.md, \"After a "
           "refusal or a fault\", before anything else";
  } else {
    int failed = flock(held, LOCK_EX | LOCK_NB);
    if (failed && errno == EWOULDBLOCK) {
      printf("cp_commit: another call is writing isk.txt's new text and holds the lock on %s: this call waits until the lock is free\n", file.c_str());
      do failed = flock(held, LOCK_EX);
      while (failed && errno == EINTR);
    }
    if (!failed) return held;
    why = std::string(strerror(errno)) + " (some shared and network file systems do not support flock())";
    then = "Do not copy or move the folder named with -isk to get past this message. If the cause can be mended where the folder is, run the same call again: it gives isk.txt its new text. If the file cannot be locked there, make a new commitment on a local disk (docs/operators-duties.md, \"A new commitment\")";
  }
  fprintf(stderr, "cp_commit: no lock could be taken on %s: %s. isk.txt is unchanged, and the leaves that were made are kept. %s\n", file.c_str(), why.c_str(), then);
  if (held >= 0) close(held);
  return -1;
}
// Replaces isk.txt's contents by `next`: what this run read at its start, with the numbers M and
// `first` that it requests (the live leaf tables are `first` .. M - 1). Under the lock, isk.txt is
// read again and left unchanged if it cannot be read, or if another run has raised M or `first`
// above this run's or changed another line; only the line "seed public" is not compared, so a
// change of it is overwritten. So concurrent runs on one machine never lower M or `first`. Runs on
// two machines still can, where a shared file system grants the lock to each machine separately.
// See also docs/limits-all-rows.tsv, row R3.4.1.
// isk is opened by its path only once (reach(), at the latest in hold_the_text()), and the lock,
// the reading, the new file and the rename go through that descriptor, so all four act on one
// directory. Returns 0, 2 (rejected) or 4 (the lock, the write or the rename failed).
static int text_moves_on(const std::string &isk, const ::cp::IskText &next) {
  const int held = hold_the_text(isk);
  if (held < 0) return 4;
  ::cp::IskText now;
  const std::string file = ::cp::isk_txt(isk);
  std::string text;
  const bool read = ::cp::read_file(via(file).fd, via(file).name, text) && ::cp::cut_isk_text(text, now);
  text = ::cp::isk_text(next);
  int rc;
  if (!read || now.M > next.M || now.first > next.first || now.version != next.version || now.keep != next.keep || now.weights != next.weights ||
      now.ranges != next.ranges || now.test_seed != next.test_seed)
    rc = refuse("cp-commit-text-moved", ::cp::isk_txt(isk) + " has changed since this call read it" +
                                            (read ? ", and says M = " + std::to_string(now.M) + ", first = " + std::to_string(now.first) : std::string()) +
                                            ": this call's text would take that back, so it stays. The leaves that were made stand" +
                                            (read ? ". Run the call again, with no number below the text's: a number that it does not give is then taken from the text as it now is" : ""));
  else
    rc = write_new(text_in_the_making(isk), text.data(), text.size()) && text_takes_the_place(isk) ? 0 : 4;
  close(held); // after the directory's sync
  return rc;
}

// ---- an unfinished leaf table ----
// -sweep is accepted only in a run on an existing commitment, which cp_commit rejects, so
// cp_commit never reaches remove_flat_folder(), sweep_leaf() or sweep_texts().
//
// Removes a directory and the files in it. It lists and empties the directory through `held`, the
// descriptor from hold_leaf(), so it empties the directory that was locked, whatever its name
// refers to by now. The directory itself is removed only while `folder` still refers to it
// (rmdir_if_same()); that fails if a subdirectory is left. Returns true if it is removed, or if
// `folder` is absent or refers to something else by now.
// It overwrites nothing: see zeros_first() and docs/limits-all-rows.tsv, row R3.4.
static bool remove_flat_folder(int held, const std::string &folder) {
  const int twin = fcntl(held, F_DUPFD_CLOEXEC, 0);
  if (DIR *const list = twin < 0 ? nullptr : fdopendir(twin)) {
    while (const struct dirent *const entry = readdir(list)) {
      const std::string name = entry->d_name;
      if (name != "." && name != "..") unlinkat(held, name.c_str(), 0);
    }
    closedir(list); // closes only the duplicate: the lock stays with `held`
  } else if (twin >= 0)
    close(twin);
  struct stat is;
  return !fstat(held, &is) && (!rmdir_if_same(folder, is.st_dev, is.st_ino) || errno == ENOENT);
}
// Locks a directory, without waiting, by an flock(): a leaf table's directory in isk, under its
// temporary or its final name, or <isk>.making and <ivk>.making, each for the whole run
// (hf_commit.cu). The lock is on the directory, not on its name, so it survives a rename, and the
// system drops it when the process exits. So a run that is at work or stalled can be told from a
// dead one, and no two runs publish one stored leaf file at the same time. Returns the descriptor,
// which the caller closes to unlock; or -1 if the directory cannot be opened or locked, or the name
// no longer refers to the directory that was opened (a run with -sweep may have removed it). A
// symbolic link under the name is not followed.
static int hold_leaf(const std::string &folder) {
  const Road::Open *const own = kept_open(folder); // a directory that the run keeps open (<isk>.making, <ivk>.making) is reopened from the run's descriptor, not by its name; the check below still compares it with what the name refers to now
  const int held = !reach(folder_of(folder)) ? -1 : own ? openat(own->fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW) : openat(via(folder).fd, via(folder).name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  struct stat locked, named;
  if (held >= 0 && !flock(held, LOCK_EX | LOCK_NB) && !fstat(held, &locked) && !fstatat(via(folder).fd, via(folder).name, &named, AT_SYMLINK_NOFOLLOW) && locked.st_dev == named.st_dev &&
      locked.st_ino == named.st_ino)
    return held;
  if (held >= 0) close(held);
  return -1;
}
// The rejection after hold_leaf() has failed on leaf table k's directory. Returns 2.
static int leaf_is_held(const std::string &folder, uint32_t k) {
  return refuse("cp-commit-leaf-begun", folder + " cannot be opened and locked. If another call of cp_commit is at work on leaf " + std::to_string(k) +
                                            ", wait until it has ended and call again. If none is: end no process and remove nothing. The name may be a symbolic link or lead to another directory by now, or the file "
                                            "system may not support flock(), as some shared and network file systems do not (then make the commitment on a local file system)");
}
// -sweep: removes what a killed run has left of leaf table k in isk/leaf/<k>.making; none of it has
// been in ivk. Call it only if isk/leaf/<k> does not exist. The run then rebuilds the table with
// its own masks, and its LINK proof with its own private randomness: each key is derived with 32
// fresh bytes from the system (Make::a_copy(), RunKeys::take_an_entrys() in hf_commit_keys.h),
// which are zeros under -test-seed; nothing is hidden then. (Also under -public-weights, in a
// production build; but Opt::parse() rejects that option together with -sweep.)
// Returns 0; 2 if the directory cannot be locked (a live run may hold it); 4 if it cannot be
// removed.
static int sweep_leaf(const std::string &isk, uint32_t k) {
  const std::string folder = beside(::cp::isk_leaf_dir(isk, k));
  if (!there(folder)) return 0;
  const int held = hold_leaf(folder);
  if (held < 0) return leaf_is_held(folder, k);
  const bool swept = remove_flat_folder(held, folder);
  if (!swept) fprintf(stderr, "cp_commit: -sweep: %s cannot be removed: %s\n", folder.c_str(), strerror(errno));
  close(held);
  return swept ? 0 : 4;
}
// -sweep: removes every isk/isk.txt.making.<id>, <id> being digits only, which a run leaves if it
// is killed between writing the file and renaming it. isk is listed, and the files are removed,
// through one descriptor. Returns false if one cannot be removed.
static bool sweep_texts(const std::string &isk) {
  const std::string begins = "isk.txt.making.";
  bool swept = true;
  const int handle = open_for_reading(isk);
  if (DIR *const list = handle < 0 ? nullptr : fdopendir(handle)) {
    while (const struct dirent *const entry = readdir(list)) {
      const std::string name = entry->d_name;
      if (name.size() > begins.size() && !name.compare(0, begins.size(), begins) && name.find_first_not_of("0123456789", begins.size()) == std::string::npos && unlinkat(handle, name.c_str(), 0) && errno != ENOENT) {
        fprintf(stderr, "cp_commit: -sweep: %s/%s cannot be removed: %s\n", isk.c_str(), name.c_str(), strerror(errno));
        swept = false;
      }
    }
    closedir(list);
  } else if (handle >= 0)
    close(handle);
  return swept;
}

// ---- cp_verify ----
// cp_commit finds and runs cp_verify with the functions of prover/common/judge.h.
using ::judge::find_the_judge; // the path of the cp_verify in this program's directory, or ""
using ::judge::INPUT;
static const char *const KEY_AS_INPUT = INPUT[0], *const SECOND_INPUT = INPUT[1];
// The time limit of one run of cp_verify under a supervisor, in milliseconds: 30 minutes.
// The runs that cp_commit starts take seconds. Only a
// test changes it, before its first call.
inline long long check_limit_ms = 1800 * 1000;
// Runs the program `judge` (cp_verify's path) with the arguments `words`, argv[0] first. Returns
// its exit status, or a number below 0 if it could not be started, did not exit by itself or was
// killed at the time limit.
// `inputs`: at most five byte strings; the child finds `inputs[i]` under the path INPUT[i]. So
// cp_verify reads the bytes that this process holds, not a file that may have changed since.
// `heard`: if not null, it receives the child's standard output, which is also printed here once
// the child has exited. If the output cannot be read back, the call returns -1.
// In a program whose main() begins with judge::supervise_if_asked(), as cp_commit's does, cp_verify
// runs under a supervisor, in a separate process group. The group is killed at the time limit, and
// when this process dies, even by SIGKILL, so that no run goes on after its caller with a key file
// or a leaf file in memory. Two exceptions: prover/common/judge.h, at `Running` and judge_says().
// At a terminal set to tostop cp_verify's first line stops the group, until the time limit.
// `judge` must then be the path that find_the_judge() returns. If it is not, or cp_verify has been
// removed or renamed since, the supervisor starts nothing, and the call returns -1.
// In any other program cp_verify is this process's child, in its group, without a time limit.
static int judge_says(const std::string &judge, std::vector<const char *> words, const std::vector<const std::string *> &inputs,
                      std::string *heard = nullptr) {
  ::judge::Running run; // this call's own list: nothing calls end_all()
  run.limit_ms = check_limit_ms;
  if (heard) heard->clear();
  const int said = ::judge::judge_says(judge, words, inputs, heard, ::judge::main_can_supervise ? &run : nullptr);
  if (heard) {
    fwrite(heard->data(), 1, heard->size(), stdout);
    fflush(stdout);
  }
  // 2 is the supervisor's status for a path that is not find_the_judge()'s, and also cp_verify's own
  // for a wrong command line.
  if (said == 2 && ::judge::main_can_supervise && find_the_judge() != judge) return -1;
  return said;
}
// Runs a part-call of cp_verify, which checks only the key file or only one leaf file:
// "cp_verify -key <config directory> <key.bin>" or, if `entry_bytes` (a leaf file) is given,
// "cp_verify -leaf <config directory> <key.bin> <leaf file>". key.bin and the leaf file are passed
// in memory (judge_says()), so that neither is written to a file before it has passed.
// Returns whether the part-call passed: exit status 3. Status 11 is read as PART PASSED WITH WORK
// (`bits` reached only with proof of work counted) and accepted too, but cp_verify never exits
// with it: such a part-call also exits with 3, and its line on proof of work says so. If `bits` is
// below BAR, the target in bits (prover/common/setting.h), which only the test option -test-bits
// requests: 14 (`NOT-A-VERIFICATION`) instead.
// cp_verify writes to this program's stdout and stderr, so the log has its last line and, if the
// part-call passes, its accounting of soundness errors. `made_here`: this run has produced the
// bytes; it only selects the error message.
// A cp_verify that could not be started, did not exit by itself, was killed at the time limit or
// ended with `NOT CHECKED` (5) has said nothing about the bytes, which may have taken hours to
// produce: it is run once more.
static bool judge_passes(const std::string &judge, const std::string &config, const std::string &key_bytes,
                         const std::string *entry_bytes, int bits, bool made_here = true) {
  const bool entry = entry_bytes != nullptr;
  const std::string threads = std::to_string(std::min(256u, std::max(1u, std::thread::hardware_concurrency()))),
                    asked = std::to_string(bits);
  std::vector<const char *> words = {"cp_verify", entry ? "-leaf" : "-key", config.c_str(), KEY_AS_INPUT};
  if (entry) words.push_back(SECOND_INPUT);
  // -q only shortens the log: a part-call leaves out two progress lines, neither with a digest, and
  // keeps its error accounting. If it rejects, cp_verify discards, with or without -q, the output
  // of the child process that read the bytes, and prints only which check failed (cpStagesRun()
  // in verifier/src/cp_verify.cpp).
  for (const char *w : {"-bits", asked.c_str(), "-j", threads.c_str(), "-q"}) words.push_back(w);
  const int wanted = bits < BAR ? 14 : 3;
  std::vector<const std::string *> inputs = {&key_bytes};
  if (entry) inputs.push_back(entry_bytes);
  int said = judge_says(judge, words, inputs);
  if (said < 0 || said == 5) {
    fprintf(stderr, "cp_commit: cp_verify %s gave no answer (%s): it is asked once more\n", entry ? "-leaf" : "-key",
            said == ::judge::LATE ? "it was killed at its time limit of 30 minutes"
            : said < 0            ? "it could not be started, or did not exit by itself"
                                  : "NOT CHECKED, status 5");
    said = judge_says(judge, words, inputs);
  }
  if (wanted == 3 && said == 11)
    printf("cp_commit: cp_verify %s passed WITH WORK (status 11): its figure reaches the asked bits only with proof of work counted\n", entry ? "-leaf" : "-key");
  if (said == wanted || (wanted == 3 && said == 11)) return true;
  fprintf(stderr, "cp_commit: cp_verify %s did not pass (%s %d; wanted: %d). %s\n", entry ? "-leaf" : "-key",
          said == ::judge::LATE ? "it was killed at its time limit of 30 minutes:"
          : said < 0            ? "it could not be started, or did not exit by itself:"
                                : "status",
          said, wanted,
          said < 0 || said == 5 ? "cp_verify has verified nothing, so nothing is said about the file. Either the machine failed (for example, too little memory): then a new call of cp_commit may pass. Or a guard inside cp_verify spoke: then its line stands above and will come again on the same files: to report the fault to anyone else, say in your own words what you saw and where the two programs come from, and send no line, length, exit code or count of that run"
          : made_here           ? "Its own last line, above, names the check. One of the two programs has a bug, or the machine a fault. To report the fault to anyone else, say in your own words what you saw "
                                  "and where the two programs come from, and send no line, length, exit code or count of that run. Never send a file of a making"
                                : "Its own last line, above, names the check");
  return false;
}
// Parts of the lines that a planning call of cp_verify (-plan) prints (verifier/src/cp_verify.cpp).
// Such a call can exit with status 4 and still report that no run with its files can be a
// verification: in a line that begins with PLAN_NOT_STANDARD (the files are outside the standard
// parameter form). A complete output ends with the line PLAN_ONLY; any other output counts as no
// plan. If cp_verify's wording of these lines changes, the two strings must change with it.
static const char *const PLAN_NOT_STANDARD = "plan: not the standard parameter form (", *const PLAN_ONLY = "PLAN ONLY: no proof was checked";
static bool plan_is_whole(const std::string &output) {
  const std::string last = std::string(PLAN_ONLY) + "\n";
  return output == last || (output.size() > last.size() && !output.compare(output.size() - last.size() - 1, std::string::npos, "\n" + last));
}
static bool plan_rules_out(const std::string &output) {
  for (size_t at = 0, end; at < output.size(); at = end + 1) {
    end = std::min(output.find('\n', at), output.size());
    const std::string line = output.substr(at, end - at);
    if (!line.compare(0, strlen(PLAN_NOT_STANDARD), PLAN_NOT_STANDARD)) return true;
  }
  return false;
}
// One planning call before the weight tables are built; it checks no proof and exits with status 4:
//   cp_verify -plan request <policy.txt> <KL.bin> <bits> [-layout <layout.txt>]
//             -prompt <request> -reply <y> -q
// It takes about a second. So these are detected before the long work and not only after the anchor
// is created: a cp_verify that cannot plan with these files (one built for another version word of
// the parameter file, say; builds are not compared otherwise), a parameter file that cannot reach
// the requested bits, and files with which cp_verify rejects every request. All its files are
// passed in memory (judge_says()), so the plan is for the bytes that are committed to, whatever the
// configuration directory contains by now.
// The request is synthetic: T = 2^lt tokens, all 0, with r = T - 1 and e = T (its range is the last
// position only), and the next token y = 0.
// `policy`: the bytes of policy.txt, the parameter file. `layer_words`: those of KL.bin, the
// quantization scales. `layout`: null if the configuration has no layout.txt. `lt`: the parameter
// file's line lt, which prover/common/parameter_file.h limits to 4 .. 16.
// NONE: lt is above 16, or the call could not be started, did not exit with status 4, or its output
// is incomplete.
// NO_VERIFICATION: it planned, and ruled a verification out.
enum class Plans { NONE, NO_VERIFICATION, GOOD };
static Plans judge_plans(const std::string &judge, const std::string &policy, const std::string &layer_words, const std::string *layout,
                         uint32_t lt, int bits) {
  const std::string asked = std::to_string(bits);
  std::string output;
  if (lt > 16) return Plans::NONE;
  std::vector<uint32_t> request(((size_t)1 << lt) + 2, 0);
  request[request.size() - 2] = ((uint32_t)1 << lt) - 1, request[request.size() - 1] = (uint32_t)1 << lt;
  const std::string request_bytes((const char *)request.data(), request.size() * 4), y_bytes(4, '\0');
  std::vector<const char *> words = {"cp_verify", "-plan", "request", INPUT[2], INPUT[3], asked.c_str()};
  std::vector<const std::string *> inputs = {&request_bytes, &y_bytes, &policy, &layer_words};
  if (layout) words.push_back("-layout"), words.push_back(INPUT[4]), inputs.push_back(layout);
  for (const char *w : {"-prompt", INPUT[0], "-reply", INPUT[1], "-q"}) words.push_back(w);
  if (judge_says(judge, words, inputs, &output) != 4 || !plan_is_whole(output)) return Plans::NONE;
  return plan_rules_out(output) ? Plans::NO_VERIFICATION : Plans::GOOD;
}
// ---- what a run that creates a commitment finds under the two temporary names ----
// After SIGKILL, a crash or a power failure a run's two temporary directories (`working` in
// identifiers) stay, or, between the run's last two renames, <isk> and <ivk>.making. The next run
// with the same names is rejected. Where temporary directories alone stay, its rejection names the
// directories that the operator can remove; beside an <isk> it names none. Printing that a directory can be removed is as
// dangerous as removing it, and whoever can write in the parent can put something else under a
// temporary name. So that is printed only for what looks like a dead run's: a directory, not a
// link; the calling user's own, with no permissions for group or others; not locked; containing
// only names that cp_commit creates (docs/limits-all-rows.tsv, row R2.4). All these checks use one
// descriptor of the directory.
// They cannot tell a dead run's directory from an empty directory of the same user, from another
// commitment's directory of that user, finished or not, or from any directory of that user with
// only such names. Nor can they tell what the path refers to when the operator acts: the checks
// are made on a descriptor, and the message names a path.
//
// `Found`: what is under one temporary name. ABSENT: nothing. LOCKED: a directory that another
// process has locked. LEFT: it looks like a dead run's. NOT_MADE: it fails another of the checks
// above, or cannot be listed. NO_LOOK: it could not be examined, opened or locked.
struct Found {
  enum Kind { ABSENT, LOCKED, LEFT, NOT_MADE, NO_LOOK } kind = ABSENT;
  std::string why; // NOT_MADE, NO_LOOK: a clause for the rejection's message
};
// The directory that foreign_name() lists: isk, isk/config, a table's (isk/anchor, isk/leaf/<k>),
// isk/leaf, ivk, ivk/leaf.
enum class Holds { SECRET, CONFIGURATION, TABLE, LEAVES, PUBLIC, ENTRIES };
// Returns the path of the first name in the open directory `at`, or below it, that cp_commit does
// not create there, or whose type (file, directory) is not the one that cp_commit creates; "" if
// there is none; `shown`, the path of `at` for messages, if `at` cannot be listed. A name counts
// with and without a final ".making". No file is opened. The known names of a table's directory are
// one list for the anchor and the leaf tables, and it also has wcard.bin (HF_RECORD_FILE), which
// cp_commit does not create. isk/audit, which a run with -keep-for-the-audit creates, is not among
// the known names, so what such a run has left counts as not created by cp_commit.
static std::string foreign_name(int at, const std::string &shown, Holds what) {
  const int twin = openat(at, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  DIR *const list = twin < 0 ? nullptr : fdopendir(twin);
  if (!list) {
    if (twin >= 0) close(twin);
    return shown;
  }
  std::string odd;
  while (const struct dirent *const entry = odd.empty() ? (errno = 0, readdir(list)) : nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    const std::string whole = shown + "/" + name;
    if (name.size() > 7 && !name.compare(name.size() - 7, 7, ".making")) name.resize(name.size() - 7);
    const auto one_of = [&name](std::initializer_list<const char *> names) { return std::find(names.begin(), names.end(), name) != names.end(); };
    const auto a_number = [](const std::string &digits) { return !digits.empty() && digits.size() <= 10 && digits.find_first_not_of("0123456789") == std::string::npos; };
    bool file = false, folder = false;
    Holds inner = Holds::TABLE;
    switch (what) {
    case Holds::SECRET:
      file = one_of({"seed.bin", "key.bin", "isk.txt"});
      folder = one_of({"config", "anchor", "leaf"});
      inner = name == "config" ? Holds::CONFIGURATION : name == "leaf" ? Holds::LEAVES : Holds::TABLE;
      break;
    case Holds::CONFIGURATION: file = one_of({"policy.txt", "KL.bin", "layout.txt"}); break;
    case Holds::TABLE: file = one_of({HF_NODES_FILE, HF_RECORD_FILE, ::cp::LEAF_COUNT_FILE, ::cp::LEAF_FRESH_FILE, ::cp::LEAF_ENTRY_FILE}); break;
    case Holds::LEAVES: folder = a_number(name); break;
    case Holds::PUBLIC:
      file = name == "key.bin";
      folder = name == "leaf", inner = Holds::ENTRIES;
      break;
    case Holds::ENTRIES: file = name.size() > 4 && !name.compare(name.size() - 4, 4, ".bin") && a_number(name.substr(0, name.size() - 4)); break;
    }
    struct stat is;
    if (fstatat(at, entry->d_name, &is, AT_SYMLINK_NOFOLLOW) || !((file && S_ISREG(is.st_mode)) || (folder && S_ISDIR(is.st_mode)))) {
      odd = whole;
    } else if (folder) {
      const int below = openat(at, entry->d_name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      odd = below < 0 ? whole : foreign_name(below, whole, inner);
      if (below >= 0) close(below);
    }
  }
  if (odd.empty() && errno) odd = shown; // the listing broke off
  closedir(list);
  return odd;
}
// Examines one temporary name; `secret`: it is <isk>.making. A parent that cannot be opened gives
// ABSENT: cp_commit has rejected such a parent before (reach_by_parts()).
static Found what_stands_at(const std::string &working, bool secret) {
  Found f;
  if (!reach(folder_of(working))) return f;
  const Via v = via(working);
  struct stat is;
  if (fstatat(v.fd, v.name, &is, AT_SYMLINK_NOFOLLOW)) {
    if (errno != ENOENT) f.kind = Found::NO_LOOK, f.why = working + " cannot be looked at: " + strerror(errno);
    return f;
  }
  if (!S_ISDIR(is.st_mode)) return f.kind = Found::NOT_MADE, f.why = working + (S_ISLNK(is.st_mode) ? " is a symbolic link" : " is no directory"), f;
  const int fd = openat(v.fd, v.name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0 || fstat(fd, &is)) {
    f.kind = Found::NO_LOOK, f.why = working + " cannot be opened: " + strerror(errno);
  } else if (is.st_uid != geteuid() || (is.st_mode & 077)) {
    f.kind = Found::NOT_MADE, f.why = working + " belongs to another user, or is open to group or others";
  } else if (flock(fd, LOCK_EX | LOCK_NB)) { // the lock is only a test: it is dropped when fd is closed below
    if (errno == EWOULDBLOCK) f.kind = Found::LOCKED;
    else f.kind = Found::NO_LOOK, f.why = working + " cannot be locked: " + strerror(errno);
  } else {
    const std::string odd = foreign_name(fd, working, secret ? Holds::SECRET : Holds::PUBLIC);
    if (odd.empty()) f.kind = Found::LEFT;
    else f.kind = Found::NOT_MADE, f.why = odd == working ? working + " cannot be listed" : "it holds " + odd + ", which a call of cp_commit does not make (one with -keep-for-the-audit makes one directory more; what a killed call with that option has left holds its seed and the bytes that it kept)";
  }
  if (fd >= 0) close(fd);
  return f;
}
// isk, ivk: the two final names; the caller has checked that neither exists. Returns 0 if neither
// temporary name exists; otherwise prints the rejection and returns 2. It creates, renames and
// removes nothing.
// Unless it finds a lock, it examines both names again a third of a second later: a running
// cp_commit that was between a mkdirat() and its lock the first time (one sync of the parent comes
// between the two) has the lock the second time, unless that sync takes longer.
static int refuse_what_is_left(const std::string &isk, const std::string &ivk) {
  const std::string working[2] = {beside(isk), beside(ivk)};
  Found found[2];
  for (int look = 0; look < 2; look++) {
    for (int i = 0; i < 2; i++) found[i] = what_stands_at(working[i], !i);
    if (found[0].kind == Found::ABSENT && found[1].kind == Found::ABSENT) return 0;
    if (look || found[0].kind == Found::LOCKED || found[1].kind == Found::LOCKED) break;
    usleep(300 * 1000);
  }
  const std::string exists = working[found[1].kind != Found::ABSENT] + " exists already";
  const std::string new_names = ". Give -isk and -ivk two new names: nothing has to be removed first. "; // no message tells the operator to remove anything; three say what can be removed
  for (int i = 0; i < 2; i++)
    if (found[i].kind == Found::LOCKED)
      return making::refuse(HF_STOP_FILE_THERE, exists + ", and a running cp_commit holds " + working[i] + " (its lock is taken). Let that call end, or end it with Ctrl-C or kill, after which it removes both directories itself; then call again");
  for (int i = 0; i < 2; i++)
    if (found[i].kind == Found::NOT_MADE)
      return making::refuse(HF_STOP_FILE_THERE, working[i] + " exists already. No call of cp_commit by this user has made this, or not this alone: " + found[i].why + ". Nothing was touched. Remove nothing on this program's word: find out what it is, or give -isk and -ivk other names" +
                                                    (i ? "" : " (if a call of cp_commit has worked in it, it holds that call's seed)"));
  for (int i = 0; i < 2; i++)
    if (found[i].kind == Found::NO_LOOK)
      return making::refuse(HF_STOP_FILE_THERE, working[i] + " exists already, and whether a cp_commit is running there cannot be told: " + found[i].why + ". Nothing was touched. End no process and remove nothing on this program's word: find out what it is, or give -isk and -ivk other names, on a local file system if it is the lock that has failed");
  if (found[0].kind == Found::ABSENT) { // the public working directory alone
    if (there(isk)) // isk has appeared since the caller's check: the state between a run's last two renames (isk first, then ivk)
      return making::refuse(HF_STOP_FILE_THERE, isk + " exists already: a commitment is made in two new folders. Give -isk and -ivk two new names: nothing has to be removed, and a folder that stands under -isk's name may hold a commitment's seed, which nothing brings back");
    return making::refuse(HF_STOP_FILE_THERE, exists + ", and neither " + isk + " nor " + working[0] + " does" + new_names + "Before anything is removed: if anybody else can write " + folder_of(working[1]) +
                                                  ", look at what has the name now. If it is what is left of an interrupted call whose secret directory is gone, it holds nothing secret and can be removed");
  }
  const bool both = found[1].kind != Found::ABSENT; // <ivk>.making is named as removable only if something was found under that name
  const std::string may_go = both ? "these two directories, they can be removed, whole, and nothing else: " + working[0] + " (it holds the seed) and " + working[1] : "this directory, it can be removed, whole, and nothing else: " + working[0] + " (it holds the seed)";
  const std::string parents = both && folder_of(working[1]) != folder_of(working[0]) ? folder_of(working[0]) + " or " + folder_of(working[1]) : folder_of(working[0]);
  return making::refuse(HF_STOP_FILE_THERE, exists + ", and no cp_commit on this machine is running in " + working[0] + " or " + working[1] + new_names + "Before anything is removed: if anybody else can write " + parents + ", look at what has " +
                                                (both ? "each name" : "the name") + " now. If a call that was killed, or a machine that failed, has left " + may_go);
}
// ---- at a run's start: the directories that it has not created ----
// Opens `folder`, in which the run is to create `name`, the path given with `option` (-isk, -ivk or
// -worst-case-table), and adds it to `kept`, so that every later look at a name in it starts from
// this descriptor ("/" excepted: via() looks a name in it up by its whole path). cp_commit calls it
// at its start, before it looks anything up in `folder`.
// The path is walked as cp_prove walks the one given with -isk (WalkedPath::open_by_parts(),
// prover/common/walked_path.h): part by part, through no symbolic link, and only root, the caller
// and the account that ran sudo may be able to swap a part. The "/." is one more part, looked up in
// `folder`, so `folder` itself must meet the same conditions: the new name is looked up there.
// So cp_commit rejects at its start a place for isk that cp_prove would reject afterwards.
// "." is not opened (see reach()); cp_commit passes no relative path.
// Returns 0, or 2 after a rejection.
static int reach_by_parts(const std::string &folder, const std::string &name, const char *option) {
  if (folder == "." || kept_open(folder)) return 0;
  std::string refused;
  const int fd = WalkedPath::open_by_parts(folder + "/.", refused);
  struct stat is;
  if (fd >= 0 && !fstat(fd, &is) && !faccessat(fd, ".", R_OK | W_OK | X_OK, 0)) { // R_OK: make_folder() syncs the new name through a descriptor opened for reading
    road().kept.push_back({folder, fd, is.st_dev, is.st_ino, false});
    return 0;
  }
  if (fd >= 0) close(fd);
  if (refused.empty()) return making::refuse("no-folder-to-write", folder + ", in which " + name + " is to be made, is not a folder that this program can read and write into");
  // The walk's advice is worded for cp_prove. The first exchange is for the added "." alone.
  const std::string last = WalkedPath::shown(folder) + ", in which ";
  for (const auto &[theirs, here] : {std::pair<std::string, std::string>{last + ". is looked up,", last + name + " is to be made,"},
                                     {"Name the folder itself that was named with -spool or -out, by its real path", std::string("Give ") + option + " a path without links: the real path"},
                                     {"If another folder was named with -spool or -out, name that one; if it was this one, run this as the user who ran cp_prove", "Name another place, or run cp_commit as that user"},
                                     {"run this as the user who ran cp_prove", "run cp_commit as that user"}})
    for (size_t at = 0; (at = refused.find(theirs, at)) != std::string::npos; at += here.size()) refused.replace(at, theirs.size(), here);
  return making::refuse("cp-commit-path", std::string("the path named with ") + option + " is refused, and nothing was touched: " + refused + ". Name a place below directories in which only you or root can write, without symbolic links on the way");
}
// Tries in the parent directory of `goal` (<isk> or <ivk>), before a weight is loaded, what the run
// needs there only after its long work. Call it after reach_by_parts() on that parent and after
// refuse_what_is_left(), which has found goal's temporary name free.
// One empty directory goes through the steps of make_folder(), hold_leaf() and rename_to(): it is
// created under the temporary name; opened without following a link; its owner and mode are checked
// (some network file systems keep neither, or map root to another user); it is locked; renamed with
// RENAME_NOREPLACE; the parent is synced; it is removed, only while its name still refers to it.
// Its second name is the temporary name with another ".making", under which no run works:
// cp_commit rejects a -isk or -ivk that ends in ".making". (So a last part of more than 241 bytes,
// with which a run could work, is rejected here.)
// The cleanup is not armed yet, so SIGINT, SIGTERM and SIGHUP are blocked until the directory is
// removed. After SIGKILL, a crash or a power failure it stays, empty, under one of the two names.
// A directory that cannot be opened, or is not the caller's own right after its creation, is left,
// and the message says so: the run cannot tell whether it is the one that it has created.
// Returns 0, or 2 after a rejection.
static int try_the_file_system(const std::string &goal) {
  struct Blocked { // the three signals, for this routine's time; one that has come ends the process at the end
    sigset_t before;
    Blocked() {
      sigset_t these;
      sigemptyset(&these);
      for (int which : {SIGINT, SIGTERM, SIGHUP}) sigaddset(&these, which);
      pthread_sigmask(SIG_BLOCK, &these, &before);
    }
    ~Blocked() { pthread_sigmask(SIG_SETMASK, &before, nullptr); }
  } blocked;
  const std::string working = beside(goal), renamed = beside(working), parent = folder_of(goal);
  const Via from = via(working), to = via(renamed);
  const auto cannot_be = [&](const char *step, const std::string &reason, const std::string &stays) {
    return making::refuse("cp-commit-file-system", "a new folder, " + working + ", cannot be " + step + " in " + parent + ": " + reason + ". A run needs this later, so the call stops before its long work: no weight was loaded, no seed was drawn and no file was written" +
                                                       stays + ". If " + parent + " is on a shared or network file system, give -isk and -ivk names on a local one");
  };
  const auto locks = [](int fd) { // another call's look at the name (what_stands_at()) holds the lock for a moment
    int failed;
    for (int i = 0; (failed = flock(fd, LOCK_EX | LOCK_NB)) && errno == EWOULDBLOCK && i < 100; i++) usleep(10 * 1000);
    return !failed;
  };
  if (mkdirat(from.fd, from.name, 0700))
    return errno == EEXIST ? making::refuse(HF_STOP_FILE_THERE, working + " exists already: it has appeared since this call looked. Give -isk and -ivk two new names: nothing has to be removed first") : cannot_be("created", strerror(errno), "");
  const std::string *own = nullptr;                // the directory's present name, once it is opened and seen to be the caller's own
  const char *failed = nullptr, *reason = nullptr; // the step that failed; why, where it is not the system's errno
  bool taken = false;                              // the second name exists
  struct stat is;
  const int fd = openat(from.fd, from.name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0 || fstat(fd, &is)) failed = "opened";
  else if (is.st_uid != geteuid() || (is.st_mode & 077))
    failed = "kept for the calling user alone", reason = "right after its creation it belongs to another user or has permissions for group or others: the file system keeps no owners or modes, or maps root to another user, or somebody has put another folder there";
  else if (own = &working, !locks(fd)) failed = "locked (flock())";
  else if (renameat2(from.fd, from.name, to.fd, to.name, RENAME_NOREPLACE)) failed = "renamed to a free name (RENAME_NOREPLACE)", taken = errno == EEXIST;
  else if (own = &renamed, !sync_dir(parent)) failed = "synced to the disk under its new name";
  int why = errno;
  const bool left = !own || rmdir_if_same(*own, is.st_dev, is.st_ino);
  if (left && !failed) failed = "removed again", why = errno;
  if (fd >= 0) close(fd); // drops the lock
  if (!failed) return 0;
  const std::string stays = left ? ". What has the name " + (own ? *own : working) + " now is left as it is: the empty folder that this call has just made, unless somebody has put something else there" : "";
  if (taken)
    return making::refuse(HF_STOP_FILE_THERE, renamed + " exists already. cp_commit uses that name for a moment at its start, and a call that was killed just then leaves an empty folder there, which rmdir removes (it removes nothing that is not empty). Or give -isk and -ivk two new names" + stays);
  return cannot_be(failed, reason ? reason : strerror(why), stays);
}
} // namespace making
} // namespace hf
