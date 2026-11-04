// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// reduced_env_start() starts a program that handles a secret key (the committer cp_commit, the prover
// cp_prove) a second time, with a reduced environment: some CUDA variables, for example, make the
// driver write GPU memory to a file. Clearing the environment in main() is too late for the dynamic
// loader (LD_PRELOAD, LD_LIBRARY_PATH, LD_AUDIT, GLIBC_TUNABLES) and for the start-up code of
// linked libraries. In the second start the loader and all libraries, also those loaded later, see
// only the reduced environment. Call reduced_env_start() early in main(): its comment has the
// conditions.
//
// The second start protects against settings that a user made by accident or habit. It does not
// protect against a malicious library built to hide itself: preloaded into the first start, it runs
// inside this program and can start it again with any environment. /etc/ld.so.preload and
// /etc/ld.so.cache still apply.
#pragma once
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
extern char **environ;
// Calls execve() on the program's own file, in the same process, with main()'s argv and only these
// variables: CUDA_VISIBLE_DEVICES, by which an operator selects the GPU; CUDA_DEVICE_ORDER, which
// decides how a number in the former is read; and those named in `more`. Returns null only in the
// second start. Any other result is an error message in static storage, and the caller must exit.
//
// Code in main() before this call runs in both starts, the first time in the original environment.
// Before the call, the caller must generate no secret key, call no CUDA function, create no file,
// print nothing unless it then exits, and leave argv, the environment and the open descriptors
// unchanged: as with any execve(), argv, the descriptors and the variables above pass to the
// second start. execve() sets the dumpable flag (PR_SET_DUMPABLE; 0 prevents core files) back
// to 1. A caller that needs 0 sets it after this call, or before it, since that code runs again in
// the second start. The second start runs the same file as the first, even if that file has been
// renamed or removed.
//
// Not supported:
// - a system without /proc;
// - a library that the loader finds only through LD_LIBRARY_PATH;
// - running the program as the loader's argument: the second start would be the loader, with the
//   program's arguments;
// - a layer that rewrites the file name on its way to the kernel (valgrind --trace-children=yes,
//   proot): every start would take itself for the first and start again, without end.
// Tools and options that work through the environment (compute-sanitizer, Nsight, a preloaded
// allocator, ASAN_OPTIONS) do not reach the second start.
static inline const char *reduced_env_start(char **argv, std::initializer_list<const char *> more) {
  // Returns the first entry for `name`, which is the one that `getenv` finds.
  auto entry_of = [](const char *name, size_t length) -> char * {
    for (char **e = environ; e && *e; e++)
      if (!strncmp(*e, name, length) && (*e)[length] == '=') return *e;
    return nullptr;
  };
  std::vector<const char *> kept = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"};
  kept.insert(kept.end(), more);
  // the second start's environment: the entries found for `kept`, each once
  std::vector<char *> list;
  for (const char *name : kept)
    if (char *entry = entry_of(name, strlen(name)))
      if (std::find(list.begin(), list.end(), entry) == list.end()) list.push_back(entry);
  // The second start recognizes itself by AT_EXECFN, the file name that the kernel recorded at
  // execve(). "/proc/self/exe" refers to the file of the process that calls execve(), so as a
  // rule a program can be started under that name only by itself. A start under that name never
  // starts the program again. It continues only if the environment has at most one entry for each
  // variable in `kept` and no other entry; otherwise reduced_env_start() returns an error message.
  const char *started_as = (const char *)getauxval(AT_EXECFN);
  if (!started_as) return "the kernel does not say under which file name the program was started";
  if (!strcmp(started_as, "/proc/self/exe")) {
    size_t entries = 0;
    for (char **e = environ; e && *e; e++) entries++;
    if (entries != list.size()) return "the second start finds more in its environment than the first gave it";
    // execve() has named the process "exe", after its path's last part; ps and top should show the
    // last part of argv[0].
    const char *slash = argv[0] ? strrchr(argv[0], '/') : nullptr;
    if (argv[0]) prctl(PR_SET_NAME, slash ? slash + 1 : argv[0]);
    return nullptr;
  }
  list.push_back(nullptr);
  execve("/proc/self/exe", argv, list.data());
  static char failed[160];
  snprintf(failed, sizeof failed, "execve() of /proc/self/exe: %s", strerror(errno));
  return failed;
}
