// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A unit test of near_host_pin() (prover/common/numa_pin.h), which pins the calling thread to the
// CPUs of a GPU's NUMA node. The test needs neither a GPU nor CUDA. It replaces what the function
// calls: cudaDeviceGetPCIBusId() (declared in numa_pin_stand_in/cuda_runtime.h), fopen(),
// sched_getaffinity(), sched_setaffinity() and printf(). So, with the header as it is, no thread is
// pinned and nothing of this machine is read.
// Each case sets what the replaced calls return, and checks the function's result, the mask that it
// passes to sched_setaffinity(), how often it calls each of the two system calls, whether it prints
// its line, and that it opens no file but the GPU's numa_node (by the bus ID in lower case) and the
// cpulist of the node that this file names. The expected numbers of calls follow the function's
// present order: the two files, then sched_getaffinity(), then the list of CPUs.
// In one case sched_getaffinity() fails and, like the system call, writes nothing. A function that
// went on with the unwritten mask is caught only because the build fills uninitialized local
// variables with a pattern (-ftrivial-auto-var-init=pattern) in which some of the node's CPUs look
// allowed; without the option, g++ 13.3's build of such a function passed.
// In two cases cpulist holds a letter. If the function never returned, the test would end itself
// after 20 seconds, name the case and exit with status 1.
// Not tested: the real system calls and sysfs files; the first two arguments of the two system
// calls; a device other than 0; the printed line's text; whether the files are closed; a cpulist
// with a bare comma or a very long range; CPU numbers of CPU_SETSIZE (1024) or more.
// Build and run, from the repository's top (SUITE.md, prover.numa_pin_test):
//   g++ -O2 -fno-ipa-modref -std=c++17 -Wall -Wextra -ftrivial-auto-var-init=pattern
//     -I prover/tests/numa_pin_stand_in -I prover -o <program> prover/tests/numa_pin_test.cpp
//   <program>
// (-fno-ipa-modref: as in this repository's other g++ calls at -O2;
// verifier/tests/compiler_pass_control.sh shows why.)
// Last line: "numa_pin_test: <n> cases: passes" (exit status 0) or
// "numa_pin_test: <k> of <n> cases differ" (1).
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sched.h>
#include <unistd.h>

// What the replaced calls return in a case, and what is expected of the function.
// The two lists are safe: in aggregate initialization their arrays live as long as the Case.
struct Case {
  const char *name;
  bool bus_id, get_works, set_works;
  const char *node, *cpus;          // the two files' text; nullptr: the file cannot be opened
  std::initializer_list<int> may;   // the CPUs that the thread may use
  int result;                       // expected: near_host_pin()'s result
  int get_calls, set_calls;         // expected: calls of sched_getaffinity() and sched_setaffinity()
  int printed;                      // expected: calls of printf()
  std::initializer_list<int> mask;  // expected: the mask passed to sched_setaffinity(), if it is called
};
static const Case *now;
static int opened, other_paths, get_calls, set_calls, printed;  // counted since the case began
static cpu_set_t passed;

static FILE *test_fopen(const char *path, const char *) {
  opened++;
  if (strcmp(path, "/sys/bus/pci/devices/0000:ab:00.0/numa_node") && strcmp(path, "/sys/devices/system/node/node1/cpulist"))
    other_paths++;
  const char *text = strstr(path, "/numa_node") ? now->node : strstr(path, "/cpulist") ? now->cpus : nullptr;
  return text ? fmemopen(const_cast<char *>(text), strlen(text), "r") : nullptr;
}
static int test_getaffinity(pid_t, size_t, cpu_set_t *mask) {
  get_calls++;
  if (!now->get_works) {
    errno = EINVAL;
    return -1;
  }
  CPU_ZERO(mask);
  for (int cpu : now->may) CPU_SET(cpu, mask);
  return 0;
}
static int test_setaffinity(pid_t, size_t, const cpu_set_t *mask) {
  set_calls++;
  passed = *mask;
  if (now->set_works) return 0;
  errno = EINVAL;
  return -1;
}
static int test_printf(const char *, ...) { return printed++, 0; }

// Every system header that numa_pin.h includes is included above, so these four names are replaced
// in numa_pin.h's own lines alone.
#define fopen test_fopen
#define sched_getaffinity test_getaffinity
#define sched_setaffinity test_setaffinity
#define printf test_printf
#include "common/numa_pin.h"
#undef fopen
#undef sched_getaffinity
#undef sched_setaffinity
#undef printf

cudaError_t cudaDeviceGetPCIBusId(char *id, int length, int) {
  snprintf(id, (size_t)length, "0000:AB:00.0");
  return now->bus_id ? cudaSuccess : 1;
}

static void too_long(int) {
  const char *said = "numa_pin_test: near_host_pin() did not return within 20 seconds in the case: ";
  (void)!write(1, said, strlen(said));
  (void)!write(1, now->name, strlen(now->name));
  (void)!write(1, "\n", 1);
  _exit(1);
}

int main() {
  const char *node1 = "1\n", *list = "0-3,6\n";  // the GPU is on node 1, whose CPUs are 0, 1, 2, 3 and 6
  const Case cases[] = {
      {"the thread may use CPUs 0 to 4", true, true, true, node1, list, {0, 1, 2, 3, 4}, 4, 1, 1, 1, {0, 1, 2, 3}},
      {"the thread may use CPUs 1, 6 and 9", true, true, true, node1, list, {1, 6, 9}, 2, 1, 1, 1, {1, 6}},
      {"the thread may use none of the node's CPUs", true, true, true, node1, list, {8, 9}, 0, 1, 0, 0, {}},
      {"sched_getaffinity() fails", true, false, true, node1, list, {}, 0, 1, 0, 0, {}},
      {"sched_setaffinity() fails", true, true, false, node1, list, {0, 1, 2, 3, 4}, 0, 1, 1, 0, {0, 1, 2, 3}},
      {"the GPU has no PCI bus ID", false, true, true, node1, list, {0, 1}, 0, 0, 0, 0, {}},
      {"numa_node cannot be opened", true, true, true, nullptr, list, {0, 1}, 0, 0, 0, 0, {}},
      {"numa_node says -1", true, true, true, "-1\n", list, {0, 1}, 0, 0, 0, 0, {}},
      {"cpulist cannot be opened", true, true, true, node1, nullptr, {0, 1}, 0, 0, 0, 0, {}},
      {"cpulist is an empty line", true, true, true, node1, "\n", {0, 1}, 0, 1, 0, 0, {}},
      {"cpulist is a letter", true, true, true, node1, "x\n", {0, 1, 2, 3, 4}, 0, 1, 0, 0, {}},
      {"cpulist has a letter after a range", true, true, true, node1, "0-3,x\n", {0, 1, 2, 3, 4}, 0, 1, 0, 0, {}},
  };
  setvbuf(stdout, nullptr, _IOLBF, 0);  // so that the cases before one that never ends are on record
  signal(SIGALRM, too_long);
  alarm(20);
  int count = 0, differ = 0;
  for (const Case &c : cases) {
    now = &c;
    opened = other_paths = get_calls = set_calls = printed = 0;
    CPU_ZERO(&passed);
    int result = near_host_pin();
    bool mask = CPU_COUNT(&passed) == (int)c.mask.size();
    for (int cpu : c.mask) mask = mask && CPU_ISSET(cpu, &passed);
    bool same = result == c.result && get_calls == c.get_calls && set_calls == c.set_calls && printed == c.printed && mask &&
                !other_paths && (c.bus_id || !opened);
    count++, differ += !same;
    printf("%s %s: result %d; sched_getaffinity() called %d time(s), sched_setaffinity() %d; %d CPU(s) in the mask passed%s; "
           "lines printed: %d; files opened: %d, %d of them by an unexpected path\n",
           same ? "as expected    " : "NOT AS EXPECTED", c.name, result, get_calls, set_calls, CPU_COUNT(&passed),
           mask ? "" : ", not the expected ones", printed, opened, other_paths);
    if (!same)
      printf("  expected: result %d; sched_getaffinity() called %d time(s), sched_setaffinity() %d; %zu CPU(s) in the mask; "
             "lines printed: %d; %s\n",
             c.result, c.get_calls, c.set_calls, c.mask.size(), c.printed,
             c.bus_id ? "no file opened by an unexpected path" : "no file opened");
  }
  if (differ) return printf("numa_pin_test: %d of %d cases differ\n", differ, count), 1;
  printf("numa_pin_test: %d cases: passes\n", count);
}
