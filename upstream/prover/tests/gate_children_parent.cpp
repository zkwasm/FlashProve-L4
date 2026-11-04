// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Started by prover/tests/gate_children.py, which kills it with SIGKILL at chosen moments and
// checks what it leaves running. It runs a program as cp_commit runs cp_verify (`judge` in
// identifiers): through hf::making::judge_says(), on a key file or a leaf file
// (prover/hash_commitment/hf_commit_making.h), or through judge::judge_says(), which that calls.
//   gate_children_parent <mode> <time limit in ms> <program> [<argument> ...]
// <mode>: supervised or direct: judge::judge_says() itself; cp_commit or cp_commit-direct, as cp_commit.
// supervised, cp_commit: main() begins with judge::supervise_if_asked(), as in cp_commit.
// judge_says() then starts this program again as a supervisor (first argument: judge::SUPERVISE),
// which starts <program> in its process group and kills the group when its caller dies. The
// supervisor starts only the path that judge::find_the_judge() returns: cp_verify in this
// program's directory. The test also starts supervisors itself.
// direct, cp_commit-direct, the test's controls: judge_says() starts <program> itself.
// <time limit>: 0: with judge::judge_says() itself, no limit; as cp_commit, the compiled limit
// (hf::making::check_limit_ms). cp_commit-direct has no time limit.
// <program> gets three inputs (judge::INPUT): "first", "second", "third".
// Prints "calls" before the call; after it "status <n>", where n is what judge_says() returned, and
// the program's standard output. As cp_commit, <program> writes to this program's standard output
// itself, so its output comes before "status <n>". Exit code 0, or 2 for wrong usage; as a
// supervisor, what supervise_if_asked() returns.
#include "common/setting.h"
#include "common/judge.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "hash_commitment/hf_commit_making.h"
int main(int argc, char **argv) {
  if (argc < 2 || (strcmp(argv[1], "direct") && strcmp(argv[1], "cp_commit-direct")))
    if (const int status = judge::supervise_if_asked(argc, argv); status >= 0) return status;
  const std::string mode = argc < 4 ? "" : argv[1];
  if (mode != "supervised" && mode != "direct" && mode != "cp_commit" && mode != "cp_commit-direct") return 2;
  const std::vector<const char *> words(argv + 3, argv + argc);
  const std::string first = "first", second = "second", third = "third";
  std::string heard;
  int status = 0;
  printf("calls\n");
  fflush(stdout);
  if (mode == "cp_commit" || mode == "cp_commit-direct") { // cp_commit calls from its main thread
    if (atoll(argv[2])) hf::making::check_limit_ms = atoll(argv[2]);
    status = hf::making::judge_says(argv[3], words, {&first, &second, &third});
  } else {
    judge::Running among;
    among.limit_ms = atoll(argv[2]);
    // From a thread other than the main thread: the kernel sends the supervisor's parent-death
    // signal when the thread that started it ends.
    std::thread([&] { status = judge::judge_says(argv[3], words, {&first, &second, &third}, &heard, &among); }).join();
  }
  printf("status %d\n%s", status, heard.c_str());
  return 0;
}
