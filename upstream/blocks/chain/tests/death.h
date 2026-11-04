// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests: runs code in a child process and reports whether it returns or aborts. A programming
// error must end the process: a failed BLK_REQUIRE (blocks/base.h) prints its message to stderr
// and, by default, calls abort().
#pragma once
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// True if `body`, run in a child, aborts, by BLK_REQUIRE or otherwise, with `text` in its stderr.
template <class Body> static bool ends_with(const char *text, Body body) {
  std::fflush(stdout);
  int fd[2];
  if (pipe(fd) != 0) return false;
  const pid_t child = fork();
  if (child < 0) { std::perror("tests: no child process"); std::abort(); } // not BLK_REQUIRE: base.h may come after this file
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    close(fd[0]);
    dup2(fd[1], 2);
    body();
    _exit(0);
  }
  close(fd[1]);
  std::string said;
  char buffer[256];
  for (ssize_t n; (n = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(fd[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child) { std::perror("tests: a child process is lost"); std::abort(); }
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && said.find(text) != std::string::npos;
}
// True if `body`, run in a child process, returns; it can call _exit(1) to make the result false.
template <class Body> static bool runs_through(Body body) {
  std::fflush(stdout);
  const pid_t child = fork();
  if (child < 0) { std::perror("tests: no child process"); std::abort(); }
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    body();
    _exit(0);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) { std::perror("tests: a child process is lost"); std::abort(); }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
