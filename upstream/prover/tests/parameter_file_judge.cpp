// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* CPU test program for `parameter_file::load` (prover/common/parameter_file.h), with which the
   programs `cp_commit` and `cp_prove` read a parameter file (`policy.txt`): `load` checks the
   file's form (`parse`), then limits on its numbers (`limit_missed`).
   Prints one line for each path read from standard input (one per line): "admitted" (accepted), or
   `load`'s rejection: a name, a colon and a description.
   It runs no other code of these programs: their later checks on the file, a plan, a table or
   another file are not made here.
   integrations/vllm/integer_model/tests/test_parameter_file.py runs it.
   Run the build command in prover/tests.
   Build: g++ -std=c++17 -O2 -fno-ipa-modref -DBLK_FIELD=2 -DBLK_HASH=1 -I .. -I ../../blocks parameter_file_judge.cpp */
#include "common/parameter_file.h"
#include <iostream>

int main() {
  for (std::string path; std::getline(std::cin, path);) {
    parameter_file::Parameters p;
    const std::string why = parameter_file::load(path, p);
    printf("%s\n", why.empty() ? "admitted" : why.c_str());
  }
  return 0;
}
