// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Reads a configuration file, and forms layout.txt's path. Host code.
#pragma once
#include "common/setting.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Reads a file of any size (parameter_file::read_once() has a limit). False if open or read fails.
inline bool whole_file(const std::string &path, std::string &bytes) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  bytes.clear();
  char piece[1 << 16];
  size_t n;
  while ((n = fread(piece, 1, sizeof piece, f)) > 0) bytes.append(piece, n);
  const bool whole = !ferror(f);
  fclose(f);
  return whole;
}

// Path of layout.txt in the directory of the parameter file (policy.txt; `policy` in identifiers).
inline std::string layout_beside(const std::string &policy_path) {
  const size_t slash = policy_path.rfind('/');
  return (slash == std::string::npos ? std::string(".") : policy_path.substr(0, slash)) + "/layout.txt";
}
