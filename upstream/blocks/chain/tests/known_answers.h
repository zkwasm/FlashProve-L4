// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Helpers for tests that compare the library (blocks/) with docs/format/format-vectors.txt.
//
// Its lines "name = value" are known answers for rules R1.x to R8.x and R12.x to R15.x of
// docs/format/FORMAT.md. They are printed by docs/format/format_rules.py, which imports nothing of
// blocks/; its head says why its answers for R15.x are not independent of blocks/. A test calls
// read_known_answers(), then expect(name, value) for each value that the library computes, and
// returns known_answers_end(). Lines for "field 1" are left out (blocks/base.h accepts only
// `BLK_FIELD` = 2); lines never asked for are only counted.
#pragma once
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "hash/hash.h"
#include "iop/message.h"

using namespace blk;

static int failures = 0, compared = 0;
static std::map<std::string, std::string> known;
// A matching line is erased: a second call with its name fails.
[[maybe_unused]] static void expect(const std::string &name, const std::string &value) {
  compared++;
  auto it = known.find(name);
  if (it == known.end())
    failures++, std::printf("FAILED: the file has no line \"%s\"\n", name.c_str());
  else if (it->second != value)
    failures++, std::printf("FAILED: %s\n  the page's: %s\n  the programs': %s\n", name.c_str(), it->second.c_str(), value.c_str());
  else
    known.erase(it);
}
[[maybe_unused]] static std::string hex(const u8 *b, size_t n) {
  std::string s;
  char two[3];
  for (size_t i = 0; i < n; i++) std::snprintf(two, sizeof two, "%02x", b[i]), s += two;
  return s;
}
[[maybe_unused]] static std::string hex(const std::vector<u8> &b) { return hex(b.data(), b.size()); }
[[maybe_unused]] static std::string hex(const Digest &d) {
  u8 b[DIGEST_BYTES];
  digest_to_bytes(d, b);
  return hex(b, DIGEST_BYTES);
}
[[maybe_unused]] static std::string text(const E &x) { return std::to_string(x.c[0].v) + " " + std::to_string(x.c[1].v) + " " + std::to_string(x.c[2].v); }
// docs/format/format-vectors.txt calls these bytes pattern(n), or pattern(n) with a i + b.
[[maybe_unused]] static std::vector<u8> pattern(size_t n, unsigned a = 7, unsigned b = 3) {
  std::vector<u8> p(n);
  for (size_t i = 0; i < n; i++) p[i] = (u8)((a * i + b) % 251);
  return p;
}
[[maybe_unused]] static Digest digest_of(const std::vector<u8> &b, size_t at = 0) {
  Digest d;
  digest_from_bytes(b.data() + at, d);
  return d;
}
[[maybe_unused]] static E element(u64 a, u64 b, u64 c) { return E{{F{a}, F{b}, F{c}}}; }
// H(168, 0, index, none; bytes), defined in docs/format/FORMAT.md, R2.1.
[[maybe_unused]] static Digest H168(u64 index, const std::vector<u8> &bytes) {
  Hasher h(Tag(Domain::STATEMENT), index);
  h.push(bytes.data(), bytes.size());
  return h.finish();
}

static size_t known_lines = 0, known_other_field = 0;
// Run from blocks/, or pass the file's path as argv[1].
// Returns 0, or the exit code that the test must return at once.
static int read_known_answers(int argc, char **argv) {
  size_t &lines = known_lines, &other_field = known_other_field;
  const std::string path = argc > 1 ? argv[1] : "../docs/format/format-vectors.txt";
  std::ifstream in(path);
  if (!in) return std::printf("FAILED: %s cannot be read (run from blocks/, or give the path)\n", path.c_str()), 1;
  {
    int field = 0; // field named in the last '#' heading; 0: none
    const std::string mine = "field " + std::to_string(BLK_FIELD), others = "field " + std::to_string(3 - BLK_FIELD);
    for (std::string line; std::getline(in, line);) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        if (line.find(mine) != std::string::npos) field = BLK_FIELD;
        else if (line.find(others) != std::string::npos) field = 3 - BLK_FIELD;
        else if (line.rfind("# sections 5 and 6", 0) != 0) field = 0; // this heading keeps `field`
        continue;
      }
      lines++;
      const size_t cut = line.rfind(" = "); // a name may contain " = " too, a value never
      if (cut == std::string::npos) return std::printf("FAILED: a line without \" = \": %s\n", line.c_str()), 1;
      std::string name = line.substr(0, cut);
      while (!name.empty() && name[0] == ' ') name.erase(0, 1);
      if (field == 3 - BLK_FIELD || name.rfind(others + ",", 0) == 0) {
        other_field++;
        continue;
      }
      // Indented lines repeat names: each is prefixed with the last unindented line's name.
      static std::string before;
      if (line[0] == ' ') name = before + ": " + name;
      else before = name;
      known[name] = line.substr(cut + 3);
    }
  }
  return 0;
}
// After a failure, or with a second command-line argument, lists the names left in `known`.
static int known_answers_end(const char *test, int argc) {
  std::printf("%s, field %d: %d lines compared, %d FAILURES; of the file's %zu lines %zu are the other field's and %zu were not asked for by this test\n", test, BLK_FIELD,
              compared, failures, known_lines, known_other_field, known.size());
  if (failures || argc > 2)
    for (const auto &left : known) std::printf("  not asked for: %s\n", left.first.substr(0, 110).c_str());
  return failures ? 1 : 0;
}
