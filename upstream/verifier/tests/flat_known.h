// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Helpers for unit tests of the verifier program cp_verify, such as
   verifier/tests/hc_forest_attention_test.cpp.
   The helpers read files of expected values ("known answers" in names and messages).
   verifier/tests/known_answers/forest_values.py generates the files
   (verifier/tests/forest_values.txt, verifier/tests/forest_weights.txt) and shares no code with
   the verifier.

   Include this header after the code under test: it needs u64 and `refuse`, which
   verifier/src/vf_common.h provides. By default `refuse` prints "NO: <name>" and exits with
   code 1. */
#pragma once
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// A parsed line "<kind> <name>=<value> .. -> <answer>"; the answer is the rest of the line.
struct KnownLine {
  std::string kind, answer;
  std::map<std::string, std::string> arg;
  // The decimal value of `name`, which the line must have.
  u64 n(const char *name) const {
    const auto it = arg.find(name);
    if (it == arg.end()) refuse(std::string("internal-flat-known-no-") + name);
    return std::stoull(it->second);
  }
};
// The file's lines of one kind, in file order. Calls refuse() if the file is unreadable, one of
// them is malformed or there are none.
static std::vector<KnownLine> knownLines(const std::string &path, const std::string &kind);
// Call this before comparing anything: it rejects a file made for another digest width. The file
// must have exactly one `FORM` line, and its `dw` (32-bit words per digest) must equal `words`:
// pass DW (verifier/src/vf_widths_and_bounds.h), or 0 if no line depends on a digest's width.
static void knownForm(const std::string &path, unsigned words) {
  const std::vector<KnownLine> form = knownLines(path, "FORM");
  if (form.size() != 1 || form[0].n("dw") != words) refuse("internal-known-answers-of-the-other-build");
}
static std::vector<KnownLine> knownLines(const std::string &path, const std::string &kind) {
  std::ifstream in(path);
  if (!in) refuse("internal-flat-known-file");
  std::vector<KnownLine> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    KnownLine k;
    ss >> k.kind;
    if (k.kind != kind) continue;
    std::string tok;
    bool arrow = false;
    while (ss >> tok) {
      if (tok == "->") {
        arrow = true;
        break;
      }
      const size_t e = tok.find('=');
      if (e == std::string::npos) refuse("internal-flat-known-line");
      k.arg[tok.substr(0, e)] = tok.substr(e + 1);
    }
    if (!arrow) refuse("internal-flat-known-line");
    std::getline(ss, k.answer);
    if (!k.answer.empty() && k.answer[0] == ' ') k.answer.erase(0, 1);
    out.push_back(k);
  }
  if (out.empty()) refuse("internal-flat-known-kind-" + kind);
  return out;
}
// A list "a,b,c" of decimal numbers. An empty text gives an empty list.
static std::vector<u64> knownList(const std::string &s) {
  std::vector<u64> v;
  std::istringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(std::stoull(t));
  return v;
}
