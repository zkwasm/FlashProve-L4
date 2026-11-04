// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The prover's view of the parameter file (policy.txt; "policy" in identifiers), a public text file
// that fixes the model's shape and the proof's parameters. prover/common/parameter_file.h parses
// and checks it; `Policy` adds access to values by name (operator[]), the vectors that
// Prover::build_header() (prover/prover/header.cuh) copies into the statement (32-bit words built
// from the public inputs), and the `call` lines split into fields. Call read() first. Host code
// only. Each line is described in verifier/docs/parameter-file.md, "All lines in their order".
#pragma once
#include "common/parameter_file.h"
#include <cstdio>
#include <cstdlib>
struct Policy : parameter_file::Parameters {
  std::vector<std::string> names; // the names of the values in w
  // Values in file order. w (63 words): two leading words (see read()), then `prof` to `seed_hi`.
  // wA: attention's `lB` to `iotaA_0`. wEH: `MF` to `F` (embedding, output head, acceptance
  // policy).
  std::vector<uint32_t> w, wA, wEH;
  // One `call` line: a lookup call's name, its public table's kind, and the names of its lookup
  // segments (its lists of keys).
  struct CL {
    std::string name, kind;
    std::vector<std::string> seg;
  };
  std::vector<CL> cl;
  // Exits with code 2 if Parameters::m has no value named n. For an optional line, use value().
  uint32_t operator[](const std::string &n) const {
    auto it = m.find(n);
    if (it == m.end()) {
      printf("policy: the line %s is missing from the parameter file\n", n.c_str());
      exit(2);
    }
    return it->second;
  }
  // Returns false, after printing the reason, if the file cannot be read, is not in the exact form
  // that parameter_file::parse() accepts, or fails a check of limit_missed(). Call read() once per
  // object: it appends to wA, wEH and cl.
  bool read(const std::string &fn) {
    const std::string why = parameter_file::load(fn, *this);
    if (!why.empty()) {
      printf("policy: %s\n", why.c_str());
      return false;
    }
    // w starts with two words that are not value lines: the statement's tag and the version word
    // (the line `version`). polSkeleton() in verifier/src/cp_verify.cpp writes the same two.
    // `prof` 20: the proof covers the whole model, attention included.
    w = {0x4D363100u, (uint32_t)frame.version}, names = {"magic", "version"};
    const size_t first = 61, attention = 11, head = 25; // accepted files (prof 20) have all three
    for (size_t i = 0; i < first + attention + head; i++)
      (i < first ? w : i < first + attention ? wA : wEH).push_back(lines[i].value);
    for (size_t i = 0; i < first; i++) names.push_back(lines[i].name);
    for (const Call &c : call) cl.push_back({c[0], c[1], std::vector<std::string>(c.begin() + 2, c.end())});
    return true;
  }
};
