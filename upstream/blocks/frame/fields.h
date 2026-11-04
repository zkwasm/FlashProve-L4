// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The fields of a key, leaf (`entry`) or proof file (a plan's `part`): name, offset, length, class.
   One per `PlanMessage`, contiguous from offset 0. `cp_verify -plan request .. -fields` (the
   verifier program) prints them; verifier/tests/proof_fields_sweep_test.py uses that list to alter
   a proof. The class states what a reader does with the bytes:
     words      compared with the reader's value (`place in the batch`: with an upper bound)
     elements   field elements, eight bytes a coordinate: rejected unless each is below the prime p
     digest     32 bytes each, any value can be read
     nonce      8 bytes, tested against the round's proof of work
     positions  codeword indices, compared with those that the reader derives
     salt       any value can be read
   Answers to draws of positions are listed at their largest length, so later offsets are upper
   bounds (`exact` false, printed `at-most`). An opening is listed as symbols, salts, paths; in the
   file each salt follows its position's symbols. */
#pragma once
#include <cstdio>
#include <string>
#include <vector>

#include "numbers/plan.h"

namespace blk {

struct Field {
  std::string name;
  u64 at = 0, bytes = 0;
  const char *cls = "";
  bool exact = true;
};

inline const char *class_of(MessageKind k) {
  switch (k) {
    case MessageKind::SCALARS:
    case MessageKind::ANSWERS:
    case MessageKind::COLUMNS: return "elements";
    case MessageKind::ROOT:
    case MessageKind::PATHS: return "digest";
    case MessageKind::NONCE: return "nonce";
    case MessageKind::SALTS: return "salt";
    case MessageKind::HEAD: return "words";
  }
  return "";
}

// Ends the process on a part planned with repeats (`count` not 1).
inline std::vector<Field> fields_of(const Planned &p, unsigned part) {
  std::vector<Field> v;
  u64 at = 0;
  bool exact = true;
  for (auto &m : p->messages_) {
    if (m.part != part) continue;
    BLK_REQUIRE(m.count == 1, "fields_of: a plan with repeats has no places");
    Field f;
    f.name = m.where.empty() ? m.name : m.where + ": " + m.name;
    f.at = at, f.bytes = m.bytes, f.cls = class_of(m.kind), f.exact = exact;
    v.push_back(f);
    at += m.bytes;
    if (m.expected != (double)m.bytes) exact = false;
  }
  return v;
}

inline void print_fields(const Planned &p, unsigned part) {
  for (auto &f : fields_of(p, part))
    std::printf("field: %s %s %llu %llu %s: %s\n", p->parts[part].c_str(), f.exact ? "at" : "at-most", (unsigned long long)f.at, (unsigned long long)f.bytes, f.cls,
                f.name.c_str());
}

}  // namespace blk
