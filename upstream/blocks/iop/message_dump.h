// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests only. Writes a `Writer` (blocks/iop/message.h: a byte buffer of typed items) to two
// files for blocks/tests/hiding/hull.cpp:
//   <stem>.bin   the bytes, unchanged;
//   <stem>.list  a line "I <kind> <offset> <bytes>" per item, then "M <offset> <words>" per mark.
// <kind> is `Writer::Item::kind`; <words> is the text given to mark(). Call keep_list() before the
// Writer's first byte: message_dump() ends the process if no item is listed, but does not catch a
// partial list, and hull.cpp uses no byte outside the items. Both files are created: if anything
// has one of the two names already, a symbolic link included, message_dump() ends the process and
// leaves it as it is; an empty file that this call created at the other name may stay behind.
#pragma once
#include <cstdio>
#include <string>

#include "base.h"
#include "iop/message.h"

BLK_MARK_NOT_FOR_SERVING(); // blocks/fence.h: marks the binary; compile error under `SERVING_BUILD`

namespace blk {

inline void message_dump(const Writer &w, const std::string &stem) {
  BLK_REQUIRE(!w.items().empty(), "message_dump: the Writer must keep its list (keep_list())");
  // x: fails if the name exists, even a symbolic link.
  FILE *bin = std::fopen((stem + ".bin").c_str(), "wbx"), *list = std::fopen((stem + ".list").c_str(), "wx");
  BLK_REQUIRE(bin && list, "message_dump: a file cannot be created (neither <stem>.bin nor <stem>.list may exist yet)");
  BLK_REQUIRE(std::fwrite(w.data(), 1, w.size(), bin) == w.size(), "message_dump: the bytes are not written");
  for (const Writer::Item &i : w.items()) std::fprintf(list, "I %c %zu %zu\n", i.kind, i.at, i.bytes);
  for (const Writer::Mark &m : w.marks()) std::fprintf(list, "M %zu %s\n", m.at, m.words.c_str());
  BLK_REQUIRE(!std::fclose(bin) && !std::fclose(list), "message_dump: a file is not closed");
}

} // namespace blk
