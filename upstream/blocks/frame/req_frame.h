// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// From the arrays (named matrices) that a request's sub-proofs declare and a layout file, builds
// the request's segments (`ReqShape::link`, blocks/iop/req.h) and their arrays' layout
// (`ReqLayout`, blocks/iop/req_rows.h). Prover and verifier declare the arrays separately and share
// this code.
// A sub-proof is the argument for one part of the forward pass. A segment (`link` in identifiers,
// messages and the layout file) is one sub-proof's part of a request's proof; a commit point
// (`moment`) is a prover message of a segment that commits arrays.
//
// layout_read() and req_frame() return null or an error string; neither exits on a file's
// content. The result must still pass req_fault() and req_layout_fault(): nothing here checks that
// blocks are aligned, stay inside their array and group, do not overlap and cover their arrays.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

#include "base.h"
#include "iop/req.h"
#include "iop/req_rows.h"

namespace blk {

// An array and a segment as their sub-proof declares them.
// F: the base field; E: its extension.
struct FrameArray {
  std::string name;
  u64 rows = 1, columns = 1;
  bool over_E = false;  // its elements are in E: it is stored in rows over E; else in rows over F
  unsigned life = 0;    // the last segment to read it is this many after its own; 0: only its own
};
struct FrameLink {
  std::string name;
  std::vector<std::vector<FrameArray>> moment;  // [moment][the array's number in the layout file]
};

// The layout file (`layout.txt`, written by prover/tools/pack_layout.py) places each array in the
// rows of one oracle message (encoded rows, sent as Merkle roots). Blank lines and lines beginning
// with # are skipped. Segments, commit points and groups are numbered from 1; arrays, rows and
// places (indices within a row) from 0.
//   a group:   link moment group class rows life
//   a block:   link moment group array R a K e low row place br bj 0
// A group is one oracle message of its commit point (elsewhere in blocks/: a set of oracle
// messages tested together). Class N: rows over F. Class B: rows over F, kept for `life` later
// segments to read; other classes have life 0. Class W: rows over E. Group lines are ordered by
// link, moment, group (req_frame() checks this; for links, only that the next link has none yet).
// A block (`LaidBlock`, blocks/iop/laid.h) holds the elements (R + i, K + c), i < 2^a, c < 2^e, of
// array number `array` of its commit point, in 2^br rows by 2^bj places: br + bj = a + e. Element
// number x = c 2^a + i (low = r) or i 2^e + c (low = c) is in row `row` + (x >> bj) of the group,
// at place `place` + (x mod 2^bj). The last number of a block line must be 0.
// A segment's mask row (random values) is not in the file: req_frame() adds it to the segment's
// first group, which cannot be class B.
struct LayoutGroup {
  unsigned link = 0, moment = 0, group = 0;  // from 0 here
  char cls = 'N';
  u64 rows = 0;
  unsigned life = 0;
};
struct LayoutBlock {
  unsigned link = 0, moment = 0, group = 0, array = 0;
  LaidBlock at;
};
struct LayoutLines {
  std::vector<LayoutGroup> groups;
  std::vector<LayoutBlock> blocks;
};

// Checks each line by itself. %llu also accepts a sign and leading zeros, so one layout has many
// spellings, each with another configuration digest.
inline const char *layout_read(const std::string &text, LayoutLines &out) {
  out = LayoutLines();
  for (size_t at = 0; at < text.size();) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    const std::string line = text.substr(at, end - at);
    at = end + 1;
    if (line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '#') continue;
    unsigned long long link, moment, group, array, R, a, K, e, row, place, br, bj, sj, rows, life;
    char c = 0, tail = 0;
    if (std::sscanf(line.c_str(), "%llu %llu %llu %llu %llu %llu %llu %llu %c %llu %llu %llu %llu %llu %c", &link, &moment, &group, &array, &R, &a,
                    &K, &e, &c, &row, &place, &br, &bj, &sj, &tail) == 14) {
      if (!link || !moment || !group || link >> 20 || moment >> 20 || group >> 20 || array >> 20) return "a block's line has a link, moment or group 0 (they are counted from 1), or a number of 2^20 or more";
      if ((c != 'r' && c != 'c') || sj != 0 || a >= 48 || e >= 48 || br >= 48 || bj >= 48) return "a block's line has another letter than r or c, a last number that is not 0, or a logarithm of 48 or more";
      LayoutBlock b;
      b.link = (unsigned)link - 1, b.moment = (unsigned)moment - 1, b.group = (unsigned)group - 1, b.array = (unsigned)array;
      b.at.first_i = R, b.at.bits_i = (unsigned)a, b.at.first_c = K, b.at.bits_c = (unsigned)e, b.at.i_is_low = c == 'r';
      b.at.first_row = row, b.at.first_place = place, b.at.row_bits = (unsigned)br, b.at.place_bits = (unsigned)bj;
      out.blocks.push_back(b);
    } else if (std::sscanf(line.c_str(), "%llu %llu %llu %c %llu %llu %c", &link, &moment, &group, &c, &rows, &life, &tail) == 6) {
      if (!link || !moment || !group || link >> 20 || moment >> 20 || group >> 20) return "a group's line has a link, moment or group 0 (they are counted from 1), or a number of 2^20 or more";
      if ((c != 'N' && c != 'B' && c != 'W') || !rows || rows >> 32 || life >> 20 || (life != 0) != (c == 'B')) return "a group's line has another class than N, B, W, no rows or 2^32 or more, or a life that does not go with its class (B alone has one)";
      LayoutGroup g;
      g.link = (unsigned)link - 1, g.moment = (unsigned)moment - 1, g.group = (unsigned)group - 1, g.cls = c, g.rows = rows, g.life = (unsigned)life;
      out.groups.push_back(g);
    } else {
      return "a line is neither a group's nor a block's";
    }
  }
  return nullptr;
}

// Fills shape.link and layout, checking the file against the declared arrays; the rest of the shape
// is the caller's. Set shape.code and shape.mask_code first: each oracle message copies them. Each
// `ReqLink::plain` (number of unmasked scalars) is reset to 0. answers: the number of answers to
// each oracle message's out-of-domain sample; req_fault() requires shape.answers to equal it.
inline const char *req_frame(const std::vector<FrameLink> &links, const LayoutLines &file, unsigned answers, ReqShape &shape, ReqLayout &layout) {
  shape.link.clear(), layout.link.clear();
  std::vector<std::vector<std::vector<size_t>>> message;  // [link][moment][group]: the oracle message's number in its link
  for (const FrameLink &l : links) {
    ReqLink x;
    x.name = l.name;
    shape.link.push_back(x);
    layout.link.emplace_back();
    message.emplace_back(l.moment.size());
  }
  for (const LayoutGroup &g : file.groups) {
    if (g.link >= links.size() || g.moment >= links[g.link].moment.size()) return "a group lies in a link or a moment that the request does not have";
    std::vector<ReqMessage> &sent = shape.link[g.link].messages;
    if (g.group != message[g.link][g.moment].size()) return "the groups of a moment are not numbered in turn";
    if (!sent.empty() && sent.back().moment > g.moment) return "the moments of a link do not come in turn";
    if (g.link + 1 < links.size() && !shape.link[g.link + 1].messages.empty()) return "the links do not come in turn";
    ReqMessage m;
    m.shape.name = links[g.link].name + ", moment " + std::to_string(g.moment + 1) + ", group " + std::to_string(g.group + 1);
    m.shape.code = shape.code, m.shape.mask_code = shape.mask_code;
    m.shape.main_rows = g.rows, m.shape.main_deg = g.cls == 'W' ? E_DEGREE : 1, m.shape.answers = answers;
    m.shape.mask_rows = sent.empty() ? 1 : 0;  // the segment's mask row
    m.moment = g.moment, m.life = g.life;
    if (sent.empty() && g.life) return "the first oracle message of a link holds its mask row and cannot be kept";
    message[g.link][g.moment].push_back(sent.size());
    sent.push_back(m);
  }
  std::vector<std::vector<std::vector<size_t>>> array;  // [link][moment][array]: its number in layout.link[link], or none yet
  for (const FrameLink &l : links) {
    array.emplace_back();
    for (const std::vector<FrameArray> &m : l.moment) array.back().emplace_back(m.size(), ~size_t(0));
  }
  for (const LayoutBlock &b : file.blocks) {
    if (b.link >= links.size() || b.moment >= links[b.link].moment.size() || b.group >= message[b.link][b.moment].size())
      return "a block lies in a group that the file does not have";
    if (b.array >= links[b.link].moment[b.moment].size()) return "a block belongs to an array that its moment does not have";
    size_t &at = array[b.link][b.moment][b.array];
    if (at == ~size_t(0)) {
      const FrameArray &a = links[b.link].moment[b.moment][b.array];
      const ReqMessage &in = shape.link[b.link].messages[message[b.link][b.moment][b.group]];
      if ((in.shape.main_deg == E_DEGREE) != a.over_E) return "an array over E lies in rows over F, or an array over F in rows over E";
      if (in.life < a.life) return "an array is kept for fewer links than its last reader is on";
      at = layout.link[b.link].size();
      layout.link[b.link].push_back(ReqArray{a.name, message[b.link][b.moment][b.group], LaidMatrix{a.rows, a.columns, {}}});
    }
    ReqArray &a = layout.link[b.link][at];
    if (a.message != message[b.link][b.moment][b.group]) return "an array lies in two groups";
    a.laid.blocks.push_back(b.at);
  }
  for (size_t l = 0; l < links.size(); l++)
    for (const std::vector<size_t> &m : array[l])
      for (size_t at : m)
        if (at == ~size_t(0)) return "an array has no block";
  return nullptr;
}

}  // namespace blk
