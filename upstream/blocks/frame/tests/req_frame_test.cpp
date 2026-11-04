// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/frame/req_frame.h: from a layout file to a request's segments (parts of its
// proof, one for each sub-proof; `link` in names and output) and their layout. `SMALL`
// must pass layout_read(), req_frame() and req_layout_fault() (blocks/iop/req_rows.h); each copy
// with one fault must fail one of them (which one is not checked). With arguments "log_ell file .."
// the program instead reports on those files, e.g. a packing tool's output: try_files().
#include <fstream>
#include <map>
#include <sstream>

#include "frame/req_frame.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// Two segments, rows of 16 entries. The first has three commit points (`moment`): cells and a
// matrix kept for the next segment (class B, life 1), a vector over E (extension field), digits.
// blocks/frame/req_frame.h defines the lines' fields.
static const char *SMALL =
    "# a small layout\n"
    "1 1 1 N 2 0\n"
    "1 1 1 0 0 1 0 3 r 0 0 0 4 0\n"   // array 0: 2 x 8 in row 0
    "1 1 1 1 0 0 0 3 c 1 0 0 3 0\n"   // array 1: 1 x 8 in the first half of row 1
    "1 1 2 B 2 1\n"
    "1 1 2 2 0 1 0 2 c 0 0 1 2 0\n"   // array 2: 2 x 4, one row in each of two rows
    "1 2 1 W 1 0\n"
    "1 2 1 0 0 0 0 3 c 0 0 0 3 0\n"
    "1 2 1 0 0 0 8 1 c 0 8 0 1 0\n"   // 10 elements: 8 and 2
    "1 3 1 N 1 0\n"
    "1 3 1 0 0 0 0 3 c 0 0 0 3 0\n"
    "2 1 1 N 1 0\n"
    "2 1 1 0 0 0 0 3 c 0 0 0 3 0\n";

static std::vector<FrameLink> small_links() {
  FrameLink a, b;
  a.name = "first", b.name = "second";
  a.moment = {{{"A", 2, 8}, {"B", 1, 8}, {"K", 2, 4, false, 1}}, {{"R", 1, 10, true}}, {{"D", 1, 8}}};
  b.moment = {{{"A", 1, 8}}};
  return {a, b};
}

// Null if all three pass.
static const char *whole(const std::string &text, const std::vector<FrameLink> &links, const Codes &codes, ReqShape &shape, ReqLayout &layout) {
  LayoutLines lines;
  if (const char *fault = layout_read(text, lines)) return fault;
  if (const char *fault = req_frame(links, lines, 1, shape, layout)) return fault;
  return req_layout_fault(codes, shape, layout);
}

static std::string replaced(std::string text, const std::string &from, const std::string &to) {
  const size_t at = text.find(from);
  BLK_REQUIRE(at != std::string::npos, "test: the line to be changed is not there");
  return text.replace(at, from.size(), to);
}

// Rows have 2^log_ell entries. Each array's size, class and life come from the file, so no
// sub-proof's declaration is compared with it. An unreadable or empty file is reported `lawful`.
static int try_files(int argc, char **argv) {
  const unsigned log_ell = (unsigned)std::atoi(argv[1]);
  Codes codes;
  codes.base.push_back(Code{log_ell + 1, size_t(1) << log_ell, 8});
  codes.mask.push_back(Code{8, 64, 8});
  for (int i = 2; i < argc; i++) {
    std::ifstream in(argv[i]);
    std::stringstream text;
    text << in.rdbuf();
    LayoutLines lines;
    const char *fault = layout_read(text.str(), lines);
    std::vector<FrameLink> links;
    if (!fault) {
      for (const LayoutGroup &g : lines.groups) {
        if (links.size() <= g.link) links.resize(g.link + 1);
        if (links[g.link].moment.size() <= g.moment) links[g.link].moment.resize(g.moment + 1);
        links[g.link].name = "link " + std::to_string(g.link + 1);
      }
      for (const LayoutBlock &b : lines.blocks) {
        if (b.link >= links.size() || b.moment >= links[b.link].moment.size()) {
          fault = "a block's line for a link or a moment without a group's line";
          break;
        }
        std::vector<FrameArray> &m = links[b.link].moment[b.moment];
        if (m.size() <= b.array) m.resize(b.array + 1);
        FrameArray &a = m[b.array];
        a.name = std::to_string(b.moment + 1) + "." + std::to_string(b.array);
        a.rows = std::max(a.rows, b.at.first_i + (u64(1) << b.at.bits_i)), a.columns = std::max(a.columns, b.at.first_c + (u64(1) << b.at.bits_c));
        for (const LayoutGroup &g : lines.groups)
          if (g.link == b.link && g.moment == b.moment && g.group == b.group) a.over_E = g.cls == 'W', a.life = g.life;
      }
    }
    ReqShape shape;
    ReqLayout layout;
    if (!fault) fault = req_frame(links, lines, 1, shape, layout);
    if (!fault) fault = req_layout_fault(codes, shape, layout);
    size_t messages = 0, arrays = 0;
    unsigned k = 0;
    for (const ReqLink &l : shape.link) messages += l.messages.size();
    for (const std::vector<ReqArray> &l : layout.link) arrays += l.size();
    if (!fault)
      for (const ReqGroup &g : req_groups(shape)) k = std::max(k, g.k);
    std::printf("%s: %s; %zu links, %zu oracle messages, %zu arrays, %zu blocks, at most %u row variables\n", argv[i], fault ? fault : "lawful",
                shape.link.size(), messages, arrays, lines.blocks.size(), k);
    if (fault) failures++;
  }
  return failures ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc > 2) return try_files(argc, argv);
  Codes codes;
  codes.base.push_back(Code{6, 16, 4});
  codes.mask.push_back(Code{6, 16, 4});
  ReqShape shape;
  ReqLayout layout;
  const char *fault = whole(SMALL, small_links(), codes, shape, layout);
  if (fault) std::printf("%s\n", fault);
  CHECK(!fault, "a lawful file is found faulty");
  if (!fault) {
    CHECK(shape.link.size() == 2 && shape.link[0].messages.size() == 4 && shape.link[1].messages.size() == 1, "another number of oracle messages");
    const std::vector<ReqMessage> &m = shape.link[0].messages;
    CHECK(m[0].shape.mask_rows == 1 && m[1].shape.mask_rows == 0 && m[1].life == 1 && m[2].shape.main_deg == E_DEGREE && m[2].moment == 1 && m[3].moment == 2,
          "an oracle message has another shape than its group's line says");
    CHECK(layout.find(0, "K") && layout.find(0, "K")->message == 1 && layout.find(0, "R")->laid.blocks.size() == 2 && !layout.find(1, "K"), "an array is not where the file puts it");
    const std::vector<ReqGroup> groups = req_groups(shape);
    CHECK(groups[1].born.size() == 1 && groups[1].born[0] == std::make_pair(size_t(0), size_t(1)) && !groups[1].target.kept[0].kept_again, "the kept rows are not handed on");
  }
  // Both pads below (random values that mask an out-of-domain sample's answer) are entries 13 to
  // 15 of their group's last row: Send::pad() (blocks/iop/send.h).
  const std::vector<std::pair<const char *, const char *>> wrong = {
      {"1 1 1 N 2 0", "1 1 1 Q 2 0"},                                  // no such class
      {"1 1 1 N 2 0", "1 1 1 N 2 1"},                                  // a life without class B
      {"1 1 2 B 2 1", "1 1 2 B 2 0"},                                  // class B without a life
      {"1 1 2 B 2 1", "1 1 3 B 2 1"},                                  // groups not in turn
      {"1 1 2 B 2 1", "1 1 2 B 2 1 7"},                                // a number too many
      {"1 2 1 W 1 0", "3 2 1 W 1 0"},                                  // no such segment
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 1 0 0 0 3 c 1 0 0 3 1"},  // the last number
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 1 0 0 0 3 c 0 0 0 3 0"},  // on top of array 0
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 1 0 0 0 3 c 2 0 0 3 0"},  // no such row in the group
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 1 0 0 0 3 c 1 4 0 3 0"},  // not aligned
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 1 0 0 0 2 c 1 0 0 2 0"},  // the array is not covered
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 1 9 0 0 0 3 c 1 0 0 3 0"},  // no such array
      {"1 1 1 1 0 0 0 3 c 1 0 0 3 0", "1 1 2 1 0 0 0 3 c 1 8 0 3 0"},  // on the kept group's pad
      {"1 3 1 0 0 0 0 3 c 0 0 0 3 0", "1 3 1 0 0 0 0 3 c 0 8 0 3 0"},  // on the digits' pad
      {"1 2 1 0 0 0 8 1 c 0 8 0 1 0", "1 1 1 0 0 0 8 1 c 1 8 0 1 0"},  // another `moment`'s array
      {"2 1 1 0 0 0 0 3 c 0 0 0 3 0", ""},                             // an array without a block
      {"1 2 1 W 1 0", "1 2 1 N 1 0"},                                  // the vector in a row over F
      {"1 3 1 N 1 0", "1 3 1 W 1 0"},                                  // digits in a row over E
      {"1 1 2 B 2 1", "1 1 2 N 2 0"},                                  // the matrix is not kept
  };
  for (const std::pair<const char *, const char *> &w : wrong) {
    ReqShape s;
    ReqLayout l;
    const char *found = whole(replaced(SMALL, w.first, w.second), small_links(), codes, s, l);
    if (!found) std::printf("passes: %s -> %s\n", w.first, w.second);
    CHECK(found != nullptr, "a file with something wrong passes");
  }
  std::printf("req_frame_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
