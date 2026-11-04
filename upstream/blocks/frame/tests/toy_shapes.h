// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The toy configuration's second public file, toy_model(), and the `ShapesOf` callback
// (blocks/frame/configure.h) that reads it for configure(), toy_shapes(). For tests on entire
// proof, key and leaf files, and for the tool blocks/frame/handover_files.cpp.
#pragma once
#include <cstdio>
#include <sstream>
#include <string>

#include "frame/configure.h"
#include "frame/tests/toy_file.h"
#include "toy/segments.h"

namespace blk {

/* States the number of a request's segments (`links`; one sub-proof each), of the linear claims
   (`rows` in `pending_rows`) that it leaves on the committed matrix (`pending`) and of matrix rows
   (`matrix`); then one identifier per segment: `configure` rejects (`refused`) any other number. */
inline std::string toy_model(const char *identifiers = "1 2 3") {
  return std::string("links 3\npending 2\nmatrix ") + "100" + "\nidentifiers " + identifiers + "\n";  // each row of the committed table holds 2^3 matrix columns
}

// files: toy_file(), then toy_model(). Text after the last identifier is ignored. The segments'
// oracle messages are those of segments_size() (blocks/toy/segments.h). No segment makes a claim on
// the matrix: `after_links` plans a message with the values of `pending` stand-in claims.
inline Shapes toy_shapes(const std::vector<std::string> &files, const Parameters &par) {
  Shapes shapes;
  unsigned LINKS = 0, pending = 0, matrix = 0;
  int read = 0;
  if (!par.rest.empty()) return shapes.refused = "the toy's parameter file has lines behind the frame's", shapes;  // `rest`: what `read_parameters` left unparsed, for the `ShapesOf` to read or reject
  if (files.size() != 2 || std::sscanf(files[1].c_str(), "links %u\npending %u\nmatrix %u\nidentifiers%n", &LINKS, &pending, &matrix, &read) != 3 || !read || LINKS > 9)
    return shapes.refused = "the toy's second file is not in its form", shapes;
  {
    std::istringstream rest(files[1].substr((size_t)read));
    for (u64 x; rest >> x;) shapes.identifiers.push_back(x);
  }
  Subproofs &s = shapes.sub;
  const unsigned vars = par.link.log_ell - 1;
  for (unsigned l = 0; l < LINKS; l++) {
    ReqLink link;
    link.name = "link " + std::to_string(l + 1);
    ReqMessage own, product;
    own.shape.name = link.name + ": b", own.shape.code = 0, own.shape.main_rows = l == 0 ? 2 : 1, own.shape.mask_rows = 1, own.shape.answers = par.answers;
    product.shape.name = link.name + ": c", product.shape.code = 0, product.shape.main_rows = 1, product.shape.answers = par.answers;
    product.life = l + 1 < LINKS ? 1 : 0;
    link.messages = {own, product};
    link.plain = l + 1 == LINKS ? 1 : 0;
    s.request.link.push_back(link);
  }
  shapes.layout.link.resize(LINKS);  // the toy's segments name no arrays
  s.open_rows.assign(LINKS, SEGMENT_ROWS);
  s.plan = [vars](Plan &p, size_t l, unsigned) { segments_plan_link(p, "link " + std::to_string(l + 1), vars); };
  s.pending_rows = pending;
  s.after_links = [pending](Plan &p) { plan_scalars(p, "the pending rows' targets", "stand-in", pending); };
  if (matrix % 2) return shapes.refused = "the toy's matrix has an even number of rows", shapes;
  s.ranges = toy_ranges(matrix);
  shapes.range_checks.assign(LINKS, RangeCheck{(u64)1 << 6, (u64)1 << 8, 128, 41, (u64)1 << 17, {10, 8}});  // synthetic numbers: the toy's segments check no range, and `hiding_conditions_missed` rejects an empty list
  return shapes;
}

}  // namespace blk
