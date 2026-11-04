// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Declares, at set-up, the arrays that a sub-proof commits to after its first commitment. The
// layout file (layout.txt, beside the parameter file) states in which rows of which oracle message
// the proof places them (read in ChainHost::links_from(), prover/prover/chain_host.cuh). In
// identifiers, `link` is a segment (one sub-proof's part of a request's proof) and `moment` is a
// commit point, numbered from 0, the sub-proof's first commitment. prover/hiding/moments.cuh
// defines both. `chain` is a request's sequence of segments; in messages, the code that runs the
// proof outside the sub-proofs.
#pragma once
#include "hiding/moments.cuh"
#include "prover/chain_host.cuh"

// Converts an array as a sub-proof declares it (ch::ArrayAt: R rows, K columns) to the form that
// ChainLink stores. Every cell type other than Word64 becomes UINT16.
inline ChainArray chain_array(const ch::ArrayAt &a) {
  ChainArray o;
  o.name = a.name;
  o.R = a.R, o.K = a.K;
  o.source.cells = a.cells == ch::ArrayAt::Word64 ? blk::ArrayCells::WORDS : blk::ArrayCells::UINT16;
  o.source.offset = a.offset, o.source.row_stride = a.row_stride, o.source.column_stride = a.column_stride;
  o.source.pointer = (unsigned)a.pointer;
  return o;
}
// A row of K elements of the extension field E, three adjacent 64-bit words each, at the first of
// the GPU pointers that the sub-proof passes to ch::moment() for the commit point.
inline ChainArray chain_vector(const std::string &name, size_t K) {
  ChainArray o;
  o.name = name;
  o.R = 1, o.K = K;
  o.source.cells = blk::ArrayCells::ELEMENTS, o.source.column_stride = EDEG;
  return o;
}
// Fills commit points 1 to units.size() + 1 of `link` and no others. `units[i]`: the name and
// length of the vector over E of commit point i + 1. `tail`: the sub-proof's approximate range
// check, after HiddenTail::init() (prover/hiding/tail_f.cuh). Its arrays ZZ and BZ form the last of
// these points. Calls exit(4) unless `tail.chainMoment` is that point's number.
inline void chain_plan_later_moments(ChainLink &link, const std::vector<std::pair<std::string, size_t>> &units, const HiddenTail &tail) {
  const int last = (int)units.size() + 1; // the range check's commit point
  if (tail.chainMoment != last) {
    printf("internal error: the chain: the tail's moment has index %d in a sub-proof with %d moments\n",
           tail.chainMoment, last + 1);
    exit(4);
  }
  if ((int)link.moment.size() < last + 1) link.moment.resize(last + 1);
  for (int i = 0; i < (int)units.size(); i++) link.moment[i + 1] = {chain_vector(units[i].first, units[i].second)};
  std::vector<ch::ArrayAt> z;
  tail.chain_arrays(z);
  link.moment[last].clear();
  for (const ch::ArrayAt &a : z) link.moment[last].push_back(chain_array(a));
}
