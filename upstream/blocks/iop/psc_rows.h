// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Turns the outputs of PSC (sumcheck with a hiding polynomial, psc.h) and LINE (merges two claims
// on a map's halves, line.h) into linear claims on oracle messages (`Row`, relation.h).
// PSC does not know which oracle messages encode its summand's maps (witness polynomials,
// summand.h): its caller gives a MapAt for each map that is the extension of a sent vector.
// Nothing here checks that a MapAt fits its map. psc_row_u() needs a MapAt for every kept map (PSC
// sends no value of it). So a caller of psc_row_u() makes a map that is never sent a shown map
// (PSC sends its masked value), and that map's claim goes on to LINE or to the next PSC.
// This file has nothing for a kept map that is never sent: BATT (batches claims on the committed
// matrix, batt.h) has such maps and builds its output claim itself, in batt_row().
#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "iop/psc.h"
#include "iop/relation.h"
#include "iop/summand.h"
#include "iop/tensor.h"

namespace blk {
// Adds coeff * map(point) to `row`, as coefficient vectors on oracle messages.
using MapAt = std::function<void(Row& row, const std::vector<E>& point, const E& coeff)>;

// block_of_table() and row_of_table() are for multilinear maps: every coordinate of the point gets
// an equality factor. A map with a variable over a set of points (`Weight::lagrange`, hp.h) needs
// a separate MapAt.

// The map whose table is an aligned block of a table of oracle message o (`part`: main or mask
// rows), read as a vector with table_bits index bits (tensor.h): entries block * 2^a ..
// (block + 1) * 2^a - 1, a = point.size().
// Not checked: block < 2^(table_bits - a); higher bits are dropped.
inline MapAt block_of_table(OracleId o, Part part, unsigned table_bits, u64 block) {
  return [=](Row& row, const std::vector<E>& point, const E& coeff) {
    const unsigned a = unsigned(point.size());
    BLK_REQUIRE(a <= table_bits, "block_of_table: the point has more coordinates than the table has index bits");
    Tensor v = Tensor::eq(point.data(), a, coeff);
    v.insert_choice(a, table_bits - a, block);
    row.add(o, part, std::move(v));
  };
}
// The map whose table is row `which` of a table of o with row_bits row bits: the point runs over
// the entries of the row.
inline MapAt row_of_table(OracleId o, Part part, unsigned row_bits, u64 which) {
  BLK_REQUIRE(row_bits >= 64 || (which >> row_bits) == 0, "row_of_table: no such row");
  return [=](Row& row, const std::vector<E>& point, const E& coeff) {
    Tensor v = Tensor::eq(point.data(), unsigned(point.size()), coeff);
    v.insert_choice(0, row_bits, which);
    row.add(o, part, std::move(v));
  };
}
// PSC's claim (U): <row_masks> + sum_i row_maps[i].coeff * (kept map row_maps[i].map)(point)
// = row_target. kept[m]: the MapAt of kept map m. Call once: it moves the mask entries (mask.h)
// out of out.row_masks.
inline Row psc_row_u(PscOutput<E>& out, const std::vector<MapAt>& kept, std::string name) {
  BLK_REQUIRE(kept.size() == out.row_maps.size(), "psc_row_u: one MapAt for each kept map");
  BLK_REQUIRE(!out.row_masks.empty(), "psc_row_u: the row has been made already");
  Row row;
  row.name = std::move(name);
  for (MaskTerm<E>& m : out.row_masks) row.add_mask(std::move(m));
  out.row_masks.clear();
  for (const KeptTerm<E>& t : out.row_maps) kept[t.map](row, out.point, t.coeff);
  row.target = out.row_target;
  return row;
}

// "map(point) + <masks> = value" as a `Row`: a PSC's claim (E_j) on shown map j
// (`PscOutput::ends`), or LINE's output.
inline Row point_row(PointClaim<E>&& claim, const MapAt& map, std::string name) {
  Row row;
  row.name = std::move(name);
  map(row, claim.point, E::one());
  for (MaskTerm<E>& m : claim.masks) row.add_mask(std::move(m));
  row.target = claim.value;
  return row;
}

}  // namespace blk
