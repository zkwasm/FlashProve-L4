// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// LINE: reduces two claims on the halves of a multilinear map V to one claim on V.
// V has k + 1 variables. V_0 = V(., 0) and V_1 = V(., 1) are its halves in the last variable
// (entries j and j + 2^k of its table). `half0` and `half1` (`PointClaim`, summand.h) claim, at one
// point, V_i(point) + <masks_i> = value_i; <masks_i> is a public linear form (`MaskTerm`s) in
// mask entries (the prover's committed random elements: mask.h).
// The verifier draws rho. V is linear in its last variable, so the two claims imply
//     V(point, rho) + (1 - rho) <masks_0> + rho <masks_1> = (1 - rho) value_0 + rho value_1.
// If either claim is false, this one holds for at most one rho in the challenge field K: error
// 1/|K|. The prover sends nothing. rho is a separate verifier message: where a sumcheck on V
// follows, rho is a coordinate of that sumcheck's point, so rho comes before its first message.
#pragma once
#include <utility>

#include "base.h"
#include "iop/summand.h"
#include "numbers/round_names.h"

namespace blk {

// For prover and verifier. `Ch`, the channel (source of challenges), needs
// close_algebraic(name).draw(). A `Rounds` channel (blocks/chain/rounds.h) requires that
// plan_line() (blocks/numbers/blocks.h) planned its next round. line() cannot report that the
// channel has rejected the proof: a verifier must ask the channel at the end of its run.
template <class K, class Ch>
PointClaim<K> line(Ch& ch, PointClaim<K>&& half0, PointClaim<K>&& half1) {
  BLK_REQUIRE(half0.point == half1.point, "line: the two claims must be at one point");
  const K rho = ch.close_algebraic(R_LINE).draw();
  const K one_minus = K::one() - rho;
  PointClaim<K> out;
  out.point = std::move(half0.point);
  out.point.push_back(rho);
  for (MaskTerm<K>& m : half0.masks) out.masks.push_back({std::move(m.slot), one_minus * m.coeff});
  for (MaskTerm<K>& m : half1.masks) out.masks.push_back({std::move(m.slot), rho * m.coeff});
  out.value = one_minus * half0.value + rho * half1.value;
  return out;
}

}  // namespace blk
