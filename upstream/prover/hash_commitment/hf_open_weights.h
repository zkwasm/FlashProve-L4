// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Types for the public coefficients of a linear claim on u ("this linear form of u has this
// value"), in factored form. The weight table encodes the model's weights as long rows, each with
// masks; u is the fold of these rows before encoding: u[i] = sum over rows r of fold[r] M[r][i],
// with M[r] row r before encoding and fold[r] in E (HfSeams::make_u, hf_open_api.h). F is the
// prime base field, E its extension of degree 3. A coefficient is called a weight here; it is not
// a model weight. As a plain vector the weights would have 2^29 elements of E at full size (the
// example configuration `target_full_f2`).
// No code uses these types; only hf_open_api.h includes this header.
#pragma once
#include <cstdint>
#include <vector>

// An element of E as three canonical coordinates (each below the modulus of the base field F),
// k0 + k1 x + k2 x^2: the same bytes as the GPU's ::Ext (prover/common/common.h) and the host's
// rf::E (prover/common/host_common.h). No code checks that. It is a separate type so that
// host-only code can include this header.
struct HfEl {
  uint64_t k0 = 0, k1 = 0, k2 = 0;
};
// One term of weight(i), the coefficient of u[i]: a product over the lc bits of the index i.
//
//   weight(i) = sum over terms t of [i < t.below] t.coef *
//               prod over l < lc of (bit l of i ? t.one[l] : t.zero[l]),
//
// where [c] is 1 if c is true and 0 otherwise. Bit l counts from the top: l = 0 is the most
// significant of the lc bits of i. Two examples, each one term with coef = 1: eq(z, i) with z in
// E^lc has one[l] = z_l, zero[l] = 1 - z_l, below = 2^lc; y^i for i < KD and 0 from KD on, with y
// in F, has one[l] = y^(2^(lc - 1 - l)), zero[l] = 1, below = KD, base_field = true (KD: the
// number of u's coefficients, hf_table_params.h). In this form the multilinear extension of
// weight() at a point of E^lc takes O(lc) operations per term, as a prefix sum of products.
struct HfProductTerm {
  HfEl coef;
  uint64_t below = 0;          // the term is 0 at i >= below; below <= 2^lc
  std::vector<HfEl> one, zero; // lc elements each, l = 0 for the top bit
  bool base_field = false;     // if true, every one[l] and zero[l] is in F (k1 = k2 = 0)
};
// All terms of one claim. `Zu` in the name is ZU, the commitment to u (sizes: hf_opening_params.h).
struct HfZuWeights {
  int lc = 0;                       // the number of bits of i
  std::vector<HfProductTerm> terms;
};
