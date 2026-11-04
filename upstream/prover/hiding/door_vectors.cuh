// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Copies a sub-proof's coefficient vectors from the GPU and writes them to a file, for tests that
// compare them with the verifier's. (A sub-proof: the argument for one part of the forward pass.)
// For small sub-proofs only: it copies whole vectors.
//
// hw::Walk::finish() (prover/hiding/walk.cuh) batches a sub-proof's linear claims into one equation
// (`Door`, prover/hiding/door.cuh): claim number j (from 0, in the order added) is multiplied by
// theta^j, for a challenge theta. The verifier computes the equation's coefficients with separate
// code (`HRow` and HFork::lin(), verifier/src/vf_subproof_engine.h, and lin()'s callers), so the claims'
// order is coded twice and must match. vectorsDifference() (verifier/tests/coefficient_vectors.h) reports the
// first difference. The equation's coefficients outside `Door::t`, such as those on the mask row
// (`Door::LS0`), are not copied.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

#include "hiding/door_terms.cuh"
namespace hd {
// One `Pair` of a committed array (a `Term`, R rows by K columns; prover/hiding/door_terms.cuh).
// Cell (r, c) of the array has the coefficient sum over its pairs of cv[r] * L[c].
// `T` (`Ext`, prover/common/common.h): an extension-field element of EDEG coordinates.
struct DoorVector {
  std::string array; // `Term::nm`, or the caller's name for the term
  unsigned pair = 0; // the pair's number in the Term's list
  std::vector<T> cv, L; // R and K entries; L in full, zeros outside [c0, c1) included
  size_t c0 = 0, c1 = 0;
};
// terms: a sub-proof's list (`Door::t`). names: null, or for each term the name, instead of
// `Term::nm`, by which vectorsDifference() finds the verifier's array. Call after the kernels that
// fill cv and after hw::Walk::finish(), which adds into L, once their streams are synchronized.
// `CK` exits with code 1 on a CUDA error.
inline std::vector<DoorVector> door_vectors(const std::vector<Term> &terms, const std::vector<std::string> *names = nullptr) {
  std::vector<DoorVector> out;
  if (names && names->size() != terms.size()) {
    printf("door_vectors: one name for each term\n");
    exit(2);
  }
  for (const Term &t : terms)
    for (size_t i = 0; i < t.v.size(); i++) {
      DoorVector x;
      x.array = names ? (*names)[&t - terms.data()] : t.nm, x.pair = (unsigned)i, x.c0 = t.v[i].c0, x.c1 = t.v[i].c1;
      x.cv.resize(t.R), x.L.resize(t.K);
      CK(cudaMemcpy(x.cv.data(), t.v[i].cv, t.R * sizeof(T), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(x.L.data(), t.v[i].L, t.K * sizeof(T), cudaMemcpyDeviceToHost));
      out.push_back(std::move(x));
    }
  return out;
}
// Format, each number as 8 bytes in host byte order: the number of pairs; then for each pair its
// array's name (length, then bytes), the pair's number, R, K, cv and L (EDEG numbers per element).
// c0 and c1 are not written; write errors are not checked.
inline void put_door_vectors(FILE *o, const std::vector<DoorVector> &v) {
  auto word = [o](unsigned long long x) { fwrite(&x, 8, 1, o); };
  auto element = [&](const T &x) {
    for (int k = 0; k < EDEG; k++) word(coord(x, k));
  };
  word(v.size());
  for (const DoorVector &x : v) {
    word(x.array.size());
    fwrite(x.array.data(), 1, x.array.size(), o);
    word(x.pair), word(x.cv.size()), word(x.L.size());
    for (const T &e : x.cv) element(e);
    for (const T &e : x.L) element(e);
  }
}
} // namespace hd
