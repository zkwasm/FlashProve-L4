// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The tree hash TH replaces each piece of THPW words of a list (the last may be shorter) by its
// digest (DW words) under the domain number TH_LIST, and each piece of every level of digests above
// under TH_UPPER, until a level is one digest. This test compares two implementations, which hash
// a piece with the same earlier_hash_h_digest() (prover/common/hash.h), with test vectors and each other:
//   rf::TH() (prover/common/host_common.h), on the host;
//   ::treehash() (prover/weights_commitment/common.cuh), on the GPU: one k_th() launch per level.
// Only this test calls that treehash(). Not covered: the treehash() that cp_prove calls
// (prover/prover/base.cuh). It launches the same kernel, k_th() (prover/hiding/product_test.cuh).
//
// A vector is a line "TH <L> <the digest, DW words of 8 hex digits>" of the file that main() opens:
// TH of list(L). The further lengths are mostly at the boundaries of a piece, of `fan` pieces
// (their digests fill one piece of the next level) and of `span` words (one piece two levels up).
// There treehash() is compared only with rf::TH(), which the vectors check. Negative control:
// one_number() must differ from every vector of more than one piece and agree with the others.
// From prover/: ./build.sh tree_hash_test; then run the program (in build.sh's `OUT`) on one H100.
#include "weights_commitment/common.cuh"

// The test list of L words: word k is ((k + 1) * 0x89ABCDEF) mod 2^32.
static std::vector<u32> list(size_t L) {
  std::vector<u32> w(L);
  for (size_t k = 0; k < L; k++) w[k] = (u32)(k + 1) * 0x89ABCDEFu;
  return w;
}
static std::string hex_of(const u32 *w) {
  std::string o;
  char c[9];
  for (int i = 0; i < DW; i++) snprintf(c, sizeof c, "%08x", w[i]), o += c;
  return o;
}
// The negative control: every level under TH_LIST.
static std::string one_number(std::vector<u32> w) {
  for (;;) {
    const size_t nc = (w.size() + THPW - 1) / THPW;
    std::vector<u32> o(nc * DW);
    for (size_t c = 0; c < nc; c++) earlier_hash_h_digest(&o[c * DW], TH_LIST, &w[c * THPW], std::min<size_t>(THPW, w.size() - c * THPW));
    w.swap(o);
    if (nc == 1) return hex_of(w.data());
  }
}
static int bad = 0;
// Runs the two implementations on the list of L words; returns the host's result.
static std::string both(size_t L) {
  const std::vector<u32> w = list(L);
  const std::string host = hex_of(rf::TH(w).data());
  u32 *in, d[DW];
  CK(cudaMalloc(&in, L * 4));
  CK(cudaMemcpy(in, w.data(), L * 4, cudaMemcpyHostToDevice));
  const size_t level1 = (L + THPW - 1) / THPW * DW, level2 = (level1 + THPW - 1) / THPW * DW;
  ::treehash_scratch(level1 * 4, level2 * 4);
  CK(cudaMemcpy(d, ::treehash(in, L), DW * 4, cudaMemcpyDeviceToHost));
  if (hex_of(d) != host) bad++, printf("TH %zu: treehash() gives another result than the host\n", L);
  cudaFree(in);
  return host;
}
int main() {
  const char *const vectors_file = "../verifier/tests/hash_vectors.txt";
  FILE *f = fopen(vectors_file, "r");
  if (!f) {
    printf("tree_hash_test: %s cannot be read: run it from prover/\n", vectors_file);
    return 2;
  }
  int vectors = 0, told_apart = 0, further = 0;
  char text[600], kind[8], hex[400];
  size_t L;
  while (fgets(text, sizeof text, f)) {
    if (sscanf(text, "%7s %zu %399s", kind, &L, hex) != 3 || strcmp(kind, "TH")) continue;
    vectors++;
    if (both(L) != hex) bad++, printf("TH %zu: the host gives another result than the test vector\n", L);
    const bool same = one_number(list(L)) == hex;
    if (same != (L <= (size_t)THPW)) bad++, printf("TH %zu: the negative control with one number for all levels did not behave as expected\n", L);
    told_apart += !same;
  }
  fclose(f);
  if (vectors != 13 || told_apart != 9)
    bad++, printf("%d lines that begin TH, %d of them of more than one piece: not 13 and 9\n", vectors, told_apart);
  const size_t piece = THPW, fan = THPW / DW, span = piece * fan * fan;
  for (size_t n : {(size_t)DW, piece - 1, piece, piece + 1, piece + DW, fan * piece - 1, fan * piece, fan * piece + 1,
                   span - 1, span, span + 1, span + piece, 2 * span - 1, 2 * span, 2 * span + 1, fan * span,
                   fan * span + 1, 10 * span + 7 * piece + 3})
    both(n), further++;
  ::treehash_scratch(0, 0);
  printf("tree_hash_test: %d vectors, %d further lengths, two implementations: %s\n", vectors, further,
         bad ? "FAILS" : "passes");
  return bad != 0;
}
