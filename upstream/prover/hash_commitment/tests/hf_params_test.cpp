// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CPU test of HfParams, which derives the weight commitment's sizes from the lines of the parameter
// file (`policy` in identifiers): numbers at full size (full(), "S1") and at toy()'s small sets;
// the map between cells of the weights W and the matrix M, whose rows, masked and encoded, are the
// weight table T; rejection names (printed: "refusal"). The numbers are typed into this file, those
// for the pad and table shapes from a specification that is not in this repository.
// With arguments "<file> <m0>" (m0: W's rows) it only prints the numbers for a parameter file; exit
// code 2 if the file cannot be read or is rejected.
// Build, from prover/: ./build.sh hc_host_test tests/hf_params_test
#include "hash_commitment/hf_params.h"
#include "frame/parameters.h" // blocks/: the frame's lines
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
static int bad = 0, done = 0;
// DIGEST_WORDS (DW) is always 8 (prover/hash_commitment/hf_table_format.h): at10 is never taken.
#define A8(at8, at10) (DIGEST_WORDS == 8 ? (long long)(at8) : (long long)(at10))
#define IS(a, b)                                                                                                        \
  do {                                                                                                                  \
    done++;                                                                                                             \
    if ((long long)(a) != (long long)(b))                                                                               \
      bad++, printf("DIFFERS line %d: %s = %lld, expected %lld\n", __LINE__, #a, (long long)(a), (long long)(b));       \
  } while (0)
static void named(const std::string &why, const char *name) { // `why` must begin with "<name>:"
  done++;
  if (why.compare(0, strlen(name), name) || why[strlen(name)] != ':')
    bad++, printf("DIFFERS: expected the refusal %s, got \"%s\"\n", name, why.c_str());
}
static void refused(HfParams P, const char *name) { named(P.derive(), name); }
// Prints the numbers for the frame's lines of a parameter file.
static int file(const char *path, size_t m0) {
  std::ifstream in(path);
  if (!in) return printf("cannot read %s\n", path), 2;
  std::stringstream bytes;
  bytes << in.rdbuf();
  blk::Parameters frame;
  std::string why;
  if (!blk::read_parameters(bytes.str(), frame, why)) return printf("%s: %s\n", path, why.c_str()), 2;
  HfParams P;
  if (P.from_policy(frame, m0)) return 2;
  printf("FILE %s ly %d R %zu n %llu l %llu KD %llu lc %d ld %d\n",
         path, P.ly, P.R, (unsigned long long)P.n, (unsigned long long)P.l, (unsigned long long)P.KD, P.lc, P.ld);
  return 0;
}
int main(int argc, char **argv) {
  if (argc == 3) return file(argv[1], strtoull(argv[2], 0, 10));
  {
    const HfParams P = HfParams::full();
    P.print();
    IS(P.ly, 22); IS(P.R, 2048); IS(P.lcl, 7); IS(P.l, 393740288ll); IS(P.KD, 446693375ll); IS(P.ld, 29); IS(P.lc, 29);
    IS(P.lead, 0); IS(P.n, 8589934592ll); IS(P.slabs, 128); IS(P.walks, 2);
    IS(P.hb, 139); IS(P.fhb, 131);
    IS(P.row_of(5, 130), 1); IS(P.coef_of(5, 130), 5 * 128 + 2);
  }
#if DIGEST_WORDS == 8
  { // Proof of work at full size: pow 20 bits (a request's files), pows 28 (a batch's shared
    // message), powc 32 (the key file's proofs and LINK proofs, `link`).
    HfParams P = HfParams::full(0);
    P.pow = 20, P.pows = 28, P.powc = 32;
    IS(P.derive().size(), 0);
    P.pows = 128;
    IS(P.derive().compare(0, 12, "policy-pows:"), 0);
  }
#endif
  { // shapes of pad commitments and of tables
    const hf::PadShape a = hf::pad_shape(80, 1, 150, 1), b = hf::pad_shape(108, 1, 150, 1), c = hf::pad_shape(80, 1, 108, 1);
    IS(a.ln, 11); IS(a.t, 215); IS(a.mk, 220); IS(a.Ks, 256); IS(a.Kc, 476); IS(b.t, 215); IS(b.Kc, 476); IS(c.t, 147);
    IS(c.mk, 152); IS(c.Kc, 408);
    const hf::TableShape z = hf::table_shape(1, 32, 3, 33, 150, 3, 16), y = hf::table_shape(1, 32, 3, 33, 108, 3, 16);
    IS(z.w, 14); IS(z.ln, 17); IS(z.t, 181); IS(z.mk, 5978); IS(z.P2, 8192); IS(z.Kp, 8192); IS(z.Kc, 14170); IS(z.cube(), 14);
    IS(y.t, 131); IS(y.mk, 4328); IS(y.Kc, 12520);
  }
  // Some of toy()'s small sets.
  const struct { int which, lc, ld; } sets[] = {{0, 18, 18}, {2, 18, 18}, {3, 19, 18}, {4, 18, 18}, {5, 15, 15}, {7, 20, 19}, {8, 20, 19}};
  for (const auto &s : sets) {
    const HfParams P = HfParams::toy(s.which);
    IS(P.lc, s.lc); IS(P.ld, s.ld);
  }
  // Toy sets 0 to 6 only: the map from W's cells (y, c) onto M's (r, i), r < R, i < l, is a
  // bijection.
  for (int which = 0; which < HfParams::TOYS; which++) {
    const HfParams P = HfParams::toy(which);
    P.print();
    if (which < 7) {
      std::vector<char> seen(P.R * P.l, 0);
      size_t twice = 0, back = 0, outside = 0, empty = 0;
      for (uint64_t y = 0; y < P.m0; y++)
        for (uint64_t c = 0; c < P.k; c++) {
          const uint32_t r = P.row_of(y, c);
          const uint64_t i = P.coef_of(y, c);
          if (r >= P.R || i >= P.l) { outside++; continue; }
          twice += seen[r * P.l + i]++;
          uint64_t y2, c2;
          P.cell_of(r, i, y2, c2);
          back += !(y2 == y && c2 == c);
        }
      for (char s : seen) empty += !s;
      IS(twice, 0); IS(back, 0); IS(outside, 0); IS(empty, 0);
    }
  }
  { // rejection names of `derive` after one or two changed lines
    HfParams P = HfParams::full(), X;
    X = P, X.b = 2, refused(X, "policy-rows");        X = P, X.b = 14, refused(X, "policy-rows");
    X = P, X.ln = 34, refused(X, "policy-fln");       X = P, X.ln = 29, refused(X, "policy-fln");
    X = P, X.pow = 33, refused(X, "policy-pow");       X = P, X.fhb = P.hb + 1, refused(X, "policy-fhb");
    X = P, X.fhb = 0, refused(X, "policy-fhb");       X = P, X.leafd = 9, refused(X, "policy-leafd");
    X = P, X.nm = 44040192, refused(X, "policy-mask-shape");
    X = P, X.i0 = 393740286, refused(X, "policy-mask-place"); X = P, X.i0 = 402653184 + 2, refused(X, "policy-mask-place");
    X = P, X.lZU = 24, refused(X, "policy-zu-lines"); X = P, X.lnZU = 25, refused(X, "policy-zu-lines");
    X = P, X.nm = 63, X.i0 = 393740288 + 64, refused(X, "policy-mask-place"); // a multiple of 2^6, not of 2^(lk - b) = 2^7
    X = HfParams::toy(0), X.b = 8, refused(X, "policy-rows");
    X = P, X.b = 18, X.ln = 40, refused(X, "policy-rows"); // fb is tested before fln
    // One to four answers, and room after the data for three pads per answer.
    X = P, X.answers = 0, refused(X, "policy-answers");
    X = P, X.answers = 5, refused(X, "policy-answers");
    X = HfParams::toy(2), X.i0 = X.l, refused(X, "policy-pad-place"); // fi0 = l
    X = P, X.answers = 2; IS(X.derive().size(), 0); IS(X.pads(), 6); IS(X.pad_entry(0), 402653184 - 6); IS(X.pad_entry(5), 402653184 - 1);
    IS(X.keep_level_most(), X.ln - hf::top_layer(X.ln, HF_COLUMNS_MOST));
    const HfParams N = HfParams::full(2);
    IS(N.leafd, 8); IS(N.lNmax, 8);
  }
  { // `derive` rejects a value of 2^30 or more under the name of that field's test
    const HfParams P = HfParams::full();
    const struct { int HfParams::*field; const char *name; } big[] = {{&HfParams::lk, "w-shape-lk"}, {&HfParams::b, "policy-rows"}, {&HfParams::ln, "policy-fln"},
        {&HfParams::hb, "hid-hb"}, {&HfParams::pow, "policy-pow"}, {&HfParams::lNmax, "w-policy-Nmax"}, {&HfParams::leafd, "policy-leafd"}, {&HfParams::fhb, "policy-fhb"},
        {&HfParams::lZU, "policy-zu-lines"}, {&HfParams::lnZU, "policy-zu-lines"}};
    for (const auto &b : big) {
      HfParams X = P;
      X.*b.field = 1 << 30;
      refused(X, b.name);
    }
  }
  { // `from_policy`: the frame's lines, and the version word, which s8::set() checks
    const HfParams P = HfParams::shape(18, 3076096, HF_HB_REFERENCE, HF_HB_REFERENCE, 11, 402653184ull, 44040191ull, 33, 6, 22, 24);
    blk::Parameters F;
    F.version = hf_version_word(), F.sec = HF_HB_REFERENCE, F.slots = 64, F.answers = 1;
    F.table_rows_log = 11, F.table_log_n = 33, F.table_data = 402653184, F.table_random = 44040191, F.block_log = 18;
    F.folded.log_ell = 22, F.folded.log_n = 24;
    F.work[3][0] = 1, F.work[3][1] = 2, F.work[3][2] = 3;
    HfParams X;
    IS(X.from_policy(F, 3076096), 0);
    IS(X.KD, P.KD); IS(X.lk, 18); IS(X.b, 11); IS(X.ln, 33); IS(X.i0, 402653184); IS(X.nm, 44040191);
    IS(X.hb, HF_HB_REFERENCE); IS(X.fhb, HF_HB_REFERENCE); IS(X.leafd, 6); IS(X.lNmax, 6); IS(X.lZU, 22); IS(X.lnZU, 24);
    IS(X.pow, 1); IS(X.pows, 2); IS(X.powc, 3); IS(X.answers, 1);
    blk::Parameters M = F;
    M.block_log = 3;
    fprintf(stderr, "(expected: w-shape-lk) ");
    IS(X.from_policy(M, 3076096), 2);
    refused(X, "w-shape-lk");
    M = F, M.version = DW == 8 ? 0x48473035u : 0x48473036u; // "HG05" or "HG06": the word of a format that these programs cannot read
    fprintf(stderr, "(expected: version-word-other-format) ");
    IS(X.from_policy(M, 3076096), 2);
    // from_policy() returns 2 whenever it rejects the file, and only prints the reason. For the
    // version word it prints what s8::set() (prover/common/state8.h) returns, so the cases below
    // also pass the same frame to s8::set() and compare the name. The printed line is not read.
    for (uint32_t older = 0x48473031u; older <= 0x48473034u; older++) { // "HG01" to "HG04"
      M = F, M.version = older;
      fprintf(stderr, "(expected: version-word-other-format) ");
      IS(X.from_policy(M, 3076096), 2);
      named(s8::set(M), "version-word-other-format");
    }
    M = F, M.version = DW == 8 ? 0x48463032u : 0x48463031u; // "HF02" or "HF01": a word outside "HG01" to "HG06", and not this build's
    fprintf(stderr, "(expected: version-word) ");
    IS(X.from_policy(M, 3076096), 2);
    named(s8::set(M), "version-word");
    M = F, M.version = 0;
    fprintf(stderr, "(expected: version-word) ");
    IS(X.from_policy(M, 3076096), 2);
    named(s8::set(M), "version-word");
  }
  printf("%d checks, %d differ\n", done, bad);
  return bad ? 1 : 0;
}
