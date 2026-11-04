// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compares this library's output with docs/format/format-vectors.txt, the known answers for
// docs/format/FORMAT.md (its rules: R4.9 ...): canonical bytes of public types, which oracle
// messages (encoded rows under Merkle roots) share a tree, the name of a committed table (`Index`
// in identifiers), the commitment ID (`published_id` in identifiers), a code's points and symbols.
#include "chain/tests/known_answers.h"
#include "code/code.h"
#include "frame/heads.h"
#include "iop/context.h"
#include "iop/foldeq.h"
#include "iop/iopp.h"

template <class T> static std::string bytes_of(const T &x) {
  Writer w;
  public_bytes(w, x);
  return hex(w.bytes());
}
// The notice (an oracle message's public shape) that the known answers write as (leaf class, a, b):
// one main row of length 2^a, one mask row of length 2^b; -1, or a length not in `codes.base`
// (`codes.mask`): no such row.
static OracleNotice notice(const Codes &codes, unsigned leaf_class, int a, int b) {
  OracleNotice n;
  n.leaf_class = leaf_class;
  n.shape.answers = 1;
  for (size_t i = 0; a >= 0 && i < codes.base.size(); i++)
    if ((int)codes.base[i].log_n == a) n.shape.code = (CodeId)i, n.shape.main_rows = 1;
  for (size_t i = 0; b >= 0 && i < codes.mask.size(); i++)
    if ((int)codes.mask[i].log_n == b) n.shape.mask_code = (CodeId)i, n.shape.mask_rows = 1;
  return n;
}
static std::string trees_of(const Codes &codes, const std::vector<OracleNotice> &sent) {
  const TreeLay lay = lay_trees(codes, sent);
  std::string s;
  for (size_t t = 0; t < lay.shapes.size(); t++) {
    s += t ? " [" : "[";
    bool first = true;
    for (size_t part = 0; part < sent.size(); part++) // in the order of the parts within the tree
      for (size_t i = 0; i < sent.size(); i++)
        if (lay.tree[i] == t && lay.part[i] == part) s += (first ? "" : ", ") + std::to_string(i), first = false;
    s += "]";
  }
  return s;
}

int main(int argc, char **argv) {
  if (int status = read_known_answers(argc, argv)) return status;

  // ---- rules R12.x: canonical bytes ----
  Codes codes;
  codes.base = {Code{10, 512, 40}, Code{6, 16, 40}}, codes.mask = {Code{10, 64, 40}};
  expect("codes: base (10, 512, 40), (6, 16, 40); mask (10, 64, 40)", bytes_of(codes));
  OracleNotice n;
  n.shape.code = 0, n.shape.mask_code = 0, n.shape.main_rows = 3, n.shape.main_deg = 1, n.shape.mask_rows = 1, n.shape.answers = 1, n.leaf_class = 0;
  expect("an oracle message's notice: base code 0, mask code 0, 3 main rows over F, 1 mask row, 1 answer, leaf class 0", bytes_of(n));
  n.shape.mask_rows = 0;
  expect("the same without a mask code", bytes_of(n));
  std::vector<IoppStage> stages(2);
  stages[0].code = 0, stages[0].mask_code = 0, stages[0].k = 2, stages[0].h = 0;
  stages[1].code = 1, stages[1].mask_code = 0, stages[1].k = 0, stages[1].h = 0;
  expect("two stages: (0, 0, 2, 0), (1, 0, 0, 0)", bytes_of(stages));
  IoppChoice choice;
  choice.masks_ride = true;
  expect("the choices: the masks ride", bytes_of(choice));

  // ---- rules R7.x: which oracle messages share a tree ----
  Codes lengths;
  lengths.base = {Code{3, 2, 1}, Code{4, 2, 1}, Code{5, 2, 1}}, lengths.mask = {Code{2, 1, 1}};
  auto N = [&](unsigned c, int a, int b) { return notice(lengths, c, a, b); };
  expect("(1, 4, -) (1, -, 2) (1, 3, -) (2, -, 2) (1, 4, 2): the trees, each as its parts", trees_of(lengths, {N(1, 4, -1), N(1, -1, 2), N(1, 3, -1), N(2, -1, 2), N(1, 4, 2)}));
  expect("(3, -, 2) (3, 5, 2) (3, -, 2): the trees", trees_of(lengths, {N(3, -1, 2), N(3, 5, 2), N(3, -1, 2)}));
  expect("(1, -, 2) (2, -, 2) (1, -, 2): the trees", trees_of(lengths, {N(1, -1, 2), N(2, -1, 2), N(1, -1, 2)}));

  // ---- rules R4.x: a committed table's name, index_name(); the commitment ID ----
  {
    // R4.9: the configuration digest of one public file, "toy\n".
    Digest configuration;
    Hasher h(Tag(Domain::PARAMETERS), 2);
    h.push_u64(1), h.push_u64(4), h.push((const u8 *)"toy\n", 4);
    configuration = h.finish();
    IndexShape table;
    table.code = Code{10, 512, 40}, table.v = 9, table.answers = 1, table.salt_bytes = 40;
    const std::vector<u8> name = index_name(digest_of(pattern(32, 13, 4)), table, configuration);
    expect("root pattern(32) with 13 i + 4, the configuration above, a = 10, 512 entries, tail 40, 2^9 rows, 1 answer, salt 40", hex(name));
    expect("the published ID of that configuration and that name with one seed, the point (7, 8, 9), (10, 11, 12) and the factor (13, 14, 15), and the two answers (1, 2, 3) and (4, 5, 6): a test of the coding alone: that name's own seed has 9 coordinates, its table 512 answers",
           hex(published_id(configuration, name, {SampleSeed{{element(7, 8, 9), element(10, 11, 12)}, element(13, 14, 15)}}, {element(1, 2, 3), element(4, 5, 6)})));
    expect("the same with that name's own seed (R4.8; field 2: 9 coordinates and the factor), sampled and not given, and the two answers (1, 2, 3) and (4, 5, 6)", hex(published_id(configuration, name, record_seeds(table, name), {element(1, 2, 3), element(4, 5, 6)})));
  }

  // ---- rules R12.x: a code's points and symbols ----
  expect("w = 7^(5 (2^32 - 1))", std::to_string(GL64_ROOT_32));
  expect("the points of (a, x) = (15, 12345), (15, 2), (14, 9999)",
         std::to_string(gl64_point(15, 12345)) + " " + std::to_string(gl64_point(15, 2)) + " " + std::to_string(gl64_point(14, 9999)));
  {
    const Code code{4, 8, 3};
    std::vector<F> message(code.ell), tail(code.r);
    for (size_t j = 0; j < code.ell; j++) message[j] = F{1000003 * j + 17};
    for (size_t j = 0; j < code.r; j++) tail[j] = F{P - 1 - 5 * j};
    for (u64 x = 0; x < code.n(); x++) {
      // The symbol as R12.11 defines it: the sum of c_j z^j, the tail's coefficients first.
      const F z = F{gl64_point(code.log_n, x)};
      F power = F::one(), by_powers = F::zero();
      for (size_t j = 0; j < code.r; j++) by_powers = by_powers + tail[j] * power, power = power * z;
      for (size_t j = 0; j < code.ell; j++) by_powers = by_powers + message[j] * power, power = power * z;
      const F symbol = symbol_at(code, x, message.data(), tail.data());
      if (symbol != by_powers) failures++, std::printf("FAILED: position %llu: the code's symbol is not the sum of powers\n", (unsigned long long)x);
      expect("code (4, 8, 3), message 1000003 j + 17, tail p - 1 - 5 j, position " + std::to_string(x) + ": the point; the symbol", std::to_string(z.v) + "; " + std::to_string(symbol.v));
    }
  }
  // blocks/chain/tests/format_vectors_test.cpp compares the hash, hash chain, channel and trees.
  // Neither test compares, among others, tree numbers (R7.13; `TreeBook` in blocks/iop/context.h),
  // the item `table`, R12.6 to R12.10, R12.12 to R12.18 or R13.x to R15.x.
  return known_answers_end("format_types_test", argc);
}
