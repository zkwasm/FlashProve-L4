// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests blocks/frame/hiding_conditions.h: six parameter files meet every condition, and
   for each condition, inputs that miss it are reported under its name. The sub-proofs and range
   checks are stand-ins (blocks/frame/files.h and main() here). `sized`, not the file, sets the number of batches, the
   leaf classes and the random tails (`Code::r`) of `proof_codes`, `key_codes` and `entry_codes`,
   so most cases alter what it returned. The file sets each group's mask message length (line
   `group`): a file that `sized` accepts can miss `mask-room` (one case). The `fold-room` case
   alters derived codes: `sized` rejects a file with mask rows that short
   (`req_fault`, blocks/iop/req.h).
   `d.link` concerns a request's segments; `Sized::link` concerns the LINK proof, which ties a
   further committed table (leaf table) to the first (anchor). `TableUse` counts the one-time tests
   and LINK proofs begun on these tables.
   Run from blocks/. */
#include <cstdio>
#include <fstream>
#include <sstream>

#include "frame/hiding_conditions.h"

using namespace blk;
static int failures = 0, made = 0;
#define CHECK(c)                                                        \
  do {                                                                  \
    ++made;                                                             \
    if (!(c)) ++failures, std::printf("%s:%d: %s\n", __FILE__, __LINE__, #c); \
  } while (0)

static std::string file(const std::string &name) {
  std::ifstream in("frame/examples/" + name, std::ios::binary);
  std::stringstream s;
  s << in.rdbuf();
  std::string t = s.str();
  if (t.size() > 11) t[11] = (char)('0' + BLK_HASH);  // the examples are written for hash 1
  return t;
}
static Sized of(const std::string &text, const Subproofs &sub) {
  Parameters par;
  std::string why;
  if (!read_parameters(text, par, why)) std::printf("%s\n", why.c_str()), ++failures;
  return sized(par, sub);
}
static bool names(const std::string &got, const char *name) {
  const bool yes = got.rfind(std::string(name) + ": ", 0) == 0;
  if (!yes) std::printf("got: %s\n", got.c_str());
  return yes;
}
static std::string with(std::string t, const std::string &a, const std::string &b) {
  const size_t at = t.find(a);
  if (at == std::string::npos) return std::printf("no %s\n", a.c_str()), ++failures, t;
  return t.replace(at, a.size(), b);
}

int main() {
  const Subproofs sub = stand_in_subproofs();
  // Stand-ins, like the sub-proofs: one approximate range check (`RangeCheck`) per segment, on
  // 2^14 rows of C = 2^16 cells below B = 41; 128 combinations; digits for [-D, D), D = 2^17.
  const std::vector<RangeCheck> checks(25, RangeCheck{(u64)1 << 14, (u64)1 << 16, 128, 41, (u64)1 << 17, {10, 8}});
  const std::string good = file("record.f2.txt");
  const std::vector<std::pair<std::string, std::string>> examples = {{"record.f2.txt", good},
                                                                     {"with work on closeness rounds", with(good, "work closeness 0 0 0", "work closeness 20 20 20")},
                                                                     {"without work", with(good, "work positions 20 28 32", "work positions 0 0 0")}};
  for (const auto &[e, text] : examples) {
    const Sized z = of(text, sub);
    CHECK(z.refused.empty());
    const std::string got = z.refused.empty() ? hiding_conditions_missed(z, sub, checks) : z.refused;
    if (!got.empty()) std::printf("%s: %s\n", e.c_str(), got.c_str());
    CHECK(got.empty());
    CHECK(hiding_conditions_missed_by_the_file(z.par).empty());  // the part on the file alone
  }
  const Sized z = of(good, sub);
  // One tail entry fewer, code by code.
  auto less = [](Code &c) { c.r -= 1; };
  // req_plan() and iopp_sized_fault() require that sizes and codes agree, so both are altered.
  auto follow = [](const Codes &codes, const std::vector<IoppStage> &stages, std::vector<Stage> &solved) {
    for (size_t j = 0; j < stages.size(); ++j) {
      const Code &main = codes.base[stages[j].code], &mask = codes.mask[stages[j].mask_code];
      solved[j].g.main.ell = main.ell, solved[j].g.main.r = main.r, solved[j].g.mask.ell = mask.ell, solved[j].g.mask.r = mask.r;
    }
  };
  auto tables_follow = [&](Sized &x) {  // tests on the table (batch, one-time), LINK proof, ranges
    for (int once = 0; once < 2; ++once) {
      LinSized &l = once ? x.once : x.batch;
      const Codes &codes = once ? x.key_codes : x.proof_codes;
      follow(codes, l.S.test_u, l.folded.stages), follow(codes, l.S.test_omega, l.omega.stages), follow(codes, l.S.test_own, l.own.stages);
    }
    follow(x.entry_codes, x.link.S.test(), x.link.linked.stages), follow(x.key_codes, x.ranges.x.stages, x.ranges.test.stages);
  };
  for (size_t i = 0; i < z.proof_codes.base.size(); ++i) {
    Sized x = z;
    less(x.proof_codes.base[i]);
    if (i == x.request.code) x.d.link.group.main.r = x.proof_codes.base[i].r;
    follow(x.proof_codes, x.request.last, x.d.last.stages), tables_follow(x);
    CHECK(names(hiding_conditions_missed(x, sub, checks), "main-rows"));
  }
  for (size_t i = 0; i < z.proof_codes.mask.size(); ++i) {
    Sized x = z;
    less(x.proof_codes.mask[i]);
    if (i == x.request.mask_code) x.d.link.group.mask.r = x.proof_codes.mask[i].r;
    follow(x.proof_codes, x.request.last, x.d.last.stages), tables_follow(x);
    CHECK(names(hiding_conditions_missed(x, sub, checks), "mask-rows"));
  }
  for (int which = 0; which < 2; ++which)
    for (size_t i = 0; i < (which ? z.entry_codes : z.key_codes).base.size(); ++i) {
      Sized x = z;
      less((which ? x.entry_codes : x.key_codes).base[i]);
      tables_follow(x);
      CHECK(names(hiding_conditions_missed(x, sub, checks), "main-rows"));
    }
  for (int which = 0; which < 2; ++which)
    for (size_t i = 0; i < (which ? z.entry_codes : z.key_codes).mask.size(); ++i) {
      Sized x = z;
      less((which ? x.entry_codes : x.key_codes).mask[i]);
      tables_follow(x);
      CHECK(names(hiding_conditions_missed(x, sub, checks), "mask-rows"));
    }
  {  /* Injected fault: the fresh mask of the fold that begins the batch's test on the
        second oracle message (`omega`) shares the leaves of `omega`. Those are opened at the
        batch's list of drawn positions (flag 2) and at every request's own (flag 4), but the mask
        is read at the batch's only: a few requests of one batch would show the whole mask. */
    const LinShape &S = z.batch.S;
    const OracleNotice second{S.omega, S.omega_leaf_class};
    const OracleNotice mask = fold_mask_notice(lin_fold_shape(S.shared_split_needed() ? S.test_omega : S.test_u, S.omega_leaf_class));
    CHECK(names(hiding_missed_in_leaves("batch", {ReadAt{second, 2 | 4}, ReadAt{mask, 2}}), "leaves"));
    const OracleNotice apart = fold_mask_notice(lin_fold_shape(S.shared_split_needed() ? S.test_omega : S.test_u, S.omega_leaf_class + 100));
    CHECK(hiding_missed_in_leaves("batch", {ReadAt{second, 2 | 4}, ReadAt{apart, 2}}).empty());
    CHECK(hiding_missed_in_the_tables_leaves("batch", S).empty() && hiding_missed_in_the_tables_leaves("one-time test", z.once.S).empty());
    Sized x = z;  // masks that split a claim on two groups, read at both lists, in `omega`'s leaves
    x.batch.S.shared_split_leaf_class = x.batch.S.omega_leaf_class;
    CHECK(!S.shared_split_needed() || names(hiding_conditions_missed(x, sub, checks), "leaves"));
  }
  {  /* Other oracle messages of the test on the table: no room for a sample's pad
        (`padding-entries`) or for mask entries. */
    Sized x = z;
    x.batch.S.fold.u.main_rows -= 1;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "padding-entries"));
    x = z, x.once.S.w.main_rows = 0;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "padding-entries"));
    x = z, x.batch.S.shared_split.mask_rows = 0;
    CHECK(!z.batch.S.shared_split_needed() || names(hiding_conditions_missed(x, sub, checks), "mask-room"));
    x = z, x.once.input.shape.mask_rows = 0;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "mask-room"));
    x = z, x.key_codes.mask[x.once.S.test_own[0].mask_code].ell = 2, tables_follow(x);
    CHECK(names(hiding_conditions_missed(x, sub, checks), "mask-room"));
  }
  {  /* The committer's range proof (`ranges`, `wf_`): its mask rows need exactly what its lookups'
        sumchecks take from them. The index-mask message (masks for claims on the table) needs two
        entries for each lookup part, 1 + 2 `k_col` for batching the parts' claims, and the pad. */
    const u64 taken = wf_own_masks(z.ranges.x, z.ranges.matrix), in_a_row = z.key_codes.mask[z.ranges.x.message.shape.mask_code].ell - z.par.answers;
    Sized x = z;
    x.ranges.x.message.shape.mask_rows = (size_t)((taken + 1 + in_a_row) / (in_a_row + 1));  // the least n with n in_a_row + (n - 1) >= taken
    CHECK(hiding_conditions_missed_in_the_ranges(x).empty());
    x.ranges.x.message.shape.mask_rows -= 1;
    const std::string got = hiding_conditions_missed_in_the_ranges(x);
    CHECK(names(got, "mask-room") && got.find(" where " + std::to_string(taken) + " are needed") != std::string::npos);
    u64 parts = 0;
    for (const WfLookup &l : z.ranges.x.lookups) parts += l.parts.size();
    CHECK(parts == z.ranges.pending && 2 * parts + 1 + 2 * z.once.S.k_col() + z.par.answers == wf_index_mask_room(z.ranges, z.par.answers));
    x = z, x.key_codes.mask[x.once.input.shape.mask_code].ell = wf_index_mask_room(z.ranges, z.par.answers) - 1, tables_follow(x);
    const std::string room = hiding_conditions_missed_on_the_table(x, sub.pending_rows);
    CHECK(names(room, "mask-room") && room.find(" where " + std::to_string(wf_index_mask_room(z.ranges, z.par.answers)) + " are needed") != std::string::npos);
  }
  {  // a mask row too short for a segment's sub-proof; below, one too short for a fold's polynomial
    const Sized room = of(with(good, "group link 1 23 22 17 13", "group link 1 23 22 17 12"), sub);
    CHECK(room.refused.empty() && names(hiding_conditions_missed(room, sub, checks), "mask-room"));
    Subproofs many = sub;  // too many pending claims (`pending_rows`) for the index-mask message
    many.pending_rows = ((u64)z.batch.input.shape.mask_rows * z.proof_codes.mask[z.batch.input.shape.mask_code].ell) - 2 * z.batch.S.k_col() - 1;
    CHECK(names(hiding_conditions_missed(z, many, checks), "mask-room"));
    many.pending_rows -= 1;
    CHECK(hiding_conditions_missed(z, many, checks).empty());
    Sized fold = z;
    fold.proof_codes.mask[fold.request.mask_code].ell = fold.d.link.group.mask.ell = 8;
    follow(fold.proof_codes, fold.request.last, fold.d.last.stages), tables_follow(fold);
    CHECK(names(hiding_conditions_missed(fold, sub, checks), "fold-room"));
  }
  {
    Sized x = z;
    x.batch.batches += 1;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "table"));
    CHECK(names(hiding_conditions_missed(z, sub, checks, TableUse{1, 1, 1}), "table"));  // `batches` is derived with no LINK proof counted: one on the anchor, beside all batches, does not fit
    x = z, x.d.set.salt_bytes = 32;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "salt"));
    x = z, x.request.at_start[0].leaf_class = 0;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "leaves"));
    x = z, x.batch.lines.requests = z.par.slots - 1;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "batch"));
    x = z, x.batch.S.omega.main_rows -= 1;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "padding-entries"));
  }
  {  // a table's row needs three spare entries per answer
    Sized x = of(file("record.f2.txt"), sub);
    x.par.table_data = (x.batch.S.matrix_rows << x.batch.S.k_lo) + 2;
    CHECK(names(hiding_conditions_missed(x, sub, checks), "padding-entries"));
  }
  {  // on the file alone: the smallest table tail that bears the tests before the first batch, and
     // one column less. A shorter tail lowers the rate and so the columns drawn, so the tail is
     // sought from the columns of the file's own table downward. Then a second one-time test.
    const TableUse use;
    const u64 start = std::max(use.one_time_tests * z.once.S.fold.columns + use.links_on_the_first * z.link.S.columns_first, use.links_on_a_copy * z.link.S.columns_further);
    Parameters few = z.par;
    auto missed = [&](u64 tail) { return few.table_random = tail, hiding_conditions_missed_by_the_file(few); };
    CHECK(missed(start).empty());
    u64 need = start;
    while (need > 1 && missed(need - 1).empty()) --need;
    CHECK(need == 59 && names(missed(need - 1), "table"));
    CHECK(names(hiding_conditions_missed_by_the_file(z.par, TableUse{2, 0, 1}), "table"));
  }
  {  /* D^2 / (C (B - 1)^2) = 2^34 / (2^16 * 1600) = 163.84; times 1.4426: 236; less 1 + 14 + 7 for
        the union: 214. */
    CHECK(range_check_bits(checks[0]) == 214);
    CHECK(names(hiding_conditions_missed(z, sub, {}), "range-checks"));
    std::vector<RangeCheck> wide = checks;
    wide[7].bound = 61;  // 72.8 * 1.4426 = 105 bits, less 22
    CHECK(range_check_bits(wide[7]) == 83 && names(hiding_conditions_missed(z, sub, wide), "range-checks"));
    wide[7].bound = 1;
    CHECK(names(hiding_conditions_missed(z, sub, wide), "range-checks"));
    /* The boundary, over a commitment's whole life: 2^34 * 1.4426 / (C * 1600) = 201 at C = 77,063
       columns and 200 at 77,064; less 22: 179, 178. Less 5 bits for 25 checks, 6 for a batch's 64
       requests, 20 for a leaf table's batches, 20 for the leaf tables that the anchor's random
       tail allows: 128 (the target) and 127. */
    wide = checks, wide[7].columns = 77063;
    CHECK(range_check_bits(wide[7]) == 179 && hiding_conditions_missed(z, sub, wide).empty());
    wide[7].columns = 77064;
    CHECK(range_check_bits(wide[7]) == 178 && names(hiding_conditions_missed(z, sub, wide), "range-checks"));
    CHECK(z.par.slots == 64 && ceil_log2(z.batch.batches) == 20 && ceil_log2(1 + (44040191 - 232) / z.link.S.columns_first) == 20);
    wide = checks, wide[3].digit_bits[1] = 9;  // digits for [-2^17, 3 * 2^17)
    CHECK(range_check_bits(wide[3]) == -1 && names(hiding_conditions_missed(z, sub, wide), "range-checks"));
    CHECK(names(hiding_conditions_missed(z, sub, checks, TableUse{2, 0, 1}), "table") && names(hiding_conditions_missed(z, sub, checks, TableUse{1, 0, 2}), "table"));
  }
  if (failures) std::printf("%d of %d checks FAILED\n", failures, made);
  else std::printf("ok, %d checks\n", made);
  return failures ? 1 : 0;
}
