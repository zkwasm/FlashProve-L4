// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs LINK at toy message lengths but at the real target of 128 bits; at full size nothing runs.
// LINK (blocks/iop/link.h) is to show that a further committed table (`Index` in identifiers), a
// leaf table, has the same entries before the cut (`LinkShape::cut()`) as the first, the anchor.
// blocks/frame/link_frame.h derives the numbers of columns and positions, the random tails of
// LINK's codes, the proximity test's stages and the hash chain's plan (its rounds with error
// bounds). The plan must hold: every round reaches the target with its proof of work. So the
// accounting printed from such a plan is that of the protocol that runs.
// No one-time test (the committer's test on the anchor) runs here: `link_check`
// (blocks/iop/tests/link_world.h) vouches that the first table has passed one.
//   link_frame_test [print]     print: also the accounting of each chain
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets blocks/iop/tests/link_world.h write plans by hand
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets it call `on_a_tests_word` (blocks/iop/foldeq.h)
#include <cstring>
#include <fstream>
#include <sstream>

#include "chain/tests/death.h"
#include "frame/files.h"
#include "frame/link_frame.h"
#include "iop/tests/link_world.h"
#include "numbers/accounting.h"

using namespace blk;

static LinkLines toy_lines() {
  LinkLines L;
  L.table_rows_log = 3, L.table_log_n = 14, L.table_data = 803, L.table_random = 1001;
  L.folded.log_n = 14, L.folded.log_ell = 9;
  L.folded.mask_log_n = 14, L.folded.mask_log_ell = 4;
  L.folded.stages = {StageChoice{2, 1, 14, 14}, StageChoice{0, 0, 16, 16}};
  return L;
}

int main(int argc, char **argv) {
  const bool print = argc > 1 && !std::strcmp(argv[1], "print");
  {
    Setting set = setting_goldilocks();
    set.set_work(ErrorKind::QUERY, Scope::ONCE, 6);
    LinkWorld w;
    Plan plan(set);
    const LinkLines lines = toy_lines();
    const LinkSized x = link_sized(set, lines, link_table(set, lines, 9000, 9000), w.codes);
    if (!x.refused.empty()) return std::printf("FAILED: no such parameters: %s\n", x.refused.c_str()), 1;
    w.S = x.S;
    TreeSize table;
    table.name = "table", table.log_n = x.table.code.log_n, table.leaf_bytes = (u64)set.f_bytes << x.table.v;
    plan.begin_scope(Scope::ONCE);
    plan_chain_start(plan, "link's chain", true);
    link_plan(plan, w.codes, x, table);
    w.plan = plan.finish();
    if (w.plan->refused()) std::printf("refused: %s\n", w.plan->why_not().c_str());
    CHECK(w.plan->holds(), "a planned round does not hold at the target");
    if (print) print_accounting(account(w.plan), "link's chain");
    {  // planning under a code with one random entry more than the stages are sized for must abort
      Codes other = w.codes;
      other.base[x.S.code].r += 1;
      CHECK(ends_with("a stage of the proximity test is sized for another main code than it names", [&] {
              Plan q(set);
              q.begin_scope(Scope::ONCE);
              link_plan(q, other, x, table);
            }),
            "a link's chain is planned under codes that its proximity test is not sized for");
    }

    LinkCase t{};
    t.name = "sized", t.data = w.S.table.plain() - 3;
    const Table rows = link_rows(t, w, 11);
    auto commit = [&](const Table &r, u64 seed) {
      LinkTable made;
      TestCoins coins(seed);
      made.index.reset(new IndexCpu(w.S.table, r, coins, w.configuration));
      made.key = made.index->key();
      return made;
    };
    LinkTable first = commit(rows, 21), further = commit(rows, 22);
    const LinkProof p = link_make(w, first, further, 31);
    const char *why = "";
    CHECK(p.made && link_check(w, first.key, further.key, p.bytes, &why), "an honest link is refused");
    Table other = rows;
    link_change(other, 1, (size_t)(w.S.cut() - 1));
    LinkTable differs = commit(other, 23);
    LinkTable first_again = commit(rows, 21);
    const LinkProof q = link_make(w, first_again, differs, 32);
    CHECK(q.made && !link_check(w, first_again.key, differs.key, q.bytes), "a copy with one cell changed is linked");
    const PartSize planned = w.plan->size_of(0);
    std::printf("columns %llu of the first table, %llu of the further one; positions %llu; a link %zu bytes (planned: at most %llu, "
                "expected %.0f) %s\n",
                (unsigned long long)w.S.columns_first, (unsigned long long)w.S.columns_further, (unsigned long long)x.linked.t,
                p.bytes.size(), (unsigned long long)planned.bytes, planned.expected, why);
    CHECK(p.bytes.size() <= planned.bytes, "a link is longer than planned");
  }
  // At full size the parameter file is only sized and planned; every plan must hold.
  {
    const std::string path = "frame/examples/record.f2.txt";
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::printf("FAILED: %s cannot be read: run the test from the folder blocks/\n", path.c_str()), 1;
    std::stringstream text;
    text << in.rdbuf();
    std::string lines = text.str();
    if (lines.size() > 11) lines[11] = (char)('0' + BLK_HASH);  // the examples are written for hash 1: the version word's fourth character
    Parameters par;
    std::string why;
    if (!read_parameters(lines, par, why)) return std::printf("FAILED: %s: %s\n", path.c_str(), why.c_str()), 1;
    const Derived d = derive(par, frame_shape_of(par, stand_in_subproofs()));
    if (!d.refused.empty()) return std::printf("FAILED: %s: %s\n", path.c_str(), d.refused.c_str()), 1;
    const Setting &set = d.set;
    Codes codes;
    const LinkSized x = link_sized(set, link_lines(par), d.table, codes);
    if (!x.refused.empty()) return std::printf("FAILED: %s: no such parameters: %s\n", path.c_str(), x.refused.c_str()), 1;
    Plan plan(set);
    TreeSize table;
    table.name = "table", table.log_n = x.table.code.log_n, table.leaf_bytes = (u64)set.f_bytes << x.table.v;
    plan.begin_scope(Scope::ONCE);
    plan_chain_start(plan, "link's chain", true);
    link_plan(plan, codes, x, table);
    if (plan.refused()) std::printf("refused: %s\n", plan.why().c_str());
    CHECK(plan.finish()->holds(), "a planned round of an example file does not hold at its target");
    const PartSize planned = plan.finish()->size_of(0);
    std::printf("%s: %s; columns %llu + %llu; positions %llu; a link at most %llu bytes, expected %.0f\n", path.c_str(), name_of(x.linked.group.rule),
                (unsigned long long)x.S.columns_first, (unsigned long long)x.S.columns_further, (unsigned long long)x.linked.t,
                (unsigned long long)planned.bytes, planned.expected);
    {  // mask rows too short for the first fold's hiding polynomial: `link_sized` must reject them
      LinkLines lines = link_lines(par);
      lines.folded.mask_log_ell = 4;
      Codes other;
      CHECK(x.linked.group.k >= 8 && !link_sized(set, lines, d.table, other).refused.empty(), "mask rows of 16 entries under a fold of 8 or more variables are admitted");
    }
  }
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
