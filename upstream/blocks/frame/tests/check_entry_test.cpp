// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes whole leaf files and checks them with blocks/frame/check_entry.h: honest files must be
// accepted; altered files, and honest files under other parameters, must be rejected. The tables
// have toy length; the target is the full 128 bits, proof of work counted.
// A leaf file (`entry` in names and messages) is a header, a leaf table's root, its record (stored
// answers to samples of its rows, if any) and the LINK proof, which ties the leaf table (`further`;
// "copy" in messages) to the anchor (`first`), the first committed table of the same data.
// main() has three parts, each described where it starts. None checks a key file (the anchor's root
// and proofs): the test itself marks the anchor as tested or, once in part 3, reads it unproved.
// Exits with code 0 and the last line "ok", or with code 1 and the last line "FAILED"; a failed
// BLK_REQUIRE aborts. The macros below must precede the first #include of a blocks/ header; a build
// with SERVING_BUILD rejects them (blocks/fence.h).
#define BLK_ENTRY_UNDER_FOR_A_TEST 1  // lets a test make an `EntryUnder` from a plan of its own
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // to build or alter plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets test code mark a table as tested (blocks/iop/foldeq.h)
#include <cstdio>
#include <string>

#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "frame/files.h"
#include "frame/link_frame.h"
#include "frame/tests/toy_file.h"
#include "frame/tests/toy_shapes.h"
#include "iop/tests/link_world.h"
#include "tests/damage.h"

using namespace blk;

// entry_under() must take only a `Configured` (configure()'s result, blocks/frame/configure.h), so
// that no caller can pass in altered values.
template <class C, class V, class = void> struct EntryUnderTakesValues : std::false_type {};
template <class C, class V> struct EntryUnderTakesValues<C, V, std::void_t<decltype(entry_under(std::declval<const C &>(), std::declval<const V &>()))>> : std::true_type {};
static_assert(!EntryUnderTakesValues<Configured, EntryFixed>::value, "entry_under() takes values from its caller");
static_assert(std::is_same_v<decltype(entry_under(std::declval<const Configured &>())), EntryUnder>, "control: entry_under(c) is no call");

// Test fixture. `w` (`LinkWorld`, blocks/iop/tests/link_world.h): configuration digest (the hash
// of the public files), codes, LINK's shape, plan (the list of all rounds, with sizes and errors).
// `fixed`: the values for `EntryUnder`. `planned`: the plan's bound on the file's length in bytes.
struct EntryWorld {
  LinkWorld w;
  EntryFixed fixed;
  u64 planned = 0;
};

// Part 1's fixture: parameters written here, with 6 bits of proof of work in the
// `ErrorKind::QUERY` rounds. The configuration digest is made up.
static EntryWorld world() {
  EntryWorld e;
  Setting set = setting_goldilocks();
  set.set_work(ErrorKind::QUERY, Scope::ONCE, 6);
  LinkLines lines;  // as `toy_lines` in link_frame_test.cpp
  lines.table_rows_log = 3, lines.table_log_n = 14, lines.table_data = 803, lines.table_random = 1001;
  lines.folded.log_n = 14, lines.folded.log_ell = 9;
  lines.folded.mask_log_n = 14, lines.folded.mask_log_ell = 4;
  lines.folded.stages = {StageChoice{2, 1, 14, 14}, StageChoice{0, 0, 16, 16}};
  const LinkSized x = link_sized(set, lines, link_table(set, lines, 9000, 9000), e.w.codes);
  if (!x.refused.empty()) std::printf("no such parameters: %s\n", x.refused.c_str());
  BLK_REQUIRE(x.refused.empty(), "the test's own lines give no link");
  e.w.S = x.S;
  Plan plan(set);
  plan.begin_scope(Scope::ONCE);
  TreeSize table;
  table.name = "table", table.log_n = x.table.code.log_n, table.leaf_bytes = (u64)set.f_bytes << x.table.v;
  plan_table_name(plan, "further table", x.table);
  plan_chain_start(plan, "link's chain", true);
  link_plan(plan, e.w.codes, x, table);
  e.w.plan = plan.finish();
  if (e.w.plan->refused()) std::printf("refused: %s\n", e.w.plan->why_not().c_str());
  e.fixed = entry_fixed(e.w.configuration, e.w.codes, x);
  e.planned = 8 + e.w.plan->size_of(0).bytes;
  return e;
}

// Part 2's fixture: parameters and plan from a parameter file's text; its hash is the configuration
// digest.
static EntryWorld world_of_file(const std::string &text) {
  EntryWorld e;
  Parameters par;
  std::string why;
  BLK_REQUIRE(read_parameters(text, par, why), "the toy parameter text is not read");
  const Derived d = derive(par, frame_shape_of(par, stand_in_subproofs()));
  BLK_REQUIRE(d.refused.empty(), "the toy parameter text gives no numbers");
  const LinkSized x = link_sized(d.set, link_lines(par), d.table, e.w.codes);
  if (!x.refused.empty()) std::printf("no such parameters: %s\n", x.refused.c_str());
  BLK_REQUIRE(x.refused.empty(), "the toy parameter text gives no link");
  e.w.S = x.S, e.w.configuration = configuration_digest({text});
  Plan plan(d.set);  // follows `file_entry` (blocks/frame/files.h)
  plan.begin_scope(Scope::ONCE);
  plan_table_name(plan, "further table", x.table);
  plan_chain_start(plan, "link's chain", true);
  link_plan(plan, e.w.codes, x, table_tree_of(d.set, par));
  e.w.plan = plan.finish();
  e.fixed = entry_fixed(e.w.configuration, e.w.codes, x);
  e.planned = 8 + e.w.plan->size_of(0).bytes;
  return e;
}

static Table rows_of(const EntryWorld &e, u64 seed) {
  const IndexShape &s = e.w.S.table;
  TestCoins coins(seed);
  Table rows(1, s.rows(), s.plain());
  for (size_t r = 0; r < s.rows(); r++)
    for (size_t j = 0; j < s.plain(); j++) {
      F x;
      coins.fill(&x, 1);
      rows.set(r, j, E::from(x));
    }
  return rows;
}

static std::vector<u8> made(const EntryWorld &e, LinkTable &first, LinkTable &further, u64 coin_seed) {
  TestCoins coins(coin_seed);
  EngineCpu engine;
  Writer file;
  CHECK(make_entry_under(EntryUnder(e.w.plan, e.fixed), *first.index, *further.index, engine, coins, nullptr, file), "an entry is not made");
  return file.bytes();
}
static std::string last_place;  // the step at which `checked` last rejected; empty if it accepted
// The committer's anchor, marked as tested (blocks/iop/foldeq.h) in place of a checked key file.
static TestedTable vouched(const IndexKey &first, const Setting &under) { return TestedTable::on_a_tests_word(first, under, {}); }  // {}: under plans built by hand the key's public values are not compared
static bool checked(const EntryWorld &e, const EntryFixed &fixed, const IndexKey &first, const std::vector<u8> &bytes, TestedTable *kept = nullptr, std::string *said = nullptr) {
  Reader file(bytes.data(), bytes.size());
  TestedTable entry;
  const char *why = nullptr;
  const char *place = nullptr;
  const bool yes = check_entry_under(EntryUnder(e.w.plan, fixed), vouched(first, e.w.plan->set), file, entry, why, nullptr, &place);
  CHECK(yes == (why == nullptr) && place && yes == !*place, "a refusal without a reason or a place, or one of them without a refusal");
  last_place = place ? place : "";
  if (kept) *kept = entry;
  if (said) *said = why ? why : "";
  return yes;
}

int main() {
  // Part 1, under plans built here with blocks/frame/link_frame.h. An
  // honest file is accepted, is no longer than planned, and yields its table's `TestedTable`
  // (blocks/iop/foldeq.h: a table's key that has passed its check). A second file for the same leaf
  // table differs and is accepted if the table's random tails (`Code::r` random coefficients per
  // row; "random entries" in messages) cover the columns of two LINK proofs; otherwise none is made
  // and nothing is written.
  // Rejected: a damaged, cut or longer file; a leaf table with one cell changed; the honest file
  // under another anchor, configuration digest or salt length, a faulty shape or a rejected plan.
  {
    const EntryWorld e = world();
    CHECK(e.w.plan->holds(), "the plan does not hold its target");
    const Table rows = rows_of(e, 11);
    LinkTable first = link_commit(e.w, rows, 21), further = link_commit(e.w, rows, 22);
    const std::vector<u8> file = made(e, first, further, 31);

    TestedTable kept;
    std::string said;
    CHECK(checked(e, e.fixed, first.key, file, &kept, &said), "an honest entry is refused");
    if (!said.empty()) std::printf("refused: %s\n", said.c_str());
    CHECK(kept.reader().commitment() == further.key.commitment() && kept.reader().record().y == further.key.rec.y && kept.reader().name() == further.key.name(),
          "what the reader keeps of the copy is not the copy's key");
    {  // optional notes (intermediate values): three from LINK, and two on the records before them
      std::vector<std::pair<std::string, E>> notes;
      Reader reader(file.data(), file.size());
      TestedTable again;
      const char *why = nullptr;
      CHECK(check_entry_under(EntryUnder(e.w.plan, e.fixed), vouched(first.key, e.w.plan->set), reader, again, why, &notes), "with notes asked for an honest entry is refused");
      std::vector<const std::pair<std::string, E> *> of_link;
      for (const auto &n : notes)
        if (n.first.rfind("LINK: ", 0) == 0) of_link.push_back(&n);
      CHECK(of_link.size() == 5u, "LINK makes five notes: two on the records, three others");
      if (of_link.size() == 5) CHECK(of_link[0]->first.find("first table's record") != std::string::npos && of_link[1]->first.find("further table's record") != std::string::npos && !(of_link[0]->second == of_link[1]->second), "the two records' notes are not there, or the two makings' folded answers are equal");
    }
    const size_t answers = e.w.S.table.rows() * e.w.S.table.answers * E_BYTES, chain_at = 8 + DIGEST_BYTES + answers;
    CHECK(file.size() > chain_at + DIGEST_BYTES && file.size() <= e.planned, "the file is longer than planned");
    std::printf("an entry of %zu bytes (planned: at most %llu): head 8, root %zu, answers %zu, the link %zu\n", file.size(), (unsigned long long)e.planned,
                (size_t)DIGEST_BYTES, answers, file.size() - chain_at);
    if (2 * e.w.S.columns_further <= e.w.S.table.code.r) {
      const std::vector<u8> again = made(e, first, further, 32);
      CHECK(again != file && checked(e, e.fixed, first.key, again), "a second entry for the same copy");
    } else {  // the leaf table's random tails cover the columns of one LINK proof only
      TestCoins coins(32);
      EngineCpu engine;
      Writer nothing;
      CHECK(!make_entry_under(EntryUnder(e.w.plan, e.fixed), *first.index, *further.index, engine, coins, nullptr, nothing) && nothing.bytes().empty(),
            "a second entry is made for a copy whose random entries bear one link's columns");
    }

    {  // the whole file (blocks/tests/damage.h): only about 40 places, or as BLK_TEST_DAMAGE sets
      const DamageChoice how = damage_choice(file.size(), 40);
      const DamageCount damaged = damage("an entry", file, how, [&](const u8 *b, size_t n) { return checked(e, e.fixed, first.key, std::vector<u8>(b, b + n)); });
      CHECK(!damaged.bad, "a damaged entry is accepted");
      std::printf("refused: %s\n", damaged.text(how).c_str());
    }
    // One bit flipped in tag, version word, root, answers, the chain's salt (the LINK proof's first
    // bytes) and later bytes; then the file cut at, or one byte before, the borders of these parts.
    std::vector<size_t> places = {0, 4, 8, 8 + DIGEST_BYTES - 1, chain_at, chain_at + DIGEST_BYTES, file.size() - 1};
    if (answers) places.push_back(8 + DIGEST_BYTES), places.push_back(chain_at - E_BYTES);
    for (size_t at : places) {
      std::vector<u8> bad = file;
      bad[at] ^= 1;
      CHECK(!checked(e, e.fixed, first.key, bad), "an entry with one byte changed is accepted");
    }
    for (size_t length : {size_t(0), size_t(7), size_t(8), 8 + (size_t)DIGEST_BYTES - 1, chain_at - 1, chain_at, chain_at + DIGEST_BYTES, file.size() - 1}) {
      const std::vector<u8> cut(file.begin(), file.begin() + (long)length);
      CHECK(!checked(e, e.fixed, first.key, cut), "an entry that is cut short is accepted");
    }
    {
      std::vector<u8> longer = file;
      longer.push_back(0);
      CHECK(!checked(e, e.fixed, first.key, longer), "an entry with a byte behind its end is accepted");
      std::vector<u8> other_kind = file;
      Writer head;
      write_head(head, TAG_KEY);
      std::copy(head.bytes().begin(), head.bytes().end(), other_kind.begin());
      CHECK(!checked(e, e.fixed, first.key, other_kind, nullptr, &said) && said == "file-of-another-kind", "a key's tag passes for an entry's");
    }
    if (answers) {
      std::vector<u8> bad = file;
      for (size_t i = 0; i < F_BYTES; i++) bad[8 + DIGEST_BYTES + i] = 0xff;  // above the modulus
      CHECK(!checked(e, e.fixed, first.key, bad, nullptr, &said) && said == "the record's answers", "an answer that is no canonical element is read");
    }

    // Other tables, other parameters. The changed cell must be caught in LINK's proximity test, at
    // its last step (BASE).
    {
      Table other = rows;
      other.set(1, 5, other.at(1, 5) + E::one());
      LinkTable changed = link_commit(e.w, other, 23);
      CHECK(!checked(e, e.fixed, first.key, made(e, first, changed, 33), nullptr, &said) && said.rfind("BASE: ", 0) == 0 && last_place == LINK_REFUSED_TEST, "a copy with one cell changed gets an entry");
      LinkTable another_first = link_commit(e.w, rows, 24);
      CHECK(!checked(e, e.fixed, another_first.key, file), "an entry is accepted under another first table");
      EntryFixed elsewhere = e.fixed;
      elsewhere.configuration.w[1] ^= 1;
      CHECK(!checked(e, elsewhere, first.key, file), "an entry is accepted under another configuration");
      elsewhere = e.fixed, elsewhere.link.table.salt_bytes += 8;
      CHECK(!checked(e, elsewhere, first.key, file), "an entry is accepted with another length of the salts in the tables' trees");
      elsewhere = e.fixed, elsewhere.link.later[0].k++;
      CHECK(!checked(e, elsewhere, first.key, file), "a shape with a fault is run");
    }
    {
      EntryWorld weak = world();
      Plan refused(weak.w.plan);
      refused.refuse("made to fail");
      weak.w.plan = refused;
      CHECK(!checked(weak, weak.fixed, first.key, file, nullptr, &said) && said == "the parameters are not admitted (the first cause is on stderr)", "an entry is checked under a plan that does not hold");
    }
  }
  // Part 2, under world_of_file()'s plan: an honest file is accepted, also with notes asked for;
  // under the digest of the text with one more blank at its end it is rejected.
  {
    const std::string text = toy_file();
    const EntryWorld e = world_of_file(text);
    CHECK(e.w.plan->holds(), "the plan from the parameter text does not hold its target");
    const Table rows = rows_of(e, 11);
    LinkTable first = link_commit(e.w, rows, 21), further = link_commit(e.w, rows, 22);
    const std::vector<u8> file = made(e, first, further, 31);
    CHECK(checked(e, e.fixed, first.key, file) && file.size() <= e.planned, "an entry from a parameter file is refused, or longer than planned");
    std::printf("from a parameter text of %zu bytes: %llu + %llu columns, an entry of %zu bytes (planned: at most %llu)\n", text.size(), (unsigned long long)e.w.S.columns_first,
                (unsigned long long)e.w.S.columns_further, file.size(), (unsigned long long)e.planned);
    EntryFixed other = e.fixed;
    other.configuration = configuration_digest({text + " "});
    CHECK(!checked(e, other, first.key, file), "an entry is accepted under a parameter text with one blank more");
    std::vector<std::pair<std::string, E>> notes;
    Reader reader(file.data(), file.size());
    TestedTable kept;
    const char *why = nullptr;
    CHECK(check_entry_under(EntryUnder(e.w.plan, e.fixed), vouched(first.key, e.w.plan->set), reader, kept, why, &notes) && notes.size() >= 3, "with notes asked for the answer is another, or no note is made");
    size_t of_link = 0;
    for (const auto &n : notes) of_link += n.first.rfind("LINK: ", 0) == 0;
    CHECK(of_link == 5, "LINK makes five notes");
  }
  // Part 3, the functions that programs call, on what configure() makes of the toy's two public
  // files (toy_file(), toy_model()). make_entry() writes the same bytes as make_entry_under() does
  // on the same plan and values; check_entry()'s verdict says whether proof of work was needed.
  // Rejected: the file under public files with one byte more, or that configure() rejects.
  // read_entry_table_unproved() reads the same root, name and record but skips the LINK proof: it
  // proves nothing. check_entry_if_tested(), for an anchor read without its proofs, answers at best
  // `IfTested::YES_IF_THE_TABLE_IS_TESTED` or `YES_WITH_WORK_IF_THE_TABLE_IS_TESTED`: no plain yes.
  {
    const std::string text = toy_file();
    const Configured c = configure({text, toy_model()}, toy_shapes);
    if (!c.refused().empty()) std::printf("refused: %s\n", c.refused().c_str());
    CHECK(c.refused().empty(), "the toy's public files are not admitted");
    if (c.refused().empty()) {
      EntryWorld e;  // only shape and configuration digest are set: enough to commit the tables
      e.w.S = c.sized().link.S, e.w.configuration = c.configuration();
      const Table rows = rows_of(e, 11);
      LinkTable first = link_commit(e.w, rows, 21), further = link_commit(e.w, rows, 22);
      Writer file;
      {
        TestCoins coins(31);
        EngineCpu engine;
        CHECK(make_entry(c, *first.index, *further.index, engine, coins, nullptr, file), "an entry is not made from the public files");
      }
      auto judged = [&](const Configured &under, std::string *said = nullptr) {
        Reader reader(file.bytes().data(), file.bytes().size());
        TestedTable kept;
        const char *why = nullptr;
        const ProofVerdict verdict = check_entry(under, on_a_tests_word(under, first.key), reader, kept, why);
        if (said) *said = why ? why : "";
        CHECK((verdict == ProofVerdict::NO) == (why != nullptr), "a refusal without a reason, or a reason without a refusal");
        CHECK(verdict == ProofVerdict::NO || (verdict == ProofVerdict::YES_WITH_WORK) == under.work_of_an_entry().needed, "the verdict does not say whether work was needed");
        return verdict != ProofVerdict::NO && kept.reader().commitment() == further.key.commitment();
      };
      std::string said;
      CHECK(judged(c, &said), "an entry from the public files is refused");
      {  // a table comes out even if the LINK proof is damaged; tag and root's length are checked
        auto unproved = [&](const std::vector<u8> &bytes, UnprovedTable &kept) {
          Reader reader(bytes.data(), bytes.size());
          const char *why = nullptr;
          return read_entry_table_unproved(c, reader, kept, why);
        };
        UnprovedTable kept;
        CHECK(unproved(file.bytes(), kept) && kept.reader().commitment() == further.key.commitment() && kept.reader().name() == further.key.name() && kept.reader().record().y == further.key.rec.y, "the reader without the link gives another key");
        std::vector<u8> spoilt = file.bytes();
        spoilt.back() ^= 1;
        CHECK(unproved(spoilt, kept) && kept.reader().commitment() == further.key.commitment(), "the reader without the link looks at the link");
        spoilt = file.bytes(), spoilt[0] ^= 1;
        CHECK(!unproved(spoilt, kept), "the reader without the link takes another kind of file");
        spoilt = file.bytes(), spoilt.resize(8 + DIGEST_BYTES - 1);
        CHECK(!unproved(spoilt, kept), "the reader without the link takes a root that is cut short");
      }
      {  // when another process checks the key file: the anchor is read from its first bytes only
        Writer head;
        write_head(head, TAG_KEY);
        head.put(first.key.root);
        for (const E &y : first.key.rec.y) head.put(y);
        Reader from(head.bytes().data(), head.bytes().size()), reader(file.bytes().data(), file.bytes().size());
        UnprovedTable unproved_first;
        const char *why = nullptr;
        CHECK(read_key_table_unproved(c, from, unproved_first, why) && unproved_first.reader().name() == first.key.name(), "a key's first bytes are not read");
        const IfTested result = check_entry_if_tested(c, unproved_first, reader, why);
        CHECK(result == (c.work_of_an_entry().needed ? IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED : IfTested::YES_IF_THE_TABLE_IS_TESTED), "an honest entry under an unproved first table");
        std::vector<u8> spoilt = file.bytes();
        spoilt.back() ^= 1;
        Reader bent(spoilt.data(), spoilt.size());
        CHECK(check_entry_if_tested(c, unproved_first, bent, why) == IfTested::NO && why && why != IF_TESTED_WHY, "an entry with its last byte changed is taken under an unproved first table");
      }
      if (!said.empty()) std::printf("refused: %s\n", said.c_str());
      std::printf("from the toy's two public files: %llu + %llu columns, an entry of %zu bytes; work needed: %s\n", (unsigned long long)e.w.S.columns_first,
                  (unsigned long long)e.w.S.columns_further, file.bytes().size(), c.work_of_an_entry().needed ? "yes" : "no");
      {
        LinkTable first_again = link_commit(e.w, rows, 21), further_again = link_commit(e.w, rows, 22);
        TestCoins coins(31);
        EngineCpu engine;
        Writer plain;
        const EntryFixed by_hand = entry_fixed(c.configuration(), c.sized().entry_codes, c.sized().link);
        CHECK(make_entry_under(EntryUnder(c.entry().chains.at(0).plan, by_hand), *first_again.index, *further_again.index, engine, coins, nullptr,
                         plain) &&
                  plain.bytes() == file.bytes(),
              "the two doors of make_entry give two files");
      }
      const Configured longer = configure({text, toy_model() + "\n"}, toy_shapes);
      CHECK(longer.refused().empty(), "control: the public files with one byte more are not admitted, so nothing is shown by a refusal under them");
      CHECK(!judged(longer, &said) && said != "the public files are not admitted", "an entry is accepted under public files with one byte more");
      CHECK(!judged(configure({text}, toy_shapes), &said) && said == "the public files are not admitted", "an entry is judged under public files that are not admitted");
    }
  }
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
