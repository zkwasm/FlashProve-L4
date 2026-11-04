// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests blocks/frame, which derives from a parameter file the sizes and plans (lists of rounds) of
   the files that a verifier reads: key file, leaf file (`entry` in identifiers), proof file. On an
   example file: `read_parameters` rejects malformed variants and other versions;
   derived sizes have known values and meet the target; compact and full plans agree; `fields_of`
   covers every byte of the three files. None of them is made or checked, and `stand_in_subproofs`
   stands in for a model's sub-proofs. Run from blocks/. */
#include <cstdio>
#include <fstream>
#include <sstream>

#include "frame/fields.h"
#include "frame/files.h"
#include "frame/heads.h"
#include "chain/rounds.h"
#include "hash/hash.h"
#include "numbers/accounting.h"

using namespace blk;

static int failures = 0, checks = 0;
#define CHECK(c)                                                   \
  do {                                                             \
    ++checks;                                                      \
    if (!(c)) ++failures, std::printf("%s:%d: %s\n", __FILE__, __LINE__, #c); \
  } while (0)

static std::string example() {
  std::ifstream in("frame/examples/record.f2.txt", std::ios::binary);
  std::stringstream s;
  s << in.rdbuf();
  std::string t = s.str();
  if (t.size() > 11) t[11] = (char)('0' + BLK_HASH);  // the examples are written for hash 1
  return t;
}
static bool reads(const std::string &t, std::string *why = nullptr) {
  Parameters p;
  std::string w;
  const bool ok = read_parameters(t, p, w);
  if (why) *why = w;
  return ok;
}
static std::vector<std::string> lines_of(const std::string &t) {
  std::vector<std::string> v;
  std::istringstream in(t);
  for (std::string l; std::getline(in, l);) v.push_back(l);
  return v;
}
static std::string join(const std::vector<std::string> &v) {
  std::string t;
  for (auto &l : v) t += l + "\n";
  return t;
}
static Digest digest_of(const std::string &t) { return parameter_file_digest(t); }

static void test_parser() {
  const std::string good = example();
  CHECK(reads(good));
  CHECK(reads(good + "a line of the statement 1\n"));  // text after the line `table` is returned unparsed in `Parameters::rest`
  const auto l = lines_of(good);
  for (size_t i = 0; i < l.size(); ++i) {
    auto drop = l, twice = l, blank = l, tail = l, two = l, zero = l, cr = l, more = l, sign = l, upper = l;
    drop.erase(drop.begin() + (long)i);
    twice.insert(twice.begin() + (long)i, l[i]);
    blank.insert(blank.begin() + (long)i, "");
    tail[i] += " ";
    two[i].insert(two[i].find(' '), " ");
    zero[i].insert(zero[i].rfind(' ') + 1, "0");
    cr[i] += "\r";
    more[i] += " 1";
    sign[i].insert(sign[i].rfind(' ') + 1, "+");
    upper[i][0] = (char)(upper[i][0] - 32);
    for (const auto &bad : {drop, blank, tail, two, zero, cr, sign, upper}) CHECK(!reads(join(bad)));
    if (i + 1 < l.size()) CHECK(!reads(join(twice)));  // a second line `table` is not parsed
    if (l[i].compare(0, 6, "stages")) CHECK(!reads(join(more)));
    if (i + 1 < l.size()) {
      auto swap = l;
      std::swap(swap[i], swap[i + 1]);
      CHECK(!reads(join(swap)));
    }
    CHECK(!(digest_of(join(more)) == digest_of(good)));
  }
  CHECK(!reads(good.substr(0, good.size() - 1)));  // the last line feed
  CHECK(!reads("\n" + good) && !reads(" " + good) && !reads(""));
  auto with_version = [&](const std::string &w) { return "version " + w + good.substr(12); };
  std::string why;
  CHECK(!reads(with_version("HG06"), &why) && !why.compare(0, 15, "version-unknown"));
  CHECK(!reads(with_version("H721"), &why) && !why.compare(0, 15, "version-unknown"));
  CHECK(!reads(with_version(std::string("H") + FORMAT_NUMBER + (char)('0' + 3 - BLK_FIELD) + (char)('0' + BLK_HASH)), &why) && !why.compare(0, 15, "version-unknown"));
  CHECK(!reads(with_version(std::string("H") + FORMAT_NUMBER + (char)('0' + BLK_FIELD) + (char)('0' + 3 - BLK_HASH)), &why) && !why.compare(0, 15, "version-unknown"));
  CHECK(!reads(with_version(std::string("H") + (char)(FORMAT_NUMBER + 1) + "11"), &why) && !why.compare(0, 15, "version-unknown"));
  CHECK(version_text(SERVED_VERSION).size() == 4 && version_from_text(version_text(SERVED_VERSION)) == SERVED_VERSION);
  // Values that are out of range or do not fit one another.
  auto with = [&](const std::string &from, const std::string &to) {
    std::string t = good;
    const size_t i = t.find(from);
    CHECK(i != std::string::npos);
    return i == std::string::npos ? t : t.replace(i, from.size(), to);
  };
  CHECK(!reads(with("group link 1 23 22", "group link 1 23 23")) && !reads(with("group link 1 23 22 17", "group link 1 23 22 24")));
  CHECK(!reads(with("group link 1", "group link 3")) && !reads(with("sec 128", "sec 0")) && !reads(with("sec 128", "sec 257")));
  CHECK(!reads(with("group once-own 1 22 18", "group once-own 1 21 18")) && !reads(with("group omega 1 22 18 13", "group omega 1 22 18 12")));
  CHECK(!reads(with("work positions 20", "work positions 128")) && !reads(with("0 0 15 15\ngroup folded", "1 0 15 15\ngroup folded")));
  CHECK(!reads(with("group own 1 22 18", "group own 1 21 18")) && !reads(with("salt 40", "salt 8")) && !reads(with("salt 40", "salt 32")) && !reads(with("salt 40", "salt 44")) && reads(with("salt 40", "salt 48")));
}

static std::string with(std::string t, const std::string &a, const std::string &b) {
  const size_t at = t.find(a);
  if (at == std::string::npos) return std::printf("no %s\n", a.c_str()), ++failures, t;
  return t.replace(at, a.size(), b);
}
static void test_refusals() {
  std::string why;
  for (const char *slots : {"slots 3", "slots 12", "slots 65"}) CHECK(!reads(with(example(), "slots 64", slots), &why) && !why.compare(0, 22, "parameter-range: slots"));
  CHECK(reads(with(example(), "slots 64", "slots 1"), &why) && reads(with(example(), "slots 64", "slots 128"), &why));
  // `answers 0`: no out-of-domain samples; the soundness argument needs them
  // (blocks/iop/req.h, req_shape_fault()). `answers 2`: above the line's range.
  for (const char *answers : {"answers 0", "answers 2"}) CHECK(!reads(with(example(), "answers 1", answers), &why) && !why.compare(0, 24, "parameter-range: answers"));
  // Three lines take one value, and the parser alone holds them to it.
  CHECK(!reads(with(example(), "reading 1", "reading 2"), &why) && !why.compare(0, 24, "parameter-range: reading") && !reads(with(example(), "reading 1", "reading 0"), &why) && !why.compare(0, 24, "parameter-range: reading"));
  CHECK(!reads(with(example(), "rowcost 1", "rowcost 2"), &why) && !why.compare(0, 24, "parameter-range: rowcost") && !reads(with(example(), "rowcost 1", "rowcost 0"), &why) && !why.compare(0, 24, "parameter-range: rowcost"));
  CHECK(!reads(with(example(), "listed 0", "listed 1"), &why) && !why.compare(0, 23, "parameter-range: listed") && !reads(with(example(), "listed 0", "listed 8"), &why) && !why.compare(0, 23, "parameter-range: listed"));
  CHECK(!reads(with(example(), "record 1", "record 0"), &why) && !why.compare(0, 23, "parameter-range: record") && !reads(with(example(), "record 1", "record 2"), &why) && !why.compare(0, 23, "parameter-range: record"));
  for (const char *rule : {"group link 0", "group link 2"}) CHECK(!reads(with(example(), "group link 1", rule), &why) && !why.compare(0, 21, "parameter-group-shape"));
  CHECK(!reads(with(example(), "44040191 0 18", "44040191 1 18"), &why) && !why.compare(0, 21, "parameter-table-shape"));
  // more than 15 digits: rejected, so that no sum of two values wraps
  CHECK(!reads(with(example(), "salt 40", "salt 1000000000000000"), &why) && !why.compare(0, 25, "parameter-number-spelling"));
  // No code is longer than 2^32: the table's (the example has 2^32), a group's, a later stage's.
  CHECK(!reads(with(example(), "table 11 32 ", "table 11 33 "), &why) && !why.compare(0, 21, "parameter-table-shape"));
  CHECK(reads(with(example(), "group link 1 23 ", "group link 1 32 ")) && !reads(with(example(), "group link 1 23 ", "group link 1 33 "), &why) && !why.compare(0, 21, "parameter-group-shape"));
  CHECK(reads(with(example(), "stages link 4 3 22 ", "stages link 4 3 32 ")) && !reads(with(example(), "stages link 4 3 22 ", "stages link 4 3 33 "), &why) && !why.compare(0, 22, "parameter-stages-shape"));
  {  // if `Sized::refused` is set, every chain's plan is rejected for the same reason
    Parameters par;
    CHECK(read_parameters(example(), par, why));
    const Subproofs sub = stand_in_subproofs();
    Sized z = sized(par, sub);
    z.refused = "made up";
    for (const FilePlan &f : {file_key(z, true), file_entry(z, true), file_proof(z, sub, false)})
      for (const ChainPlan &c : f.chains) CHECK(c.plan->refused() && !c.plan->holds() && c.plan->why_not() == "made up");
  }
}

static void test_derive_and_plans() {
  Parameters par;
  std::string why;
  CHECK(read_parameters(example(), par, why));
  const Subproofs sub = stand_in_subproofs();
  const Sized z = sized(par, sub);
  CHECK(z.refused.empty());
  if (!z.refused.empty()) return (void)std::printf("%s\n", z.refused.c_str());
  const Derived &d = z.d;
  const Setting &s = d.set;
  // `d.link`: the group of one segment of a request's proof (`link` in identifiers); `z.link`: the
  // LINK proof, which ties a leaf table to the first committed table. `t`: positions tested; `r`: a
  // row's random coordinates (its random tail), which are to hide them.
  CHECK(d.link.t == 219 && analysis(s, d.link.group).at.source == CountSource::JOHNSON);  // proof of work only on rounds that draw positions
  CHECK(hidden_positions(s, d.link.group.main) >= d.link.t && hidden_positions(s, d.link.group.mask) >= d.link.t);
  CHECK(d.link.group.main.r <= d.link.t + 1);  // and at most t + 1 random coordinates
  // `z.batch`, `z.once`: LIN, the test on the committed table, in a batch and in the key file
  for (const LinSized *x : {&z.batch, &z.once}) {  // one oracle message in groups omega and own: random coordinates for the batch's list and every request's
    CHECK(x->omega.group.main.r == x->own.group.main.r);
    CHECK(hidden_positions(s, x->omega.group.main) >= x->omega.t + x->lines.requests * x->own.t);
    CHECK(hidden_positions(s, x->folded.group.mask) >= x->folded.t + x->omega.t);  // the mask message `shared_split` is in the groups folded and omega
  }
  // All groups go up to the Johnson radius, where the proximity term (`closeness`) rests on the
  // theorem on codes, taken as given; no proof of it verified by us (blocks/numbers/codes.h).
  CHECK(d.link.group.rule == Radius::JOHNSON && z.batch.folded.group.rule == Radius::JOHNSON && analysis(s, d.table).once.source == CountSource::JOHNSON);
  for (const TestSizes *x : {&z.once.folded, &z.once.omega, &z.once.own, &z.link.linked})
    for (const Stage &st : x->stages) CHECK(st.g.rule == Radius::JOHNSON);
  // table columns that each test opens; batches that the table's random coordinates allow
  const u64 batch = 64, once = 62;  // the table's code has length 2^32
  CHECK(z.batch.S.fold.columns == batch && z.once.S.fold.columns == once && z.link.S.columns_first == once && z.link.S.columns_further == once);
  CHECK(z.batch.batches == 688127);
  /* Where the mask code has the larger rate it limits the distance. File lengths do not show this:
     many distances give the same number of positions. First loop: the distance is below both codes'
     Johnson radius. Second loop (stages 2 to 4 of the test on a batch's folded row): the first fold
     round's error is within one bit of the target and the mask code's term is over half of it; the
     main code alone would allow a distance beyond the mask code's radius, where no lemma covers
     mask rows. */
  for (const TestSizes *x : {&d.last, &z.batch.folded, &z.batch.omega, &z.batch.own})
    for (const Stage &st : x->stages) CHECK(surely_lt(analysis(s, st.g).at.delta, Bound(1) - Bound::sqrt(rho_of(s, st.g.mask))) && surely_lt(analysis(s, st.g).at.delta, Bound(1) - Bound::sqrt(rho_of(s, st.g.main))));
  for (size_t i = 1; i <= 3; ++i) {
    const GroupSize &g = z.batch.folded.stages[i].g;
    const Analysis a = analysis(s, g);
    const Bound radius = Bound(1) - Bound::sqrt(rho_of(s, g.mask));
    CHECK(mask_rules(s, g) && z.batch.folded.stages[i].t == 54 && g.h == 2);
    CHECK(g.mask.log_n == 12 && g.mask.k(s) == 309);  // 256 message entries and 54 random ones: dimension 310, reduced rate 309 / 4096
    CHECK(std::fabs(radius.hi.value() - (1 - std::sqrt(309.0 / 4096))) < 1e-9 && a.at.delta.hi.value() > 0.7230 && a.at.delta.hi.value() < 0.7250);
    CHECK(surely_lt(radius, johnson(s, g.main, 100).delta));
    const double with = -fold_round_error(s, g, a, 1).hi.log2();
    Analysis without = a;
    without.mask_at.count = Bound((u64)0);
    CHECK(with >= 128 && with < 129 && -fold_round_error(s, g, without, 1).hi.log2() > with + 1);
    // The slack (radius minus distance) is sqrt(rate) / m: with m a quarter larger the stage
    // misses the target.
    CHECK(!group_holds(s, Scope::SHARED, group_of_m(g, g.m + g.m / 4 + 1), GroupLoad()));
  }
  Accounting acc[2];
  PartSize size[2][3];
  for (int compact = 0; compact < 2; ++compact) {
    const FilePlan key = file_key(z, compact != 0), entry = file_entry(z, compact != 0), proof = file_proof(z, sub, compact != 0);
    CHECK(key.chains.size() == 3 && entry.chains.size() == 1 && proof.chains.size() == 3);
    const Planned p = merged(s, {key, entry, proof});
    CHECK(p->holds());
    for (const char *file : {"key", "entry"}) {  // with the example file the committer's files rely on the theorem on codes, taken as given (index 1: `CountSource::JOHNSON`)
      const Accounting f = account(p, p->part_index(file));
      CHECK(f.closeness[0] == 0 && f.closeness[1] > 0);
    }
    CHECK(account(p, p->part_index("proof")).closeness[1] > 0);
    if (!compact)  // a `Cursor` goes through every chain's plan from its first round to its last
      for (const FilePlan *f : {&key, &entry, &proof})
        for (const ChainPlan &c : f->chains) {
          Cursor walk(c.plan);
          u64 n = 0;
          for (auto &r : c.plan->rounds_) {
            if (r.kind == RoundKind::RECORD) continue;
            CHECK(&walk.next_round(r.name.c_str()) == &r);
            ++n;
          }
          CHECK(walk.at_end() && n <= c.plan->queries());
        }
    acc[compact] = account(p);
    for (int i = 0; i < 3; ++i) size[compact][i] = p->size_of((unsigned)p->part_index(i == 0 ? "key" : i == 1 ? "entry" : "proof"));
    if (compact) continue;
    for (unsigned part = 0; part < p->parts.size(); ++part) {  // the fields are contiguous and cover the file
      u64 at = 0;
      for (auto &f : fields_of(p, part)) {
        CHECK(f.at == at && f.bytes > 0 && f.cls[0]);
        at += f.bytes;
      }
      CHECK(at == p->size_of(part).bytes);
    }
    for (auto &r : p->rounds_)  // a segment's list is no longer than what its rows' random coordinates were sized for
      if (r.name == R_LOC && r.where.find("link") == 0) CHECK(r.draw_positions <= d.link.t);
  }
  CHECK(acc[0].rounds == acc[1].rounds && acc[0].queries == acc[1].queries && std::fabs(acc[0].sum - acc[1].sum) < 1e-9 &&
        std::fabs(acc[0].sum_lists - acc[1].sum_lists) < 1e-9 && acc[0].planned_calls == acc[1].planned_calls);
  for (int i = 0; i < 3; ++i) CHECK(size[0][i].bytes == size[1][i].bytes);
  // A target that the file's other lines cannot reach is rejected, not planned.
  Parameters hard = par;
  hard.sec = 200;
  CHECK(!sized(hard, sub).refused.empty());
  // The table's distances are derived, up to the Johnson radius.
  CHECK(d.table.distances.m_later <= d.table.distances.m_once && d.table.distances.m_later >= 3);
  CHECK(merged(s, {file_key(z, true), file_entry(z, true), file_proof(z, sub, true)})->holds());
}

static void test_heads() {
  // proof file's header: tag, version word, leaf table's number; 4 bytes each
  Writer w;
  write_head(w, TAG_PROOF, 7);
  CHECK(w.size() == 12);
  {
    Reader r(w.data(), w.size());
    u32 entry = 0;
    CHECK(!read_head(r, TAG_PROOF, entry) && entry == 7 && r.at_end());
  }
  auto why = [&](std::vector<u8> bytes, u32 tag) {
    Reader r(bytes.data(), bytes.size());
    u32 entry = 0;
    const char *x = read_head(r, tag, entry);
    CHECK(!x || !r.ok());
    return std::string(x ? x : "");
  };
  std::vector<u8> good = w.bytes(), field = good, hash = good, older = good, cut(good.begin(), good.end() - 1);
  field[6] = (u8)('0' + 3 - BLK_FIELD), hash[7] = (u8)('0' + 3 - BLK_HASH), older[5] = 'G';
  CHECK(why(good, TAG_KEY) == "file-of-another-kind" && why(good, TAG_ENTRY) == "file-of-another-kind" && why(cut, TAG_PROOF) == "file-too-short");
  CHECK(why(field, TAG_PROOF) == "version-unknown" && why(hash, TAG_PROOF) == "version-unknown" && why(older, TAG_PROOF) == "version-unknown");
  const Digest p = parameter_file_digest(example()), q = parameter_file_digest(example() + " "), a = parameter_file_digest("a"), b = parameter_file_digest("b");
  CHECK(!(configuration_digest({"ab", "c"}) == configuration_digest({"a", "bc"})) && !(configuration_digest({"ab"}) == configuration_digest({"ab", ""})) &&
        !(configuration_digest({"ab"}) == parameter_file_digest("ab")));
  {  // the commitment ID (`published_id`) depends on configuration, name, name's length, seeds and answers
    const std::vector<u8> name(120, 7), other(120, 8), shorter(119, 7);
    const std::vector<E> none, one{E::one()}, two{E::one(), E::zero()};
    const std::vector<SampleSeed> no_seed, a_seed{SampleSeed{{E::one()}, E::one()}}, other_point{SampleSeed{{E::zero()}, E::one()}}, other_factor{SampleSeed{{E::one()}, E::one() + E::one()}};
    std::vector<Digest> ids{published_id(p, name, no_seed, none),    published_id(q, name, no_seed, none),     published_id(p, other, no_seed, none),
                            published_id(p, shorter, no_seed, none), published_id(p, name, no_seed, one),      published_id(p, name, no_seed, two),
                            published_id(p, name, a_seed, none),     published_id(p, name, other_point, none), published_id(p, name, other_factor, none)};
    for (size_t i = 0; i < ids.size(); ++i)
      for (size_t j = 0; j < i; ++j) CHECK(!(ids[i] == ids[j]));
    CHECK(published_id(p, name, no_seed, none) == ids[0] && !(ids[0] == configuration_digest({std::string(name.begin(), name.end())})));
  }
  {  /* Known answers: docs/format/format-vectors.txt, the output of
        docs/format/format_rules.py (Python's hashlib). The name follows `index_name`
        (blocks/iop/foldeq.h): root, configuration digest, six 64-bit shape words. */
    auto hex = [](const Digest &d) {
      std::string o;
      for (u32 w : d.w)
        for (int i = 0; i < 4; ++i) o += "0123456789abcdef"[(w >> (8 * i + 4)) & 15], o += "0123456789abcdef"[(w >> (8 * i)) & 15];
      return o;
    };
    const Digest configuration = configuration_digest({"toy\n"});
    CHECK(hex(configuration) == "e7b508817ead50bb3f386f68d442b114119e93e8ef2b14a7d70190075348db5f");
    std::vector<u8> name;
    for (unsigned i = 0; i < 32; ++i) name.push_back((u8)((13 * i + 4) % 251));
    for (u32 w : configuration.w)
      for (int i = 0; i < 4; ++i) name.push_back((u8)(w >> 8 * i));
    for (u64 word : {10, 512, 40, 9, 1, 40})
      for (int i = 0; i < 8; ++i) name.push_back((u8)(word >> 8 * i));
    CHECK(name.size() == 112);
    CHECK(hex(published_id(configuration, name, {}, {})) == "80d59c8b07aaa71829c9765057082a108618402daf661caf2c390c09dfafddd3");
    auto element = [](u64 a, u64 b, u64 c) {
      E x = E::zero();
      x.c[0] = from_u64(a), x.c[1] = from_u64(b), x.c[2] = from_u64(c);
      return x;
    };
    CHECK(hex(published_id(configuration, name, {SampleSeed{{element(7, 8, 9), element(10, 11, 12)}, element(13, 14, 15)}}, {element(1, 2, 3), element(4, 5, 6)})) == "4fb918a30faa7643cf8e6bc431c6643d7b5b4cc298dd0531cc0206bb4f756a0b");
  }
  // The first challenge depends on the label, the configuration digest and the statement.
  Plan one_round(setting_of_this_build());
  plan_line(one_round, "");
  const Planned plan = one_round.finish();
  std::vector<E> challenges;
  auto challenge = [&](u64 label, const Digest &configuration, const Digest &what) {
    Binder binder(label, plan);
    binder.configuration(configuration);
    binder.statement("what is proved", what);
    Writer proof;
    ProverRounds ch(binder, a, proof);
    challenges.push_back(ch.close_algebraic(R_LINE).draw());
    ch.finish();
  };
  challenge(LABEL_FIRST_CHAIN, p, a), challenge(LABEL_FIRST_CHAIN, p, b), challenge(LABEL_FIRST_CHAIN, q, a), challenge(LABEL_ONE_TIME_CHAIN, p, a);
  for (size_t i = 0; i < challenges.size(); ++i)
    for (size_t j = 0; j < i; ++j) CHECK(!(challenges[i] == challenges[j]));
}

int main() {
  test_heads();
  test_parser();
  test_refusals();
  test_derive_and_plans();
  if (failures) std::printf("%d of %d checks FAILED\n", failures, checks);
  else std::printf("ok, %d checks\n", checks);
  return failures ? 1 : 0;
}
