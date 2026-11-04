// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests three blocks of the protocol, prover against verifier, on the library's channel
// (`ProverRounds`, `VerifierRounds`) and trees: SEND sends oracle messages (rows sent as codewords
// under Merkle roots) and takes out-of-domain samples (random linear forms) of them; BAT batches
// linear claims into one by powers of a challenge, in both its forms (bat_prove(),
// bat_sigma_prove()); SPLIT splits a claim on two groups into one on each. A group is a set of
// oracle messages that one proximity test reads. The code and the printed lines call a linear claim
// a row (`Row`).
//
// whole_run() sends six oracle messages (`Config`) in two prover messages and forms two groups with
// different base codes. One claim is on the first group; two claims across the groups are batched
// into one, which is split. These blocks only pass claims on, so in place of a proximity test each
// group's claims are batched and checked by test_close_verify() (blocks/iop/tests/send_world.h):
// the prover sends its messages unmasked; the verifier evaluates the claim on them ("row") and
// compares their encodings with the committed codewords at drawn positions ("positions"). main()
// first runs layout_case(), merged_runs_case() and run_input_case().
//
// The program returns 0 only if the verifier accepts, both sides derive the same batched claims,
// and those hold on the prover's messages. Every run is honest, so nothing here shows that a false
// claim is rejected.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets a test write plans by hand (blocks/numbers/plan.h)
#include "iop/split.h"
#include "iop/tests/send_world.h"

using namespace blk;

namespace {

constexpr u64 POSITIONS = 7;

// `Code` (blocks/code/code.h): log2 of the length, message entries, random entries.
Codes the_codes() {
  Codes c;
  c.base = {Code{8, 8, 7}, Code{7, 8, 7}};
  c.mask = {Code{7, 4, 15}};
  return c;
}

OracleShape shape(const char *name, CodeId code, size_t main_rows, unsigned deg, size_t mask_rows, unsigned answers = 1) {
  OracleShape s;
  s.name = name, s.code = code, s.main_rows = main_rows, s.main_deg = deg, s.mask_rows = mask_rows, s.answers = answers;
  return s;
}

// The public inputs: each oracle message's shape and leaf class (only messages of one class can
// share a Merkle tree), the groups, the claims' points. The messages vary: with and without mask
// rows (rows of random values), over F and over E (the field, its cubic extension); o1 has three
// main rows (not a power of two); `both` has mask rows only and is in both groups: SPLIT takes its
// mask entry from it; `unsampled` gets no sample.
struct Config {
  unsigned answers;  // per sampled message
  std::vector<OracleNotice> first, second;
  std::vector<E> points;
  // A group: base code, then each member's number in sending order and whether the member is kept
  // (`GroupLayout`: its main rows are not folded with the others).
  struct Group {
    const char *name;
    CodeId code;
    std::vector<std::pair<unsigned, bool>> members;
  };
  const std::vector<Group> group = {{"first", 0, {{0, false}, {1, false}, {3, false}}},
                                    {"second", 1, {{1, false}, {4, false}, {5, true}}}};
  // For the first query (the hash input that fixes the public inputs): the number of oracle
  // messages in the first prover message, then the groups.
  std::vector<u64> numbers() const {
    std::vector<u64> n = {first.size()};
    for (const Group &g : group) {
      n.push_back(g.code), n.push_back(g.members.size());
      for (const auto &m : g.members) n.push_back(m.first), n.push_back(m.second);
    }
    return n;
  }
  explicit Config(unsigned s) : answers(s) {
    first = {{shape("o1", 0, 3, 1, 1, s), 0}, {shape("both", NONE_ID, 0, 1, 1, s), 2}, {shape("unsampled", NONE_ID, 0, 1, 1, 0), 3}};
    second = {{shape("o2", 0, 1, 3, 0, s), 0}, {shape("o3", 1, 2, 1, 2, s), 1}, {shape("o4", 1, 2, 1, 0, s), 1}};
    ToyCoins pub(5);
    points = random_point(64, pub);
  }
};
struct Ids {
  OracleId o1, both, unsampled, o2, o3, o4;
  GroupId g1, g2;
};

// Returns the three claims, values (`target`) unset: the prover sends them. It takes a mask entry
// from S, so each side calls it once, at the same step.
std::vector<Row> forms(Statement &S, const Ids &id, const std::vector<E> &points) {
  const E *z = points.data();
  std::vector<Row> rows(3);
  rows[0].name = "on the first group";
  rows[0].add(id.o1, Part::MAIN, Tensor::eq(z, S.table_bits(id.o1, Part::MAIN), z[20]));
  rows[0].add_mask(S.masks(id.o1).take(), z[21]);
  rows[0].add_run(id.o2, Part::MAIN, 2, 1, {z[22], z[23], z[24]});
  rows[0].add_unit(id.o2, Part::MAIN, 1, z[25]);
  rows[1].name = "across, one";
  rows[1].add(id.o1, Part::MAIN, Tensor::eq(z + 5, S.table_bits(id.o1, Part::MAIN)));
  rows[1].add(id.o3, Part::MAIN, Tensor::eq(z + 9, S.table_bits(id.o3, Part::MAIN)));
  rows[1].add_unit(id.o3, Part::MASK, 3, z[26]);
  rows[2].name = "across, two";
  // o2 and o4 have their pads in their main rows, and a claim must have coefficient 0 on every pad.
  // So these two tensors read the first half of o2's row and the first entry of each of o4's rows.
  Tensor on_o2 = Tensor::eq(z + 12, 2), on_o4 = Tensor::eq(z + 15, 1, z[27]);
  on_o2.insert_choice(2, 1, 0);
  on_o4.insert_choice(1, S.place_bits(id.o4, Part::MAIN), 0);
  rows[2].add(id.o2, Part::MAIN, on_o2);
  rows[2].add(id.o4, Part::MAIN, on_o4);
  rows[2].add_run(id.o3, Part::MASK, 0, 2, {z[28], z[29]});
  Form whole;  // one coefficient vector for a whole main table: on o1 and on o4, which is kept
  whole.add(Tensor::eq(z + 30, S.table_bits(id.o1, Part::MAIN), z[40]));
  whole.add_entry(4, z[41]);  // row 0, place 1
  rows[1].add_form(id.o1, whole);
  Form on_kept;
  on_kept.add(on_o4);
  on_kept.add_entry(0, z[42]);
  rows[2].add_form(id.o4, on_kept);
  rows[2].add_run(id.o4, Part::MAIN, 0, 2, {z[43], z[44], z[45]});  // along row 0 of the kept member
  return rows;
}

// Forms the two groups from Config::numbers() after its first entry, as fixed by the first query.
Ids groups(Statement &S, const std::vector<u64> &numbers, const std::vector<OracleId> &a, const std::vector<OracleId> &b) {
  Ids id{a[0], a[1], a[2], b[0], b[1], b[2], 0, 0};
  const OracleId sent[6] = {a[0], a[1], a[2], b[0], b[1], b[2]};
  GroupId made[2];
  size_t at = 1;
  for (size_t g = 0; g < 2; g++) {
    made[g] = S.new_group(g ? "second" : "first", (CodeId)numbers[at]);
    const u64 members = numbers[at + 1];
    at += 2;
    for (u64 m = 0; m < members; m++, at += 2) S.join(made[g], sent[numbers[at]], numbers[at + 1] != 0);
  }
  BLK_REQUIRE(at == numbers.size(), "test: the groups' numbers");
  id.g1 = made[0], id.g2 = made[1];
  return id;
}

std::vector<SampleSize> sizes(const Codes &codes, const std::vector<OracleNotice> &sent) {
  std::vector<SampleSize> s;
  for (const OracleNotice &n : sent)
    if (n.shape.answers) s.push_back(SampleSize{Send::cube(codes, n.shape).bits, no_sized_group()});
  return s;
}

u64 shown(const Codes &c, const std::vector<OracleShape> &members) {
  u64 n = 0;
  for (const OracleShape &s : members) {
    if (s.main_rows) n += s.main_rows * (c.base[s.code].ell + c.base[s.code].r);
    if (s.mask_rows) n += s.mask_rows * (c.mask[s.mask_code].ell + c.mask[s.mask_code].r);
  }
  return n;
}

void plan(Plan &p, const Codes &codes, const Config &cfg) {
  plan_send(p, "first message", sizes(codes, cfg.first), (unsigned)lay_trees(codes, cfg.first).shapes.size());
  plan_send(p, "second message", sizes(codes, cfg.second), (unsigned)lay_trees(codes, cfg.second).shapes.size());
  plan_scalars(p, "targets", "", 3);
  plan_bat_sigma(p, "", 2);
  plan_split(p, "");
  plan_bat(p, "first group", no_sized_group(), 2 + 3 * cfg.answers);   // its own row, a share, three samples
  plan_test_close(p, shown(codes, {cfg.first[0].shape, cfg.first[1].shape, cfg.second[0].shape}), POSITIONS, codes.base[0].log_n);
  plan_bat(p, "second group", no_sized_group(), 1 + 3 * cfg.answers);  // a share, three samples
  plan_test_close(p, shown(codes, {cfg.first[1].shape, cfg.second[1].shape, cfg.second[2].shape}), POSITIONS, codes.base[1].log_n);
}

OracleDraft draft(const Codes &codes, const OracleNotice &n, Engine &engine, Coins &coins) {
  OracleDraft d;
  d.shape = n.shape;
  d.leaf_class = n.leaf_class;
  if (n.shape.main_rows) {
    Table t = random_table(n.shape.main_deg, n.shape.main_rows, codes.base[n.shape.code].ell, coins);
    test_leave_pad_empty(codes, n.shape, t);
    d.main = engine.put(t);
  }
  if (n.shape.mask_rows) d.mask = Send::fresh_mask(codes, n.shape, coins);
  return d;
}

struct ProverSaw {
  Notes rows;             // both groups' batched claims, recorded as `decide` records them
  PositionList at1, at2;  // the positions at which the prover opened
  bool rows_true = true;
};

int prove(ProverContext &pr, PlantEngine &engine, const Config &cfg, const TestBound &x, ProverSaw &saw) {
  const Codes &codes = pr.st.codes();
  std::vector<OracleDraft> d1, d2;
  for (const OracleNotice &n : cfg.first) d1.push_back(draft(codes, n, engine, pr.coins));
  const std::vector<OracleId> a = send_prove(pr, std::move(d1));
  for (const OracleNotice &n : cfg.second) d2.push_back(draft(codes, n, engine, pr.coins));
  const std::vector<OracleId> b = send_prove(pr, std::move(d2));
  const Ids id = groups(pr.st, pr.ch.get(x.numbers), a, b);
  TEST_CHECK(!pr.st.oracle(id.unsampled).sampled && pr.st.oracle(id.both).sampled);

  std::vector<Row> rows = forms(pr.st, id, pr.ch.get(x.stated));
  for (Row &r : rows) r.target = row_value(pr, r);
  for (Row &r : rows) pr.ch.send(r.target);
  const Row across = bat_sigma_prove(pr, {rows[1], rows[2]}, "across");
  TEST_CHECK(row_value(pr, across) == across.target);
  split_prove(pr, across, id.g1, id.g2, id.both);
  pr.st.add_row(id.g1, rows[0]);

  for (GroupId g : {id.g1, id.g2}) {
    GroupLayout L;
    const GroupRow row = bat_prove(pr, g, &L);
    std::vector<Claimed> claims;
    for (OracleId o : L.members) claims.push_back(honest_claim(pr, engine, o));
    if (group_row_on_claims(pr.st, L, row, claims) != row.target) saw.rows_true = false;
    (g == id.g1 ? saw.at1 : saw.at2) = test_close_prove(pr, L, claims);
    note_group_row(&saw.rows, g == id.g1 ? "row 1" : "row 2", pr.st, L, row);
  }
  pr.ch.finish();
  pr.st.require_all_tested();
  return 0;
}

// The verifier, called by run_verifier() (blocks/chain/closed.h) with only the channel and handles
// to the public inputs. It records in `notes` what the test compares afterwards.
bool decide(VerifierRounds &ch, const TestBound &x, Notes *notes) {
  VerifierContext vf(x.codes, ch);
  const std::vector<OracleNotice> &sent = ch.get(x.sent);
  const std::vector<u64> &numbers = ch.get(x.numbers);
  const std::vector<OracleNotice> first(sent.begin(), sent.begin() + numbers[0]), second(sent.begin() + numbers[0], sent.end());
  std::vector<OracleId> a, b;
  if (!send_verify(vf, first, a)) return false;
  if (!send_verify(vf, second, b)) return false;
  const Ids id = groups(vf.st, numbers, a, b);
  std::vector<Row> rows = forms(vf.st, id, ch.get(x.stated));
  for (Row &r : rows)
    if (!vf.ch.recv(r.target)) return false;
  Row across;
  if (!bat_sigma_verify(vf, {rows[1], rows[2]}, "across", across)) return false;
  if (!split_verify(vf, across, id.g1, id.g2, id.both)) return false;
  vf.st.add_row(id.g1, rows[0]);
  bool yes = true;
  for (GroupId g : {id.g1, id.g2}) {
    GroupLayout L;
    GroupRow row;
    if (!bat_verify(vf, g, row, &L)) return false;
    note_group_row(notes, g == id.g1 ? "row 1" : "row 2", vf.st, L, row);
    const CloseVerdict verdict = test_close_verify(vf, L, row);
    note_verdict(notes, g == id.g1 ? "close 1" : "close 2", verdict);
    yes = yes && verdict.accept();  // goes on after a rejection: the test checks both verdicts
  }
  vf.st.require_all_tested();
  return yes;
}
std::vector<E> as_notes(const PositionList &at) {
  std::vector<E> v;
  for (size_t i = 0; i < at.size(); i++) v.push_back(E::from(from_u64(at[i])));
  return v;
}

// A group's layout: aligned blocks of rows by decreasing size, ties in sending order. Expected:
// tables of 2^k main and 2^h mask rows, k >= max(h, 1); each member's first row in either. Each
// case runs SEND.
int layout_case(const std::vector<size_t> &main_rows, const std::vector<size_t> &mask_rows, unsigned k, unsigned h,
                const std::vector<u64> &main_first, const std::vector<u64> &mask_first) {
  const Codes codes = the_codes();
  std::vector<OracleNotice> sent;
  for (size_t i = 0; i < main_rows.size(); i++) sent.push_back({shape("o", main_rows[i] ? 0 : NONE_ID, main_rows[i], 1, mask_rows[i]), 0});
  Plan p(test_setting());
  plan_send(p, "", sizes(codes, sent), (unsigned)lay_trees(codes, sent).shapes.size());
  ToyCoins coins(3);
  PlantEngine engine;
  Writer proof;
  const TestBinder binder(p, codes, sent);
  ProverRounds ch(binder.b, test_bytes(coins), proof);
  ProverContext pr(binder.codes, ch, engine, coins);
  std::vector<OracleDraft> drafts;
  for (const OracleNotice &n : sent) drafts.push_back(draft(codes, n, engine, coins));
  const std::vector<OracleId> ids = send_prove(pr, std::move(drafts));
  const GroupId g = pr.st.new_group("g", 0);
  for (size_t i = ids.size(); i-- > 0;) pr.st.join(g, ids[i]);  // the order of joining does not matter
  const GroupLayout L = pr.st.layout(g);
  TEST_CHECK(L.members == ids);
  TEST_CHECK(L.k == k && L.h == h);
  TEST_CHECK(L.main_first == main_first && L.mask_first == mask_first);
  // `MaskLedger` gives out each mask entry but the pad once, in increasing order.
  for (size_t i = 0; i < ids.size(); i++) {
    if (!mask_rows[i]) continue;
    const u64 pad = Send::pad(codes, sent[i].shape)[0].index;
    MaskLedger &ledger = pr.st.masks(ids[i]);
    TEST_CHECK(ledger.left() == mask_rows[i] * codes.mask[0].ell - 1);
    u64 last = 0;
    bool first = true;
    while (ledger.left()) {
      const MaskSlot s = ledger.take();
      TEST_CHECK(s.oracle() == ids[i] && s.index() != pad && s.index() < pad);
      TEST_CHECK((s.index() & ((u64(1) << ceil_log2(mask_rows[i])) - 1)) < mask_rows[i]);
      TEST_CHECK(first || s.index() > last);
      last = s.index(), first = false;
    }
  }
  return 0;
}

// Bat::combine() adds runs of coefficients (`Run`) on the same entries of one table into one run,
// entry by entry; a run with another length, start or table stays separate.
int merged_runs_case() {
  ToyCoins coins(9);
  auto some = [&](size_t n) {
    std::vector<E> v(n);
    for (E &x : v) x = coins.e();
    return v;
  };
  std::vector<Row> rows(4);
  std::vector<std::vector<E>> on(4);
  for (size_t i = 0; i < 4; i++) rows[i].add_run(7, Part::MASK, 3, 2, on[i] = some(5)), rows[i].target = coins.e();
  const std::vector<E> shorter = some(4), elsewhere = some(5), other_part = some(5);
  rows[1].add_run(7, Part::MASK, 3, 2, shorter);
  rows[2].add_run(7, Part::MASK, 4, 2, elsewhere);
  rows[3].add_run(7, Part::MAIN, 3, 2, other_part);
  const E x = coins.e();
  const Row sum = Bat::combine(rows, x, "sum");
  TEST_CHECK(sum.runs.size() == 4);
  TEST_CHECK(sum.runs[0].size() == 5 && sum.runs[0].first == 3 && sum.runs[0].step == 2 && sum.runs[0].part == Part::MASK);
  for (size_t j = 0; j < 5; j++) TEST_CHECK(sum.runs[0].at(j) == on[0][j] + x * (on[1][j] + x * (on[2][j] + x * on[3][j])));
  for (size_t j = 0; j < 4; j++) TEST_CHECK(sum.runs[1].at(j) == x * shorter[j]);
  for (size_t j = 0; j < 5; j++) TEST_CHECK(sum.runs[2].at(j) == x * x * elsewhere[j] && sum.runs[3].at(j) == x * x * x * other_part[j]);
  TEST_CHECK(sum.target == rows[0].target + x * (rows[1].target + x * (rows[2].target + x * rows[3].target)));
  // A run that reads a range of a shared vector over F acts like a run that owns these values: when
  // read, when batched, and when added into a claim that has such a run. The vector is not changed.
  auto tail = std::make_shared<std::vector<F>>(9);
  coins.fill(tail->data(), tail->size());
  const std::vector<F> before = *tail;
  std::vector<Row> shared(3), own(3);
  for (size_t i = 0; i < 3; i++) {
    std::vector<E> lifted;
    for (size_t j = 0; j < 5; j++) lifted.push_back(E::from((*tail)[i + j]));
    shared[i].add_run(7, Part::MASK, 3, 2, tail, i, 5), own[i].add_run(7, Part::MASK, 3, 2, lifted);
    for (size_t j = 0; j < 5; j++) TEST_CHECK(shared[i].runs[0].at(j) == lifted[j] && shared[i].coefficient(7, Part::MASK, 3 + 2 * j) == lifted[j]);
    TEST_CHECK(shared[i].runs[0].size() == 5 && is_zero(shared[i].coefficient(7, Part::MASK, 4)) && is_zero(shared[i].coefficient(7, Part::MASK, 13)));
  }
  const Row of_shared = Bat::combine(shared, x, "sum"), of_own = Bat::combine(own, x, "sum");
  TEST_CHECK(of_shared.runs.size() == 1 && of_own.runs.size() == 1);
  for (size_t j = 0; j < 5; j++) TEST_CHECK(of_shared.runs[0].at(j) == of_own.runs[0].at(j));
  Row onto = shared[0];
  onto.add_scaled(shared[1], x);
  TEST_CHECK(onto.runs.size() == 1);
  for (size_t j = 0; j < 5; j++) TEST_CHECK(onto.runs[0].at(j) == E::from(before[j]) + x * before[1 + j]);
  TEST_CHECK(*tail == before);
  return 0;
}

// What begin_run() passes to a proximity test, in order: the plain claims as created; every
// member's sample claims, all answers, by oracle message in sending order; the generator claims
// (`RowKind::GENERATOR`: a codeword's symbol at an opened position) as created.
int run_input_case() {
  const Codes codes = the_codes();
  const std::vector<OracleNotice> sent = {{shape("a", 0, 2, 1, 1, 2), 0}, {shape("b", NONE_ID, 0, 1, 2, 2), 0}, {shape("c", 0, 1, 3, 0, 2), 0}};
  Setting set = test_setting();
  set.samples = 2;
  Plan p(set);
  plan_send(p, "", sizes(codes, sent), (unsigned)lay_trees(codes, sent).shapes.size());
  ToyCoins coins(4);
  PlantEngine engine;
  Writer proof;
  const TestBinder binder(p, codes, sent);
  ProverRounds ch(binder.b, test_bytes(coins), proof);
  ProverContext pr(binder.codes, ch, engine, coins);
  std::vector<OracleDraft> drafts;
  for (const OracleNotice &n : sent) drafts.push_back(draft(codes, n, engine, coins));
  const std::vector<OracleId> ids = send_prove(pr, std::move(drafts));
  const GroupId g = pr.st.new_group("g", 0);
  pr.st.join(g, ids[2], true);
  pr.st.join(g, ids[0]);
  pr.st.join(g, ids[1]);
  for (const char *name : {"g0", "p0", "g1", "p1"}) {
    Row r;
    r.name = name;
    r.kind = name[0] == 'g' ? RowKind::GENERATOR : RowKind::PLAIN;
    r.add_unit(ids[0], Part::MAIN, 0, E::one());
    // Runs on the same entries in every claim: on main rows that are not kept, on mask rows and on
    // kept main rows. g1 and p1 get one further run.
    for (const auto &where : {std::make_pair(ids[0], Part::MAIN), std::make_pair(ids[1], Part::MASK), std::make_pair(ids[2], Part::MAIN)})
      r.add_run(where.first, where.second, 0, 2, {coins.e(), coins.e(), coins.e()});  // low places: clear of every pad
    if (name[1] == '1') r.add_run(ids[0], Part::MAIN, 2, 1, {coins.e(), coins.e()});
    pr.st.add_row(g, r);
  }
  std::vector<E> answers;  // as SEND has left them with the oracle messages
  for (OracleId o : ids)
    for (const Row &r : pr.st.oracle(o).sample_rows) answers.push_back(r.target);
  const RunInput in = pr.st.begin_run(g);
  const std::vector<std::string> names = {"p0", "p1", "sample of a", "sample of a", "sample of b", "sample of b", "sample of c", "sample of c", "g0", "g1"};
  TEST_CHECK(in.rows().size() == names.size() && answers.size() == 6);
  for (size_t i = 0; i < names.size(); i++) {
    TEST_CHECK(in.rows()[i].name == names[i]);
    TEST_CHECK(in.rows()[i].kind == (i < 2 ? RowKind::PLAIN : i < 8 ? RowKind::SAMPLE : RowKind::GENERATOR));
    if (i >= 2 && i < 8) TEST_CHECK(in.rows()[i].target == answers[i - 2]);
  }
  for (size_t i = 2; i < 8; i += 2) TEST_CHECK(in.rows()[i].target != in.rows()[i + 1].target);  // two seeds, two answers

  // On a group, Bat::combine() adds up the claims' runs before it maps them to the group's tables.
  // The result must equal the sum taken claim by claim.
  const E x = coins.e();
  const GroupRow fast = Bat::combine(pr.st, g, in, x);
  GroupRow slow;
  E power = E::one();
  for (const Row &r : in.rows()) add_to_group_row(pr.st, in.layout, r, power, slow), power *= x;
  auto written = [](const Form &c, unsigned vars) {
    std::vector<E> v(size_t(1) << vars, E::zero());
    c.add_to(v.data(), vars);
    return v;
  };
  const unsigned main_bits = in.layout.k + in.layout.log_ell;
  TEST_CHECK(written(fast.c, main_bits) == written(slow.c, main_bits) && fast.d == slow.d && fast.target == slow.target);
  TEST_CHECK(fast.kept.size() == 1 && slow.kept.size() == 1 && fast.kept[0].oracle == ids[2]);
  const unsigned kept_bits = pr.st.table_bits(ids[2], Part::MAIN);
  TEST_CHECK(written(fast.kept[0].c, kept_bits) == written(slow.kept[0].c, kept_bits));
  TEST_CHECK(fast.c.entries.size() < slow.c.entries.size());  // one entry per index, claims merged
  return 0;
}

int whole_run(unsigned answers) {
  const Codes codes = the_codes();
  const Config cfg(answers);
  Setting set = test_setting();
  set.samples = answers;
  Plan p(set);
  plan(p, codes, cfg);

  ToyCoins coins(7);
  PlantEngine engine;
  Writer proof;
  TestInputs in{codes, cfg.first, cfg.numbers(), cfg.points};
  in.messages.insert(in.messages.end(), cfg.second.begin(), cfg.second.end());
  const TestBinder binder(p, in);
  ProverRounds pch(binder.b, test_bytes(coins), proof);
  ProverContext pr(binder.codes, pch, engine, coins);
  ProverSaw ps;
  if (prove(pr, engine, cfg, binder.x, ps)) return 1;

  Notes vs;
  const bool accepted = run_verifier(TEST_LABEL, p, in, test_bind, decide, proof.data(), proof.size(), nullptr, &vs);
  const bool same_rows = noted(ps.rows, "row 1") == noted(vs, "row 1") && noted(ps.rows, "row 2") == noted(vs, "row 2") && !noted(vs, "row 2").empty();
  auto yes = [&vs](const char *what) { return noted_yes(vs, what); };

  // Both sides derive the same batched claims and draw the same positions.
  TEST_CHECK(same_rows);
  TEST_CHECK(noted(vs, "close 1: drawn") == as_notes(ps.at1) && noted(vs, "close 2: drawn") == as_notes(ps.at2));
  TEST_CHECK(yes("close 1: read") && yes("close 2: read"));
  TEST_CHECK(accepted && ps.rows_true);
  std::printf("ok: accepted with %u answer(s) to a sample; proof of %zu bytes\n", answers, proof.size());
  return 0;
}

}  // namespace

int main() {
  const u64 none = GroupLayout::NONE;
  if (layout_case({1}, {0}, 1, 0, {0}, {none})) return 1;
  if (layout_case({1, 3, 2, 0}, {1, 0, 2, 1}, 3, 2, {6, 0, 4, none}, {2, none, 0, 3})) return 1;
  if (layout_case({5, 1}, {1, 1}, 4, 1, {0, 8}, {0, 1})) return 1;
  if (layout_case({0, 0}, {3, 1}, 3, 3, {none, none}, {0, 4})) return 1;
  if (merged_runs_case()) return 1;
  if (run_input_case()) return 1;
  if (whole_run(1)) return 1;
  if (whole_run(2)) return 1;
  return 0;
}
