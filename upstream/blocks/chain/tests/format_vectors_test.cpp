// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compares the library's hash, hash chain (`Chain`: a protocol's Fiat-Shamir transcript), channel
// (`ProverRounds`, `VerifierRounds`: one side's sending, receiving and challenges) and Merkle trees
// with stored known answers. docs/format/FORMAT.md specifies, byte for byte, what is hashed
// and what is written to files; rule numbers below are that document's.
// docs/format/format-vectors.txt is the output of docs/format/format_rules.py, a Python
// implementation of those rules. This test computes its cases with the library's functions and
// compares. It skips field 1's lines: the library has only field 2, Goldilocks (R1.2; `BLK_FIELD`).
// Lines that it does not compute are no failure: it counts them. Run it from blocks/, or give the
// path of format-vectors.txt as the first argument. Exit code: 1 after a failure, otherwise 0.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // this test builds its plans by hand (blocks/numbers/plan.h)
#include "chain/rounds.h"
#include "chain/tests/known_answers.h"
#include "tree/opening.h"

// Byte i of leaf x's 40-byte salt is (3 i + 9 + x) mod 251. bytes() relies on `OracleTree` drawing
// all salts in one call.
struct PageSalts : Coins {
  void fill(F *, size_t) override { BLK_REQUIRE(false, "no element is asked for"); }
  void bytes(u8 *out, size_t n) override {
    for (size_t i = 0; i < n; i++) out[i] = (u8)((3 * (i % 40) + 9 + i / 40) % 251);
  }
};

// For the cases on failing words (R6.4) below. `Given` is `Blake2bF`, but the words from which a
// round's verifier's message is drawn are `given_row`, then 0x11 bytes. It fits only those cases'
// round: a prover's message of three words and no proof of work, so one block per hash call.
static std::vector<u64> given_row;
struct Given : Blake2bF {
  static void compress(u32 cv[CVW], const u32 block[BLKW], const Tweak &t) {
    Blake2bF::compress(cv, block, t);
    if ((block[0] & 0xFFFF) != (u32)Domain::CHAIN_QUERY) return; // higher bits: the chain's label
    const u64 j = block[10] >> 16 | (u64)block[11] << 16;
    for (int w = j == 0 ? 4 : 0; w < 8; w++) {
      const size_t i = (size_t)(8 * j + w) - 4;
      const u64 word = i < given_row.size() ? given_row[i] : 0x1111111111111111ull;
      cv[2 * w] = (u32)word, cv[2 * w + 1] = (u32)(word >> 32);
    }
  }
};

// From known_answers.h: expect(name, value) compares `value` with the stored line of that name;
// byte i of pattern(n, a, b) is (a i + b) mod 251, by default (7 i + 3) mod 251.
int main(int argc, char **argv) {
  if (int status = read_known_answers(argc, argv)) return status;

  // ---- rules R2.x, R3.x: the hash ----
  expect("H(168, 0, 0, none; no byte)", hex(H168(0, {})));
  expect("H(168, 0, 5, none; pattern(128))", hex(H168(5, pattern(128))));
  expect("H(168, 0, 2^33 + 5, none; pattern(129))", hex(H168(((u64)1 << 33) + 5, pattern(129))));
  {
    Hasher h(Tag(Domain::LEAF_MAIN, 9, digest_of(pattern(32)), 3, 40), 12); // three symbols and a salt of 40 bytes are 64 bytes
    h.push(pattern(64).data(), 64);
    expect("H(164, 9, 12, start = pattern(32); pattern(64)), W = 3, s8 = 5", hex(h.finish()));
  }
  {
    // One hash call (R5.2) from each of two rounds' queries, made directly with chain_call().
    auto call = [](u64 label, const std::vector<u8> &message, ChainKind kind, unsigned work, u64 nonce, u64 j) {
      ChainAbsorb a(digest_of(pattern(32)), 3, label);
      a.push(message.data(), message.size());
      const ChainCall c = chain_call(Chain::prefix(a, kind, work), nonce, j);
      Writer w;
      for (u64 word : c.w) w.put_u64(word);
      return hex(w.bytes());
    };
    const std::string plain = "H(162, 0, 3, start = pattern(32); no message, kind 1, w = 0, j = 0, nonce 0), 64 bytes";
    const std::string full = "H(162, 2^32, 3, start = pattern(32); the message pattern(8), kind 4, w = 8, j = 1, nonce 5), 64 bytes";
    expect(plain, call(0, {}, ChainKind::ALGEBRAIC, 0, 0, 0));
    expect(full, call((u64)1 << 32, pattern(8), ChainKind::POSITIONS, 8, 5, 1));
  }
  expect("a vertex of the binding tree: H(178, 0, 0, start = pattern(32); pattern(64))",
         hex(vertex(Tag(Domain::BINDING_NODE, 0, digest_of(pattern(32))), digest_of(pattern(64)), digest_of(pattern(64), 32))));
  expect("a vertex: H(166, 0, 0, start = pattern(32); pattern(64))", hex(vertex(digest_of(pattern(64)), digest_of(pattern(64), 32), digest_of(pattern(32)))));

  // ---- rules R1.x: a field element as eight bytes ----
  for (u64 w : std::vector<u64>{0, P - 1, P, ~(u64)0}) {
    u8 b[8];
    for (int i = 0; i < 8; i++) b[i] = (u8)(w >> (8 * i));
    const std::string name = "field " + std::to_string(BLK_FIELD) + ", the bytes " + hex(b, 8);
    Reader r(b, 8);
    F f;
    expect(name, r.get(f) ? std::to_string(f.v) : "-");
  }

  // ---- rules R4.x: the first query, built by a `Binder` ----
  // The first query is the hash input that fixes a chain's label, parameters, statement and random
  // bytes; its hash is rho_0, the chain's first value (R4.6).
  // The configuration digest of one public file (R4.9): the number of files, its length, its bytes.
  Digest configuration;
  {
    Hasher h(Tag(Domain::PARAMETERS), 2);
    h.push_u64(1), h.push_u64(4), h.push((const u8 *)"toy\n", 4);
    configuration = h.finish();
    expect("the configuration's digest of one file, the four bytes 'toy' and a line feed", hex(configuration));
  }
  // A toy plan of six rounds. A plan lists a chain's rounds and is part of the parameters (R4.3).
  Setting set = setting_goldilocks();
  set.salt_bytes = 40, set.nonce_bytes = 8, set.position_bytes = 0, set.samples = 1, set.saved_coefficient = true;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  Plan plan(set);
  plan.send_scalars("two elements", 2);
  plan.close("toy: challenge", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = 1;
  plan.send_scalars("one element", 1);
  plan.close_none("toy: nothing");
  plan.send_root("a root");
  {
    PlanRound &r = plan.close("toy: positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    r.draw_E = 1, r.draw_positions = 5, r.log_n = 10, r.rows = 3;
  }
  {
    PlanRound &r = plan.close("toy: seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING);
    r.draw_E = 3, r.draw_nonzero = 1;
  }
  plan.send_answers("an answer", 1);
  plan.close("toy: matrix", RoundKind::BITS, ErrorKind::SMALL, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_trits = 40;
  plan.close("toy: seal", RoundKind::SEAL, ErrorKind::ZERO, Bound((u64)0), Priced::NOTHING).message_bytes = 8; // one word
  auto bind = [&](Binder &b) {
    b.configuration(configuration);
    b.parameter("toy: a table's length", (u64)1024);
    b.statement("toy: claim", element(1, 2, 3));
    b.statement("toy: name", std::string("abc"));
  };
  Binder prover_side(7, plan), verifier_side(7, plan), for_the_digest(7, plan);
  bind(prover_side), bind(verifier_side), bind(for_the_digest);
  expect("the parameters' bytes", hex(for_the_digest.parameter_bytes()));
  const Digest random_bytes = digest_of(pattern(32, 5, 1));
  // A `Binder` whose parameters' digest is not the known answer counts as one failure. The bare
  // `Chain` is then built on the known digest, and the channel is not compared with that chain.
  Digest parameters = for_the_digest.parameters_digest();
  bool binder_agrees = true;
  if (known.count("the parameters' digest") && known["the parameters' digest"] != hex(parameters)) {
    const std::string page = known["the parameters' digest"];
    std::vector<u8> b(DIGEST_BYTES);
    for (size_t i = 0; i < DIGEST_BYTES; i++) b[i] = (u8)std::stoul(page.substr(2 * i, 2), nullptr, 16);
    expect("the parameters' digest", hex(parameters));
    parameters = digest_of(b), binder_agrees = false;
  } else
    expect("the parameters' digest", hex(parameters));
  expect("the run's value (label 7, random bytes pattern(32) with 5 i + 1)", hex(run_value(7, parameters, random_bytes)));
  // The statement's bytes by hand (R4.1). An item: name (length, bytes), value's length, value.
  Writer statement;
  auto item = [&](const std::string &name, const std::vector<u8> &value) {
    statement.put_u64(name.size()), statement.put_bytes((const u8 *)name.data(), name.size());
    statement.put_u64(value.size()), statement.put_bytes(value.data(), value.size());
  };
  {
    Writer claim, name;
    claim.put(element(1, 2, 3));
    name.put_u64(3), name.put_bytes((const u8 *)"abc", 3);
    item("toy: claim", claim.bytes()), item("toy: name", name.bytes());
  }
  expect("the statement's bytes", hex(statement.bytes()));
  FirstQuery q;
  q.label = 7, q.parameters = parameters, q.statement = statement.bytes(), q.salt = random_bytes;
  Chain c(q);
  expect("rho_0", hex(c.value()));
  {
    FirstQuery bare = q;
    bare.salted = false;
    expect("rho_0 of the same chain without random bytes", hex(Chain(bare).value()));
  }

  // ---- rules R5.x, R6.x: six rounds on a bare `Chain` and through a channel (`ProverRounds`) ----
  const Digest root = digest_of(pattern(32, 11, 2));
  Writer proof;
  ProverRounds ch(prover_side, random_bytes, proof);
  auto same = [&](const E &from_the_channel, const E &from_the_chain) {
    if (binder_agrees && from_the_channel != from_the_chain) failures++, std::printf("FAILED: the channel draws another element than the chain by itself\n");
    return text(from_the_chain);
  };
  {
    ChainAbsorb m = c.message();
    m.push(element(4, 5, 6)), m.push(element(7, 8, 9));
    c.close(m, ChainKind::ALGEBRAIC);
    expect("round 1, kind 1, message (4, 5, 6), (7, 8, 9): rho_1", hex(c.value()));
    ch.send(element(4, 5, 6)), ch.send(element(7, 8, 9));
    expect("round 1: the element drawn", same(ch.close_algebraic("toy: challenge").draw(), c.draw_E()));
  }
  {
    ChainAbsorb m = c.message();
    m.push(element(1, 1, 1));
    c.close(m, ChainKind::EMPTY);
    expect("round 2, kind 7, message (1, 1, 1): rho_2", hex(c.value()));
    ch.send(element(1, 1, 1));
    ch.close_empty("toy: nothing");
  }
  {
    ChainAbsorb m = c.message();
    m.push(root);
    const ChainPrefix p = Chain::prefix(m, ChainKind::POSITIONS, 8);
    const u64 nonce = search_nonce(p, 8);
    expect("round 3, kind 4, message one root, pattern(32) with 11 i + 2, 8 bits of work: the least nonce", std::to_string(nonce));
    expect("round 3: block 0, the test block", hex(chain_block(chain_call(p, nonce, 0), 0)));
    if (!c.step(p, nonce, 8)) failures++, std::printf("FAILED: the least nonce has not the work\n");
    expect("round 3: rho_3", hex(c.value()));
    ch.send_root(root);
    PositionsDraw d = ch.close_positions("toy: positions");
    d.require_rows(3);
    const PositionList list = d.positions(10);
    std::string drawn;
    for (u64 x : list.draws()) {
      const u64 by_itself = c.draw_position(10);
      if (binder_agrees && x != by_itself) failures++, std::printf("FAILED: the channel draws another position than the chain by itself\n");
      drawn += (drawn.empty() ? "" : " ") + std::to_string(by_itself);
    }
    expect("round 3: five positions in [0, 2^10), as drawn", drawn);
    expect("round 3: the element behind them", same(d.batching(), c.draw_E()));
  }
  {
    c.round(ChainKind::SEED);
    expect("round 4, kind 3, empty message: rho_4", hex(c.value()));
    const SampleSeed seed = ch.close_seed("toy: seed").seed(2, 1);
    const std::string z0 = same(seed.z[0], c.draw_E()), z1 = same(seed.z[1], c.draw_E());
    expect("round 4: a seed with a point of two coordinates", z0 + "; " + z1 + "; " + same(seed.beta, c.draw_E_nonzero()));
  }
  {
    ChainAbsorb m = c.message();
    m.push(element(9, 9, 9));
    c.close(m, ChainKind::BITS);
    expect("round 5, kind 5, message the answer (9, 9, 9): rho_5", hex(c.value()));
    ch.send_answer(element(9, 9, 9));
    // `entries`: decoded from the chain's words (R6.7), 32 per word, two bits each, low bits first.
    std::string entries, from_the_channel;
    for (signed char t : ch.close_bits("toy: matrix").trits(40)) from_the_channel += (from_the_channel.empty() ? "" : " ") + std::to_string((int)t);
    for (int i = 0; i < 40;) {
      const u64 word = c.next_word();
      for (int k = 0; k < 32 && i < 40; k++, i++) {
        const unsigned two = (unsigned)(word >> (2 * k)) & 3;
        entries += (entries.empty() ? "" : " ") + std::to_string(two == 1 ? 1 : two == 3 ? -1 : 0);
      }
    }
    if (binder_agrees && entries != from_the_channel) failures++, std::printf("FAILED: the channel draws other entries than the chain by itself\n");
    expect("round 5: forty entries", entries);
  }
  {
    const u64 word[1] = {77};
    c.round(word, ChainKind::SEAL);
    expect("round 6, kind 6, message the word 77: rho_6", hex(c.value()));
    ch.send_u64(77);
    Sealed sealed;
    if (!ch.seal("toy: seal", sealed)) failures++, std::printf("FAILED: a prover's seal\n");
    Digest by_itself;
    for (int i = 0; i < 4; i++) {
      const u64 w = c.next_word();
      by_itself.w[2 * i] = (u32)w, by_itself.w[2 * i + 1] = (u32)(w >> 32);
    }
    if (binder_agrees && by_itself != sealed.digest()) failures++, std::printf("FAILED: the channel's seal is not the chain's\n");
    expect("round 6: the seal", hex(by_itself));
    ch.finish();
  }
  {
    // `VerifierRounds` must accept `proof`: random bytes, items as sent, the nonce after the root.
    Reader r(proof.data(), proof.size());
    VerifierSession v_session;
    VerifierRounds v(verifier_side, r, v_session);
    E x;
    SentRoot d;
    u64 w = 0;
    bool good = v.recv(x) && v.recv(x);
    v.close_algebraic("toy: challenge").draw();
    good = good && v.recv(x);
    v.close_empty("toy: nothing");
    good = good && v.recv_root(d);
    {
      PositionsDraw p = v.close_positions("toy: positions");
      p.require_rows(3); // required on both sides, because the plan entry (`PlanRound`) sets `rows`
      p.positions(10), p.batching();
    }
    v.close_seed("toy: seed").seed(2, 1);
    good = good && v.recv_answer(x);
    v.close_bits("toy: matrix").trits(40);
    good = good && v.recv_u64(w);
    Sealed sealed;
    good = good && v.seal("toy: seal", sealed) && v.finish();
    if (!good) failures++, std::printf("FAILED: the verifier's channel refuses the prover's file: %s\n", v.why() ? v.why() : "?");
  }
  {
    // A committed table's name (R4.10): the root, the configuration digest, six words on its shape.
    Writer name;
    name.put(digest_of(pattern(32, 13, 4))), name.put(configuration);
    for (u64 word : {10, 3, 40, 9, 1, 40}) name.put_u64(word);
    const SampleSeed seed = record_seeds(name.data(), name.size(), 2, 1)[0];
    expect("record seed of the name with root pattern(32) with 13 i + 4, the configuration of one file 'toy' and a line feed, a = 10, 3 entries, tail 40, 2^9 rows, 1 answer, salt 40; two "
           "coordinates: the point",
           text(seed.z[0]) + "; " + text(seed.z[1]));
    expect("the same: the factor", text(seed.beta));
  }

  // ---- rules R6.x: rejected words, the cap on tries, the largest number of words ----
  const std::string field = "field " + std::to_string(BLK_FIELD);
  for (u64 w : std::vector<u64>{0, P - 1, P, ~(u64)0}) {
    const std::string name = field + ", word " + std::to_string(w);
    F f;
    expect(name, F_from_word(w, f) ? std::to_string(f.v) : "-");
  }
  // A word not below `P` fails (R6.3). draw_F() takes the first word that does not fail, or gives
  // 0 after DRAW_TRIES failures (R6.4).
  for (size_t fails : {0, 1, 5, 6, 7}) {
    given_row.assign(fails, ~(u64)0);
    given_row.push_back(5);
    ChainT<Given> g(q);
    ChainAbsorbT<Given> m = g.message();
    m.push(E::one());
    g.close(m, ChainKind::ALGEBRAIC);
    const F f = g.draw_F();
    size_t used = 0; // words `draw_F` used (at most DRAW_TRIES = 6), inferred from the next word
    const u64 next = g.next_word();
    for (used = 0; used < given_row.size() && !(used >= (fails < 6 ? fails + 1 : 6) && given_row[used] == next); used++) {
    }
    if (next == 0x1111111111111111ull) used = given_row.size();
    expect(field + ", " + std::to_string(fails) + " words that fail, then the word 5", std::to_string(f.v) + "; " + std::to_string(used));
  }
  {
    auto most = [](u64 e, u64 nonzero, u64 positions, u64 trits, bool seal) {
      PlanRound r;
      r.draw_E = e, r.draw_nonzero = nonzero, r.draw_positions = positions, r.draw_trits = trits, r.kind = seal ? RoundKind::SEAL : RoundKind::ALGEBRAIC;
      return std::to_string(most_words(r));
    };
    expect("one element of E; one that is not 0; a seed with two coordinates; five positions and an element; forty entries; a seal",
           most(1, 0, 0, 0, false) + "; " + most(1, 1, 0, 0, false) + "; " + most(3, 1, 0, 0, false) + "; " + most(1, 0, 5, 0, false) + "; " + most(0, 0, 0, 40, false) + "; " +
               most(0, 0, 0, 0, true));
  }

  // ---- rules R7.x: leaves, a tree, its openings ----
  const std::vector<u8> salt = pattern(40, 3, 9);
  OracleTreeShape s;
  s.log_n = 3, s.tree_number = 5, s.salt_bytes = 40, s.run = s.parameters = digest_of(pattern(32, 5, 1));
  for (size_t count : {3, 9, 10, 11, 14, 16, 25, 26}) {
    std::vector<F> column;
    for (size_t i = 1; i <= count; i++) column.push_back(F{i});
    s.main_symbols = {count}, s.mask_symbols = {0};
    const std::string name = "column 1 .. " + std::to_string(count) + ": bytes; the byte at which the salt begins";
    expect(name, std::to_string(leaf_bytes(count, 40) - LABEL_BYTES) + "; " + std::to_string(leaf_salt_at(count, 40) - LABEL_BYTES)); // a leaf's bytes (R7.3) begin after the label
    expect(name + ": its leaf under 164: tree 5, a = 3, position 6", hex(oracle_leaf(s, 6, column.data(), count, salt.data())));
    if (count == 3) {
      expect("a binding leaf: depth 2, place 3, that start as the configuration's digest, seal pattern(32), that salt",
             hex(leaf(Tag(Domain::LEAF_BINDING, 0, s.parameters, 4, 40), ((u64)1 << 2) + 3, digest_of(pattern(32)), salt.data(), 40)));
      OracleTreeShape kept = s;
      kept.stored = true, kept.tree_number = 0; // a stored table's tree has the number 0
      expect("the leaf of the column 1 .. 3 under 174, tree 0 (R7.4), the same position, with that start in the place of the configuration's digest",
             hex(oracle_leaf(kept, 6, column.data(), count, salt.data())));
    }
  }
  {
    s.log_n_zk = 1, s.main_symbols = {2}, s.mask_symbols = {1}, s.parameters = configuration;
    auto column = [&](u64 x, std::vector<F> &out) { out = {F{100 + x}, F{200 + x}, F{300 + short_position(3, 1, x)}}; };
    PageSalts salts;
    OracleTree tree(s, salts);
    std::vector<F> col;
    for (u64 x = 0; x < 8; x++) column(x, col), tree.set_leaf(x, col.data(), col.size());
    const Digest top = tree.finish();
    expect("column of x: 100 + x, 200 + x, 300 + (x mod 2); salt pattern(40) with 3 i + 9 + x: the root", hex(top));
    const std::vector<std::pair<std::string, std::vector<u64>>> cases = {{"[6, 1, 6, 3]", {6, 1, 6, 3}}, {"[5]", {5}}, {"[0, 4, 2, 6]", {0, 4, 2, 6}}, {"[2, 3]", {2, 3}}};
    for (const auto &drawn : cases) {
      const PositionList list = PositionList::from_draws(drawn.second, 3);
      struct Order {
        struct Value {};
        std::string said;
        Value leaf(size_t) { return {}; }
        bool given(unsigned d, u64 place, Value &) {
          said += (said.empty() ? "(" : " (") + std::to_string(d) + ", " + std::to_string(place) + ")";
          return true;
        }
        Value node(unsigned, u64, const Value &, const Value &) { return {}; }
      } order;
      Order::Value nothing;
      tree_walk(s.tree(), list, order, nothing);
      const std::string name = "drawn " + drawn.first + ": the given digests, as (depth, place)";
      expect(name, order.said);
      Opening o;
      tree.open(list, o, column);
      Writer w;
      write_opening(w, s, list, o);
      expect(name + ": the opening: bytes; H(168, 0, 0, none; the opening)", std::to_string(w.size()) + "; " + hex(H168(0, w.bytes())));
      if (!check_opening(top, s, list, o)) failures++, std::printf("FAILED: the opening at %s does not lead to the root\n", drawn.first.c_str());
    }
    // One value per vertex (R7.14), on the same tree. A session has one record per root's 32 bytes.
    VerifierSession session;
    TreeValues &record = session.record(top);
    bool admitted = true;
    for (const std::vector<u64> &at : {std::vector<u64>{1, 3, 6}, std::vector<u64>{3, 5}}) {
      const PositionList list = PositionList::from_draws(at, 3);
      Opening o;
      tree.open(list, o, column);
      admitted = admitted && check_opening(top, s, list, o, &record);
    }
    expect("the honest openings at 1 3 6 and then at 3 5: values entered, computed or given; all admitted", std::to_string(record.entered()) + "; " + (admitted ? "yes" : "no"));
    expect("vertices in the record", std::to_string(record.vertices()));
    // Offers a value as tree_check() does and withdraws it as after a rejected opening, so no case
    // below leaves a value in a record for the next one.
    auto zeros_at_2_1 = [](TreeValues &r) {
      r.begin();
      const bool taken = r.put(2, 1, Digest{});
      r.take_back();
      return std::string(taken ? "yes" : "no");
    };
    expect("then the vertex (2, 1), which the first opening computed, given as 32 zero bytes: admitted", zeros_at_2_1(record));
    const Digest the_same_bytes = top; // as if another tree had been sent with this root
    expect("the same value at the same vertex of another tree, whose root is the same 32 bytes: admitted", zeros_at_2_1(session.record(the_same_bytes)));
    expect("the same value at the same vertex under another root: admitted", zeros_at_2_1(session.record(digest_of(pattern(32, 5, 9)))));
  }

  // ---- rules R8.x: a whole chain in a file: channel, tree and opening together ----
  {
    Plan file_plan(set);
    file_plan.send_root("the tree");
    {
      PlanRound &r = file_plan.close("toy file: seed", RoundKind::SEED, ErrorKind::SAMPLE, Bound((u64)0), Priced::NOTHING);
      r.draw_E = 3, r.draw_nonzero = 1;
    }
    file_plan.send_answers("the answer", 1);
    file_plan.close_none("toy file: answer");
    file_plan.send_scalars("an element", 1);
    {
      PlanRound &r = file_plan.close("toy file: positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
      r.draw_positions = 4, r.log_n = 3;
    }
    Binder sends(9, file_plan), reads(9, file_plan);
    sends.configuration(configuration), reads.configuration(configuration);
    expect("label 9, empty statement, parameters: version, a plan of three rounds (kinds 3, 7, 4 with 8 bits of work and four positions in [0, 2^3)), configuration: the parameters' digest",
           hex(sends.parameters_digest()));
    Writer file;
    ProverRounds p(sends, random_bytes, file);
    OracleTreeShape t;
    t.log_n = 3, t.log_n_zk = 1, t.main_symbols = {2}, t.mask_symbols = {1}, t.tree_number = 0, t.salt_bytes = 40;
    t.run = p.run(), t.parameters = p.configuration();
    auto column = [&](u64 x, std::vector<F> &out) { out = {F{100 + x}, F{200 + x}, F{300 + short_position(3, 1, x)}}; };
    PageSalts salts;
    OracleTree tree(t, salts);
    std::vector<F> col;
    for (u64 x = 0; x < 8; x++) column(x, col), tree.set_leaf(x, col.data(), col.size());
    const Digest top = tree.finish();
    expect("the tree of section 7's columns and salts, number 0, under this chain's run's value: the root", hex(top));
    p.send_root(top);
    p.close_seed("toy file: seed").seed(2, 1);
    p.send_answer(element(9, 9, 9));
    p.close_empty("toy file: answer");
    p.send(element(4, 5, 6));
    const PositionList list = p.close_positions("toy file: positions").positions(3);
    Opening o;
    tree.open(list, o, column);
    write_opening(p.file_only(), t, list, o);
    p.finish();
    expect("the file: random bytes | root | answer (9, 9, 9) | element (4, 5, 6) | nonce | opening: bytes", std::to_string(file.size()));
    expect("the file", hex(file.bytes()));
    {
      struct Order {
        struct Value {};
        std::string said;
        Value leaf(size_t) { return {}; }
        bool given(unsigned d, u64 place, Value &) {
          said += (said.empty() ? "(" : " (") + std::to_string(d) + ", " + std::to_string(place) + ")";
          return true;
        }
        Value node(unsigned, u64, const Value &, const Value &) { return {}; }
      } order;
      Order::Value nothing;
      tree_walk(t.tree(), list, order, nothing);
      u64 nonce = 0; // it follows the random bytes, the root and two elements
      for (int i = 0; i < 8; i++) nonce |= (u64)file.bytes()[32 + 32 + 24 + 24 + i] << (8 * i);
      std::string draws;
      for (u64 x : list.draws()) draws += (draws.empty() ? "" : " ") + std::to_string(x);
      expect("the nonce; the draws; the given digests", std::to_string(nonce) + "; " + draws + "; " + order.said);
    }
    Reader r(file.data(), file.size());
    VerifierSession v_session;
    VerifierRounds v(reads, r, v_session);
    SentRoot got_root;
    E x;
    bool good = v.recv_root(got_root);
    v.close_seed("toy file: seed").seed(2, 1);
    good = good && v.recv_answer(x);
    v.close_empty("toy file: answer");
    good = good && v.recv(x);
    const PositionList again = v.close_positions("toy file: positions").positions(3);
    OracleTreeShape own = t;
    own.run = v.run(), own.parameters = v.configuration();
    Opening got;
    good = good && receive_opening(v, got_root, own, again, got) && v.finish();
    if (!good) failures++, std::printf("FAILED: the verifier's channel refuses the file: %s\n", v.why() ? v.why() : "?");
  }

  // Among the lines not computed here: the two named "the oracle's string of ..." (R2.4) and those
  // on types of other directories of blocks/, some of which blocks/tests/format_types_test.cpp
  // computes.
  return known_answers_end("format_vectors_test", argc);
}
