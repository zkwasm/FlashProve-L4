// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests `TestedTable` (blocks/iop/foldeq.h): the type in which `check_key` and `check_entry` return
   a committed table (`Index` in identifiers) whose key file or leaf file (`entry`) they accepted,
   and which `check_entry` and `check_proof` require as input. Toy size, synthetic data, CPU only.
   1. Honest files pass all three checks; the test vouches for no table (`on_a_tests_word`). After a
      rejected key file the later checks get an empty table: they must reject, not end the process.
   2. A function of the caller (`ShapesOf`, blocks/frame/configure.h) derives from the public files
      the ranges that a key file states for the matrix; the configuration digest covers only the
      files. So `toy_shapes` and `lax_shapes` give two configurations with one digest and one
      commitment ID (`published_id`). A table that passed `check_key` under the lax one must end the
      process in `check_entry` and `check_proof` under the strict one. With a form of `TestedTable`
      that does not carry the key file's public values (`key_public`, e.g. ranges, rows, codes), both
      accepted it.
   3. A program that runs the checks in separate processes calls `check_proof_if_tested`. Its result
      is never a plain yes, and it fills `why` and `at`, so that no null reason passes for a yes.
      On a table that was never read, it and `check_entry_if_tested` must reject, as in 1. */
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // blocks/iop/tests/lin_world.h builds plans by hand
#define BLK_TESTED_TABLE_FOR_A_TEST   // the same header requires it; it enables `on_a_tests_word`
#include "chain/tests/death.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "frame/tests/seam_world.h"
#include "iop/tests/wf_world.h"

// toy_shapes() without the lookup on two-digit weights; every row keeps the one on single digits.
static Shapes lax_shapes(const std::vector<std::string> &files, const Parameters &par) {
  Shapes s = toy_shapes(files, par);
  if (s.refused.empty()) s.sub.ranges.lookups.pop_back();
  return s;
}

int main() {
  const std::vector<std::string> files = {seam_file(4), toy_model(3, {})};
  const Configured c = configure(files, toy_shapes), lax = configure(files, lax_shapes);
  if (!c.refused().empty() || !lax.refused().empty()) return std::printf("FAILED: %s | %s\n", c.refused().c_str(), lax.refused().c_str()), 1;
  const LinShape &S = c.sized().once.S;
  // out_of_range: one weight is 3 + 8 * 3 = 27, outside -16 .. 15, though each digit is valid.
  auto matrix = [&](bool out_of_range) {
    LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
    M.a.resize(M.rows * M.columns);
    TestCoins data(3);
    const WfRun *weight = nullptr;
    for (F &cell : M.a) cell = wf_signed((long long)(data.word() % 8) - 4);
    for (const WfLookup &l : c.sized().ranges.x.lookups)
      if (l.digit_weight.size() == 2)
        for (const WfPart &p : l.parts)
          for (const WfRun &r : p.runs) {
            if (!weight) weight = &r;
            for (u64 i = 0; i < r.places; i++)
              for (u64 col = 0; col < M.columns; col++) {
                const long long value = (long long)(data.word() % 32) - 16, low = ((value + 4) % 8 + 8) % 8 - 4;
                M.a[(r.first_row + i) * M.columns + col] = wf_signed(low), M.a[(r.first_row + i + r.stride) * M.columns + col] = wf_signed((value - low) / 8);
              }
          }
    BLK_REQUIRE(weight, "the toy's ranges have no look-up of two digits");
    if (out_of_range) M.a[weight->first_row * M.columns] = wf_signed(3), M.a[(weight->first_row + weight->stride) * M.columns] = wf_signed(3);
    return M;
  };
  EngineCpu engine;
  const NonceSearch search = [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };
  const std::vector<std::vector<u8>> requests = {{1, 2, 3, 4}, {5, 6, 7}, {8, 9}};
  const std::vector<E> replies = {E::from(from_u64(1001)), E::from(from_u64(1002)), E::from(from_u64(1003))};
  const char *why = nullptr;
  auto key_made_under = [&](const Configured &under, const LinMatrix &M, IndexCpu &table, Writer &out) {
    const LinMatrix *m = &M;
    WfCpu ranges(under.sized().ranges.x, wf_matrix_of(under.sized().once.S), [m](u64 row, u64 column) { return m->at(row, column); });
    HeavyCpu products(under.sized().once.S, M);
    TestCoins coins(44);
    return make_key(under, table, ranges, products, engine, coins, search, out);
  };
  auto key_says = [&](const Configured &under, const Writer &file, TestedTable &out) {
    Reader r(file.bytes().data(), file.bytes().size());
    return check_key(under, r, out, why) != ProofVerdict::NO;
  };
  auto entry_says = [&](const Configured &under, const TestedTable &first, const Writer &file, TestedTable &out) {
    Reader r(file.bytes().data(), file.bytes().size());
    return check_entry(under, first, r, out, why) != ProofVerdict::NO;
  };
  auto proof_says = [&](const Configured &under, const TestedTable &table, u32 number, const std::vector<u8> &bytes) {
    Reader file(bytes.data(), bytes.size());
    return check_proof(under, table, number, requests[0], requests[0], replies[0], file, toy_links, nullptr, why) != ProofVerdict::NO;
  };

  {  // 1 and 3: honest files
    const LinMatrix M = matrix(false);
    TestCoins table_coins(24), copy_coins(25), link_coins(45);
    IndexCpu first(S.fold.index, lin_table_rows(S, M), table_coins, c.configuration()), further(c.sized().link.S.table, lin_table_rows(S, M), copy_coins, c.configuration());
    Writer key_file, entry_file;
    CHECK(key_made_under(c, M, first, key_file) && make_entry(c, first, further, engine, link_coins, search, entry_file), "no key or no entry is made of an honest matrix");
    const std::vector<std::vector<u8>> made = prove_batch(c, further, IndexName{further.commitment(), further.record()}, M, 1, requests, replies, 62);
    TestedTable key, copy;
    CHECK(!key.tested_under() && key_says(c, key_file, key) && key.tested_under() && key.reader().name() == first.name(), "an honest key");
    CHECK(entry_says(c, key, entry_file, copy) && copy.tested_under() && copy.reader().name() == further.name() && copy.key_public() == key.key_public(), "an honest entry under the key's table");
    CHECK(made.size() == 3 && proof_says(c, copy, 1, made[0]), "an honest proof on the copy");
    TestedTable one = key;
    CHECK(entry_says(c, one, entry_file, one) && proof_says(c, one, 1, made[0]), "the entry's table written into the variable that held the key's");
    TestedTable moved = std::move(one);
    CHECK(proof_says(c, moved, 1, made[0]), "a moved table");
    const Configured again = configure(files, toy_shapes);
    CHECK(proof_says(again, copy, 1, made[0]), "a second Configured from the same files by the same routine");
    CHECK(!proof_says(c, key, 1, made[0]), "a proof on the copy is accepted on the key's own table");
    {  // One bit of the key file is changed. The bytes of a file must not be able to end a process.
      std::vector<u8> bent_bytes = key_file.bytes();
      bent_bytes.back() ^= 1;
      Writer bent_key;
      bent_key.put_bytes(bent_bytes.data(), bent_bytes.size());
      TestedTable none, unused;
      CHECK(!key_says(c, bent_key, none) && !none.tested_under(), "a key file with one bit changed is accepted, or hands out a table");
      CHECK(runs_through([&] { if (entry_says(c, none, entry_file, unused) || std::string(why).find("has passed no check") == std::string::npos) _exit(1); }), "behind a refused key, check_entry does not refuse in words");
      CHECK(runs_through([&] { if (proof_says(c, none, 1, made[0]) || std::string(why).find("has passed no check") == std::string::npos) _exit(1); }), "behind a refused key, check_proof does not refuse in words");
    }

    UnprovedTable unproved;
    Reader head(entry_file.bytes().data(), entry_file.bytes().size()), file(made[0].data(), made[0].size());
    CHECK(read_entry_table_unproved(c, head, unproved, why) && unproved.reader().name() == further.name(), "the copy's table is not read from the entry's first bytes");
    const char *at = nullptr;
    why = nullptr;
    const IfTested result = check_proof_if_tested(c, unproved, 1, requests[0], requests[0], replies[0], file, toy_links, nullptr, why, nullptr, &at);
    CHECK(result == (c.needs_work() ? IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED : IfTested::YES_IF_THE_TABLE_IS_TESTED), "an honest proof on a table that was read without its proofs");
    CHECK(why == IF_TESTED_WHY && at == IF_TESTED_WHY, "behind a result that is no plain yes, no reason is given");
    std::vector<u8> bent = made[0];
    bent.back() ^= 1;
    Reader bent_file(bent.data(), bent.size());
    CHECK(check_proof_if_tested(c, unproved, 1, requests[0], requests[0], replies[0], bent_file, toy_links, nullptr, why) == IfTested::NO && why != IF_TESTED_WHY, "a proof with its last byte changed, on a table read without its proofs");
    // `never` was not filled, as when a caller ignores a rejection by read_key_table_unproved() or
    // read_entry_table_unproved(). A file's bytes can cause that rejection, so the two checks must
    // reject, not end the process.
    UnprovedTable never;
    CHECK(runs_through([&] {
            Reader proof_again(made[0].data(), made[0].size()), entry_again(entry_file.bytes().data(), entry_file.bytes().size());
            if (check_proof_if_tested(c, never, 1, requests[0], requests[0], replies[0], proof_again, toy_links, nullptr, why) != IfTested::NO) _exit(1);
            if (check_entry_if_tested(c, never, entry_again, why) != IfTested::NO) _exit(1);
          }), "a table that was never read is not refused by the two routines that take it");
  }
  {  // 2: two configurations from the same files
    const LinMatrix M = matrix(true);
    CHECK(c.configuration() == lax.configuration() && Plan::same_setting(c.sized().d.set, lax.sized().d.set) && c.sized().once.S.fold.index.distances == lax.sized().once.S.fold.index.distances,
          "control: the two configurations differ in digest, setting or distances, so that a stop on those would part them");
    TestCoins table_coins(24), again_coins(24), copy_coins(25), link_coins(45);
    IndexCpu first(S.fold.index, lin_table_rows(S, M), table_coins, c.configuration()), again(S.fold.index, lin_table_rows(S, M), again_coins, c.configuration());
    IndexCpu further(c.sized().link.S.table, lin_table_rows(S, M), copy_coins, c.configuration());
    Writer none, key_file, entry_file;
    CHECK(!key_made_under(c, M, again, none), "control: the strict configuration makes a key of a matrix that is out of its ranges");
    CHECK(key_made_under(lax, M, first, key_file) && make_entry(c, first, further, engine, link_coins, search, entry_file), "control: the lax configuration makes no key");
    const std::vector<std::vector<u8>> on_first = prove_batch(c, first, IndexName{first.commitment(), first.record()}, M, 0, requests, replies, 51);
    TestedTable refused, under_lax, unused;
    CHECK(!key_says(c, key_file, refused) && !refused.tested_under(), "the strict check accepts the lax key");
    CHECK(key_says(lax, key_file, under_lax) && published_id(c, under_lax.reader()) == published_id(lax, under_lax.reader()), "control: the lax check refuses its own key, or the two commitment IDs differ");
    CHECK(ends_with("checked under other public values", [&] { (void)entry_says(c, under_lax, entry_file, unused); }), "check_entry takes a first table whose key was checked under laxer ranges");
    CHECK(on_first.size() == 3 && ends_with("checked under other public values", [&] { (void)proof_says(c, under_lax, 0, on_first[0]); }), "check_proof takes a table whose key was checked under laxer ranges");
    CHECK(entry_says(lax, under_lax, entry_file, unused), "control: under the lax configuration the same entry is refused");
    // `handed` is passed by reference, so check_proof() must re-check it after calling `swapping`.
    Configured handed = lax;
    const LinksCheck swapping = [&](VerifierContext &V, const RequestPublic &x, LinksResult &out) {
      const bool yes = toy_links(V, x, out);
      handed = c;
      return yes;
    };
    CHECK(ends_with("has changed the configuration", [&] {
            Reader file(on_first[0].data(), on_first[0].size());
            (void)check_proof(handed, under_lax, 0, requests[0], requests[0], replies[0], file, swapping, nullptr, why);
          }), "check_proof goes on under a configuration that the caller's routine has swapped in");
  }
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
