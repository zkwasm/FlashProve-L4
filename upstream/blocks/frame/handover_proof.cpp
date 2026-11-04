// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test tool. It proves two toy batches on the CPU and writes their proof files, with what a second
// verifier (another implementation) needs to check them.
//   handover_proof <folder> [<links: 1 to 3; default 3> [arrays] [class] [later]]
//   handover_proof <folder> [<links> [arrays] [class] [later]]
//                  judge <on-the-first-table | on-the-copy> <n> <file>
// The data are synthetic and every random value comes from small constants in this source:
// nothing in any of these files is hidden.
// The tool overwrites nothing. At the first name that exists it exits with code 2, and what it has
// written by then stays: use a new directory <folder>, with on-the-first-table/ and on-the-copy/
// made in it.
// What the toy proves and does not prove, its arguments, its files and its exit codes: see the
// comment above main().
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // test headers build plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // a test header needs it; this tool marks no table as tested
#include "frame/tests/seam_world.h"  // the toy's public files, prover and verifier's callback

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "iop/tests/wf_world.h"

static void put(const std::string &path, const u8 *data, size_t bytes) {
  std::FILE *out = std::fopen(path.c_str(), "wbx");  // x: fails if the name exists, symlinks too
  if (!out && errno == EEXIST) std::printf("%s exists already, and this tool overwrites nothing: name another folder\n", path.c_str()), std::exit(2);
  if (!out || std::fwrite(data, 1, bytes, out) != bytes || std::fclose(out)) std::printf("cannot write %s: do the folder and its two folders on-the-first-table and on-the-copy stand?\n", path.c_str()), std::exit(2);
}
static void put(const std::string &path, const std::vector<u8> &b) { put(path, b.data(), b.size()); }
static void put(const std::string &path, const std::string &s) { put(path, reinterpret_cast<const u8 *>(s.data()), s.size()); }
static std::string decimal(const E &x) {
  Writer w;
  w.put(x);
  std::string text;
  for (size_t t = 0; t * 8 < w.bytes().size(); t++) {
    u64 v = 0;
    for (int b = 7; b >= 0; b--) v = v << 8 | w.bytes()[8 * t + (size_t)b];
    text += (t ? " " : "") + std::to_string(v);
  }
  return text;
}
// Returns null, or what is wrong with the directory to write into. An empty shell variable on the
// command line ("$W", "$W/") must not make the tool write into the root of the file system.
static const char *folder_fault(const char *given) {
  std::error_code e;
  return !*given                                        ? "the folder's name is empty"
         : !std::filesystem::is_directory(given, e)     ? "this is no folder: make one first"
         : std::filesystem::equivalent(given, "/", e)   ? "this is the root of the file system"
                                                        : nullptr;
}
// The path for a `judge` call's verdict: `judged` with its last suffix replaced by .txt. Returns an
// empty string in the cases that main() then prints; a name's leading dot starts no suffix, and a
// symbolic link at the verdict's path counts as a file that exists.
static std::string verdict_file(const std::string &judged) {
  const size_t slash = judged.rfind('/'), name = slash == std::string::npos ? 0 : slash + 1, dot = judged.rfind('.');
  std::error_code e;
  if (dot == std::string::npos || dot <= name || !std::filesystem::is_regular_file(judged, e)) return "";
  const std::string verdict = judged.substr(0, dot) + ".txt";
  return verdict == judged || std::filesystem::symlink_status(verdict, e).type() != std::filesystem::file_type::not_found ? "" : verdict;
}
// `YES_WITH_WORK` (blocks/frame/configure.h): accepted, but some round meets the target (line
// `sec 128` of the first public file) only with proof of work counted. Below, any verdict but `NO`
// counts as accepted.
static const char *word(ProofVerdict v) { return v == ProofVerdict::NO ? "NO" : v == ProofVerdict::YES ? "YES" : "YES; proof of work counted"; }

// The toy is that of blocks/frame/tests/seam_world.h, shared with the test
// blocks/frame/tests/seam_test.cpp. A request's proof has <links> segments (`links`, `shape.link`
// in identifiers). Each multiplies two committed vectors entry by entry and makes `CLAIMS` claims
// on the committed matrix, which batt_shown_prove() reduces to one pending claim, left for LIN, the
// batch's test on the committed table (`Index` in identifiers). The claims' forms are a public
// function of the request's bytes and stand in for a model. The reply is one unmasked scalar, about
// which the segments prove nothing.
// A proof file has three hash chains (Fiat-Shamir transcripts): the request's first chain, which
// has the segments, the batch's shared chain, and the request's second chain. LIN has a part on
// each.
//   <links> = 1   one pending claim, so member_first_prove() has nothing to batch
//   arrays        the item "layout" of the first chain's first query (the hash input that starts a
//                 chain) lists two arrays per segment
//   class         the second segment (the only one, if <links> = 1) sends one more oracle message,
//                 with a leaf class of its own (a number that each oracle message has: only
//                 messages of one class share a Merkle tree), and so in a tree of its own
//   later         that segment sends one more oracle message at a later commit point (`moment`)
//
// main() commits to one matrix twice: as the anchor, a commitment's first table (`first`,
// on-the-first-table/, table number 0), and as a leaf table, a further table of the same rows
// (`further`, on-the-copy/, number 1). It makes and checks the key file, which is about the anchor,
// and the leaf file (`entry` in identifiers), whose LINK proof (`link_coins`, `LINK_LABEL`) ties
// the leaf table to the anchor. Then it proves a batch of three requests on each table (the library
// allows a batch on the anchor) and checks each proof file with check_proof()
// (blocks/frame/check.h). A batch's binding tree, a Merkle tree over the seals that end its first
// chains, has depth 2 (line `slots 4` of the first public file), so one place (a request's leaf
// number in it) stays empty.
// Written into <folder>:
//   public-file-1.txt, public-file-2.txt   what configure() reads; the second has <links> and the
//                                   words arrays, class, later
//   key.file, entry.file            the key file, the leaf file. A key file (defined in
//                                   blocks/frame/check_key.h) has three chains: the one-time chain,
//                                   which has the committer's range proof (WF) and the first part
//                                   of the committer's own, one-time run of LIN, then a shared and
//                                   a second chain as in a proof file
//   key.one-time.parameters, .statement; key.shared.parameters; key.second.parameters;
//   entry.parameters, .statement    first queries of these two files' chains, under the names of
//                                   blocks/frame/handover_files.cpp, the companion tool for key
//                                   files and leaf files
// and into each table's subdirectory, for n = 0, 1, 2:
//   <n>.proof, <n>.request, <n>.reply (its coordinates in decimal)
//   <n>.place.txt                   place, the tree's depth, table number (`stored copy`)
//   <n>.claims                      the claims' forms: two 64-bit words (segments, claims in each);
//                                   for each claim its point and its row weights written out, each
//                                   as a count word and that many elements
//   <n>.parts.tsv                   offset and length of the five parts: header (`head`), first
//                                   chain, place, shared chain, second chain; a proof file does not
//                                   store them
//   <n>.targets.tsv                 values that check_proof() notes on the first chain, in order;
//                                   no decision reads them
//   <n>.first.parameters, .statement       the first chain's first query, as `Binder`
//                                   (blocks/chain/public.h) has its bytes
//   shared.parameters, second.parameters   the same, once per batch, without the statements: these
//                                   have the binding tree's root (shared chain) and two seals
//                                   (second chain), which a verifier computes as it goes
//   0.altered-NN.proof              altered copies of request 0's file
//   altered.tsv                     what was changed in each and why check_proof() rejects it; also
//                                   calls with the honest file under another request, reply, number
//                                   or table, which write no file
// Exit code 0 only if the key file, the leaf file and all six honest proof files are accepted and
// every altered copy and call is rejected; 2 for bad arguments or a file that cannot be written.
//
// judge: checks <file>, made or altered by someone else, as the proof of request <n> in the named
// batch (any word other than on-the-copy means the anchor's), and writes check_proof()'s reason, or
// ACCEPTED, to the path from verdict_file(). <links> and the words must be those of the run that
// made the batch. Exit code 0 only if <file> is accepted, 1 if it is rejected, and 2, with the file
// unchecked and nothing written, if verdict_file() returns no path, the file cannot be read, or the
// tool's own key file or leaf file cannot be made or is rejected; 2 also if the write fails.
int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: handover_proof <folder> [<links: 1 to 3> [arrays] [class] [later]]\n");
    return 2;
  }
  if (const char *fault = folder_fault(argv[1])) {
    std::printf("\"%s\": %s\n", argv[1], fault);
    return 2;
  }
  const std::string folder = std::string(argv[1]) + "/";
  // `judging` is true even for a bare `judge`, which gets the usage line.
  std::vector<std::string> words(argv + 2, argv + argc), to_judge; bool judging = false;
  for (size_t i = 0; i < words.size(); i++)
    if (words[i] == "judge") judging = true, to_judge.assign(words.begin() + (long)i + 1, words.end()), words.resize(i);
  const unsigned links = !words.empty() ? (unsigned)std::atoi(words[0].c_str()) : 3;
  const std::vector<std::string> more(words.begin() + (words.empty() ? 0 : 1), words.end());  // arrays, class, later: `configure` rejects anything else
  if (links < 1 || links > 3 || (judging && (to_judge.size() != 3 || to_judge[1].size() != 1 || to_judge[1][0] < '0' || to_judge[1][0] > '2')))
    return std::printf("usage: handover_proof <folder> [<links: 1 to 3> [arrays] [class] [later]] [judge <on-the-first-table | on-the-copy> <n: 0 to 2> <file>]\n"), 2;
  const std::vector<std::string> files = {seam_file(4), toy_model(links, more)};
  const Configured c = configure(files, toy_shapes);
  if (!c.refused().empty()) {
    std::printf("the public files are not admitted: %s\n", c.refused().c_str());
    return 2;
  }
  const LinShape &S = c.sized().once.S;
  const WfPublic &x = c.sized().ranges.x;
  const WfMatrix size = wf_matrix_of(S);
  // A matrix within the ranges of the committer's range proof (WF): every cell is in -4 .. 3; where
  // a lookup has two digits, a row and the row `stride` below it have the digits low and high of a
  // weight low + 8 high in -16 .. 15.
  LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
  M.a.resize(M.rows * M.columns);
  TestCoins data(3);
  for (F &cell : M.a) cell = wf_signed((long long)(data.word() % 8) - 4);
  for (const WfLookup &l : x.lookups)
    if (l.digit_weight.size() == 2)
      for (const WfPart &p : l.parts)
        for (const WfRun &r : p.runs)
          for (u64 i = 0; i < r.places; i++)
            for (u64 col = 0; col < M.columns; col++) {
              const long long weight = (long long)(data.word() % 32) - 16, low = ((weight + 4) % 8 + 8) % 8 - 4;
              M.a[(r.first_row + i) * M.columns + col] = wf_signed(low), M.a[(r.first_row + i + r.stride) * M.columns + col] = wf_signed((weight - low) / 8);
            }
  EngineCpu engine;
  const NonceSearch search = [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };

  // ---- key file and leaf file; proofs are checked on what check_key(), check_entry() give ----
  TestCoins commit_coins(24), coins(44), copy_coins(25), link_coins(45);
  IndexCpu first(S.fold.index, lin_table_rows(S, M), commit_coins, c.configuration());
  const LinMatrix *matrix = &M;
  WfCpu ranges(x, size, [matrix](u64 row, u64 column) { return matrix->at(row, column); });
  HeavyCpu products(S, M);
  Writer key_file, entry_file;
  if (!make_key(c, first, ranges, products, engine, coins, search, key_file)) return std::printf("no key is made\n"), judging ? 2 : 1;
  IndexCpu further(c.sized().link.S.table, lin_table_rows(S, M), copy_coins, c.configuration());
  if (!make_entry(c, first, further, engine, link_coins, search, entry_file)) return std::printf("no entry is made\n"), judging ? 2 : 1;
  TestedTable key, copy;
  const char *why = nullptr;
  {
    Reader r(key_file.bytes().data(), key_file.bytes().size());
    const ProofVerdict v = check_key(c, r, key, why);
    std::printf("key: %zu bytes: %s%s%s\n", key_file.bytes().size(), word(v), why ? ": " : "", why ? why : "");
    if (v == ProofVerdict::NO) return judging ? 2 : 1;
  }
  {
    Reader r(entry_file.bytes().data(), entry_file.bytes().size());
    const ProofVerdict v = check_entry(c, key, r, copy, why);
    std::printf("entry: %zu bytes: %s%s%s\n", entry_file.bytes().size(), word(v), why ? ": " : "", why ? why : "");
    if (v == ProofVerdict::NO) return judging ? 2 : 1;
  }
  const std::vector<std::vector<u8>> requests = {{1, 2, 3, 4}, {5, 6, 7}, {8, 9}};
  const std::vector<E> replies = {E::from(from_u64(1001)), E::from(from_u64(1002)), E::from(from_u64(1003))};
  auto judge = [&](const TestedTable &table, u32 number, const std::vector<u8> &request, const E &reply, const std::vector<u8> &bytes, const char *&reason, u64 *place = nullptr, Notes *listen = nullptr) {
    Reader file(bytes.data(), bytes.size());
    reason = nullptr;
    ProofFacts facts;
    // check_proof() takes no list for notes. Its callback gets the first chain's context, so the
    // list is attached there; a caller cannot reach the other two chains' contexts.
    const LinksCheck with_notes = [listen](VerifierContext &V, const RequestPublic &pub, LinksResult &out) { return V.notes = listen, toy_links(V, pub, out); };
    const ProofVerdict v = check_proof(c, table, number, request, request, reply, file, with_notes, nullptr, reason, &facts);
    if (place && v != ProofVerdict::NO) *place = facts.place;
    return v;
  };
  if (!to_judge.empty()) {
    const bool on_the_copy = to_judge[0] == "on-the-copy";
    const unsigned n = unsigned(to_judge[1][0] - '0');
    const std::string &path = to_judge[2];
    const std::string verdict = verdict_file(path);
    if (verdict.empty()) {
      std::printf("%s: not judged. The verdict would go to the same name ending in .txt. Either this is no file, or its name has no ending after a dot, or it ends in .txt itself, or that .txt file exists already: judge a copy under another name\n", path.c_str());
      return 2;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::printf("%s: not judged: the file cannot be read\n", path.c_str());
      return 2;
    }
    const std::vector<u8> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const ProofVerdict v = judge(on_the_copy ? copy : key, on_the_copy ? 1 : 0, requests[n], replies[n], bytes, why);
    put(verdict, std::string(v == ProofVerdict::NO ? why : "ACCEPTED") + "\n");
    std::printf("%s: %zu bytes: %s\n", path.c_str(), bytes.size(), v == ProofVerdict::NO ? why : "ACCEPTED");
    return v == ProofVerdict::NO ? 1 : 0;
  }
  put(folder + "public-file-1.txt", files[0]), put(folder + "public-file-2.txt", files[1]);
  put(folder + "key.file", key_file.bytes()), put(folder + "entry.file", entry_file.bytes());
  {  // first queries of the key file's and leaf file's chains, for second verifiers of these files
    const KeyFixed fixed = key_fixed(c);
    const IndexName name{first.commitment(), first.record()};
    Binder one_time(LABEL_ONE_TIME_CHAIN, c.key().chains[0].plan), shared(LIN_LABEL_SHARED, c.key().chains[1].plan), second(LIN_LABEL_SECOND, c.key().chains[2].plan), link(LINK_LABEL, c.entry().chains.at(0).plan);
    bind_one_time_chain(one_time, fixed, name);
    put(folder + "key.one-time.parameters", one_time.parameter_bytes()), put(folder + "key.one-time.statement", one_time.statement_bytes());
    shared.configuration(fixed.configuration);
    lin_bind_shared(shared, fixed.codes, fixed.lin, name, LinBindingRoot());  // an empty root: the parameters' bytes do not contain it
    put(folder + "key.shared.parameters", shared.parameter_bytes());
    second.configuration(fixed.configuration);
    put(folder + "key.second.parameters", second.parameter_bytes());
    link_bind(link, c.configuration(), c.sized().entry_codes, c.sized().link.S, name, IndexName{further.commitment(), further.record()});
    put(folder + "entry.parameters", link.parameter_bytes()), put(folder + "entry.statement", link.statement_bytes());
  }
  unsigned accepted_false = 0, made_false = 0, refused_honest = 0;
  // keeper: the prover's table; table: what the verifier has of it.
  struct Batch {
    const char *where;
    IndexCpu *keeper;
    const TestedTable *table;
    u32 number;
    u64 coin_seed;
  };
  for (const Batch &b : {Batch{"on-the-first-table/", &first, &key, 0, 51}, Batch{"on-the-copy/", &further, &copy, 1, 62}}) {  // request n's first chain seeds its random bytes with coin_seed + 100 + n, its second with + 200 + n: all 12 seeds differ
    const std::string to = folder + b.where;
    const IndexName name{b.keeper->commitment(), b.keeper->record()};
    std::vector<Parts> parts;
    const std::vector<std::vector<u8>> made = prove_batch(c, *b.keeper, name, M, b.number, requests, replies, b.coin_seed, nullptr, &parts);
    if (made.size() != requests.size()) return std::printf("%s the keeper does not serve\n", b.where), 1;
    for (unsigned n = 0; n < made.size(); n++) {
      const std::string stem = to + std::to_string(n);
      u64 place = ~(u64)0;
      Notes notes;
      const ProofVerdict v = judge(*b.table, b.number, requests[n], replies[n], made[n], why, &place, &notes);
      {
        std::string list = "what\tthe element's coordinates\n";
        for (const auto &note : notes) list += note.first + "\t" + decimal(note.second) + "\n";
        put(stem + ".targets.tsv", list);
      }
      std::printf("%s%u.proof: %zu bytes: %s%s%s; place %llu\n", b.where, n, made[n].size(), word(v), why ? ": " : "", why ? why : "", (unsigned long long)place);
      refused_honest += v == ProofVerdict::NO || place != n;
      put(stem + ".proof", made[n]), put(stem + ".request", requests[n]), put(stem + ".reply", decimal(replies[n]) + "\n");
      put(stem + ".place.txt", "place " + std::to_string(n) + "\ndepth " + std::to_string(c.fixed().binding.log_leaves) + "\nstored copy " + std::to_string(b.number) + "\n");
      std::string borders = "part\tfirst byte\tbytes\n";
      size_t at = 0;
      const char *const names[5] = {"head", "first chain", "place", "shared chain", "second chain"};
      for (unsigned p = 0; p < 5; at += parts[n][p].size(), p++) borders += std::string(names[p]) + "\t" + std::to_string(at) + "\t" + std::to_string(parts[n][p].size()) + "\n";
      put(stem + ".parts.tsv", borders);
      {
        Writer forms;
        const LinShape &lin = c.fixed().lin;
        forms.put_u64(c.fixed().shape.link.size()), forms.put_u64(CLAIMS);
        for (size_t l = 0; l < c.fixed().shape.link.size(); l++)
          for (size_t j = 0; j < CLAIMS; j++) {
            const TensorRow row = member_row_form(lin, seed_of(requests[n]), unsigned(l), j);
            LinClaim f;
            f.g = row.g;
            const std::vector<E> g = lin_weights(lin, f);
            forms.put_u64(row.psi.size());
            for (const E &e : row.psi) forms.put(e);
            forms.put_u64(g.size());
            for (const E &e : g) forms.put(e);
          }
        put(stem + ".claims", forms.bytes());
      }
      Binder q(LABEL_FIRST_CHAIN, c.proof().chains[0].plan);
      RequestPublic pub = bind_request_parameters(q, c);
      bind_request_statement(q, pub, name, b.number, requests[n], requests[n]);  // in the toy a request is its own file
      put(stem + ".first.parameters", q.parameter_bytes()), put(stem + ".first.statement", q.statement_bytes());
    }
    {
      Binder q(LABEL_PROOF_SHARED_CHAIN, c.proof().chains[1].plan);
      bind_shared_chain(q, c, name, LinBindingRoot());  // an empty root: the parameters' bytes do not contain it
      put(to + "shared.parameters", q.parameter_bytes());
    }
    {
      Binder q(LABEL_PROOF_SECOND_CHAIN, c.proof().chains[2].plan);
      q.configuration(c.fixed().configuration);
      put(to + "second.parameters", q.parameter_bytes());
    }
    // ---- altered copies of request 0's file; the honest file under other arguments ----
    std::string table = "copy\twhat was changed\tat byte\tthe reader's words\n";
    unsigned number = 0;
    auto tell = [&](const char *what, long at, const std::vector<u8> *bytes, ProofVerdict v, const char *reason) {
      char nn[8] = "-";
      if (bytes) std::snprintf(nn, sizeof nn, "%02u", ++number), put(to + "0.altered-" + nn + ".proof", *bytes);
      table += std::string(nn) + "\t" + what + "\t" + (at >= 0 ? std::to_string(at) : std::string("-")) + "\t" + (v == ProofVerdict::NO ? reason : "ACCEPTED") + "\n";
      accepted_false += v != ProofVerdict::NO, made_false++;
    };
    const std::vector<u8> &good = made[0];
    size_t begin[6] = {0};
    for (unsigned p = 0; p < 5; p++) begin[p + 1] = begin[p] + parts[0][p].size();
    const size_t salt = c.fixed().binding.salt_bytes;
    struct Change {
      const char *what;
      size_t at;
    };
    for (const Change &ch : {Change{"the head's tag", 0}, Change{"the head's version", 4}, Change{"the head's third word: the stored copy's number", 8}, Change{"the first chain's random bytes", begin[1]},
                             Change{"a byte at the middle of the first chain", (begin[1] + begin[2]) / 2}, Change{"the first chain's last byte", begin[2] - 1}, Change{"the place's word", begin[2]},
                             Change{"the place's salt", begin[2] + 8}, Change{"the place's first sibling", begin[2] + 8 + salt}, Change{"the place's last sibling", begin[3] - 1},
                             Change{"the shared chain's first byte", begin[3]}, Change{"a byte at the middle of the shared chain", (begin[3] + begin[4]) / 2}, Change{"the second chain's random bytes", begin[4]},
                             Change{"a byte at the middle of the second chain", (begin[4] + begin[5]) / 2}, Change{"the last byte", begin[5] - 1}}) {
      std::vector<u8> bent = good;
      bent[ch.at] ^= 1;
      const ProofVerdict v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell(ch.what, (long)ch.at, &bent, v, why);
    }
    {  // three more, for checks that the one-bit changes above do not reach
      std::vector<u8> bent = good;
      bent[8] ^= 1;  // header and call agree; the chain fails: its first query has the number
      ProofVerdict v = judge(*b.table, b.number ^ 1, requests[0], replies[0], bent, why);
      tell("the head's third word, and the call asks for that number", 8, &bent, v, why);
      bent = good, bent[begin[2] + (c.fixed().binding.log_leaves >> 3)] ^= u8(1u << (c.fixed().binding.log_leaves & 7));  // the honest place is below 2^depth: that bit is 0
      v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell("the place's word + 2^depth: no such place", (long)begin[2], &bent, v, why);
      // The reply is (1001, 0, 0). Its second coordinate is written as p, the field's modulus
      // (`P`): the same element, not in canonical form. The channel hashes the parsed element, so
      // only parsing can reject it.
      Writer spelled;
      spelled.put(replies[0]);
      const std::vector<u8> &r = spelled.bytes();
      const auto at = std::search(good.begin() + (long)begin[1], good.begin() + (long)begin[2], r.begin(), r.end());
      BLK_REQUIRE(at != good.begin() + (long)begin[2] && std::search(at + 1, good.begin() + (long)begin[2], r.begin(), r.end()) == good.begin() + (long)begin[2], "the reply's bytes stand once in the first chain");
      const size_t second = size_t(at - good.begin()) + 8;
      bent = good;
      for (unsigned i = 0; i < 8; i++) bent[second + i] = u8(P >> (8 * i));
      v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell("the reply's second coordinate, 0, spelled p", (long)second, &bent, v, why);
    }
    {
      std::vector<u8> bent = good;
      bent.pop_back();
      ProofVerdict v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell("the last byte is cut off", -1, &bent, v, why);
      bent = good, bent.push_back(0);
      v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell("a byte is added", -1, &bent, v, why);
    }
    const char *const part_names[5] = {"", "request 1's first chain in request 0's file", "request 1's place in request 0's file", "", "request 1's second chain in request 0's file"};
    for (unsigned swap : {1u, 2u, 4u}) {  // the shared chain is the same bytes in all files of a batch
      std::vector<u8> bent;
      for (unsigned p = 0; p < 5; p++) bent.insert(bent.end(), parts[p == swap ? 1 : 0][p].begin(), parts[p == swap ? 1 : 0][p].end());
      const ProofVerdict v = judge(*b.table, b.number, requests[0], replies[0], bent, why);
      tell(part_names[swap], -1, &bent, v, why);
    }
    struct Call {
      const char *what;
      const TestedTable *table;
      u32 number;
      unsigned request, reply;
    };
    for (const Call &call : {Call{"the honest file, asked about request 1's bytes", b.table, b.number, 1, 0}, Call{"the honest file, asked about another reply", b.table, b.number, 0, 1},
                             Call{"the honest file, asked about another stored copy's number", b.table, b.number + 1, 0, 0}, Call{"the honest file, checked on the other table", b.number ? &key : &copy, b.number, 0, 0}}) {
      const ProofVerdict v = judge(*call.table, call.number, requests[call.request], replies[call.reply], good, why);  // first: it sets `why`
      tell(call.what, -1, nullptr, v, why);
    }
    put(to + "altered.tsv", table);
  }
  std::printf("%u altered copies and calls, %u accepted; %u honest files refused\n", made_false, accepted_false, refused_honest);
  return accepted_false || refused_honest ? 1 : 0;
}
