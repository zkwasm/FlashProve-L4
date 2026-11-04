// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes a key file and a leaf file (`entry` in names) at toy size, with the extra files that the
// second implementation of the verifier (tests/new/second/) needs to check them. A key file has the
// Merkle root of a commitment's first table (the anchor) and the committer's proofs about it. A
// leaf file has the root of a further table of the same matrix (a leaf table; `copy` in
// identifiers) and the LINK proof (`link`) that ties it to the anchor. make_key() and make_entry()
// build the two files; check_key() and check_entry() check them. All data is synthetic: the matrix
// and all randomness (`coins` in identifiers) come from constants in this source.
//   handover_files <folder>                      writes the files listed at main()
//   handover_files <folder> judge-key <file>     checks a key file made or altered elsewhere
//   handover_files <folder> judge-entry <file>   likewise a leaf file, under this program's anchor
// The first form returns 0 only if both honest files are accepted and every altered copy is
// rejected. A `judge-` call returns 0 only if <file> is accepted, 1 if it is rejected, and 2 if it
// checked nothing (see verdict_file()). The program overwrites nothing: it exits with code 2 at the
// first name that exists. What it has written stays, an incomplete set: use a new, empty folder.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // test headers build plans by hand (blocks/numbers/plan.h)
// check_entry() takes the anchor as a `TestedTable` (blocks/iop/foldeq.h), made by check_key().
// Where no key file is made or read, it comes from IndexCpu::key() instead: this program, not
// check_key(), vouches for that anchor.
#define BLK_TESTED_TABLE_FOR_A_TEST   // enables `on_a_tests_word` (blocks/frame/configure.h)
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
#include "frame/tests/toy_shapes.h"
#include "iop/tests/wf_world.h"

using namespace blk;

static void put(const std::string &path, const u8 *data, size_t bytes) {
  std::FILE *out = std::fopen(path.c_str(), "wbx");  // x: fails if the name exists, even a symlink
  if (!out && errno == EEXIST) std::printf("%s exists already, and this tool overwrites nothing: name another folder\n", path.c_str()), std::exit(2);
  if (!out || std::fwrite(data, 1, bytes, out) != bytes || std::fclose(out)) std::printf("cannot write %s\n", path.c_str()), std::exit(2);
}
static void put(const std::string &path, const std::vector<u8> &b) { put(path, b.data(), b.size()); }
static void put(const std::string &path, const std::string &s) { put(path, reinterpret_cast<const u8 *>(s.data()), s.size()); }
// The coordinates of x as decimal numbers, separated by spaces.
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
// Checks the output folder, so that a shell variable that is empty in the command ("$W", "$W/")
// cannot make the program write into the root of the file system. Returns null or the fault.
static const char *folder_fault(const char *given) {
  std::error_code e;
  return !*given                                        ? "the folder's name is empty"
         : !std::filesystem::is_directory(given, e)     ? "this is no folder: make one first"
         : std::filesystem::equivalent(given, "/", e)   ? "this is the root of the file system"
                                                        : nullptr;
}
// The file to which a `judge-` call writes its result: the checked file's path with its last
// suffix replaced by .txt. Returns an empty string, and the caller then checks and writes nothing,
// if the path is not a regular file, has no suffix after its name's first character or ends in
// .txt, or if the result's name exists, even as a symbolic link.
static std::string verdict_file(const std::string &judged) {
  const size_t slash = judged.rfind('/'), name = slash == std::string::npos ? 0 : slash + 1, dot = judged.rfind('.');
  std::error_code e;
  if (dot == std::string::npos || dot <= name || !std::filesystem::is_regular_file(judged, e)) return "";
  const std::string verdict = judged.substr(0, dot) + ".txt";
  return verdict == judged || std::filesystem::symlink_status(verdict, e).type() != std::filesystem::file_type::not_found ? "" : verdict;
}
// `YES_WITH_WORK` (blocks/frame/configure.h): some round meets the target (line `sec 128` of the
// first public file) only with its proof of work counted. In this program every verdict but `NO`
// counts as accepted.
static const char *word(ProofVerdict v) { return v == ProofVerdict::NO ? "NO" : v == ProofVerdict::YES ? "YES" : "YES; proof of work counted"; }

// The first form writes into <folder>:
//   public-file-1.txt, public-file-2.txt   the only data that configure() gets; the configuration
//       digest is their hash
//   key.file, entry.file                   the key file and the leaf file
//   key.one-time.parameters, .statement; key.shared.parameters; key.second.parameters;
//   entry.parameters, .statement           the parameter and statement bytes of the hash chains'
//       first queries, as `Binder` (blocks/chain/public.h) forms them. A chain is one protocol's
//       Fiat-Shamir transcript; its first query is the hash input that fixes the statement and the
//       parameters' digest; its seal is the digest that ends it. Two statements are not written,
//       since a verifier computes them as it reads the file: the shared chain's has the root of the
//       binding tree (a Merkle tree, here with one leaf: the one-time chain's seal), the second
//       chain's has the other two chains' seals.
//   key.targets.tsv, entry.targets.tsv     intermediate values that check_key(), in all three
//       chains, and check_entry() note as they run (`Notes`), in order; no verdict depends on them
//   key.altered-NN.file, entry.altered-NN.file, altered.tsv   altered copies, what was changed, and
//       the reason for each rejection
int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: handover_files <folder>\n");
    return 2;
  }
  if (const char *fault = folder_fault(argv[1])) {
    std::printf("\"%s\": %s\n", argv[1], fault);
    return 2;
  }
  const std::string folder = std::string(argv[1]) + "/";
  const std::vector<std::string> files = {toy_file(), toy_model()};
  const Configured c = configure(files, toy_shapes);
  if (!c.refused().empty()) {
    std::printf("the public files are not admitted: %s\n", c.refused().c_str());
    return 2;
  }
  const LinShape &S = c.sized().once.S;
  const WfPublic &x = c.sized().ranges.x;
  const WfMatrix size = wf_matrix_of(S);
  // A matrix within its ranges, for the shape that the public files give. Every cell is a toy
  // byte, -4 .. 3. Where a lookup has two digits, a pair of cells is a weight: low + 8 * high in
  // -16 .. 15.
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
  LinMatrix committed = M;  // a copy of M; nothing below reads it
  EngineCpu engine;
  const NonceSearch search = [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };

  // ---- the anchor and the key file ----
  TestCoins commit_coins(24), coins(44);
  IndexCpu first(S.fold.index, lin_table_rows(S, M), commit_coins, c.configuration());
  if (argc >= 3 && (std::string(argv[2]) == "judge-entry" || std::string(argv[2]) == "judge-key")) {
    const std::string path = argc >= 4 ? argv[3] : "(no file named)";  // this name has no suffix, so nothing is checked: exit code 2, below
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
    Reader r(bytes.data(), bytes.size());
    TestedTable kept;
    const char *reason = nullptr;
    const ProofVerdict v = std::string(argv[2]) == "judge-key" ? check_key(c, r, kept, reason) : check_entry(c, on_a_tests_word(c, first.key()), r, kept, reason);
    put(verdict, std::string(v == ProofVerdict::NO ? reason : "ACCEPTED") + "\n");
    std::printf("%s: %zu bytes: %s\n", path.c_str(), bytes.size(), v == ProofVerdict::NO ? reason : "ACCEPTED");
    return v == ProofVerdict::NO ? 1 : 0;
  }
  put(folder + "public-file-1.txt", files[0]), put(folder + "public-file-2.txt", files[1]);  // after the `judge-` calls, which write only their result
  const LinMatrix *matrix = &M;
  WfCpu ranges(x, size, [matrix](u64 row, u64 column) { return matrix->at(row, column); });
  HeavyCpu products(S, M);
  Writer key_file;
  if (!make_key(c, first, ranges, products, engine, coins, search, key_file)) {
    std::printf("no key is made\n");
    return 1;
  }
  auto judge_key = [&](const std::vector<u8> &bytes, TestedTable &key, const char *&why, Notes *listen) {
    Reader r(bytes.data(), bytes.size());
    why = nullptr;
    return check_key(c, r, key, why, listen);
  };
  auto listed = [](const Notes &notes) {
    std::string list = "what\tthe element's coordinates\n";
    for (const auto &n : notes) list += n.first + "\t" + decimal(n.second) + "\n";
    return list;
  };
  TestedTable key;
  const char *why = nullptr;
  Notes key_notes;
  const ProofVerdict key_verdict = judge_key(key_file.bytes(), key, why, &key_notes);
  put(folder + "key.file", key_file.bytes());
  put(folder + "key.targets.tsv", listed(key_notes));
  const KeyFixed fixed = key_fixed(c);
  const IndexName name{first.commitment(), first.record()};
  {
    Binder b(LABEL_ONE_TIME_CHAIN, c.key().chains[0].plan);
    bind_one_time_chain(b, fixed, name);
    put(folder + "key.one-time.parameters", b.parameter_bytes()), put(folder + "key.one-time.statement", b.statement_bytes());
  }
  {
    Binder b(LIN_LABEL_SHARED, c.key().chains[1].plan);
    b.configuration(fixed.configuration);
    lin_bind_shared(b, fixed.codes, fixed.lin, name, LinBindingRoot());  // an empty root: the parameters' bytes do not contain it
    put(folder + "key.shared.parameters", b.parameter_bytes());
  }
  {
    Binder b(LIN_LABEL_SECOND, c.key().chains[2].plan);
    b.configuration(fixed.configuration);
    put(folder + "key.second.parameters", b.parameter_bytes());
  }
  std::printf("key: %zu bytes: %s%s%s\n", key_file.bytes().size(), word(key_verdict), why ? ": " : "", why ? why : "");

  // ---- the leaf file: a leaf table of the same rows, other randomness, tied to the anchor ----
  TestCoins copy_coins(25), link_coins(45);
  Table copy_rows = lin_table_rows(S, M);
  IndexCpu further(c.sized().link.S.table, copy_rows, copy_coins, c.configuration());
  Writer entry_file;
  if (!make_entry(c, first, further, engine, link_coins, search, entry_file)) {
    std::printf("no entry is made\n");
    return 1;
  }
  Notes notes;
  auto judge_entry = [&](const std::vector<u8> &bytes, const char *&reason, Notes *listen) {
    Reader r(bytes.data(), bytes.size());
    TestedTable kept;
    reason = nullptr;
    return check_entry(c, key, r, kept, reason, listen);  // `key`: the anchor's `TestedTable`
  };
  const ProofVerdict entry_verdict = judge_entry(entry_file.bytes(), why, &notes);
  put(folder + "entry.file", entry_file.bytes());
  {
    Binder b(LINK_LABEL, c.entry().chains.at(0).plan);
    link_bind(b, c.configuration(), c.sized().entry_codes, c.sized().link.S, name, IndexName{further.commitment(), further.record()});
    put(folder + "entry.parameters", b.parameter_bytes()), put(folder + "entry.statement", b.statement_bytes());
  }
  put(folder + "entry.targets.tsv", listed(notes));
  std::printf("entry: %zu bytes: %s%s%s\n", entry_file.bytes().size(), word(entry_verdict), why ? ": " : "", why ? why : "");
  if (key_verdict == ProofVerdict::NO || entry_verdict == ProofVerdict::NO) return 1;

  // ---- altered copies ----
  std::string table = "file\tcopy\twhat was changed\tat byte\tthe reader's words\n";
  unsigned accepted = 0, made = 0;
  // Both files begin with a header of 8 bytes (`head` in altered.tsv: tag, version word), the
  // table's root and its record's stored sample answers.
  auto alter = [&](const char *which, const std::vector<u8> &honest, size_t answers_bytes) {
    const size_t chain = 8 + DIGEST_BYTES + answers_bytes;
    struct Change {
      const char *what;
      long at;  // byte whose low bit is flipped; -1: the last byte is cut off; -2: a byte is added
    };
    std::vector<Change> changes = {{"the head's tag", 0}, {"the head's version", 4}, {"the table's root", 8}, {"the chain's random bytes", (long)chain}, {"the first root behind them", (long)(chain + DIGEST_BYTES)},
                                   {"a byte at a quarter of the file", (long)(honest.size() / 4)}, {"a byte at the middle of the file", (long)(honest.size() / 2)},
                                   {"a byte at three quarters of the file", (long)(3 * honest.size() / 4)}, {"the last byte", (long)honest.size() - 1}, {"the last byte is cut off", -1}, {"a byte is added", -2}};
    changes.push_back({"the record's first answer", 8 + DIGEST_BYTES});
    unsigned n = 0;
    for (const Change &ch : changes) {
      std::vector<u8> bent = honest;
      if (ch.at == -1) bent.pop_back();
      else if (ch.at == -2) bent.push_back(0);
      else bent[(size_t)ch.at] ^= 1;
      const char *reason = nullptr;
      TestedTable unused;
      const ProofVerdict v = std::string(which) == "key" ? judge_key(bent, unused, reason, nullptr) : judge_entry(bent, reason, nullptr);
      char number[8];
      std::snprintf(number, sizeof number, "%02u", ++n);
      put(folder + which + ".altered-" + number + ".file", bent);
      table += std::string(which) + "\t" + number + "\t" + ch.what + "\t" + (ch.at >= 0 ? std::to_string(ch.at) : std::string("-")) + "\t" + (v == ProofVerdict::NO ? reason : "ACCEPTED") + "\n";
      accepted += v != ProofVerdict::NO, made++;
    }
  };
  const size_t answers_bytes = first.record().y.size() * E_BYTES;
  alter("key", key_file.bytes(), answers_bytes), alter("entry", entry_file.bytes(), answers_bytes);
  put(folder + "altered.tsv", table);
  std::printf("%u altered copies, %u accepted\n", made, accepted);
  return accepted ? 1 : 0;
}
