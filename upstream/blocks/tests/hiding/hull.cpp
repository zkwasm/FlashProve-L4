// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The sampled hull: a test of hiding for provers whose `coins` (private random values) cannot be
// set one by one, as exact.h needs, and whose output is too long for one matrix.
// Input: a directory (`folder` in the usage text and identifiers) for each of two witnesses of
// one public statement, and in it a directory for each run, with the files that message_dump()
// wrote. All runs need fresh coins and the same challenges (see FreshFieldCoins).
// The view is the vector of field elements that a run sent. A set W of its entries that is affine
// in the coins is uniform on base + span (its hull), and a few more runs than |W| give the span.
// Two witnesses have the same distribution on W if and only if the spans are equal and contain the
// bases' difference. The same runs serve all sets of up to (runs - margin) entries: see look().
// A pass says nothing about: a relation that no tested set contains whole; entries of degree above
// one in the coins (equal hulls are then necessary, not sufficient); digests, salts, lengths, time.
// Exit status 0: no set tells the witnesses apart; 1: some set does; 2: wrong usage, or nothing can
// be said (for example, the challenges differ or the runs are too few).
// Call: hull <folder of witness 0> <folder of witness 1> [-blind <n>] [-wide <n>] [-margin <n>: 20]
//   [-runs <n>] [-from <n>] [-v]. Without arguments, or with -v alone: own_test().
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>

#include "chain/tests/death.h"
#include "engine/engine_cpu.h"
#include "iop/message_dump.h"
#include "tests/hiding/exact.h"
#include "toy/product.h"
#include "toy/segments.h"
#include "toy/toy_coins.h"

using namespace blk;

namespace {

struct Mark {
  size_t entry; // the number of entries of the view before it
  size_t file;
  std::string words;
};
struct Run {
  std::vector<F> view;
  std::string frame;      // what must be the same in all runs
  size_t long_bytes = 0;  // items of plain bytes that are longer than 8
  std::vector<size_t> salts;   // for each of these: how many entries of the view come before it,
  std::vector<size_t> salt_file; // and which file it is in
  std::vector<bool> scalar;    // for each entry: it belongs to an element of E
  std::vector<Mark> marks;
  std::vector<size_t> file_begins; // entry at which each file begins, and the end
  std::vector<std::string> names;
};

std::vector<std::string> sorted(const std::string &folder, bool folders, const char *ending) {
  std::vector<std::string> out;
  for (const auto &e : std::filesystem::directory_iterator(folder))
    if (e.is_directory() == folders && (!ending || e.path().extension() == ending)) out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

// An item's kind (Writer::Item): F, E: an element of the base or extension field; D: digest;
// W: 64-bit word; B: plain bytes.
Run read_run(const std::string &folder) {
  Run r;
  for (const std::string &list : sorted(folder, false, ".list")) {
    const std::string stem = list.substr(0, list.size() - 5), name = std::filesystem::path(stem).filename().string();
    std::ifstream b(stem + ".bin", std::ios::binary);
    BLK_REQUIRE(b.good(), "hull: a list without its bytes");
    const std::string bytes((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
    r.file_begins.push_back(r.view.size()), r.names.push_back(name);
    r.frame += "file " + name + "\n";
    std::vector<std::pair<size_t, size_t>> begins; // (byte, entry) of every item
    std::vector<std::pair<size_t, std::string>> marks;
    std::ifstream l(list);
    for (std::string line; std::getline(l, line);) {
      std::istringstream s(line);
      char what = 0;
      s >> what;
      if (what == 'M') {
        size_t at = 0;
        s >> at;
        std::string words;
        std::getline(s, words);
        marks.emplace_back(at, words.empty() ? words : words.substr(1));
        continue;
      }
      char kind = 0;
      size_t at = 0, n = 0;
      s >> kind >> at >> n;
      BLK_REQUIRE(what == 'I' && !s.fail() && at + n <= bytes.size(), "hull: a line of a list is not understood");
      begins.emplace_back(at, r.view.size());
      r.frame += line + "\n";
      const u8 *p = (const u8 *)bytes.data() + at;
      if (kind == 'F' || kind == 'E') {
        for (size_t c = 0; c < n / F_BYTES; c++) {
          F x;
          BLK_REQUIRE(F_from_bytes(p + c * F_BYTES, x), "hull: an element is not in canonical form");
          r.view.push_back(x);
          r.scalar.push_back(kind == 'E');
        }
      } else if (kind == 'W' || (kind == 'B' && n <= 8)) {
        r.frame.append((const char *)p, n);
      } else if (kind == 'B') {
        r.long_bytes++;
        r.salts.push_back(r.view.size()), r.salt_file.push_back(r.names.size() - 1);
      }
    }
    for (const auto &m : marks) {
      const auto it = std::lower_bound(begins.begin(), begins.end(), std::make_pair(m.first, size_t(0)));
      r.marks.push_back(Mark{it == begins.end() ? r.view.size() : it->second, r.names.size() - 1, m.second});
      r.frame += "mark " + std::to_string(m.first) + " " + m.second + "\n";
    }
  }
  r.file_begins.push_back(r.view.size());
  std::stable_sort(r.marks.begin(), r.marks.end(), [](const Mark &a, const Mark &b) { return a.entry < b.entry; });
  return r;
}

struct Witness {
  std::vector<std::vector<F>> view;
};

using Set = std::vector<size_t>;

// base: a witness's run 0; span: of its other runs' differences from it
struct Judged {
  bool same;
  size_t one, two, together, with_bases;
};
Judged judge(const Witness &a, const Witness &b, const Set &w) {
  auto diff = [&w](const std::vector<F> &x, const std::vector<F> &y) {
    std::vector<F> d(w.size());
    for (size_t j = 0; j < w.size(); j++) d[j] = x[w[j]] - y[w[j]];
    return d;
  };
  Echelon one(w.size(), w.size()), two(w.size(), w.size()), both(w.size(), w.size());
  for (size_t i = 1; i < a.view.size(); i++) one.add(diff(a.view[i], a.view[0])), both.add(diff(a.view[i], a.view[0]));
  for (size_t i = 1; i < b.view.size(); i++) two.add(diff(b.view[i], b.view[0])), both.add(diff(b.view[i], b.view[0]));
  Judged j{false, one.rank(), two.rank(), both.rank(), 0};
  both.add(diff(b.view[0], a.view[0]));
  j.with_bases = both.rank();
  j.same = j.one == j.two && j.two == j.together && j.with_bases == j.together;
  return j;
}

// For a set with more entries than one witness has runs: tests only whether the bases' difference
// lies in both spans together.
Judged judge_wide(const Witness &a, const Witness &b, const Set &w) {
  auto diff = [&w](const std::vector<F> &x, const std::vector<F> &y) {
    std::vector<F> d(w.size());
    for (size_t j = 0; j < w.size(); j++) d[j] = x[w[j]] - y[w[j]];
    return d;
  };
  Echelon both(w.size(), w.size());
  for (size_t i = 1; i < a.view.size(); i++) both.add(diff(a.view[i], a.view[0]));
  for (size_t i = 1; i < b.view.size(); i++) both.add(diff(b.view[i], b.view[0]));
  Judged j{false, both.rank(), both.rank(), both.rank(), 0};
  both.add(diff(b.view[0], a.view[0]));
  j.with_bases = both.rank();
  j.same = j.with_bases == j.together;
  return j;
}

u64 word_of(F x) {
  u8 b[8] = {};
  static_assert(F_BYTES <= 8, "an element of the base field fits a word");
  F_to_bytes(x, b);
  u64 v = 0;
  for (int i = 0; i < 8; i++) v |= (u64)b[i] << (8 * i);
  return v;
}

std::string stretches(const Set &s) { // "3-7, 12, 40-41": at most eight of them
  std::string out;
  size_t shown = 0;
  for (size_t i = 0; i < s.size();) {
    size_t j = i;
    while (j + 1 < s.size() && s[j + 1] == s[j] + 1) j++;
    if (shown++ == 8) return out + ", ..";
    out += (out.empty() ? "" : ", ") + std::to_string(s[i]) + (j > i ? "-" + std::to_string(s[j]) : "");
    i = j + 1;
  }
  return out;
}

bool silent = false;  // set by `own_test` without -v: SAY then prints nothing
#define SAY(...) ((void)(silent || std::printf(__VA_ARGS__)))

// Options. -margin n (default 20): outside the step "wide", a set has at most k - n entries, where
// k is the lesser of the two witnesses' numbers of runs; these n are the "few more runs" of the
// comment at the top of this file. Status 2 unless n >= 2 and k >= n + 4, and k >= 30 if n is 20.
// -from n: skip each witness's first n runs, in the order of the directories' names.
// -runs n: read at most n of the rest.
// -blind n, -wide n: add the steps "blind" and "wide". -v: print a line for every set.
// Steps, in order: frame, still, pairs, plan, leaves, blind, wide, whole.
int look(int argc, const char *const *argv) {
  size_t blind = 0, wide = 0, margin = 20, most_runs = 0, from_run = 0;
  bool verbose = false;
  std::vector<std::string> folders;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "-blind") && i + 1 < argc) blind = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "-wide") && i + 1 < argc) wide = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "-margin") && i + 1 < argc) margin = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "-runs") && i + 1 < argc) most_runs = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "-from") && i + 1 < argc) from_run = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "-v")) verbose = true;
    else folders.push_back(argv[i]);
  }
  if (folders.size() != 2) return std::fprintf(stderr, "usage: hull <folder of witness 0> <folder of witness 1> [-blind <n>] [-wide <n>] [-margin <n>] [-runs <n>] [-from <n>] [-v]\n"), 2;

  // frame: the only check on the challenges, and only through sent items that change with them
  // (proof-of-work nonces, listed positions); fresh coins are not checked
  Witness w[2];
  Run first;
  for (int x = 0; x < 2; x++) {
    size_t passed = 0;
    for (const std::string &folder : sorted(folders[x], true, nullptr)) {
      if (passed++ < from_run) continue;
      if (most_runs && w[x].view.size() == most_runs) break; // runs go by name, never by content
      Run r = read_run(folder);
      if (w[0].view.empty()) first = r;
      if (r.frame != first.frame) {
        size_t at = 0;
        while (at < r.frame.size() && at < first.frame.size() && r.frame[at] == first.frame[at]) at++;
        const size_t file = first.frame.rfind("file ", at), item = first.frame.rfind("\nI ", at);
        auto line = [](const std::string &frame, size_t from) { return from == std::string::npos ? std::string("?") : frame.substr(from, std::min(size_t(60), frame.find('\n', from) - from)); };
        const size_t next = item == std::string::npos ? item : first.frame.find('\n', item + 1);
        SAY("hull: NOTHING CAN BE SAID: the frame of %s is not that of the first run: they part in %s, in or behind the item \"%s\" (what follows: \"%s\" there, \"%s\" here): the verifier's messages have moved, or the "
                    "statement is another\n",
                    folder.c_str(), line(first.frame, file).c_str(), line(first.frame, item == std::string::npos ? item : item + 1).c_str(), line(first.frame, next == std::string::npos ? next : next + 1).c_str(),
                    line(r.frame, next == std::string::npos ? next : next + 1).c_str());
        return 2;
      }
      w[x].view.push_back(std::move(r.view));
    }
  }
  const size_t m = first.view.size(), k = std::min(w[0].view.size(), w[1].view.size());
  if (margin < 2 || k < margin + 4 || (margin == 20 && k < 30) || !m) return SAY("hull: NOTHING CAN BE SAID: %zu runs, %zu entries, a margin of %zu\n", k, m, margin), 2;
  const size_t widest = k - margin;
  SAY("hull: frame: the same in %zu + %zu runs: %zu files, %zu entries of the view, %zu marks; %zu items of plain bytes longer than 8 and all digests are not looked at; a set has at most %zu entries\n",
              w[0].view.size(), w[1].view.size(), first.names.size(), m, first.marks.size(), first.long_bytes, widest);
  size_t apart = 0, less = 0; // less: the current step's sets of less than full rank for a witness
  auto look = [&](const char *kind, const std::string &name, Set s, size_t &sets, size_t &told) {
    // A set that is too wide is cut into pieces that overlap by half.
    for (size_t from = 0; from < s.size(); from += widest / 2) {
      const Set piece(s.begin() + from, s.begin() + std::min(s.size(), from + widest));
      const Judged j = judge(w[0], w[1], piece);
      sets++;
      if (j.one < piece.size() || j.two < piece.size()) less++;
      if (!j.same) told++, apart++;
      if ((!j.same && told <= 12) || verbose)
        SAY("hull: %s: %s: entries %s: %s (dimensions %zu, %zu, together %zu, with the bases' difference %zu; %zu entries)\n", kind, name.c_str(), stretches(piece).c_str(),
                    j.same ? "the same hull" : "TOLD APART", j.one, j.two, j.together, j.with_bases, piece.size());
      if (from + widest >= s.size()) break;
    }
  };

  // still: entries that no coin moves
  std::vector<bool> moves(m, false);
  {
    Set fixed, told;
    for (size_t j = 0; j < m; j++) {
      bool still[2] = {true, true};
      for (int x = 0; x < 2; x++)
        for (size_t i = 1; i < w[x].view.size() && still[x]; i++) still[x] = w[x].view[i][j] == w[x].view[0][j];
      moves[j] = !still[0] && !still[1];
      if (still[0] && still[1] && w[0].view[0][j] == w[1].view[0][j]) fixed.push_back(j);
      else if (still[0] || still[1]) told.push_back(j);
    }
    apart += told.size();
    SAY("hull: still: %zu entries are moved by no coin and are the same for both witnesses%s%s; %zu are moved by no coin for one witness at least and tell them apart%s%s\n", fixed.size(), fixed.empty() ? "" : ": ",
                stretches(fixed).c_str(), told.size(), told.empty() ? "" : ": TOLD APART: ", stretches(told).c_str());
  }

  // pairs: entries that move in proportion, as when one coin masks them all ("under one mask" in
  // the output)
  {
    constexpr size_t R = 4;
    std::vector<std::pair<std::array<u64, R>, size_t>> key;
    for (size_t j = 0; j < m; j++) {
      if (!moves[j]) continue;
      std::array<F, R> d;
      F scale = F::zero();
      for (size_t i = 0; i < R; i++) {
        d[i] = w[0].view[i + 1][j] - w[0].view[0][j];
        if (scale == F::zero() && d[i] != F::zero()) scale = inv(d[i]);
      }
      if (scale == F::zero()) continue; // moves, but not in these runs: left to later steps
      std::array<u64, R> v;
      for (size_t i = 0; i < R; i++) v[i] = word_of(d[i] * scale);
      key.emplace_back(v, j);
    }
    std::sort(key.begin(), key.end());
    size_t sets = 0, told = 0, entries = 0;
    for (size_t i = 0; i < key.size();) {
      size_t j = i;
      Set s;
      while (j < key.size() && key[j].first == key[i].first) s.push_back(key[j++].second);
      if (s.size() > 1) entries += s.size(), look("pairs", "one mask", s, sets, told);
      i = j;
    }
    SAY("hull: pairs: %zu entries stand in %zu sets under one mask each; %zu sets tell the witnesses apart\n", entries, sets, told);
    less = 0;
  }

  // plan: the sets that the files' marks (Writer::mark()) name
  {
    size_t sets = 0, told = 0;
    std::map<std::string, Set> trees; // all openings of one tree of one file
    std::map<std::string, std::vector<Set>> rows; // by row: entry r of every position, all openings
    std::vector<bool> opened(m, false);
    size_t in_openings = 0, in_no_row = 0;
    for (size_t i = 0; i < first.marks.size(); i++) {
      const Mark &a = first.marks[i];
      const size_t end = i + 1 < first.marks.size() && first.marks[i + 1].file == a.file ? first.marks[i + 1].entry : first.file_begins[a.file + 1];
      Set s;
      for (size_t j = a.entry; j < end; j++) s.push_back(j);
      if (s.empty()) continue;
      const size_t tree = a.words.find("opening of tree ");
      if (tree != std::string::npos) {
        Set &all = trees[first.names[a.file] + ": " + a.words.substr(tree, a.words.find(" at ", tree) - tree)];
        all.insert(all.end(), s.begin(), s.end());
        for (size_t j : s) opened[j] = true;
        // An opening shows the positions of one list in turn, each with the same number of entries.
        const size_t at = a.words.find(" at ", tree), positions = at == std::string::npos ? 0 : std::strtoull(a.words.c_str() + at + 4, nullptr, 10);
        in_openings += s.size();
        if (positions > 1 && s.size() % positions == 0) {
          const size_t width = s.size() / positions;
          std::vector<Set> &by_row = rows[first.names[a.file] + ": " + a.words.substr(tree, at - tree) + " of width " + std::to_string(width)];
          by_row.resize(width);
          for (size_t j = 0; j < s.size(); j++) by_row[j % width].push_back(s[j]);
        } else {
          in_no_row += s.size();
        }
      }
      look("plan", first.names[a.file] + ": behind the mark \"" + a.words + "\"", s, sets, told);
    }
    for (const auto &t : trees) look("plan", t.first + ", all lists", t.second, sets, told);
    for (const auto &t : rows)
      for (size_t r = 0; r < t.second.size(); r++) look("plan", t.first + ", entry " + std::to_string(r) + " of every position, all lists", t.second[r], sets, told);
    for (size_t f = 0; f < first.names.size(); f++) {
      Set outside;
      for (size_t j = first.file_begins[f]; j < first.file_begins[f + 1]; j++)
        if (!opened[j]) outside.push_back(j);
      if (outside.empty()) continue;
      look("plan", first.names[f] + ": outside all openings", outside, sets, told);
      for (const auto &t : trees) {
        if (t.first.compare(0, first.names[f].size() + 2, first.names[f] + ": ")) continue;
        Set s = outside;
        s.insert(s.end(), t.second.begin(), t.second.end());
        if (s.size() <= widest) look("plan", first.names[f] + ": outside all openings, with " + t.first.substr(first.names[f].size() + 2), s, sets, told);
      }
    }
    SAY("hull: plan: %zu sets named by the marks; %zu had less than full rank; %zu tell the witnesses apart; %zu of the view's %zu entries lie in marked openings, %zu of these in no row set\n", sets, less, told,
                in_openings, m, in_no_row);
    less = 0;
  }

  // leaves: the sets by row again, without marks: salts at equal distances end leaves of one width
  {
    size_t sets = 0, told = 0, runs = 0, covered = 0;
    const std::vector<size_t> &salt = first.salts;
    for (size_t i = 1; i < salt.size();) {
      const size_t width = salt[i] - salt[i - 1], file = first.salt_file[i];
      if (first.salt_file[i - 1] != file) {
        i++;
        continue;
      }
      size_t j = i;
      while (j + 1 < salt.size() && first.salt_file[j + 1] == file && salt[j + 1] - salt[j] == width) j++;
      // The leaf that ends at salt[i - 1] counts too if `width` entries fit between it and the salt
      // before it, or its file's beginning.
      if (width > 0 && j > i) {
        std::vector<size_t> ends;
        if (salt[i - 1] >= first.file_begins[file] + width && (i < 2 || first.salt_file[i - 2] != file || salt[i - 1] - salt[i - 2] >= width)) ends.push_back(salt[i - 1]);
        for (size_t l = i; l <= j; l++) ends.push_back(salt[l]);
        runs++, covered += ends.size() * width;
        for (size_t r = 0; r < width; r++) {
          Set s;
          for (size_t e : ends) s.push_back(e - width + r);
          look("leaves", first.names[file] + ": " + std::to_string(ends.size()) + " leaves of width " + std::to_string(width) + " that end at entry " + std::to_string(ends.back()) + ", place " + std::to_string(r), s, sets, told);
        }
      }
      i = j + 1;
    }
    SAY("hull: leaves: %zu runs of leaves of one width, %zu entries in them; %zu sets; %zu had less than full rank; %zu tell the witnesses apart\n", runs, covered, sets, less, told);
    less = 0;
  }

  // blind: sets of n neighbours
  if (blind) {
    blind = std::min(blind, widest);
    size_t sets = 0, told = 0;
    for (size_t from = 0; from < m; from += blind / 2) {
      Set s;
      for (size_t j = from; j < std::min(m, from + blind); j++) s.push_back(j);
      look("blind", "neighbours", s, sets, told);
      if (from + blind >= m) break;
    }
    SAY("hull: blind: %zu sets of %zu neighbours; %zu had less than full rank; %zu tell the witnesses apart\n", sets, blind, less, told);
    less = 0;
  }

  // wide: sets of up to n neighbours, n at most (both witnesses' runs together - margin - 2), first
  // in the whole view, then among the entries of elements of E alone; tested by judge_wide()
  if (wide) {
    const size_t most = w[0].view.size() + w[1].view.size() - margin - 2;
    wide = std::min(wide, most);
    auto walk = [&](const char *name, const Set &all) {
      size_t sets = 0, told = 0, short_of = 0;
      for (size_t from = 0; from < all.size(); from += wide / 2) {
        const Set s(all.begin() + from, all.begin() + std::min(all.size(), from + wide));
        const Judged j = judge_wide(w[0], w[1], s);
        sets++;
        if (j.together < s.size()) short_of++;
        if (!j.same) told++, apart++;
        if ((!j.same && told <= 12) || verbose)
          SAY("hull: wide: %s: entries %s: %s (both witnesses' differences together: dimension %zu, with the bases' difference %zu; %zu entries)\n", name, stretches(s).c_str(),
                      j.same ? "the bases' difference lies in the span" : "TOLD APART", j.together, j.with_bases, s.size());
        if (from + wide >= all.size()) break;
      }
      SAY("hull: wide: %s: %zu sets of up to %zu entries; %zu had less than full rank; %zu tell the witnesses apart\n", name, sets, wide, short_of, told);
    };
    Set all(m), scalars;
    for (size_t j = 0; j < m; j++) {
      all[j] = j;
      if (first.scalar[j]) scalars.push_back(j);
    }
    walk("neighbours", all);
    walk("neighbours among the elements of E", scalars);
  }

  // whole: all entries of the view as one set; skipped if there are more than `widest`
  if (m <= widest) {
    Set s(m);
    for (size_t j = 0; j < m; j++) s[j] = j;
    size_t sets = 0, told = 0;
    verbose = true;
    look("whole", "the whole view", s, sets, told);
  } else {
    SAY("hull: whole: not looked at: %zu entries want %zu runs\n", m, m + margin);
  }
  SAY("hull: %s\n", apart ? "TOLD APART" : "no set that was looked at tells the two witnesses apart");
  return apart ? 1 : 0;
}

// ---- its own test ----

Setting tiny_setting() {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 10; // bits: no security level; hiding does not depend on it
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 6);
  return set;
}

// Field elements are fresh in every run. Bytes (salts, and a hash chain's 32 random bytes, from
// which its challenges start) are the same in all runs, and under
// BLK_TEST_PROVER_MESSAGES_OUTSIDE nothing that is sent enters a hash chain: so the challenges are
// the same too.
class FreshFieldCoins : public Coins {
public:
  explicit FreshFieldCoins(u64 run) : field_(run), bytes_(7) {}
  void fill(F *out, size_t n) override { field_.fill(out, n); }
  void bytes(u8 *out, size_t n) override { bytes_.bytes(out, n); }

private:
  ToyCoins field_, bytes_;
};

// Writes both witnesses' runs of the prover `world`.
void lay(const std::string &out, const std::string &world, size_t runs) {
  const Setting set = tiny_setting();
  const ToyRun one = toy_size(set, 4, 7, 7, {StageChoice{1, 1, 8, 8}, StageChoice{0, 0, 9, 9}}, "product, tiny edition: a test size that reaches no security level");
  const SegmentsRun chain = segments_size(set, 3, 6, 9, 9, {StageChoice{1, 1, 9, 9}, StageChoice{0, 0, 10, 10}}, "links, tiny edition: a test size that reaches no security level");
  for (unsigned witness = 0; witness < 2; witness++)
    for (size_t run = 0; run < runs; run++) {
      char name[64];
      std::snprintf(name, sizeof name, "/witness%u/run%06zu", witness, run);
      std::filesystem::create_directories(out + name);
      FreshFieldCoins coins(1000 + run + 1000000 * witness);
      EngineCpu engine;
      Writer proof;
      proof.keep_list();
      if (world == "product") {
        Table abc(1, 3, one.shape.codes.base[0].ell);
        ToyCoins(100 + witness).fill(abc.a.data(), 2 * abc.len);
        for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
        toy_prove(one, abc, engine, coins, proof);
      } else if (world == "links") {
        std::vector<F> x(size_t(1) << chain.vars());
        std::vector<std::vector<F>> b(chain.links(), x);
        ToyCoins from(100 + witness);
        from.fill(x.data(), x.size());
        for (auto &v : b) from.fill(v.data(), v.size());
        segments_prove(chain, x, b, engine, coins, proof);
      } else {
        // 300 values of the witness, each under a coin of its own; entry 5 is a constant. Entry
        // 250, too far from entry 10 for a set of neighbours, is three times entry 10, which tells
        // nothing new ("made-up-lawful"), or another value under entry 10's coin (the fault).
        std::vector<F> value(300), mask(300);
        ToyCoins(100 + witness).fill(value.data(), value.size());
        coins.fill(mask.data(), mask.size());
        const F three = F::one() + F::one() + F::one();
        for (size_t j = 0; j < value.size(); j++)
          proof.put(j == 5 ? three : j != 250 ? value[j] + mask[j] : world == "made-up-lawful" ? three * (value[10] + mask[10]) : value[j] + mask[10]);
      }
      message_dump(proof, out + name + "/proof");
    }
}

// message_dump() (blocks/iop/message_dump.h) creates its two files. Three scenes below `work`, each
// call in a child process: a file stands at <stem>.bin; <stem>.list is a symbolic link to a file;
// <stem>.bin is a symbolic link to nothing. Each call must end the process with its message, and
// what stood there must be as it was (the call may leave an empty file at the other name).
// Not shown: a file at <stem>.list, a link to a file at <stem>.bin, or a name that is taken between
// a look and the call. Returns the number of scenes that came out otherwise.
int standing_names(const std::filesystem::path &work) {
  Writer sent;
  sent.keep_list();
  sent.put(F::one());
  const auto text = [](const std::filesystem::path &file) {
    std::ifstream in(file);
    std::stringstream all;
    all << in.rdbuf();
    return all.str();
  };
  const auto refused = [&](const char *stem) {
    return ends_with("message_dump: a file cannot be created", [&] { message_dump(sent, (work / stem).string()); });
  };
  const auto differs = [](const char *scene) { std::printf("hull: its own test: message_dump() with %s\n", scene); };
  int wrong = 0;
  std::ofstream(work / "stands.bin") << "mine";
  if (!refused("stands") || text(work / "stands.bin") != "mine")
    wrong++, differs("a file at <stem>.bin: not refused as it must be, or the file has changed");
  // The links' texts are names beside the links, so they point inside `work` whatever TMPDIR says.
  std::ofstream(work / "target") << "mine";
  std::filesystem::create_symlink("target", work / "linked.list");
  if (!refused("linked") || text(work / "target") != "mine")
    wrong++, differs("a link to a file at <stem>.list: not refused as it must be, or the file has changed");
  std::filesystem::create_symlink("nowhere", work / "dangling.bin");
  if (!refused("dangling") || std::filesystem::exists(work / "nowhere"))
    wrong++, differs("a dangling link at <stem>.bin: not refused as it must be, or a file was made through it");
  return wrong;
}

// Runs look() on small provers, then standing_names().
int own_test(bool with_the_steps) {
  silent = !with_the_steps;
  struct Case {
    const char *world;
    int wanted;
  };
  const std::vector<Case> cases = {{"product", 0}, {"links", 0}, {"made-up-lawful", 0}, {"made-up-fault", 1}};
  std::string made = (std::filesystem::temp_directory_path() / "hull-own-test-XXXXXX").string();
  BLK_REQUIRE(mkdtemp(made.data()), "hull: no folder for its own test can be made"); // the call makes the folder or fails: what is removed below is this process's own
  const std::filesystem::path work = made;
  int failures = 0;
  for (const Case &c : cases) {
    const std::string folder = (work / c.world).string(), zero = folder + "/witness0", one = folder + "/witness1";
    lay(folder, c.world, 40);
    const char *call[] = {"hull", zero.c_str(), one.c_str(), "-margin", "4", "-blind", "36", "-wide", "74"};
    const int status = look(9, call);
    if (status != c.wanted) failures++, std::printf("hull: its own test: %s: status %d, wanted %d\n", c.world, status, c.wanted);
  }
  failures += standing_names(work);
  std::filesystem::remove_all(work);
  std::printf("hull: %s (its own test: %zu small provers, 40 + 40 runs each)\n", failures ? "FAILED" : "ok", cases.size());
  return failures ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  const bool v_alone = argc == 2 && !std::strcmp(argv[1], "-v");
  return argc == 1 || v_alone ? own_test(v_alone) : look(argc, argv);
}
