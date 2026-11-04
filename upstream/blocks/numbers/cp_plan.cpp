// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* cp_plan computes parameters, round errors and message sizes from a text file that gives a setting
   (`Setting`: byte widths, target, proof of work) and lists a protocol's
   blocks (steps). It runs no protocol and reads no proof. The field is Goldilocks.

     cp_plan FILE [--rounds] [--messages]
       --rounds    print every group of rounds with a non-zero error, largest first, work counted
       --messages  also print every message by name and place (digit runs as '#'): count, bytes

   FILE has lines "word key=value ..."; '#' starts a comment. blocks/numbers/settings/README.md
   describes every line. The blocks are planned by the functions of blocks/numbers/blocks.h, which
   prover and verifier use too, so equal shapes give equal figures. Nothing checks that FILE matches
   what they run. The one test, blocks/numbers/tests/cp_plan_files.py, checks the exit status and
   the start and order of some printed lines; of the printed figures it checks only the field's p
   and the target, and it runs no option.

   Exit status: 0 only if every round's error is at most 2^-target, proof of work counted, rounds on
   the same main and mask code have one distance (delta, the proximity radius at which their error
   bounds are computed; see `run`) and every hiding budget (`Budget`) is met; 2 if FILE or the
   command line cannot be read; otherwise 3, also if planning rejects the parameters. A FILE of
   codes and groups alone plans no round and gets 0; the line "the plan's own verdict:" says "none".
   A failed internal check (`BLK_REQUIRE`, blocks/base.h) aborts instead. */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "numbers/accounting.h"
#include "numbers/blocks.h"

using namespace blk;

namespace {

struct Line {
  int number = 0;
  std::string word;
  std::vector<std::string> bare;
  std::map<std::string, std::string> kv;
};

[[noreturn]] void fail(const Line &l, const std::string &why) {
  std::fprintf(stderr, "cp_plan: line %d (%s): %s\n", l.number, l.word.c_str(), why.c_str());
  std::exit(2);
}

// A number below 2^63: digits, 2^a, or a sum or difference of these; no partial sum is below 0.
u64 number(const Line &l, const std::string &s) {
  const u64 most = (u64)1 << 63;
  u64 total = 0;
  int sign = 1;
  if (s.empty() || s.back() == '+' || s.back() == '-') fail(l, "not a number: '" + s + "'");
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '+' || s[i] == '-') {
      if (!i || s[i - 1] == '+' || s[i - 1] == '-') fail(l, "not a number: " + s);
      sign = s[i] == '+' ? 1 : -1;
      ++i;
      continue;
    }
    u64 x = 0;
    size_t j = i;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
      if (x > most / 10 - 1) fail(l, "number too large: " + s);
      x = 10 * x + (u64)(s[j++] - '0');
    }
    if (j == i) fail(l, "not a number: " + s);
    if (j < s.size() && s[j] == '^') {
      u64 a = 0;
      size_t q = ++j;
      while (j < s.size() && s[j] >= '0' && s[j] <= '9') a = 10 * a + (u64)(s[j++] - '0');
      if (q == j || j - q > 2 || x != 2 || a > 62) fail(l, "only powers 2^a with a < 63: " + s);
      x = (u64)1 << a;
    }
    if (sign > 0 ? x >= most - total : x > total) fail(l, "number too large or below 0: " + s);
    total = sign > 0 ? total + x : total - x;
    i = j;
  }
  return total;
}
std::vector<std::string> split(const std::string &s, char c) {
  std::vector<std::string> o;
  std::string t;
  std::istringstream in(s);
  while (std::getline(in, t, c)) o.push_back(t);
  return o;
}

// A committed table (`table` line). A stored copy is one commitment to it under its own masks; the
// committer is the party that makes the copies (cp_commit).
struct Table {
  TableSize size;
  u64 spent = 0;  // stored copy's columns opened by the committer's last `lin one_time=1` or `link`
};

/* Parsing is strict, to catch typing mistakes: an unknown word or name, a key or word that the line
   does not take (not checked on the `end` that closes a repeat), a repeated key, a malformed or too
   large number, or a setting line after any other line gives exit status 2. */
struct Run {
  std::vector<Line> lines;
  Setting set;
  std::map<std::string, CodeSize> codes;
  std::map<std::string, GroupSize> groups;
  std::map<std::string, Scope> group_scope;
  std::map<std::string, GroupLoad> group_load;  // what the line requires of the group's rounds
  std::map<std::string, bool> group_named;      // FILE sets the distance; the plan tests its rounds
  std::map<std::string, TreeSize> trees;
  std::map<std::string, Table> tables;
  std::vector<std::pair<std::string, std::vector<std::string>>> sums;
  std::vector<std::string> notes;
  std::vector<std::string> group_order;
  std::set<std::string> work_lines;
  bool setting_closed = false, budgets_met = true;
  std::string part = "proof";
  mutable std::set<std::string> asked;   // the current line's keys that were asked for
  mutable size_t bare_asked = 0;

  std::string get(const Line &l, const char *k) const {
    asked.insert(k);
    auto i = l.kv.find(k);
    if (i == l.kv.end()) fail(l, std::string("needs ") + k + "=");
    return i->second;
  }
  std::string get(const Line &l, const char *k, const std::string &d) const {
    asked.insert(k);
    auto i = l.kv.find(k);
    return i == l.kv.end() ? d : i->second;
  }
  u64 num(const Line &l, const char *k) const { return number(l, get(l, k)); }
  u64 num(const Line &l, const char *k, u64 d) const { return asked.insert(k), l.kv.count(k) ? number(l, get(l, k)) : d; }
  // Key k's value, which must not exceed `most`.
  unsigned small(const Line &l, const char *k, u64 most = 1 << 20) const { return within(l, num(l, k), most); }
  unsigned small(const Line &l, const char *k, u64 d, u64 most) const { return within(l, num(l, k, d), most); }
  unsigned within(const Line &l, u64 x, u64 most) const {
    if (x > most) fail(l, "number too large here: " + std::to_string(x));
    return (unsigned)x;
  }
  bool flag(const Line &l, const char *k, u64 d) const { return within(l, num(l, k, d), 1) != 0; }
  // The line's i-th word without a key.
  const std::string &bare(const Line &l, size_t i = 0) const {
    if (i >= l.bare.size()) fail(l, "a word is missing after " + l.word);
    bare_asked = std::max(bare_asked, i + 1);
    return l.bare[i];
  }
  // Fails if l has a key or word that nothing asked for.
  void all_read(const Line &l) const {
    for (auto &kv : l.kv)
      if (!asked.count(kv.first)) fail(l, "this line takes no key " + kv.first);
    if (l.bare.size() > bare_asked) fail(l, "a word too many: " + l.bare[bare_asked]);
  }
  template <class T>
  T &find_to_change(const Line &l, std::map<std::string, T> &m, const std::string &name) const {
    auto i = m.find(name);
    if (i == m.end()) fail(l, "unknown name " + name);
    return i->second;
  }
  template <class T>
  const T &find(const Line &l, const std::map<std::string, T> &m, const std::string &name) const {
    auto i = m.find(name);
    if (i == m.end()) fail(l, "unknown name " + name);
    return i->second;
  }
  Scope scope_of(const Line &l, const std::string &s) const {
    if (s == "request") return Scope::REQUEST;
    if (s == "shared") return Scope::SHARED;
    if (s == "once") return Scope::ONCE;
    fail(l, "scope is request, shared or once");
  }
  std::vector<TreeSize> tree_list(const Line &l, const std::string &s) const {
    std::vector<TreeSize> o;
    for (auto &n : split(s, ',')) o.push_back(find(l, trees, n));
    return o;
  }
  SampleSize sample(const Line &l, const std::string &s) const {
    auto x = split(s, ':');
    if (x.size() != 2) fail(l, "a sample is group:cube_bits");
    return SampleSize{within(l, number(l, x[1]), 64), {find(l, groups, x[0])}};
  }
  // A distance that a line gives as a share of a code's length, in whole positions, rounded down.
  u64 positions_of(const Line &l, const std::string &s, const CodeSize &c) const { return (fraction(l, s) * Bound(Big(c.n()))).lo.floor().to_u64(); }
  Bound fraction(const Line &l, const std::string &s) const {
    auto x = split(s, '/');
    if (x.size() != 2 || !number(l, x[1])) fail(l, "a fraction is a/b");
    return Bound::ratio(Big(number(l, x[0])), Big(number(l, x[1])));
  }
  /* The hiding budget of one code in one part of FILE: met if the code's random coordinates hide
     `drawn` positions. The draws of the `loc` and `iopp` lines whose group has the code add up, as
     if all opened the same rows; no other line counts (`lin` only prints a note), and `iopp`
     ignores `opens` for a mask code. A line inside a repeat counts once, as if its rows were fresh
     each time. */
  struct Budget {
    std::string part;
    CodeSize code;
    u64 drawn = 0;
  };
  std::map<std::string, Budget> budgets;
  void budget(const CodeSize &c, u64 draws) {
    Budget &b = budgets[part + "/" + c.name];
    b.part = part, b.code = c, b.drawn += draws;
  }
  void budgets_to_notes() {
    for (auto &x : budgets) {
      const Budget &b = x.second;
      const u64 hidden = hidden_positions(set, b.code);
      if (hidden < b.drawn) budgets_met = false;
      notes.push_back("hiding budget: " + b.part + ": rows of code " + b.code.name + " have " + std::to_string(b.code.r) + " random coordinates, which hide " +
                      std::to_string(hidden) + " positions, and are opened at " + std::to_string(b.drawn) + " drawn positions in all: " +
                      (hidden >= b.drawn ? "enough" : "TOO FEW"));
    }
  }

  static bool of_the_setting(const std::string &word) {
    static const std::set<std::string> words = {"field", "target", "instances", "samples", "saved_coefficient", "bytes", "work"};
    return words.count(word) != 0;
  }
  // Reads a setting, `code`, `group`, `tree`, `table` or `sum` line; else returns false.
  bool define(const Line &l) {
    const std::string &w = l.word;
    // One setting throughout: it sizes codes and groups, `Plan` copies it, run() prints it.
    if (of_the_setting(w) && setting_closed) fail(l, "the setting's lines stand before all others");
    auto one_of = [&](const char *a, const char *b) {
      if (bare(l) != a && bare(l) != b) fail(l, w + " " + a + " | " + b);
      return bare(l) == a;
    };
    if (w == "field") {
      if (bare(l) != "goldilocks") fail(l, "field goldilocks");  // the one field; the line may be left out
    } else if (w == "target") set.target = within(l, number(l, bare(l)), 512);
    else if (w == "instances") {
      if (!(set.instances = number(l, bare(l)))) fail(l, "at least one instance");
    } else if (w == "samples") {
      if (!(set.samples = within(l, number(l, bare(l)), 8))) fail(l, "at least one answer");
    } else if (w == "saved_coefficient") set.saved_coefficient = one_of("1", "0");
    else if (w == "bytes") {
      set.salt_bytes = small(l, "salt", set.salt_bytes, 1024), set.nonce_bytes = small(l, "nonce", set.nonce_bytes, 8);
      set.digest_bytes = small(l, "digest", set.digest_bytes, 64);
      set.hash_block_bytes = small(l, "block", set.hash_block_bytes, 1024);
      if (!set.hash_block_bytes || !set.digest_bytes) fail(l, "a width of 0");
    } else if (w == "work") {
      static const std::map<std::string, ErrorKind> kinds = {{"field", ErrorKind::FIELD}, {"sample", ErrorKind::SAMPLE},
          {"closeness", ErrorKind::CLOSENESS}, {"positions", ErrorKind::QUERY}, {"smallness", ErrorKind::SMALL}};
      const ErrorKind k = find(l, kinds, bare(l));
      if (!work_lines.insert(bare(l)).second) fail(l, "a second line for this kind of work");
      set.set_work(k, Scope::REQUEST, small(l, "request", 0, 64));
      set.set_work(k, Scope::SHARED, small(l, "shared", 0, 64));
      set.set_work(k, Scope::ONCE, small(l, "once", 0, 64));
    } else if (w == "code") {
      CodeSize c;
      c.name = bare(l), c.log_n = small(l, "log_n", 48), c.ell = num(l, "ell"), c.r = num(l, "r", 0);
      c.over_E = flag(l, "E", 0);
      if (!c.ell || c.ell > c.n() || c.r > c.n() || c.dim(set) > c.n()) fail(l, "message and randomness exceed the length, or there is no message");
      codes[c.name] = c;
    } else if (w == "group") {
      GroupSize g;
      g.name = bare(l), g.main = find(l, codes, get(l, "main"));
      if (get(l, "mask", "none") != "none") g.mask = find(l, codes, get(l, "mask"));
      g.k = small(l, "k", 0, 32), g.h = small(l, "h", 0, 32), g.slots = small(l, "slots", 1, 1 << 20);
      if (!g.slots) fail(l, "at least one slot");
      const Scope sc = scope_of(l, get(l, "scope", "request"));
      GroupLoad load;
      load.cube_bits = small(l, "cube", load.cube_bits, 64), load.batched_rows = num(l, "rows", load.batched_rows);
      load.messages = num(l, "messages", load.messages);
      const std::string regime = get(l, "regime", "johnson");
      const bool named = !regime.rfind("m:", 0) || !regime.rfind("delta:", 0);  // FILE's distance
      if (!regime.rfind("m:", 0)) g = group_of_m(g, number(l, regime.substr(2)));
      else if (!regime.rfind("delta:", 0)) g = group_at(g, positions_of(l, regime.substr(6), g.main));
      else if (regime == "johnson") g = size_group(set, sc, g, load);
      else fail(l, "a rule for the distance is johnson, m:N or delta:a/b");
      if (!groups.count(g.name)) group_order.push_back(g.name);
      groups[g.name] = g, group_scope[g.name] = sc, group_load[g.name] = load, group_named[g.name] = named;
    } else if (w == "tree") {
      TreeSize t;
      t.name = bare(l), t.log_n = small(l, "log_n", 48);
      t.leaf_bytes = (num(l, "rows_F", 0) + num(l, "copied_rows_F", 0)) * set.f_bytes + (num(l, "rows_E", 0) + num(l, "copied_rows_E", 0)) * set.e_bytes;
      trees[t.name] = t;
    } else if (w == "table") {
      Table T;
      T.size.code = find(l, codes, get(l, "code")), T.size.v = small(l, "v", 30);
      T.size.answers = small(l, "answers", 1, 8);
      if (!T.size.answers) fail(l, "answers must be from 1 to 8");
      T.size.distances.radius = positions_of(l, get(l, "delta_once"), T.size.code);
      T.size.distances.later_radius = positions_of(l, get(l, "delta_later", get(l, "delta_once")), T.size.code);
      tables[bare(l)] = T;
    } else if (w == "sum") {
      sums.push_back({bare(l), split(get(l, "of"), '+')});
    } else
      return false;
    return true;
  }

  // Reads the setting lines at the top of FILE; returns the next line's index.
  size_t head() {
    size_t at = 0;
    for (; at < lines.size() && of_the_setting(lines[at].word); ++at) {
      asked.clear(), bare_asked = 0;
      define(lines[at]);
      all_read(lines[at]);
    }
    setting_closed = true;
    return at;
  }
  size_t block(Plan &p, size_t at, size_t end) {
    for (; at < end; ++at) {
      const Line &l = lines[at];
      asked.clear(), bare_asked = 0;
      const std::string &w = l.word;
      if (define(l)) {
        all_read(l);
        continue;
      }
      if (w == "part") {
        p.begin_part(part = bare(l));
      } else if (w == "scope") {
        p.begin_scope(scope_of(l, bare(l)));
      } else if (w == "repeat") {
        size_t depth = 1, j = at + 1;
        for (; j < end && depth; ++j) depth += (lines[j].word == "repeat") - (lines[j].word == "end");
        if (depth) fail(l, "repeat without end");
        const u64 n = number(l, bare(l));
        if (!n) fail(l, "repeat at least once");
        all_read(l);
        p.push_repeat(n);
        block(p, at + 1, j - 1);
        p.pop_repeat();
        at = j - 1;
        continue;
      } else if (w == "end") {
        fail(l, "end without repeat");
      } else
        call(p, l, get(l, "where", w));
      all_read(l);
    }
    return at;
  }

  // Plans one protocol block. where: its label in the output.
  void call(Plan &p, const Line &l, const std::string &where) {
    const std::string &w = l.word;
    if (w == "chain") plan_chain_start(p, where, flag(l, "salted", 1));
    else if (w == "send") {
      std::vector<SampleSize> o;
      for (auto &x : split(get(l, "samples"), ',')) o.push_back(sample(l, x));
      plan_send(p, where, o, small(l, "roots", 1, 64));
    } else if (w == "bat") plan_bat(p, where, {find(l, groups, get(l, "group"))}, num(l, "rows"));
    else if (w == "bat_sigma") plan_bat_sigma(p, where, num(l, "rows"));
    else if (w == "split") plan_split(p, where);
    else if (w == "seal") plan_seal(p, where);
    else if (w == "psc") {
      PscSize x;
      x.degree.assign(small(l, "variables", 64), small(l, "degree", 3, 64));
      if (!x.degree.empty() && x.degree[0] < 2) fail(l, "a round polynomial has degree below 'degree', at least 2");
      x.degree_Y = small(l, "degree_Y", 2, 64), x.values = small(l, "values", 2, 1 << 20), x.with_Y = flag(l, "Y", 1);
      plan_psc(p, where, x);
    } else if (w == "zero") plan_zero(p, where, small(l, "variables", 64));
    else if (w == "line") plan_line(p, where);
    else if (w == "shift") plan_shift(p, where, num(l, "entries"));
    else if (w == "join") plan_join(p, where, small(l, "variables", 64));
    else if (w == "lkp") plan_lkp(p, where, small(l, "keys_log", 62));
    else if (w == "lookup") plan_lookup(p, where, LookupSize{small(l, "v", 62), small(l, "d", 62), small(l, "mu", 6), flag(l, "shifted", 0)});
    else if (w == "product_tree") plan_product_tree(p, where, small(l, "levels", 62), small(l, "more", 0, 62));
    else if (w == "rng") plan_rng(p, where, num(l, "columns"));
    else if (w == "scalars") plan_scalars(p, "scalars", where, num(l, "n"));
    else if (w == "fold") plan_fold(p, where, find(l, groups, get(l, "group")), small(l, "roots", 0, 1));
    else if (w == "loc") {
      const GroupSize &g = find(l, groups, get(l, "group"));
      const u64 t = plan_loc(p, where, g, tree_list(l, get(l, "trees"))), opens = num(l, "opens", 1);
      budget(g.main, t * opens);
      if (g.has_mask()) budget(g.mask, t * opens);
    } else if (w == "iopp") {
      const GroupSize &g = find(l, groups, get(l, "group"));
      std::vector<StageChoice> c;
      for (auto &x : split(get(l, "stages"), ',')) {
        auto y = split(x, ':');
        if (y.size() != 4) fail(l, "a stage is k:h:log_n:mask_log_n");
        c.push_back(StageChoice{within(l, number(l, y[0]), 32), within(l, number(l, y[1]), 32), within(l, number(l, y[2]), 48), within(l, number(l, y[3]), 48)});
      }
      // t1 leaves 2^-(target + 30) of the error for the batching term of the proximity test's code
      // switch (CS in blocks.h). If that round needs more positions than t1, plan_iopp() rejects
      // the plan.
      const u64 t1 = plan_positions(p, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30))), opens = num(l, "opens", 1);
      auto st = solve_stages(p, g, t1, c);
      plan_iopp(p, where, st, tree_list(l, get(l, "trees")), num(l, "rows"), small(l, "roots", 1, 1), flag(l, "masks_ride", 0));
      budget(g.main, t1 * opens);
      if (g.has_mask()) budget(g.mask, t1);
      for (size_t i = 1; i < st.size(); ++i) {
        char b[320];
        const GroupSize &x = st[i].g;
        std::snprintf(b, sizeof b, "stage: %s: 2^%u rows of 2^%u in 2^%u (message %llu + %llu random), 2^%u mask rows of %llu + %llu in 2^%u, m = %llu, "
                      "list %s, %llu positions", x.name.c_str(), x.k, ceil_log2(x.main.ell), x.main.log_n, (unsigned long long)x.main.ell,
                      (unsigned long long)x.main.r, x.h, (unsigned long long)x.mask.ell, (unsigned long long)x.mask.r, x.mask.log_n,
                      (unsigned long long)x.m, analysis(set, x).list.dec().c_str(), (unsigned long long)st[i].t);
        bool seen = false;
        for (auto &n : notes) seen = seen || n == b;
        if (!seen) notes.push_back(b);
      }
    } else if (w == "record") plan_record(p, where, find(l, tables, get(l, "table")).size);
    else if (w == "lin") {
      Table &T = find_to_change(l, tables, get(l, "table"));
      LinSize x{sample(l, get(l, "w")), sample(l, get(l, "u")), sample(l, get(l, "omega"))};
      x.instances = num(l, "instances"), x.k_lo = small(l, "k_lo", 64), x.rows_log = small(l, "rows_log", 64);
      if (!x.instances) fail(l, "at least one instance");
      x.one_time = flag(l, "one_time", 0);
      const Scope outer = p.scope();
      if (!x.one_time) p.begin_scope(Scope::REQUEST);
      plan_lin_first(p, where, x);
      plan_seal(p, where);  // the request's chain ends here; the batch's begins
      if (!x.one_time) p.begin_scope(Scope::SHARED);
      const u64 nu = plan_lin_shared(p, where, T.size, x, find(l, trees, get(l, "tree")));
      p.begin_scope(outer);
      char b[256];
      if (x.one_time) T.spent = nu;
      else if (nu) {
        std::snprintf(b, sizeof b, "hiding budget: %s: %llu columns a batch; a stored copy with %llu random coordinates a row, %llu of them spent by "
                      "the committer's tests, bears %llu batches", where.c_str(), (unsigned long long)nu, (unsigned long long)T.size.code.r,
                      (unsigned long long)T.spent, (unsigned long long)((T.size.code.r - std::min(T.size.code.r, T.spent)) / nu));
        notes.push_back(b);
      }
    } else if (w == "link") {
      Table &T = find_to_change(l, tables, get(l, "table"));
      u64 a = 0, c = 0;
      plan_link(p, where, T.size, sample(l, get(l, "u")), find(l, trees, get(l, "tree")), a, c);
      T.spent = c;
    } else
      fail(l, "unknown word");
  }
};

void print_groups(const Run &r) {
  for (auto &name : r.group_order) {
    const GroupSize &g = r.groups.at(name);
    const Setting &s = r.set;
    const Scope sc = r.group_scope.at(name);
    const unsigned w = s.work_of(ErrorKind::QUERY, sc), bits = s.target > w ? s.target - w : 0;
    std::printf("code: %s: n = 2^%u, message %llu + %llu random, rate %.6f, %llu x 2^%u rows", name.c_str(), g.main.log_n,
                (unsigned long long)g.main.ell, (unsigned long long)g.main.r, rho_of(s, g.main).hi.value(), (unsigned long long)g.slots, g.k);
    if (r.group_named.at(name) ? analysis(s, g).bad() : !group_holds(s, sc, g, r.group_load.at(name))) {
      std::printf(": NO DISTANCE HOLDS ITS ROUNDS AT THE TARGET\n");
      continue;
    }
    const Analysis a = analysis(s, g);
    std::printf("; closeness count: %s", name_of(a.at.source));
    if (a.at.source == CountSource::JOHNSON) {
      std::printf(", eta = %.6f, m = %llu as printed", a.at.eta.hi.value(), (unsigned long long)a.at.m);
    }
    std::printf(", delta = %.6f, count a = 2^%.2f, closeness 2^-%.2f, list %s, positions for %u bits: %llu\n", a.at.delta.lo.value(), a.at.count.hi.log2(),
                cut2(bits_of((g.k ? fold_round_error(s, g, a, 1) : base_error(s, g, a)).error())), a.list.dec().c_str(), bits,
                (unsigned long long)positions(a.at.agree, bits, Bound((u64)0), sc == Scope::REQUEST ? s.instances : 1));
    if (g.has_mask())
      std::printf("code:   its mask code: n = 2^%u, message %llu + %llu random, rate %.6f, 2^%u rows; at that distance: %s, count 2^%.2f\n", g.mask.log_n,
                  (unsigned long long)g.mask.ell, (unsigned long long)g.mask.r, rho_of(s, g.mask).hi.value(), g.h, name_of(a.mask_at.source),
                  a.mask_at.count.hi.log2());
  }
}

void print_sizes(const Run &r, const Planned &p, bool messages) {
  static const char *const kind[] = {"scalars", "sample answers", "roots", "nonces", "opened symbols", "salts", "paths", "heads"};
  for (unsigned part = 0; part < p->parts.size(); ++part) {
    const PartSize all = p->size_of(part);
    if (!all.bytes) continue;
    std::printf("size: %s: %.0f bytes expected (%.3f MB), %llu at most\n", p->parts[part].c_str(), all.expected, all.expected / 1e6, (unsigned long long)all.bytes);
    double by[8] = {};
    for (auto &m : p->messages_)
      if (m.part == part) by[(int)m.kind] += m.expected * (double)m.count;
    for (int k = 0; k < 8; ++k)
      if (by[k] > 0) std::printf("size:   %-16s %14.0f  (%5.1f %%)\n", kind[k], by[k], 100.0 * by[k] / all.expected);
    {  // by block call: `where` up to the first comma
      std::map<std::string, std::pair<double, u64>> use;
      std::vector<std::string> order;
      for (auto &m : p->messages_)
        if (m.part == part) {
          const std::string n = m.where.substr(0, m.where.find(','));
          if (!use.count(n)) order.push_back(n);
          use[n].first += m.expected * (double)m.count, use[n].second += m.bytes * m.count;
        }
      for (auto &n : order)
        std::printf("size:   in %-20s %12.0f expected, %12llu at most\n", n.c_str(), use[n].first, (unsigned long long)use[n].second);
    }
    if (!messages) continue;
    std::map<std::string, std::pair<double, u64>> named;
    std::vector<std::string> order;
    for (auto &m : p->messages_)
      if (m.part == part) {
        const std::string n = digits_as_hash(m.name + (m.where.empty() ? "" : " (" + m.where + ")"));
        if (!named.count(n)) order.push_back(n);
        named[n].first += m.expected * (double)m.count, named[n].second += m.count;
      }
    for (auto &n : order) std::printf("message: %s: %s: %llu time(s), %.0f bytes in all\n", p->parts[part].c_str(), n.c_str(), (unsigned long long)named[n].second, named[n].first);
  }
  for (auto &s : r.sums) {
    double e = 0;
    u64 most = 0;
    for (auto &n : s.second) {
      const int i = p->part_index(n);
      if (i < 0) {
        std::fprintf(stderr, "cp_plan: sum %s: there is no part %s\n", s.first.c_str(), n.c_str());
        std::exit(2);
      }
      e += p->size_of((unsigned)i).expected, most += p->size_of((unsigned)i).bytes;
    }
    std::printf("size: %s: %.0f bytes expected (%.3f MB), %llu at most\n", s.first.c_str(), e, e / 1e6, (unsigned long long)most);
  }
}

int run(const std::vector<Line> &lines, bool rounds, bool messages) {
  Run r;
  r.lines = lines;
  r.set = setting_goldilocks();
  const size_t head = r.head();
  Plan plan(r.set);
  plan.compact = true;
  r.block(plan, head, r.lines.size());
  const Planned p = plan.finish();
  r.budgets_to_notes();
  const Setting &s = r.set;
  std::printf("setting: field p = %s, |E| = p^%u = 2^%.4f, %s codes; target %u; m as Theorem 4.6 of ePrint 2025/2055 prints it; rows of an interleaved word cost nothing; "
              "%u sample answer(s); %llu instance(s) a session\n", s.field_name.c_str(), s.degree, Real(s.card_E()).log2(), s.circle ? "circle" : "plain Reed-Solomon", s.target,
              s.samples, (unsigned long long)s.instances);
  std::printf("setting: bytes: F %u, E %u, digest %u, salt %u, nonce %u, hash block %u\n", s.f_bytes, s.e_bytes, s.digest_bytes, s.salt_bytes,
              s.nonce_bytes, s.hash_block_bytes);
  print_groups(r);
  for (auto &n : r.notes) std::printf("%s\n", n.c_str());
  Accounting a = account(p);
  // The plan's verdict (`Cause` in plan.h) asks one thing more than account() (accounting.h): one
  // distance for all rounds on the same main and mask code. Asked of the whole file as one plan,
  // that can give a false NO: two proximity-test stages that share both codes may validly differ.
  // The verdict line comes before the accounting, and the two never contradict each other: a NO for
  // two distances is copied into the accounting here, and account() finds the other causes itself.
  // The exit status follows the verdict and the hiding budgets; a verdict other than NO also
  // reports a missed budget in its line. If p->holds() but no round has an error (as in a FILE of
  // codes and groups alone), the line says "none", not "holds".
  if (p->cause() == Cause::TWO_DISTANCES) a.holds = false, a.why_not = "by the plan's own verdict, which cp_plan asks of the whole file and this account does not: " + p->why_not();
  std::printf("the plan's own verdict: %s%s\n", !p->holds() ? ("NO: " + p->why_not() + (p->cause() == Cause::TWO_DISTANCES ? " (asked of the whole file as one plan: numbers/plan.h says where that is too strict)" : "")).c_str()
                                                : a.formed  ? "holds: every round is at the target, and rounds on the same two codes have one distance"
                                                            : "none: the file lays no round with an error",
              !p->holds() || r.budgets_met ? "" : "; BUT A HIDING BUDGET IS NOT MET (the lines \"hiding budget:\"): this run ends with status 3");
  if (rounds) print_rounds(a);
  print_accounting(a, "the whole plan", rounds);
  if (!r.budgets_met) std::printf("hiding budget: NOT MET: some rows are opened at more positions than their random coordinates hide\n");
  print_sizes(r, p, messages);
  return p->holds() && r.budgets_met ? 0 : 3;  // `holds` is false whenever the accounting says NO
}

}  // namespace

int main(int argc, char **argv) {
  const char *file = nullptr;
  bool rounds = false, messages = false;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--rounds")) rounds = true;
    else if (!std::strcmp(argv[i], "--messages")) messages = true;
    else if (!file && argv[i][0] != '-') file = argv[i];
    else file = "", i = argc;  // a second file or an unknown option
  std::ifstream in(file ? file : "");
  if (!file || !*file || !in) {
    std::fprintf(stderr, "usage: cp_plan FILE [--rounds] [--messages]\n");
    return 2;
  }
  std::vector<Line> lines;
  std::string text;
  for (int n = 1; std::getline(in, text); ++n) {
    text = text.substr(0, text.find('#'));
    std::istringstream t(text);
    Line l;
    l.number = n;
    if (!(t >> l.word)) continue;
    for (std::string x; t >> x;) {
      const size_t e = x.find('=');
      if (e == std::string::npos) l.bare.push_back(x);
      else if (!l.kv.emplace(x.substr(0, e), x.substr(e + 1)).second) fail(l, "a key twice: " + x);
    }
    lines.push_back(l);
  }
  return run(lines, rounds, messages);
}
