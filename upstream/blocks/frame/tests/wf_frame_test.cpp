// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/frame/wf_frame.h, which derives the sizes of WF: the committer's proof, by lookups,
// that the committed matrix's values are in range. It prints sizes; it makes and checks no proof.
// Run it from blocks/: the parameter file (argv[1]) defaults to a relative path. Exit code 0 with a
// last line "ok: ...", or 1 with "FAILED:...".
// An entry is the matrix rows, `stride` apart, that hold the digits of a row of values. A lookup's
// keys are cut into parts of 2^log_places places; a place reads an entry or, if `log_rows_apart` is
// 1, one row of it, and gives a key for each of the matrix's 2^k_col columns. A full part reads
// 2^part_log rows (`dial` below; `ranges-part` in a parameter file). A lookup's multiplicities
// (how often each table value is a key) have the fewest bits (planes), at least 2 and more than
// log_places + k_col, that count all its keys, padding included.
// main() calls cuts(), tests wf_planes_of(), then a matrix of a large model's size at four part
// sizes, then settings that must be rejected (`refused` in the code), all but the last by exact
// message.
#include <cstdio>
#include <fstream>
#include <sstream>

#include "frame/wf_frame.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

static u64 mix(u64 &s) {  // SplitMix64 (Steele, Lea, Flood, OOPSLA 2014)
  u64 z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static std::vector<std::pair<u64, u64>> entries_of(const std::vector<WfRun> &runs) {  // (first row, stride), entry by entry
  std::vector<std::pair<u64, u64>> p;
  for (const WfRun &r : runs)
    for (u64 i = 0; i < r.places; i++) p.push_back({r.first_row + i, r.stride});
  return p;
}
// Tests wf_cut() on pseudo-random lists of entries. The rule: with m (`most`) entries to a part,
// entry i of part j is the list's entry j * m + i, and its row q is at place q * m + i; so each
// part but the last is full.
static void cuts() {
  u64 s = 3;
  for (int turn = 0; turn < 40; turn++) {
    std::vector<WfRun> all;
    for (u64 n = 1 + mix(s) % 6, row = 0; n; n--) {
      const u64 places = 1 + mix(s) % 37;
      all.push_back(WfRun{0, row, places, places});
      row += 2 * places + mix(s) % 3;
    }
    const std::vector<std::pair<u64, u64>> want = entries_of(all);
    for (unsigned apart = 0; apart < 2; apart++)
      for (unsigned log_places = apart; log_places < 8; log_places++) {
        const std::vector<WfPart> parts = wf_cut(all, log_places, apart);
        const u64 most = u64(1) << (log_places - apart);
        CHECK(parts.size() == (want.size() + most - 1) / most, "the number of parts");
        for (size_t j = 0; j < parts.size(); j++) {
          std::vector<long long> row_at(size_t(1) << log_places, -1), stride_at(size_t(1) << log_places, -1);
          u64 upto = 0;
          for (const WfRun &r : parts[j].runs) {
            CHECK(r.first_place >= upto && r.places >= 1 && r.first_place + r.places <= (u64(1) << log_places), "runs overlap, fall, or leave the part");
            upto = r.first_place + r.places;
            for (u64 i = 0; i < r.places; i++) row_at[r.first_place + i] = (long long)(r.first_row + i), stride_at[r.first_place + i] = (long long)r.stride;
          }
          for (u64 q = 0; q < u64(1) << apart; q++)
            for (u64 i = 0; i < most; i++) {
              const u64 entry = j * most + i;
              const bool there = entry < want.size();
              CHECK(row_at[q * most + i] == (there ? (long long)(want[entry].first + q * want[entry].second) : -1), "a place reads another row than the rule says");
              if (there) CHECK(stride_at[q * most + i] == (apart ? 0 : (long long)want[entry].second), "a place has another stride than the rule says");
            }
        }
      }
  }
}

// Two-byte weights in units of unequal size. A unit is a run of entries: the rows with their low
// bytes, then the rows with their high bytes. `bytes` looks up every row's cells in a table of
// 2^d = 2^8 values, `weights` each two-byte weight in one of 2^13.
static WfShape large_shape() {
  WfShape S;
  S.k_col = 18, S.matrix_rows = 3076096;
  const u64 weight_rows = 1527800;
  WfShapeLookup in_bytes, in_weights;
  WfLookup &bytes = in_bytes.lookup, &weights = in_weights.lookup;
  bytes.name = "bytes", bytes.d = 8, bytes.digit_weight = {F::one()}, bytes.offset = from_u64(128);
  in_bytes.log_rows_apart = 1;
  bytes.first_bound = {2, 1, 0, 5, 4, 3, 8, 7, 6};
  weights.name = "weights", weights.d = 13, weights.digit_weight = {F::one(), from_u64(256)}, weights.offset = from_u64(4096);
  weights.first_bound = bytes.first_bound;
  u64 s = 7, row = 0;
  for (u64 left = weight_rows; left;) {
    const u64 entries = std::min<u64>(left, 1000 + mix(s) % 7000);
    in_weights.entries.push_back(WfRun{0, row, entries, entries});
    row += 2 * entries, left -= entries;
  }
  in_bytes.entries = in_weights.entries;
  in_bytes.entries.push_back(WfRun{0, row, (S.matrix_rows - row) / 2, (S.matrix_rows - row) / 2});  // the rows of single bytes, as one more unit
  S.lookups = {in_bytes, in_weights};
  return S;
}
// The lines `group once-ranges` and `stages once-ranges`, fixed here: main() uses only setting_of()
// of its parameter file.
static GroupLine a_line() {
  GroupLine g;
  g.log_n = 22, g.log_ell = 18, g.mask_log_n = 16, g.mask_log_ell = 12;
  g.stages = {StageChoice{4, 3, 19, 17}, StageChoice{3, 3, 17, 17}, StageChoice{0, 0, 20, 20}};
  return g;
}

int main(int argc, char **argv) {
  cuts();
  const std::string path = argc > 1 ? argv[1] : "frame/examples/record.f2.txt";
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  Parameters par;
  std::string why;
  if (!in || !read_parameters(text.str(), par, why)) return std::printf("FAILED: %s: %s\n", path.c_str(), in ? why.c_str() : "cannot be read"), 1;
  const Setting s = setting_of(par);
  const WfShape shape = large_shape();
  u64 before = ~u64(0);
  std::string sizes;
  {  // the planes, by the definition
    for (unsigned k_col : {0u, 2u, 5u})
      for (unsigned log_places : {1u, 3u})
        for (size_t parts = 1; parts <= 9; parts++) {
          const std::vector<WfPart> list(parts, WfPart{log_places, {WfRun{0, 0, 1, 0}}});
          const unsigned planes = wf_planes_of(list, k_col), v = log_places + k_col;
          const u64 keys = (u64)parts << v;
          CHECK(planes >= 2 && planes > v && !(keys >> planes), "too few planes");
          CHECK(planes == 2 || planes - 1 <= v || (keys >> (planes - 1)), "a plane too many");
        }
    WfShape named = shape;
    named.lookups[0].lookup.live = 40, named.lookups[0].lookup.mu = 6;
    Codes codes;
    CHECK(wf_sized(s, wf_lines(named, a_line(), 18), codes).refused == "ranges: a look-up's planes follow from its parts: whoever gives the shape leaves them 0", "a shape may name its planes");
  }
  for (unsigned part_log : {17u, 18u, 19u, 20u}) {
    Codes codes;
    const WfSized z = wf_sized(s, wf_lines(shape, a_line(), part_log), codes);
    if (!z.refused.empty()) std::printf("part_log %u: %s\n", part_log, z.refused.c_str());
    CHECK(z.refused.empty(), "the large shape is not admitted");
    if (!z.refused.empty()) continue;
    CHECK(wf_covers_all_rows(z.x, 0, z.matrix), "the bytes' look-up does not cover every row");
    // The 3,076,096 x 2^18 keys of `bytes` lie between 2^39 and 2^40; `weights` has about half as
    // many. Padding fills each lookup's 24, 12, 6 or 3 parts to 2^(log_places + k_col) keys each:
    // 3 * 2^38 and 3 * 2^37 keys (worked out, not printed). So 40 and 39 planes (`live`) in
    // 2^mu = 2^6 slots. `live` 0, which means all slots, is not tested here.
    CHECK(z.x.lookups[0].mu == 6 && z.x.lookups[0].live == 40 && z.x.lookups[1].mu == 6 && z.x.lookups[1].live == 39, "the planes of the large shape");
    Plan p(s);
    p.begin_scope(Scope::ONCE);
    wf_plan(p, codes, z);
    if (p.refused()) std::printf("part_log %u: %s\n", part_log, p.why().c_str());
    CHECK(!p.refused() && p.finish()->holds(), "the plan is refused, or a round does not hold at the target");
    u64 bytes = 0, lookup_scalars = 0;
    const Planned laid = p.finish();
    for (const PlanMessage &m : laid->messages_) {
      bytes += m.bytes * m.count;
      if (m.kind == MessageKind::SCALARS && (m.where.rfind("ranges: bytes", 0) == 0 || m.where.rfind("ranges: weights", 0) == 0)) lookup_scalars += m.bytes * m.count;
    }
    CHECK(lookup_scalars == (wf_own_masks(z.x, z.matrix) + wf_index_masks(z.x, z.matrix)) * s.e_bytes, "the look-ups' planned scalars are not the mask entries that are announced");
    CHECK(wf_index_mask_room(z, s.samples) == 2 * z.pending + 1 + 2 * 18 + s.samples, "the room in the index-mask message");
    CHECK(bytes < before, "larger parts do not make the proof smaller");
    before = bytes;
    sizes += " 2^" + std::to_string(part_log) + ": " + std::to_string(z.pending) + " parts, " + std::to_string(bytes) + " bytes, " + std::to_string(p.finish()->rounds()) + " rounds;";
  }
  {
    Codes codes;
    GroupLine shorter = a_line();
    shorter.log_ell = 17, shorter.log_n = 21;
    CHECK(wf_sized(s, wf_lines(shape, shorter, 18), codes).refused == "once-ranges: a block of planes is longer than a main row", "a block longer than a row is not refused by name");
    CHECK(wf_sized(s, wf_lines(shape, a_line(), 0), codes).refused == "ranges: a look-up's table, planes or parts are not admitted", "a dial below a look-up's digits is not refused by name");
    // The message for too many parts: expected at every `dial` from 2 to `least` - 1.
    const std::string many = "ranges: a look-up would have more than 256 parts: the line ranges-part is too small for this matrix";
    static_assert(WF_MOST_PARTS == 256, "the words of the refusal name the number");
    // The message for too large a part: expected at every `dial` from `greatest` + 1 to 62.
    const std::string long_parts = "ranges: a part would read more rows than the matrix has: the line ranges-part is too large for this matrix";
    const unsigned greatest = ceil_log2(shape.matrix_rows);
    CHECK(greatest == 22 && wf_lines(shape, a_line(), greatest).fault == nullptr, "the dial of a part as long as the whole matrix is refused");
    for (const WfLookup &l : wf_lines(shape, a_line(), greatest).lookups) CHECK(l.parts.size() == 1, "a part as long as the whole matrix is not the only one");
    for (unsigned dial = greatest + 1; dial <= 62; dial++) {
      const WfLines lines = wf_lines(shape, a_line(), dial);
      bool cut = false;
      for (const WfLookup &l : lines.lookups) cut = cut || !l.parts.empty();
      CHECK(!cut && wf_sized(s, lines, codes).refused == long_parts, "a dial above the matrix's row bits is not refused by name, or something was cut for it");
    }
    unsigned least = greatest;
    while (least > 1 && wf_lines(shape, a_line(), least - 1).fault == nullptr) least--;
    CHECK(least > 2 && least < 17, "no dial of the large shape is refused for its number of parts, or one of the example files' kind is");
    for (const WfLookup &l : wf_lines(shape, a_line(), least).lookups) CHECK(!l.parts.empty() && l.parts.size() <= WF_MOST_PARTS, "the least dial that is admitted has too many parts, or none");
    bool above = false;
    for (const WfShapeLookup &l : shape.lookups) {
      const unsigned digits = ceil_log2(std::max<size_t>(l.lookup.digit_weight.size(), 1));
      for (unsigned dial = least; dial <= 20; dial++) CHECK(wf_parts_of(l.entries, dial - digits, l.log_rows_apart) == wf_cut(l.entries, dial - digits, l.log_rows_apart).size(), "the count of the parts is not the number that the cut makes");
      above = above || wf_parts_of(l.entries, least - 1 - digits, l.log_rows_apart) > WF_MOST_PARTS;
    }
    CHECK(above, "the dial below the least has no look-up with too many parts");
    for (unsigned dial = least - 1; dial >= 2; dial--) CHECK(wf_sized(s, wf_lines(shape, a_line(), dial), codes).refused == many, "a dial that gives very many parts is not refused by name");
    {  // `wf_plan` needs the list of codes that `wf_sized` filled: an altered list is rejected
      Codes sized_with;
      const WfSized z = wf_sized(s, wf_lines(shape, a_line(), 18), sized_with);
      Codes other = sized_with;
      other.base[z.x.stages[0].code].r += 1;
      Plan with_it(s), with_another(s);
      with_it.begin_scope(Scope::ONCE), with_another.begin_scope(Scope::ONCE);
      wf_plan(with_it, sized_with, z), wf_plan(with_another, other, z);
      CHECK(z.refused.empty() && !with_it.refused(), "the plan with the list of codes that was sized with is refused");
      CHECK(with_another.refused() && with_another.why() == "ranges: a stage of the proximity test is sized for another main code than it names", "a plan with another list of codes is not refused by name");
    }
    GroupLine tight = a_line();
    tight.log_n = 18;
    CHECK(!wf_sized(s, wf_lines(shape, tight, 18), codes).refused.empty(), "a code as long as its message is admitted");
  }
  std::printf("%s:%s\n", failures ? "FAILED" : "ok", sizes.c_str());
  return failures != 0;
}
