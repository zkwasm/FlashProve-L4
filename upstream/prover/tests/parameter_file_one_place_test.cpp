// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the rule `prover-only-stages` of limit_missed() (prover/common/parameter_file.h): s < l.
// Here l is the third number of a line `group <group>`, and s is the sum of the first numbers of
// the "fours" (the four numbers of each stage of a proximity test but the first) of the line
// `stages <group>`. That test's last message has 2^(l - s) entries ("places").
//   parameter_file_one_place_test <parameter file>
// The file must pass parse() and limit_missed(), and each `stages` line must have two fours or
// more. The test raises s to l - 1, l and l + 1. A four's first number is at most 8, so a `stages`
// line with few fours cannot reach a large l. Such a line is tried in two ways: at the file's own l,
// with the line's first four repeated (with_sum()); and with the line's own number of fours, at a
// lowered l (within_reach()), for which s of that line, and of the other line of its pair, must be
// below the lowered l. The test runs on the CPU only
// and writes no file. It does not run cp_verify on the changed files (its planning call accepts at
// least one file that the rule rejects), nor a proximity test, whose limit the rule restates.
#include "common/parameter_file.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
static int failures = 0;
typedef std::vector<std::string> Lines;
// The index of the one line that begins with `words`.
static size_t at(const Lines &lines, const std::string &words) {
  size_t found = 0, number = 0;
  for (size_t i = 0; i < lines.size(); i++)
    if (!lines[i].compare(0, words.size() + 1, words + " ")) found = i, number++;
  if (number != 1) {
    printf("the parameter file does not have exactly one line %s\n", words.c_str());
    exit(2);
  }
  return found;
}
static std::vector<unsigned> numbers(const Lines &lines, const std::string &words) {
  std::istringstream in(lines[at(lines, words)].substr(words.size()));
  std::vector<unsigned> found;
  for (unsigned x; in >> x;) found.push_back(x);
  return found;
}
static Lines with(Lines lines, const std::string &words, const std::vector<unsigned> &values) {
  std::string &line = lines[at(lines, words)] = words;
  for (unsigned x : values) line += " " + std::to_string(x);
  return lines;
}
// Raises the fours' first numbers in `stages <group>` until they add up to `sum`, within what
// parse() allows: at most 8, and 0 in the last four. If the line has too few fours for that, its
// first four is repeated first.
static Lines with_sum(const Lines &lines, const std::string &group, unsigned sum) {
  std::vector<unsigned> s = numbers(lines, "stages " + group);
  const std::vector<unsigned> first(s.begin(), s.begin() + 4);
  while (8 * (s.size() / 4 - 1) < sum) s.insert(s.begin(), first.begin(), first.end());
  unsigned now = 0;
  for (size_t i = 0; i < s.size(); i += 4) now += s[i];
  for (size_t i = 0; i + 4 < s.size(); i += 4)
    while (now < sum && s[i] < 8) s[i]++, now++;
  if (now != sum) {
    printf("the line stages %s has no room for the sum %u\n", group.c_str(), sum);
    exit(2);
  }
  return with(lines, "stages " + group, s);
}
// The file with l of `group` lowered to the largest value for which with_sum() reaches l + 1 without
// repeating a four; unchanged if l is no larger. parse() wants the same l in omega and own, and in
// once-omega and once-own ("parameter-own-code"), so the other of such a pair gets the same l.
static Lines within_reach(Lines lines, const std::string &group) {
  const size_t fours = numbers(lines, "stages " + group).size() / 4;
  if (fours < 2) {
    printf("this test cannot use this parameter file: its line stages %s has only four numbers, so their sum "
           "cannot be raised\n",
           group.c_str());
    exit(2);
  }
  const size_t omega = group.find("omega"), own = group.find("own");
  const std::string other = omega != std::string::npos ? group.substr(0, omega) + "own"
                            : own != std::string::npos ? group.substr(0, own) + "omega"
                                                       : group;
  const unsigned most = 8 * (unsigned)(fours - 1) - 1;
  for (const std::string &name : {group, other}) {
    std::vector<unsigned> g = numbers(lines, "group " + name);
    if (g[2] > most)
      printf("%s: tried with the third number of the line group %s lowered from %u to %u\n", group.c_str(),
             name.c_str(), g[2], most);
    g[2] = std::min(g[2], most);
    lines = with(lines, "group " + name, g);
  }
  return lines;
}
// `refused`: the group whose `stages` line the rejection must name; empty: the file must pass.
static void expect(const std::string &what, const Lines &lines, const std::string &refused) {
  std::string bytes;
  for (const std::string &line : lines) bytes += line + "\n";
  parameter_file::Parameters p;
  std::string got = parameter_file::parse(bytes, p);
  if (got.empty()) got = parameter_file::limit_missed(p);
  const std::string wanted =
      refused.empty() ? ""
                      : "prover-only-stages: line " + std::to_string(at(lines, "stages " + refused) + 1) +
                            ": the 1st, 5th, 9th, .. numbers of this line must add up to less than the third number of "
                            "the line above it: these programs cannot run a proximity test whose last message has "
                            "one place";
  const bool good = got == wanted;
  printf("%s: %s\n", what.c_str(), good ? "as expected" : "FAILS");
  if (!good) {
    failures++;
    printf("    wanted: [%s]\n    got: [%s]\n", wanted.c_str(), got.c_str());
  }
}
int main(int argc, char **argv) {
  std::string bytes;
  using namespace parameter_file;
  if (argc != 2 || read_once(argv[1], bytes, PARAMETER_FILE_MOST) != Found::Read) {
    printf("usage: parameter_file_one_place_test <parameter file>\n");
    return 2;
  }
  Lines file;
  for (size_t from = 0, end; (end = bytes.find('\n', from)) != std::string::npos; from = end + 1)
    file.push_back(bytes.substr(from, end - from));
  expect("control: the file as it is", file, "");
  for (const char *group : {"link", "folded", "omega", "own", "once-folded", "once-omega", "once-own", "once-ranges"}) {
    const unsigned places_log = numbers(file, std::string("group ") + group)[2];
    expect(std::string(group) + ": two places", with_sum(file, group, places_log - 1), "");
    expect(std::string(group) + ": one place", with_sum(file, group, places_log), group);
    expect(std::string(group) + ": a sum above the group's number", with_sum(file, group, places_log + 1), group);
    const Lines lowered = within_reach(file, group);
    const unsigned low = numbers(lowered, std::string("group ") + group)[2];
    if (low == places_log) continue;
    const std::string own_fours = std::string(group) + ": the line's own fours: ";
    expect(own_fours + "two places", with_sum(lowered, group, low - 1), "");
    expect(own_fours + "one place", with_sum(lowered, group, low), group);
    expect(own_fours + "a sum above the group's number", with_sum(lowered, group, low + 1), group);
  }
  // With one four (two stages) s is 0, so the rule rejects only l = 0 (one place).
  std::vector<unsigned> ranges = numbers(file, "group once-ranges"), last = numbers(file, "stages once-ranges");
  last.erase(last.begin(), last.end() - 4);
  ranges[2] = 1;
  const Lines short_test = with(file, "stages once-ranges", last);
  expect("one four, two places", with(short_test, "group once-ranges", ranges), "");
  ranges[2] = 0;
  expect("one four, one place", with(short_test, "group once-ranges", ranges), "once-ranges");
  printf("parameter_file_one_place_test: %s\n", failures ? "some checks failed" : "all checks passed");
  return failures != 0;
}
