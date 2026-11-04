// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Tests rules of limit_missed() (prover/common/parameter_file.h): `prover-only-f` and
// `prover-only-V` reject a parameter file (policy.txt) whose f (a dense layer's hidden width), or
// whose V (vocabulary size) rounded up to a multiple of 2^(la - lt), has more than eight one-bits;
// `prover-only-f` also rejects an f of 2^16 or more.
//   parameter_file_one_bits_test <parameter file>
// The file must pass parse() and limit_missed() and have experts, la - lt = 2 and 11 as the last
// number of its line `table` (V may be at most 2^11), as prover/examples/target_small_f2/policy.txt
// does. The cases change value lines in memory; each calls both functions and compares the whole
// result. With experts f is a power of two, so the cases on f use the file without experts. Its
// call and callH lines stay: apart from their form, only the name and kind of each are checked.
// The test runs on the CPU only and writes no file. It does not run the forward-pass kernels whose
// limit the rules restate.
#include "common/parameter_file.h"
#include <cstdlib>
static int failures = 0;
typedef std::vector<std::string> Lines;
static bool begins(const std::string &line, const std::string &with) { return !line.compare(0, with.size(), with); }
// The index of the line that begins with `name` and a space. Exits unless exactly one line does.
static size_t at(const Lines &lines, const std::string &name) {
  size_t found = 0, number = 0;
  for (size_t i = 0; i < lines.size(); i++)
    if (begins(lines[i], name + " ")) found = i, number++;
  if (number != 1) {
    printf("the parameter file does not have exactly one line %s\n", name.c_str());
    exit(2);
  }
  return found;
}
static Lines with(Lines lines, const std::string &name, unsigned value) {
  lines[at(lines, name)] = name + " " + std::to_string(value);
  return lines;
}
// The file without experts: NE, hr (the router's digit form) and the experts' and router's weight
// widths are 0; rcl, gcl and the four public tables' digests, which need experts, are removed.
static Lines without_experts(const Lines &lines) {
  Lines kept;
  for (const std::string &line : lines) {
    bool goes = false;
    for (const char *name : {"rcl ", "gcl ", "tsilu", "tsig", "trope", "texp"}) goes = goes || begins(line, name);
    if (!goes) kept.push_back(line);
  }
  for (const char *name : {"NE", "hr", "wb_gs", "wb_us", "wb_gr", "wb_ur", "wb_dnr", "wb_r"})
    kept = with(kept, name, 0);
  return kept;
}
// `refused` names the line that the rejection must point to; empty if the file must be accepted.
static void expect(const char *what, const Lines &lines, const std::string &refused, const std::string &sentence) {
  std::string bytes;
  for (const std::string &line : lines) bytes += line + "\n";
  parameter_file::Parameters p;
  std::string got = parameter_file::parse(bytes, p);
  if (got.empty()) got = parameter_file::limit_missed(p);
  const std::string place = refused.empty() ? "" : ": line " + std::to_string(at(lines, refused) + 1) + ": ";
  const std::string wanted = refused.empty() ? "" : "prover-only-" + refused + place + sentence;
  const bool good = got == wanted;
  printf("%s: %s\n", what, good ? "as expected" : "FAILS");
  if (!good) {
    failures++;
    printf("    wanted: [%s]\n    got: [%s]\n", wanted.c_str(), got.c_str());
  }
}
int main(int argc, char **argv) {
  std::string bytes;
  using namespace parameter_file;
  if (argc != 2 || read_once(argv[1], bytes, PARAMETER_FILE_MOST) != Found::Read) {
    printf("usage: parameter_file_one_bits_test <parameter file>\n");
    return 2;
  }
  Lines file;
  for (size_t from = 0, end; (end = bytes.find('\n', from)) != std::string::npos; from = end + 1)
    file.push_back(bytes.substr(from, end - from));
  const Lines dense = without_experts(file);
  const std::string f_says = "f must have at most eight one-bits in binary",
                    V_says = "V, rounded up to a multiple of 2^(la - lt), must have at most eight one-bits in binary";
  expect("control: the file as it is", file, "", "");
  expect("control: the file without experts", dense, "", "");
  // 2044 has nine one-bits and 2040 eight. 2041 is rounded up to 2044. 2047 has eleven and is rounded up to 2048, which
  // has one.
  expect("without experts, f 2044", with(dense, "f", 2044), "f", f_says);
  expect("without experts, f 2040", with(dense, "f", 2040), "", "");
  expect("V 2044", with(file, "V", 2044), "V", V_says);
  expect("V 2040", with(file, "V", 2040), "", "");
  expect("V 2041", with(file, "V", 2041), "V", V_says);
  expect("V 2047", with(file, "V", 2047), "", "");
  // The width over which the forward pass adds byte products into 32-bit sums. 65,280 has eight one-bits.
  expect("without experts, f 65280", with(dense, "f", 65280), "", "");
  expect("without experts, f 65536", with(dense, "f", 65536), "f",
         "f must be below 2^16: the forward pass adds 2 f products of signed bytes into one 32-bit sum");
  printf("parameter_file_one_bits_test: %s\n", failures ? "some checks failed" : "all checks passed");
  return failures != 0;
}
