// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// CPU-only test of rf::HfRecord (prover/hash_commitment/hf_record.h), a weight table's commitment
// record (`card` in `h_card` and wcard.bin): writing and reading back, each check of fails() on an
// altered word, read_bytes(), and the two valid first words: the anchor's (a commitment's first
// weight table) and a leaf table's (a further table of the same weights; `copy` in identifiers and
// messages). It uses one shape, HfParams::toy(0), and does not test that header's hf_unit_list_words().
// Build, from prover/: ./build.sh hc_host_test tests/hf_record_test
// Run: hf_record_test <directory> (printed: "folder"). The test creates a new directory there
// (mkdtemp), writes its one file into it, and removes both.
// Exit code 1: a check failed; 2: wrong usage, or mkdtemp() or TH() failed.
#include "common/hash.h" // earlier_hash_h_digest()
#include "common/setting.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace rf {
// The tree hash, as in prover/common/host_common.h; the tested header calls it without defining it.
// DW (prover/common/setting.h) is a digest's length in 32-bit words.
static std::vector<uint32_t> TH(std::vector<uint32_t> w) {
  if (w.size() < (size_t)DW) exit(2);
  for (uint32_t num = TH_LIST;; num = TH_UPPER) {
    const size_t nc = (w.size() + THPW - 1) / THPW;
    std::vector<uint32_t> o(nc * DW);
    for (size_t c = 0; c < nc; c++) earlier_hash_h_digest(&o[c * DW], num, &w[c * THPW], std::min<size_t>(THPW, w.size() - c * THPW));
    w.swap(o);
    if (nc == 1) return w;
  }
}
} // namespace rf
#include "hash_commitment/hf_record.h"
#include <unistd.h>
static int bad = 0, done = 0;
static void is(bool ok, const char *what) {
  done++;
  if (!ok) bad++, printf("FAILS: %s\n", what);
}
int main(int argc, char **argv) {
  if (argc != 2 || !*argv[1]) {
    printf("usage: hf_record_test <a folder in which the test makes a folder of its own>\n");
    return 2;
  }
  std::string dir = std::string(argv[1]) + "/hf_record_test.XXXXXX";
  if (!mkdtemp(dir.data())) {
    printf("hf_record_test: no folder of its own can be made in %s: %s\n", argv[1], strerror(errno));
    return 2;
  }
  const HfParams P = HfParams::toy(0);
  // `units`, `hpar`, `RT` and `once` are arbitrary words, not digests; `once` becomes h_once, the
  // record's last digest. `a` keeps rec's h_0, the tree hash of its words before h_once; `b` ends
  // up as its h_card, the tree hash of all its words.
  uint32_t units[DW], hpar[DW], RT[DW], once[DW], a[DW], b[DW];
  for (int i = 0; i < DW; i++) units[i] = 100 + i, hpar[i] = 200 + i, RT[i] = 300 + i, once[i] = 400 + i;
  rf::HfRecord rec(HF_TIER_HIDDEN, P, units, hpar, RT);
  rec.h_0(a);
  rec.set_h_once(once);
  rec.h_0(b);
  is(!memcmp(a, b, sizeof a), "h_0 does not depend on h_once");
  rec.h_card(b);
  is(memcmp(a, b, sizeof a), "h_card differs from h_0");
  is(!rec.fails(P, units, hpar), "the record passes its own checks");
  const std::string path = dir + "/wcard.bin";
  is(rec.write(path), "written");
  is(!rec.write(path), "not written over");
  rf::HfRecord back;
  is(back.read(path) && back.w == rec.w && back.tw == rec.tw, "read back");
  unlink(path.c_str());
  is(!back.read(path) && !strcmp(back.refusal, "input-unreadable-card"), "a missing file");
  const struct { int at; const char *name; } faults[] = {
      {HF_REC_MAGIC, "hc-record-head"}, {HF_REC_VERSION, "hc-record-head"}, {HF_REC_TIER, "hc-record-tier"},
      {HF_REC_LK, "hc-record-shape"}, {HF_REC_M0, "hc-record-shape"}, {HF_REC_FB, "hc-record-shape"},
      {HF_REC_FLN, "hc-record-shape"}, {HF_REC_FI0, "hc-record-shape"}, {HF_REC_FNM, "hc-record-shape"},
      {HF_REC_UNIT_LIST + DW - 1, "hc-record-unit-list"}, {HF_REC_H_PAR, "hc-record-hpar"}, {HF_REC_TH_TF + 3, "hc-record-tw"}};
  for (const auto &f : faults) {
    rf::HfRecord x = rec;
    x.w[f.at] += 7;
    const char *name = x.fails(P, units, hpar);
    is(name && !strcmp(name, f.name), f.name);
  }
  for (const uint32_t tier : {0u, 2u}) { // HF_TIER_HIDDEN, 1, is the only valid tier word
    rf::HfRecord x = rec;
    x.w[HF_REC_TIER] = tier;
    const char *name = x.fails(P, units, hpar);
    is(name && !strcmp(name, "hc-record-tier"), "the tier word is 1 alone");
  }
  { rf::HfRecord x = rec; x.tw[0]++; const char *name = x.fails(P, units, hpar); is(name && !strcmp(name, "hc-record-tw"), "RT changed"); }
  { // a leaf table's first word
    rf::HfRecord x = rec;
    x.w[HF_REC_MAGIC] = HF_RECORD_COPY_WORD;
    is(!x.fails(P, units, hpar) && x.is_copy() && !rec.is_copy(), "a copy's first word is admitted and told apart");
    is(rec.level() == 0 && x.level() == 1, "the first word says the level");
    x.w[HF_REC_MAGIC] = 0x48435232u; // "HCR2": no record begins with it
    is(x.fails(P, units, hpar) != nullptr && x.level() == -1, "there is no level 2");
    x.set_copy();
    is(x.w[HF_REC_MAGIC] == 0x4843524Cu && x.level() == 1, "set_copy() lays HCRL");
    uint32_t c[DW];
    x.h_0(c);
    is(memcmp(a, c, sizeof a), "a copy's h_0 is not the first root's");
    x.h_card(c);
    is(memcmp(b, c, sizeof b), "a copy's h_card is not the first root's");
  }
  { // from bytes
    const std::vector<uint32_t> file = rec.file_words();
    rf::HfRecord x;
    is(file.size() == (size_t)HF_RECORD_FILE_WORDS && x.read_bytes((const uint8_t *)file.data(), file.size() * 4) && x.w == rec.w && x.tw == rec.tw,
       "file_words() and read_bytes() are each other's inverse");
    is(!x.read_bytes((const uint8_t *)file.data(), file.size() * 4 - 1) && !strcmp(x.refusal, "hc-record-length") && x.w.empty(), "a byte less");
    is(!x.read_bytes((const uint8_t *)file.data(), file.size() * 4 - 4) && !strcmp(x.refusal, "hc-record-length") && x.w.empty(), "a word less");
    std::vector<uint32_t> longer = file;
    longer.push_back(0);
    is(!x.read_bytes((const uint8_t *)longer.data(), longer.size() * 4) && !strcmp(x.refusal, "hc-record-length") && x.w.empty(), "a word more");
    is(!x.read_bytes(nullptr, file.size() * 4), "no bytes");
  }
  is(!rmdir(dir.c_str()), "the test's own folder is removed at its end");
  printf("%d checks, %d fail\n", done, bad);
  return bad ? 1 : 0;
}
