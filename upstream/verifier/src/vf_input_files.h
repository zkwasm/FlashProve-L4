// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Reads the configuration and checks two headers before cp_verify (the verifier) forks its stages.
     key.bin       tag "CPIK", version word (4 characters); then what blocks/frame/check_key.h reads
     a proof file  tag "CPPF", version word, k: proved on leaf/<k>.bin; then blocks/frame/check.h
   No file stores a length; that code rejects a missing or extra byte. Nothing here changes a global
   variable. The configuration comes from the caller's directory, not a key file; in practice the
   provider writes it. The commitment ID (-id) identifies it and the anchor (key.bin's table). */
#pragma once
#include "vf_common.h"
#include "frame/heads.h" // ../../blocks: the files' tags
// Header lengths in 32-bit words.
static constexpr size_t CP_KEY_HEADW = 2, CP_PROOF_HEADW = 3;

// Returns little-endian 32-bit word i of a file's bytes b. Not checked: 4 * i + 4 <= b.size().
static u32 cpWord(const string &b, size_t i) {
  const u8 *p = (const u8 *)b.data() + 4 * i;
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}
// To reject, this file calls refuse() (vf_common.h), which prints "NO: <name>" and exits with code 1.
// Checks key.bin's header. ver: the version word of the caller's policy.txt (cpPolicyWord()).
static void cpKeyHead(const string &b, u32 ver) {
  if (b.size() < 4 * CP_KEY_HEADW) refuse("cp-key-length");
  if (cpWord(b, 0) != blk::TAG_KEY || cpWord(b, 1) != ver) refuse("cp-key-head");
}
// Configuration files. No layout.txt: a call is rejected (plan-layout) unless `-plan` lists arrays.
struct CpConf {
  string pol, layoutB;
  bool hasLayout = false;
  vector<u32> KL;
  const string *layout() const { return hasLayout ? &layoutB : nullptr; }
};
static CpConf cpConfRead(const string &folder) {
  CpConf c;
  needBytes((folder + "/policy.txt").c_str(), c.pol, "policy");
  bool odd = false;
  mustReadWords((folder + "/KL.bin").c_str(), c.KL, odd, "KL");
  if (odd) refuse("KL-length");
  const string lay = folder + "/layout.txt";
  struct stat sb;
  c.hasLayout = !lstat(lay.c_str(), &sb); // a name that exists must be readable
  if (c.hasLayout) needBytes(lay.c_str(), c.layoutB, "layout");
  return c;
}
// Checks a proof file's header. Returns k unchecked: a k without a leaf file is cp-entry-missing.
static u32 cpProofHead(const string &b, u32 ver) {
  if (b.size() < 4 * CP_PROOF_HEADW) refuse("cp-proof-length");
  if (cpWord(b, 0) != blk::TAG_PROOF || cpWord(b, 1) != ver) refuse("cp-proof-head");
  return cpWord(b, 2);
}
