// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Writes the public inputs with which the audit (tools/audit/cut.py) replays seven hash chains
// (Fiat-Shamir transcripts; key file: one-time, shared, second; leaf file, `entry`: one; proof
// file: first, shared, second):
//   <chain>.parameters: the parameter bytes of its first query, the hash input that starts it;
//   <chain>.rounds.tsv, <chain>.messages.tsv: the rounds and messages of its plan.
// It reads only a configuration's three public files, through the verifier's own
// cp_with_configured() (verifier/src/cp_configured.h). Replayed with other parameter bytes than
// the prover's, a chain fails at or before its first proof of work or opening.
//   pubtool <config folder> <asked bits> <folder that stands>
// <asked bits>: a configuration with a lower target (`sec` in `policy.txt`) is rejected:
// `NO: security-bits`, exit code 1.
// <folder that stands>: an existing directory without any of these files; nothing is overwritten.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "cp_configured.h"
#include "frame/check.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"

static std::string rd(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fprintf(stderr, "pubtool: cannot read %s\n", path.c_str()), exit(2);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
static FILE *create(const std::string &path) {  // "x": the open fails if anything has the name, a symbolic link too, so the tool overwrites nothing and writes through no link
  FILE *f = fopen(path.c_str(), "wbx");
  if (!f) fprintf(stderr, "pubtool: cannot create %s: %s%s\n", path.c_str(), strerror(errno), errno == EEXIST ? " (this tool overwrites nothing: name an empty directory)" : ""), exit(2);
  return f;
}
static void wr(const std::string &path, const std::vector<blk::u8> &b) {
  FILE *out = create(path);
  if (fwrite(b.data(), 1, b.size(), out) != b.size() || fclose(out)) fprintf(stderr, "pubtool: cannot write %s\n", path.c_str()), exit(2);
  printf("%s: %zu bytes\n", path.c_str(), b.size());
}
// pieces() in tools/audit/cut.py reads the columns by number.
static void lists(const std::string &stem, const blk::Planned &p) {
  FILE *f = create(stem + ".rounds.tsv");
  for (const blk::PlanRound &r : p->rounds_)
    fprintf(f, "%s\t%s\t%d\t%llu\t%u\t%zu\t%u\t%u\t%llu\t%llu\t%llu\t%u\t%llu\t%llu\t%llu\n", r.name.c_str(), r.where.c_str(), (int)r.kind, (unsigned long long)r.count, r.work_bits, r.message_bytes, r.roots,
            r.answers, (unsigned long long)r.draw_E, (unsigned long long)r.draw_nonzero, (unsigned long long)r.draw_positions, r.log_n, (unsigned long long)r.draw_trits, (unsigned long long)r.rows, (unsigned long long)r.first_list);
  fclose(f);
  f = create(stem + ".messages.tsv");
  for (const blk::PlanMessage &m : p->messages_)
    fprintf(f, "%s\t%s\t%d\t%llu\t%llu\t%d\n", m.name.c_str(), m.where.c_str(), (int)m.kind, (unsigned long long)m.bytes, (unsigned long long)m.count, (int)m.in_chain);
  fclose(f);
  printf("%s: %zu entries of rounds, %zu of messages\n", stem.c_str(), p->rounds_.size(), p->messages_.size());
}
int main(int argc, char **argv) {
  if (argc != 4) return fprintf(stderr, "usage: pubtool <config folder> <asked bits> <folder that stands>\n"), 2;
  std::error_code e;  // a variable that is empty on the command line ("$W", "$W/") must not make the tool write into the root of the file system
  if (!std::filesystem::is_directory(argv[3], e) || std::filesystem::equivalent(argv[3], "/", e))
    return fprintf(stderr, "pubtool: \"%s\" is not a directory, or it is the root of the file system: name an empty directory that exists\n", argv[3]), 2;
  const std::string config = argv[1], to = std::string(argv[3]) + "/";
  const std::string kl = rd(config + "/KL.bin");
  if (kl.size() % 4) return 2;
  std::vector<uint32_t> words(kl.size() / 4);
  for (size_t i = 0; i < words.size(); i++)
    words[i] = (uint32_t)(unsigned char)kl[4 * i] | (uint32_t)(unsigned char)kl[4 * i + 1] << 8 | (uint32_t)(unsigned char)kl[4 * i + 2] << 16 | (uint32_t)(unsigned char)kl[4 * i + 3] << 24;
  return cp_with_configured(rd(config + "/policy.txt"), rd(config + "/layout.txt"), words, atof(argv[2]), true, [&](const blk::Configured &c) {
    const blk::IndexName none{std::vector<blk::u8>(blk::DIGEST_BYTES, 0), blk::IndexRecord()};  // placeholder: enters only the statement, which is not written
    {
      blk::Binder q(blk::LABEL_FIRST_CHAIN, c.proof().chains[0].plan);
      (void)blk::bind_request_parameters(q, c);
      wr(to + "proof.first.parameters", q.parameter_bytes()), lists(to + "proof.first", c.proof().chains[0].plan);
    }
    {
      blk::Binder q(blk::LABEL_PROOF_SHARED_CHAIN, c.proof().chains[1].plan);
      q.configuration(c.fixed().configuration);
      blk::LinPublic x;
      blk::lin_bind_parameters(q, c.fixed().codes, c.fixed().lin, x);
      wr(to + "proof.shared.parameters", q.parameter_bytes()), lists(to + "proof.shared", c.proof().chains[1].plan);
    }
    {
      blk::Binder q(blk::LABEL_PROOF_SECOND_CHAIN, c.proof().chains[2].plan);
      q.configuration(c.fixed().configuration);
      wr(to + "proof.second.parameters", q.parameter_bytes()), lists(to + "proof.second", c.proof().chains[2].plan);
    }
    const blk::KeyFixed fixed = blk::key_fixed(c);
    {
      blk::Binder q(blk::LABEL_ONE_TIME_CHAIN, c.key().chains[0].plan);
      (void)blk::bind_one_time_chain(q, fixed, none);
      wr(to + "key.one-time.parameters", q.parameter_bytes()), lists(to + "key.one-time", c.key().chains[0].plan);
    }
    {
      blk::Binder q(blk::LIN_LABEL_SHARED, c.key().chains[1].plan);
      q.configuration(fixed.configuration);
      blk::LinPublic x;
      blk::lin_bind_parameters(q, fixed.codes, fixed.lin, x);
      wr(to + "key.shared.parameters", q.parameter_bytes()), lists(to + "key.shared", c.key().chains[1].plan);
    }
    {
      blk::Binder q(blk::LIN_LABEL_SECOND, c.key().chains[2].plan);
      q.configuration(fixed.configuration);
      wr(to + "key.second.parameters", q.parameter_bytes()), lists(to + "key.second", c.key().chains[2].plan);
    }
    {
      blk::Binder q(blk::LINK_LABEL, c.entry().chains.at(0).plan);
      (void)blk::link_bind(q, c.configuration(), c.sized().entry_codes, c.sized().link.S, none, none);
      wr(to + "entry.parameters", q.parameter_bytes()), lists(to + "entry", c.entry().chains.at(0).plan);
    }
    return 0;
  });
}
