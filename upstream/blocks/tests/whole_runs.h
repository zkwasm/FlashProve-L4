// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// WholeRun: one interface for tests to the two complete toy-size protocols in blocks/toy/
// (entrywise products of vectors). prove() derives witness and randomness (`coins` in identifiers)
// from `key`, so it is repeatable and its proofs hide nothing.
// Includes blocks/tests/damage.h for the tests: child processes, damage to bytes.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#define BLK_REPEATABLE_COINS_HIDE_NOTHING 1 // a test regenerates its proofs from a key
#include "coins/tests/repeatable_coins.h"
#include "engine/engine_cpu.h"
#include "tests/damage.h"
#include "toy/product.h"
#include "toy/segments.h"
#include "toy/toy_coins.h"

namespace blk {

// A run's free parameters, passed to toy_size() or segments_size() (whose `log_ell` is `log_len`).
struct RunChoice {
  Setting set;
  unsigned log_len = 10, log_n = 13, mask_log_n = 9;
  std::vector<StageChoice> further = {StageChoice{1, 1, 12, 10}, StageChoice{0, 0, 12, 12}};
  size_t links = 3; // segments (`links` in identifiers and output) of a SegmentsWholeRun
};

// Little proof of work and a low target (in bits, for each round's error and work together), so
// that proofs are short and a test can verify many.
inline RunChoice small_choice() {
  RunChoice c;
  c.set = setting_goldilocks();
  c.set.hash_block_bytes = HASH_BLOCK_BYTES;
  c.set.target = 32;
  c.set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  return c;
}

class WholeRun {
public:
  virtual ~WholeRun() = default;
  virtual const char *name() const = 0;
  // False if sizing rejected the choice (reason in `why`); then prove() must not be called.
  virtual bool planned(std::string &why) const = 0;
  // The digest of the run's parameters in the first query of the hash chain (Fiat-Shamir
  // transcript): it ties a proof to them.
  virtual std::vector<u8> bound() const = 0;
  // Appends a newline to the run's public file (parameters as text, toy_text()): the file's digest
  // changes, nothing else in the run.
  virtual void another_text() = 0;
  // Verifies from only the proof and the public file's text with `more` appended: sizes a new run
  // from that text, and rejects a text that the new run would not write.
  virtual bool verify_from_text(const std::string &more, const std::vector<u8> &proof) const = 0;
  virtual std::vector<u8> prove(u64 key) const = 0;
  virtual bool verify(const u8 *proof, size_t bytes, const char **why = nullptr) const = 0;
  bool verify(const std::vector<u8> &proof, const char **why = nullptr) const { return verify(proof.data(), proof.size(), why); }
};

class ProductRun : public WholeRun {
public:
  explicit ProductRun(const RunChoice &c) : run_(toy_size(c.set, c.log_len, c.log_n, c.mask_log_n, c.further)) {}
  const char *name() const override { return "product"; }
  void another_text() override { run_.shape.text += "\n"; }
  bool verify_from_text(const std::string &more, const std::vector<u8> &proof) const override { return toy_verify_file(run_.shape.text + more, proof.data(), proof.size()); }
  bool planned(std::string &why) const override {
    if (run_.plan->refused()) why = run_.plan->why_not();
    return !run_.plan->refused();
  }
  std::vector<u8> bound() const override {
    Binder binder(TOY_LABEL, run_.plan);
    toy_bind(binder, run_.shape, run_.stages);
    Writer w;
    w.put(binder.parameters_digest()); // the run's statement is empty
    return w.bytes();
  }
  std::vector<u8> prove(u64 key) const override {
    ToyCoins made_up(key); // the witness, and nothing else
    Table abc(1, 3, run_.shape.codes.base[0].ell);
    made_up.fill(abc.a.data(), 2 * abc.len);
    for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
    const auto coins = run_coins_that_hide_nothing(key, CoinInputs().add("the public file", run_.shape.text).add("the witness", abc.a.data(), abc.a.size()).finish());
    EngineCpu engine;
    Writer proof;
    toy_prove(run_, abc, engine, *coins, proof);
    return proof.bytes();
  }
  bool verify(const u8 *proof, size_t bytes, const char **why) const override { return toy_verify(run_, proof, bytes, why); }
  using WholeRun::verify;

private:
  ToyRun run_;
};

class SegmentsWholeRun : public WholeRun {
public:
  explicit SegmentsWholeRun(const RunChoice &c) : run_(segments_size(c.set, c.links, c.log_len, c.log_n, c.mask_log_n, c.further)) {}
  const char *name() const override { return "links"; }
  void another_text() override { run_.text += "\n"; }
  bool verify_from_text(const std::string &more, const std::vector<u8> &proof) const override { return segments_verify_file(run_.text + more, proof.data(), proof.size()); }
  bool planned(std::string &why) const override {
    if (run_.plan->refused()) why = run_.plan->why_not();
    return !run_.plan->refused();
  }
  std::vector<u8> bound() const override {
    Binder binder(SEGMENTS_LABEL, run_.plan);
    segments_bind(binder, run_);
    Writer w;
    w.put(binder.parameters_digest()); // the run's statement is empty
    return w.bytes();
  }
  std::vector<u8> prove(u64 key) const override {
    ToyCoins made_up(key); // the witness, and nothing else
    std::vector<F> x(size_t(1) << run_.vars());
    std::vector<std::vector<F>> b(run_.links(), x);
    made_up.fill(x.data(), x.size());
    for (std::vector<F> &v : b) made_up.fill(v.data(), v.size());
    CoinInputs all;
    all.add("the public file", run_.text).add("x", x.data(), x.size());
    for (const std::vector<F> &v : b) all.add("b", v.data(), v.size());
    const auto coins = run_coins_that_hide_nothing(key, all.finish());
    EngineCpu engine;
    Writer proof;
    segments_prove(run_, x, b, engine, *coins, proof);
    return proof.bytes();
  }
  bool verify(const u8 *proof, size_t bytes, const char **why) const override { return segments_verify(run_, proof, bytes, why); }
  using WholeRun::verify;

private:
  SegmentsRun run_;
};

constexpr size_t WHOLE_RUNS = 2;
inline std::unique_ptr<WholeRun> whole_run(size_t which, const RunChoice &c) {
  if (which == 0) return std::make_unique<ProductRun>(c);
  return std::make_unique<SegmentsWholeRun>(c);
}

} // namespace blk
