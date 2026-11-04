// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// BatchOpening runs a batch's opening in the required order. The opening is the prover's side of
// the protocol block LIN (blocks/iop/lin.h): one linear claim per request on the committed weight
// table (`Index` in identifiers; `keeper` is the object that owns it). Plain C++. Unit test:
// prover/tests/batch_opening_test.cpp (toy size).
//
// A proof has three hash chains (Fiat-Shamir transcripts): the request's first chain, which ends in
// LIN's first part and a digest, its seal; the batch's shared chain; and the request's second
// chain. The seals are leaves of a Merkle tree, the binding tree. Its root is in the shared chain's
// first query (the hash input that fixes a chain's statement); a second chain's first query
// contains the request's own seal and the shared chain's.
#pragma once
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "hash_commitment/hf_open_lin.h"
#include "iop/lin.h"
#include "prover/request_part.h"

// The caller's functions that write the shared chain's and a second chain's first query. The prover
// program cp_prove uses bind_shared_chain() and bind_second_chain() (blocks/frame/check.h), as the
// verifier does.
struct BatchBinds {
  hf::HfLinBind shared;                                                                            // (binder, the binding tree's root)
  std::function<void(blk::Binder &, const blk::Sealed &own, const blk::Sealed &shared)> second;    // (binder, the request's own seal, the shared chain's)
};

// Calls, in order: begin(); put(r) for each request; open(); second(r) for each request; end().
// A call out of order ends the process. `r` (prover/prover/request_part.h): a request's state
// between its two chains. Batches that overlap in time need separate objects, each with its own
// `engine` and `coins`.
class BatchOpening {
 public:
  // `setup`, `keeper`, `engine` and `coins` must outlive the object. `coins`: the batch's own
  // private randomness, for the binding tree's salts and unused leaves, the places (the requests'
  // leaf numbers) and the shared chain. `configuration`: the digest in every chain's first query;
  // the binding tree's hashes include it.
  BatchOpening(const hf::HfLinSetup &setup, const blk::LinBindingShape &binding, const blk::Digest &configuration, const blk::Planned &second_plan, BatchBinds binds, blk::IndexHolder &keeper,
               blk::Engine &engine, blk::Coins &coins, blk::NonceSearch search = nullptr)
      : setup_(setup), second_plan_(second_plan), binds_(std::move(binds)), keeper_(keeper), engine_(engine), coins_(coins), search_(std::move(search)),
        binding_(binding, configuration, coins) {
    // A random arrangement, so that a place says nothing of the order of arrival.
    free_.resize(size_t(1) << binding.log_leaves);
    for (size_t i = 0; i < free_.size(); i++) free_[i] = i;
    for (size_t i = free_.size(); i > 1; i--) std::swap(free_[i - 1], free_[below(i)]);
  }

  // Has `keeper` count the columns that this batch opens against the weight table's limit. False:
  // the limit allows no further batch. An abandoned batch stays counted. Call it before any
  // request's first chain starts, whose first query names the table (not checked).
  [[nodiscard]] bool begin() {
    BLK_REQUIRE(at_ == At::NEW, "the batch's opening: begin() comes first, once");
    if (!blk::lin_begin(keeper_, setup_.sized.S, begun_)) return false;
    at_ = At::FIRST_PARTS;
    return true;
  }
  // Call it when r's first chain (prover/prover/chain_run.cuh) is sealed. It sets r.place.
  void put(RequestPart &r) {
    BLK_REQUIRE(at_ == At::FIRST_PARTS && !free_.empty(), "the batch's opening: no request can join now, or the batch is full");
    r.place = free_.back();
    free_.pop_back();
    binding_.put(r.place, r.seal.digest());
  }
  // Closes the binding tree and runs the shared chain. In cp_prove, calls of `keeper` inside it
  // block until the thread that called Prover::fold_begin() (prover/prover/fold_walk.cuh) makes a
  // pass over the weights, so fold_begin() runs open() on a separate thread. If open() throws,
  // every later call except is_open() ends the process.
  void open() {
    BLK_REQUIRE(at_ == At::FIRST_PARTS, "the batch's opening: open() comes behind begin(), once");
    at_ = At::OPENING;
    hf::hf_open_lin(setup_, blk::TableTest::LATER, keeper_, std::move(begun_), binding_.root(), binds_.shared, engine_, coins_, search_, opened_);
    at_ = At::SECOND_PARTS;
  }
  // Runs r's second chain, on which r's context continues: statement, trees, and the oracle
  // messages that still await their proximity test, such as LIN's W and the index-mask message
  // (masks for r's claims on the weight table). The chain's 32 random bytes, and whatever the
  // context draws after them, come from r.context->coins: in cp_prove the request's own private
  // randomness (Prover::chain_coins(), prover/prover/chain_coins.cuh), not the batch's.
  void second(RequestPart &r) {
    BLK_REQUIRE(at_ == At::SECOND_PARTS && r.context, "the batch's opening: a second part is not due");
    blk::Binder b(blk::LABEL_PROOF_SECOND_CHAIN, second_plan_);
    binds_.second(b, r.seal, opened_.seal);
    blk::u8 bytes[blk::DIGEST_BYTES];
    r.context->coins.bytes(bytes, sizeof bytes);
    blk::Digest random_bytes;
    blk::digest_from_bytes(bytes, random_bytes);
    blk::Writer second;
    blk::ProverRounds channel(b, random_bytes, second, search_);
    {
      blk::ProverContext P = blk::lin_go_on(std::move(*r.context), channel);
      const blk::LinSecond x = blk::lin_second_prove(P, r.lin, r.lin_first, *opened_.context, opened_.shared);
      blk::lin_second_tests_prove(P, r.lin, r.lin_first, x);
      channel.finish();
      P.st.require_all_tested();  // no group and no sampled oracle message is left untested
    }
    r.second = second.bytes();
    blk::Writer path;
    binding_.write_path(path, r.place);
    r.path = path.bytes();
    r.let_go();
  }
  // The shared chain's bytes: the same in every proof of the batch.
  const std::vector<blk::u8> &shared() const {
    BLK_REQUIRE(at_ == At::SECOND_PARTS || at_ == At::OVER, "the batch's opening: the shared chain has not run");
    return opened_.bytes.bytes();
  }
  // The shared chain has run and end() is still due.
  bool is_open() const { return at_ == At::SECOND_PARTS; }
  // Releases Omega, the shared chain's oracle message that the second chains read.
  void end() {
    BLK_REQUIRE(at_ == At::SECOND_PARTS, "the batch's opening: end() comes behind open(), once");
    hf::hf_open_lin_end(opened_);
    at_ = At::OVER;
  }

 private:
  enum class At { NEW, FIRST_PARTS, OPENING, SECOND_PARTS, OVER };
  // Uniform in [0, n).
  // Source for the shuffle (the constructor) and the rejection rule: Lemire, "Fast Random Integer Generation in an
  // Interval", arXiv:1805.10941, which reviews both as known techniques: Algorithm 1; section 3.1, Algorithm 3 with
  // the variation in the section's last paragraph.
  size_t below(size_t n) {
    const blk::u64 most = ~blk::u64(0) - ~blk::u64(0) % n;  // a multiple of n
    for (;;) {
      blk::u8 b[8];
      coins_.bytes(b, sizeof b);
      const blk::u64 x = blk::word_from_bytes(b);
      if (x < most) return (size_t)(x % n);
    }
  }
  const hf::HfLinSetup &setup_;
  const blk::Planned second_plan_;
  BatchBinds binds_;
  blk::IndexHolder &keeper_;
  blk::Engine &engine_;
  blk::Coins &coins_;
  blk::NonceSearch search_;
  blk::LinBinding binding_;
  std::vector<blk::u64> free_;
  std::unique_ptr<blk::LinBegun> begun_;
  hf::HfLinOpened opened_;
  At at_ = At::NEW;
};
