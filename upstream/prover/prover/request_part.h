// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// What a request keeps between the seal (closing digest) of its first hash chain (Fiat-Shamir
// transcript) and its second chain (chain order: batch_opening.h). `context` refers to an `Engine`
// and `Coins`, which must outlive it: the engine keeps the tables and trees that the second chain
// tests and may recompute opened leaves' salts from `Coins`.
#pragma once
#include <cstring>
#include <memory>
#include <vector>

#include "iop/lin.h"

struct RequestPart {
  std::unique_ptr<blk::Binder> binder;           // of the first chain
  std::unique_ptr<blk::Writer> first;            // the first chain's bytes
  std::unique_ptr<blk::ProverRounds> channel;    // the first chain, sealed and finished
  std::unique_ptr<blk::ProverContext> context;
  blk::LinPublic lin;                            // as in the first chain's first query
  blk::LinFirst lin_first;                       // from the first part of the weight table's test
  blk::Sealed seal;
  blk::u64 place = 0;                            // in the binding tree: set by put()
  std::vector<blk::u8> path, second;             // made by second(): the request's place, salt and path; the second chain's bytes
  // Zeroes the proof bytes; the slot's failure word (slot.h), not this call, withholds the proof
  // file. After finish_request()'s call, BatchOpening::put() and second() still run; second()
  // refills `path` and `second`.
  void withhold() {
    if (first) explicit_bzero(const_cast<blk::u8 *>(first->data()), first->size()), first->clear();
    for (std::vector<blk::u8> *v : {&path, &second}) {
      if (!v->empty()) explicit_bzero(v->data(), v->size());
      v->clear();
    }
  }
  // Frees everything but the proof bytes, `lin`, `seal` and `place`.
  void let_go() { context.reset(), channel.reset(), binder.reset(), lin_first = blk::LinFirst(); }
};
