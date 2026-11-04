// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Settings for prover/longrow/reference.h, which only tests include: it derives private random
// values. `working_word`, the version word, goes into the random values' key; GPU tests set it to
// hz::COINS_WORD. No code reads `digest_bytes`.
#pragma once
#include "hiding/long_rows_words.h"
namespace lr {
struct Params {
  uint32_t working_word = lrw::WORKING_WORD;
  int digest_bytes = lrw::DIGEST_BYTES;
};
} // namespace lr
