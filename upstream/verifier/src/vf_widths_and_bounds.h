// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Integer type names, and sizes and bounds used across the verifier program cp_verify: constants
// of the format, not settings.
// HD marks a function defined in a header.
#pragma once
#include "vf_digest_words.h"
#include <cstdint>
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint8_t u8;
#define HD static inline
// Sizes of the hash H (vf_earlier_hash.h; built on BLAKE2b's compression function), in 32-bit words.
// DW: a digest, the state's first 8 words (vf_digest_words.h). SW: the state (chaining value).
// BLKW: one input block.
static constexpr u32 DW = DIGEST_WORDS, SW = 16, BLKW = 32;
// TBLW: optional lines of the parameter file with the digests of four public tables
// (SILU, sigmoid, RoPE, softmax), one word per line.
// HEXW: hex digits of a printed digest.
// THPW: words in a full chunk of TH() (vf_common.h), the tree hash of files, requests and statements.
static constexpr u32 TBLW = 4 * DW, HEXW = 8 * DW, THPW = 16 * DW;
static_assert(2 * DW <= BLKW && TBLW == 32 && HEXW == 64 && THPW == 128, "the lengths of the format");
// A weight of any kind except the embedding is committed as two base-256 digits and has WEIGHT_WIDTH_BITS bits:
// -WEIGHT_BOUND .. WEIGHT_BOUND - 1. Three things are tied to these constants: the parameter file's widths `wb_X`,
// one for each weight kind X, which must be WEIGHT_WIDTH_BITS for every kind that the model has except the
// embedding (else cp_verify rejects with the name "policy-weight-kind"); the bound on a weight in
// the no-overflow conditions of the layers and the output head (q9() and h9() in cp_verify.cpp);
// and the width in the range proof of the key file `key.bin` (asserted in vf_segments.inc).
static constexpr u32 WEIGHT_WIDTH_BITS = 13, WEIGHT_BOUND = 1u << (WEIGHT_WIDTH_BITS - 1);
// The condition "down" in q9() takes f * (SMAXTOP + 1) * 2^LBU * WEIGHT_BOUND as the bound on the down
// projection's sum (f: the feed-forward width).
// SMAXTOP: largest absolute value in table SILU (gateTableEntry()), which the parameter-file line `smax`
// can only lower.
// LBU: the lookup key of u, the up projection's output, is u + 2^LBU, a row number in a
// 2^16-row table (lookup segment "UT", vf_layers_check.inc), so |u| <= 2^LBU.
static constexpr u32 SMAXTOP = 4095, LBU = 15;
