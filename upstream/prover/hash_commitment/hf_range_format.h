// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Constants and formulas for HfRangePlan, the plan of lookup calls (hf_range_plan.h). WF, the range
// proof that the committer program `cp_commit` makes on the weight matrix W, has two lookups. The
// keys of each are cut by rows of W into parts, and each part is one call of a plan
// (hf_wf_keys.cuh).
#pragma once
#include "common/field_host.h" // HD
#include "hash_commitment/hf_format.h"
#include <cstddef>
#include <cstdint>

// A call has 2^lwr places and, at each place, one key for each of W's 2^lk columns: 2^n keys,
// n = lwr + lk. An identity is one of 2^iota equal shares of a call's keys, those in the columns c
// with one value of c mod 2^iota; iota = max(0, n - 28). The 28 is the default of
// HfRangePlan::make()'s identity_bits (`cp_commit` passes 30). It is neither the parameter file's
// line `lidb`, which concerns a request's lookups only, nor ELIDO (prover/common/setting.h).
constexpr int HF_RANGE_IDENTITY_BITS = 28;
// N_F is the number of calls. HfRangePlan::make() checks the bounds on lwr (HF_LWR_LEAST,
// HF_LWR_MOST of hf_message_format.h), on n and on N_F. HF_RANGE_N_MOST limits the size of a call
// that `cp_commit` accepts, and no verifier code uses it; at full size a call has up to 2^36 keys.
constexpr int HF_RANGE_N_LEAST = 10, HF_RANGE_N_MOST = 40;
constexpr int HF_RANGE_CALLS_LEAST = 2, HF_RANGE_CALLS_MOST = 4096;
// The calls cover the rows y < hf_ranged_rows(m0, m_e) of W, which are all m0 rows. The rows from
// m_e on, where the last unit (one or two tensors' rows) ends, belong to no tensor; the honest
// committer sets them to 0, and HfRangePlan::pair_map() pairs them as neighbours. Only
// HfRangePlan::make() calls this function; it stores the result as mR and mR / 2, the number of
// pairs of rows, as mP.
HD constexpr uint64_t hf_ranged_rows(uint64_t m0, uint64_t /* m_e */) {
  return m0;
}
// A call of kind B is for the claim that cells of W are in [-128, 127], one of kind W for the claim
// that weights stored in two cells, low + 256 * high, are in [-4,096, 4,095]. The lookup table has
// 2^k rows, and off is added to a value to make its key. The padding key is 0.
enum HfRangeKind : uint32_t { HF_RANGE_B = 0, HF_RANGE_W = 1 };
HD constexpr int hf_range_k(uint32_t kind) {
  return kind == HF_RANGE_B ? 8 : 13;
}
HD constexpr uint32_t hf_range_off(uint32_t kind) {
  return kind == HF_RANGE_B ? 128 : 4096;
}
constexpr uint32_t HF_RANGE_SECOND_DIGIT = 256; // factor of a pair's second row, the high cell
// What one call adds to HfRangePlan's totals, by n, iota and its k, in elements of the extension
// field E unless said. The totals describe a range file (see HfRangePlan). Outside tests, only the
// numbers of BW's columns are read (BW holds the multiplicities' bits), by RangeWindows
// (hf_range_windows.cuh). Its count_all() counts keys in 32 counters for each column, and only a
// test calls it.
constexpr uint64_t hf_range_elements(int n, int iota, int k) { // e_f: the roots Z, Y; the key side; the table side
  return 33 * ((uint64_t)1 << iota) +
         (uint64_t)(n * (n + 1) - iota * (iota + 1) + (k + 1) * (k + 12) + 2 * k + 12 + 2 * iota * (k + 2));
}
constexpr uint64_t hf_range_nrel(int n, int iota, int k) { // its product relations
  return (uint64_t)(n - iota + k + 3);
}
constexpr uint64_t hf_range_nop(int n, int iota, int k) { // its openings
  return (uint64_t)(2 * (n - iota + k + 2) + 93);
}
constexpr uint64_t hf_range_bw_columns(int iota, int k) { // Kb_f, its columns of BW, which has HF_BW_ROWS rows
  return (uint64_t)1 << (k + iota - 5);
}
constexpr uint64_t hf_range_ra_entries(int iota) { // its entries of RA
  return (uint64_t)62 << iota;
}
constexpr int HF_BW_ROWS = 1024;
// vL_f and vR_f, the two end values of the key side's last layer, have the pad slots
// base_f + hf_range_sv() and the next one, where base_f is the first pad slot of call f.
constexpr uint64_t hf_range_sv(int n, int iota) {
  return 33 * ((uint64_t)1 << iota) + (uint64_t)(n * (n + 1) - iota * (iota + 1) - 2);
}
