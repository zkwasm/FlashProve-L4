// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Headers shared by several programs pass a weight table's masks (each row's random entries),
// Merkle-leaf salts and pads through this macro. The program cp_commit's first include,
// prover/hash_commitment/hf_commit_derivation_check.cuh, defines it to compare them with a second
// implementation of their derivation (described there). Unless defined earlier, the macro is empty
// and compares nothing. Definitions supply the semicolon; a use has none, so the empty form leaves
// no token:
//     HELD_TO_THE_DERIVATION_CHECK(table_salts(..))
#pragma once
#ifndef HELD_TO_THE_DERIVATION_CHECK
#define HELD_TO_THE_DERIVATION_CHECK(call)
#endif
