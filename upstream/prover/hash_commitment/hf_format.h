// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Format constants of the weight commitment for the GPU programs `cp_commit` and `cp_prove`, in two
// headers that share one numbering of sections:
//   hf_table_format.h (sections 0 to 5, 9, 10): the format as a whole and the weight table (`T`).
//   hf_message_format.h (sections 1, 2, 4, 6, 8 to 11): what belongs to a batch, to a link (a message about
//     two weight tables) or to the key file's proofs, which are made once for a commitment. Neither
//     program writes the messages described there.
// The namespace `hf` and the prefixes hf_ (files too), Hf and HF_ mark these two programs' code for
// the weight commitment.
#pragma once
#include "hash_commitment/hf_table_format.h"
#include "hash_commitment/hf_message_format.h"
