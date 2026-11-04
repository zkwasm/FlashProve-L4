// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The weight commitment's parameters. hf_table_params.h: HfTableParams, for the weight table T (the
// encoded weights). hf_opening_params.h: HfParams extends it; cp_commit and cp_prove use HfParams.
#pragma once
#include "hash_commitment/hf_table_params.h"
#include "hash_commitment/hf_opening_params.h"
