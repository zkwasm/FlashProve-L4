// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The control for plan_types_test.cpp. Each expression that it compares with `BY_HAND` must be
// ill-formed there and well-formed here, so a misspelled one fails this build. The macro makes
// `Plan`'s private members public and adds conversions between `Plan` and `Planned`.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // read by blocks/numbers/plan.h
#include "numbers/tests/plan_types_test.cpp"
