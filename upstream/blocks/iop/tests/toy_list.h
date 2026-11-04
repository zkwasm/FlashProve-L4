// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Two names for rows over F, for tests only (send_world.h).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "base.h"
#include "code/code.h"
#include "field/field.h"

namespace blk {

using ToyRow = std::vector<F>;      // one row over F: n symbols, or ell + r coordinates
using ToyRows = std::vector<ToyRow>;  // several rows; a row over E is three of these

}  // namespace blk
