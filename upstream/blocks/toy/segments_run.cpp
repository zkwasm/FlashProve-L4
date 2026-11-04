// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Toy three-segment proof, CPU engine; usage in segments_main.h.
#include "engine/engine_cpu.h"
#include "toy/segments_main.h"

int main(int argc, char **argv) { return blk::segments_main<blk::EngineCpu>(argc, argv); }
