// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Toy proof, CPU engine; usage in blocks/toy/toy_main.h.
#include "engine/engine_cpu.h"
#include "toy/toy_main.h"

int main(int argc, char **argv) { return blk::toy_main<blk::EngineCpu>(argc, argv); }
