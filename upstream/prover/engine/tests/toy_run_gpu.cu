// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A toy-size proof and its verification (blocks/toy/toy_main.h), with the prover on the GPU engine.
// On the same two arguments (the second fixes witness and randomness, so such a run hides nothing),
// blocks/toy/toy_run.cpp (CPU engine) must print the same proof digest;
// prover/tests/engine_twins.sh tests this, for the arguments "0 5" and "20 5" only.
// For the same key and order of draws, `CardStreamCoins` gives the same private random values as
// the CPU generator, also in GPU memory.
// coins_own_test_card() ends the process if the generator's GPU self-test fails.
#include "engine/engine_gpu.cuh"
#include "engine/stream_coins.cuh"
#include "toy/toy_main.h"

int main(int argc, char **argv) {
  blk::coins_own_test_card();
  return blk::toy_main<blk::EngineGpu, blk::CardStreamCoins>(argc, argv);
}
