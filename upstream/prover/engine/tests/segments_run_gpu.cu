// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the toy three-segment proof, segments_main(), with the prover on the GPU engine. With the
// same two arguments (the second fixes witness and randomness, so such a run hides nothing),
// blocks/toy/segments_run.cpp (CPU engine) must print the same proof digest;
// prover/tests/engine_twins.sh tests this, for the arguments "0 5" and "20 5" only.
// `CardStreamCoins` is the CPU run's generator of private randomness, `StreamCoins`, with added
// calls that write its values into GPU memory.
// coins_own_test_card() aborts if the generator's GPU self-test fails.
#include "engine/engine_gpu.cuh"
#include "engine/stream_coins.cuh"
#include "toy/segments_main.h"

int main(int argc, char **argv) {
  blk::coins_own_test_card();
  return blk::segments_main<blk::EngineGpu, blk::CardStreamCoins>(argc, argv);
}
