// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Test tool: a command line around two headers of the GPU prover, compiled for the CPU.
// prover/weights_blocks/fold_rule.h: fr::weight(), the rule that converts a stored weight to its
// committed integer.
// prover/weights_blocks/made_up_blocks.h: the synthetic model's blocks (`made_up` in identifiers).
//
// Built by the fixture `programs` in integrations/vllm/integer_model/tests/conftest.py.
#include "weights_blocks/fold_rule.h"
#include "weights_blocks/made_up_blocks.h"
#include "weights_commitment/sha256.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Usage. Arguments are not validated. Multipliers and integers are in host byte order.
//
//   block_reference decode TYPE EXPONENT BLOCKS_IN_A_ROW MULTIPLIERS
//     If MULTIPLIERS is not 0, standard input starts with one binary32 multiplier for each element
//     of a row (a block has 256 elements); otherwise every multiplier is 1. Whole rows of blocks
//     follow; a partial last row is ignored. Writes each element's integer to standard output as a
//     32-bit number, INT_MIN where the rule rejects the value.
//     TYPE is 0 for Q4_K, 1 for Q5_K, 2 for Q6_K, 3 for F32, 4 for Q8_0, 5 for F16 and 6 for BF16.
//
//   block_reference synthetic SEED STREAM LAYER FIRST_EXPERT EXPERTS ROWS_OF_ONE_EXPERT
//                             BLOCKS_IN_A_ROW
//     Prints, for the given experts of one tensor of the synthetic model, the tensor's type, the
//     SHA-256 of the blocks' bytes and the SHA-256 of the integers at the synthetic model's
//     exponent, 12, each cast to 16 bits (multipliers: the gains of the norm that feeds the tensor,
//     if any). STREAM is the prover's tensor index: 2 to 8 for attn_q, attn_k, attn_v,
//     attn_output, ffn_gate, ffn_up and ffn_down; 10 to 12 for the routed experts' ffn_gate_exps,
//     ffn_up_exps and ffn_down_exps. Other tensors take FIRST_EXPERT 0 and EXPERTS 1.

static int decode(int type, int exponent, int in_a_row, bool with_multipliers) {
  std::vector<float> multipliers((size_t)256 * in_a_row, 1.f);
  if (with_multipliers &&
      fread(multipliers.data(), 4, multipliers.size(), stdin) != multipliers.size()) {
    fprintf(stderr, "block_reference: standard input ended before all multipliers were read\n");
    return 2;
  }
  std::vector<uint8_t> row((size_t)in_a_row * fr::block_bytes(type));
  std::vector<int32_t> integers(multipliers.size());
  while (fread(row.data(), 1, row.size(), stdin) == row.size()) {
    for (size_t j = 0; j < integers.size(); j++)
      integers[j] = fr::weight(type, row.data() + (j >> 8) * fr::block_bytes(type), (int)(j & 255),
                               multipliers[j], exponent);
    if (fwrite(integers.data(), 4, integers.size(), stdout) != integers.size()) {
      fprintf(stderr, "block_reference: standard output could not be written\n");
      return 2;
    }
  }
  return 0;
}

static int synthetic(uint64_t seed, int stream, int layer, int first_expert, int experts, int rows,
                     int in_a_row) {
  const int type = made_up::type_of(stream, layer), size = fr::block_bytes(type),
            norm = made_up::norm_of(stream);
  std::vector<float> multipliers((size_t)256 * in_a_row, 1.f);
  for (size_t j = 0; norm >= 0 && j < multipliers.size(); j++) {
    const uint32_t bits = made_up::gain_bits(seed, norm, layer, j);
    memcpy(&multipliers[j], &bits, 4);
  }
  Sha256 of_bytes, of_integers;
  std::vector<uint8_t> block(size);
  for (int expert = first_expert; expert < first_expert + experts; expert++)
    for (int row = 0; row < rows; row++)
      for (int k = 0; k < in_a_row; k++) {
        made_up::block(type, seed, stream, layer, expert, row, k, block.data());
        of_bytes.update(block.data(), block.size());
        int16_t integers[256];
        for (int el = 0; el < 256; el++)
          integers[el] =
              (int16_t)fr::weight(type, block.data(), el, multipliers[(size_t)256 * k + el], 12);
        of_integers.update(integers, sizeof integers);
      }
  printf("%d %s %s\n", type, of_bytes.hex().c_str(), of_integers.hex().c_str());
  return 0;
}

int main(int argc, char **argv) {
  auto number = [&](int i) {
    return atoi(argv[i]);
  };
  if (argc == 6 && !strcmp(argv[1], "decode"))
    return decode(number(2), number(3), number(4), number(5) != 0);
  if (argc == 9 && !strcmp(argv[1], "synthetic"))
    return synthetic(strtoull(argv[2], nullptr, 10), number(3), number(4), number(5), number(6),
                     number(7), number(8));
  fprintf(stderr, "usage: block_reference decode TYPE EXPONENT BLOCKS_IN_A_ROW MULTIPLIERS\n"
                  "       block_reference synthetic SEED STREAM LAYER FIRST_EXPERT EXPERTS "
                  "ROWS_OF_ONE_EXPERT BLOCKS_IN_A_ROW\n");
  return 2;
}
