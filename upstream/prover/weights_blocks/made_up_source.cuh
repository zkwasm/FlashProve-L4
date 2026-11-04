// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A BlockSource (set-up reads a model through it: block_weights.h, which lists the tensors' stream
// numbers) for the synthetic model of made_up_blocks.h: tensor() computes the bytes on the GPU. The
// lie:: calls are test hooks (prover/common/lie.cuh): in this repository they alter no weight.
#pragma once
#include "common/lie.cuh"
#include "weights_blocks/block_weights.h"
#include "weights_blocks/made_up_blocks.h"

namespace wb {

// Writes a synthetic tensor's quantized blocks to out. A block has made_up::block_bytes(type) bytes.
// A row of the tensor has per_row blocks of 256 inputs. Each expert has rows_per_expert consecutive
// rows. GRID_LOOP (prover/prover/forward.cuh) is a grid-stride loop.
__global__ void k_made_up_blocks(int type, u64 seed, int stream, int layer, size_t rows_per_expert, int per_row,
                                 size_t blocks, uint8_t *out
) {
  GRID_LOOP(i, blocks) {
    const size_t row = i / per_row;
    made_up::block(type, seed, stream, layer, row / rows_per_expert, row % rows_per_expert, i % per_row,
                   out + i * made_up::block_bytes(type));
  }
}
// Sets out[row][input] to synth_weight() / 4096. synth_weight() (prover/prover/forward.cuh) maps an index to
// a signed integer of `bits` bits. Indices start at base and run as [input][row] if transposed.
__global__ void k_made_up_f32(u64 seed, int stream, int bits, u64 base, size_t rows, int inputs, bool transposed,
                              float *out) {
  GRID_LOOP(i, rows * inputs) {
    const size_t row = i / inputs, in = i % inputs;
    // Test hook, the identity in this repository; it is told this integer's row, input and stream.
    out[i] = (float)lie::at(lie::CODE, (long long)row, (long long)in, synth_weight(seed, stream, base + (transposed ? in * rows + row : i), bits), stream) *
             (1.f / 4096.f);
  }
}
// Writes the n F32 gains of a layer's norm to out. norm: 0 attention's, 1 the feed-forward norm.
__global__ void k_made_up_gains(u64 seed, int norm, int layer, int n, float *out) {
  GRID_LOOP(j, n) out[j] = __uint_as_float(made_up::gain_bits(seed, norm, layer, j));
}

// seed and shape must be set before the first call. Gains are never freed. Router (stream 9),
// embedding (13) and head (14) are F32: the integers that the prover uses with GeneratedWeights
// (prover/prover/weights_generated.cuh; router: parameter file without "cw 14") divided by scale().
struct MadeUpSource : BlockSource {
  u64 seed = 0;
  Shape shape;

  int type(int stream, int layer) override {
    return stream == 9 || stream >= 13 ? (int)F32 : made_up::type_of(stream, layer);
  }
  // False for one of each layer's seven matrices of streams 2 to 8. BlockWeights::fill() stores it
  // as byte planes and cells, not as blocks (see Form), so tests cover both cases.
  bool as_blocks(int stream, int layer) override { return (layer + stream) % 7 != 3; }
  // Computes the tensor on the CUDA stream `reader`. If `into` is null, it first waits for `reader`
  // and uses the source's own buffer, which the next such call overwrites and all_read() frees.
  // Not checked here: t.inputs must be a multiple of 256, or k_made_up_f32() overruns the buffer.
  FileTensor tensor(int stream, int layer, uint8_t *into, size_t room, cudaStream_t reader, uint8_t *) override {
    FileTensor t;
    t.type = type(stream, layer);
    t.rows = shape.rows(stream, layer) * (stream >= 10 && stream <= 12 ? shape.NE : 1);
    t.inputs = shape.inputs(stream, layer);
    const size_t blocks = t.rows * (t.inputs / 256), bytes = blocks * block_bytes(t.type);
    if (into && bytes > room) {
      printf("internal error: the synthetic tensor of stream %d is larger than the buffer given for it\n", stream);
      exit(2);
    }
    if (!into) {
      CK(cudaStreamSynchronize(reader));
      if (bytes > room_bytes_) {
        if (room_) CK(fork_free(room_));
        CK(fork_alloc(&room_, bytes, Place::Own));
        room_bytes_ = bytes;
      }
      into = room_;
    }
    lie::here(layer, lie::WEIGHTS, reader);
    if (t.type == F32)
      k_made_up_f32<<<grid_blocks(t.rows * t.inputs), 256, 0, reader>>>(seed, stream, stream == 13 ? 20 : 13,
                                                                 stream == 9 ? (u64)layer * shape.d * shape.NE : 0,
                                                                 t.rows, t.inputs, stream != 13, (float *)into);
    else
      k_made_up_blocks<<<grid_blocks(blocks), 256, 0, reader>>>(t.type, seed, stream, layer, shape.rows(stream, layer),
                                                         t.inputs / 256, blocks, into);
    // Test hook, empty in this repository, so the lambda is never called. For Q4_K or Q5_K it gives
    // the address of the byte whose low half holds the low 4 bits of a weight's code (only inputs 0
    // to 31 of each 64; layout's source: top of decode.cuh).
    lie::in_memory<uint8_t>(layer, lie::WEIGHTS, lie::CODE, reader, [&](long long row, long long input, long long of_stream) -> uint8_t * {
      if (of_stream != stream || t.type > Q5_K || row < 0 || (size_t)row >= t.rows || input < 0 || input >= t.inputs || input % 64 >= 32) return nullptr;
      return into + ((size_t)row * (t.inputs / 256) + input / 256) * block_bytes(t.type) + (t.type == Q5_K ? 48 : 16) + 32 * (input % 256 / 64) + input % 32;
    });
    t.bytes = into;
    return t;
  }
  // The full reason for the NaNs below: GgufSource::gain() in gguf_source.cuh.
  const float *gain(int stream, int layer) override {
    lie::place(stream, layer, cudaStreamPerThread); // a swapped tensor needs its partner's gains
    const int norm = made_up::norm_of(stream);
    if (norm < 0) return nullptr;
    float *&g = gains_[{norm, layer}];
    if (!g) {
      CK(fork_alloc(&g, (size_t)shape.d * 4, Place::Own));
      CK(cudaMemset(g, 0xFF, (size_t)shape.d * 4)); // NaNs: set-up rejects them if the launch fails
      k_made_up_gains<<<grid_blocks(shape.d), 256>>>(seed, norm, layer, shape.d, g);
      CK(cudaStreamSynchronize(cudaStreamPerThread));
    }
    return g;
  }
  float scale(int, int) override { return 4096.f; }
  void all_read() override {
    if (room_) CK(fork_free(room_));
    room_ = nullptr, room_bytes_ = 0;
  }

private:
  uint8_t *room_ = nullptr;
  size_t room_bytes_ = 0;
  std::map<std::pair<int, int>, float *> gains_;
};

} // namespace wb
