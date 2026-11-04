// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The interface through which the parts of the committer (the program `cp_commit`) receive the
// model's weights. PinnedWeights (prover/hash_commitment/hf_commit_weights.cuh) implements
// WeightWalker: it keeps each tensor once in page-locked host memory, as its source stores it. A
// pass over the weights (`walk` in identifiers) copies the tensors to the GPU chunk by chunk
// (`Piece`), converts each chunk there to the committed integers (the integer model's weights) and
// calls a Visit for it. All calls act on the calling thread's current GPU; PinnedWeights has
// buffers for one GPU only. Code that receives the weights must keep no mutable static state, must
// launch on the stream that it is given, and must allocate only in a set-up call (not checked).
//
// A unit (rf::UnitW) is one tensor, or a layer's q and k tensors together; it fills consecutive
// rows of the committed weight matrix W. Its field X is its kind: rf::unit_tensors()
// (prover/weights_commitment/fold_source.cuh) lists the tensors of each kind.
// Include prover/weights_commitment/common.cuh first: it provides rf::UnitW
// (prover/common/weight_layout.h).
#pragma once
#include "hash_commitment/hf_seed.h" // hf_key32
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
namespace hf {
// A chunk of one unit as integers in GPU memory: int16, or int32 for the embedding (kind 12), in
// the model file's order: input index fastest, then output row, then expert.
//   kinds 7, 8, 10 (routed experts)   the experts first .. first + count - 1
//   kinds 12, 13 (embedding, head)    the vocabulary rows first .. first + count - 1
//   every other kind                  the entire unit: first = 0, count = 1
// For kind 0, w0 is q and w1 is k, and k's rows follow q's rows in one buffer; for every other kind
// w1 is null. `whole`: the chunk is the entire unit.
//
// Byte planes: on request (see walk_units()), a chunk of a unit of kind 1 to 11 that is stored in
// GGUF's Q4_K or Q5_K blocks has w0 null. Then lo and hi contain the two bytes of each weight
// w = lo + 256 * hi, with lo in [-128, 127], as qblocks::decode_planes()
// (prover/weights_blocks/decode.cuh) writes them. The bytes for input kk, expert e, output x are at
// index x * ld + (e - first) * inputs + kk, with ld = count * inputs (inputs: a row's length).
// A chunk of a tensor of a wider type that decode_planes() reads (Q8_0, F16, BF16) has the same form, if
// the tensor's rows are whole runs of 256 inputs.
struct Piece {
  size_t unit = 0;              // the unit's index in the layout's list
  const rf::UnitW *u = nullptr; // the unit
  const void *w0 = nullptr, *w1 = nullptr;
  const int8_t *lo = nullptr, *hi = nullptr;
  size_t ld = 0;
  size_t first = 0, count = 1;
  bool whole = true;
};
// A tensor in Q4_K or Q5_K blocks (`type` 0 or 1), in page-locked host memory as a model's file
// stores it: file_rows rows (output row fastest, then expert) of inputs / 256 blocks each. The
// committed integer of a code (a weight's 4 or 5 stored bits) is qblocks::weight() with its
// argument G = gains[input] * scale, rounded to binary32, or G = scale if gains is null.
// Or a tensor of a wider type, divided into the same runs of 256 inputs, which
// prover/weights_blocks/decode.cuh calls blocks too: the committed integer of a decoded value v is
// qblocks::wide_weight(v, G).
struct PackedBlocks {
  const uint8_t *host = nullptr;
  int type = 0; // as in weights_blocks/decode.cuh: 0 Q4_K, 1 Q5_K, 4 Q8_0, 5 F16, 6 BF16
  size_t inputs = 0, file_rows = 0;
  const float *gains = nullptr; // on the GPU until `PinnedWeights::end`: a multiplier per input
  float scale = 1.f;            // 2^e
  size_t block_bytes() const {
    switch (type) {
    case 0: return 144;
    case 1: return 176;
    case 4: return 272;
    case 5:
    case 6: return 512;
    }
    fprintf(stderr, "cp_commit: internal error: a tensor's blocks have the type number %d, which is none of 0 (Q4_K), "
                    "1 (Q5_K), 4 (Q8_0), 5 (F16) and 6 (BF16)\n", type);
    exit(4);
  }
  size_t row_bytes() const { return inputs / 256 * block_bytes(); }
};
// Called for each chunk of a pass, in increasing order of unit, then of `first`. The visit owns
// the chunk's memory until the work that it has queued on the stream is done: the next chunk is
// decoded into the same buffer on that stream. A visit that reads the chunk on another stream, or
// on the host, must wait for that read before it returns.
typedef std::function<void(const Piece &, cudaStream_t)> Visit;
struct WeightWalker {
  static constexpr size_t WHOLE_UNITS = ~(size_t)0;
  // One pass over all units. A unit that can be split (see Piece) comes in chunks of at most
  // piece_bytes bytes of integers, but of at least one expert or vocabulary row; WHOLE_UNITS gives
  // one chunk per unit (at most 10.7 GB at full size, Llama 4 Maverick's shape). Returns when the
  // visits' stream is idle. After all visits, PinnedWeights exits with code 3 if the conversion to
  // integers reported a fault, such as a value that is not finite or an integer out of range.
  virtual void walk(size_t piece_bytes, const Visit &visit) = 0;
  // Allocates the GPU buffers that passes with this piece_bytes need, from the driver and not from
  // the pools of prover/weights_commitment/lent_memory.h. Code that then takes all free GPU memory
  // must call this first. For WHOLE_UNITS at full size: 12.1 GB (the largest unit's integers and
  // three small staging buffers). The staging buffers hold stored bytes, so they grow with the
  // file's type; the figure is for a file of the 4-bit types.
  virtual void reserve(size_t piece_bytes) = 0;
  // Zeroes and frees the GPU buffers for the chunks; the next pass allocates them again.
  // PackedBlocks::gains stays valid.
  virtual void release() = 0;
  // Like walk(), but over the units unit_from .. unit_to - 1 only and on the caller's stream. With
  // `planes` set and piece_bytes other than WHOLE_UNITS, the chunks that can come as byte planes do
  // (see Piece). Byte planes are decoded without the conversion's fault tests: ask for them only
  // after a pass without planes has converted the same weights.
  virtual void walk_units(cudaStream_t stream, size_t piece_bytes, size_t unit_from, size_t unit_to, bool planes,
                          const Visit &visit) = 0;
  // Describes the unit's tensor in host memory, for code that copies and decodes parts of it itself
  // (hf_range_windows.cuh). Returns false if the unit has two tensors, if its tensor is not stored
  // in one of PackedBlocks' types, has int32 integers or a row length that is no multiple of 256, or if
  // a test's fault-injection hook (hf_commit_doors.h) changes one of its integers: such a unit can
  // be read only through a pass.
  virtual bool blocks(size_t unit, PackedBlocks &out) = 0;
  virtual ~WeightWalker() {}
};
} // namespace hf
