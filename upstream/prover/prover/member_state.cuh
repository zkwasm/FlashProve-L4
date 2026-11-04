// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Saves and restores the GPU state of a request (`member` in identifiers) between its turns. A turn
// is its start or end, or its work on a layer group: the layers that a sub-proof (`fork`) covers
// together. Requests of a pass over the weights (batch pass) share the sub-proofs' working buffers.
#pragma once
#include "common/side_streams.h"
#include "common/slot_flags.h"
#include "prover/fork_memory.h"
#include <algorithm>
#include <map>
#include <vector>

struct StateCopy {
  char *dst;
  const char *src;
  size_t bytes;
  unsigned first; // the sum of state_pieces() over the earlier entries of its launch
};
// An entry of a copy table (a StateCopy) is copied in pieces of this many 16-byte words, one thread
// block per piece. state_pieces() returns an entry's number of pieces, at least 1.
constexpr size_t state_piece_words = 16384;
static inline unsigned state_pieces(size_t bytes) {
  return (unsigned)std::max<size_t>(1, (bytes / 16 + state_piece_words - 1) / state_piece_words);
}
// Copies the table's n entries: for each, `bytes` bytes from src to dst, including those after the
// last full 16-byte word. dst and src must be multiples of 16 (the 128-bit loads and stores need
// aligned addresses), and n at least 1. Launch 256 threads per block and one block per piece (the
// sum of state_pieces() over the entries). No dst range may overlap any src range or another dst
// range. The kernel checks none of this; MemberState::build() checks its entries' alignment.
// Public sources for the techniques of this function.
// The idea is public: vectorized loads and stores via cast pointers, the rest copied singly: "CUDA
// Pro Tip: Increase Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The compiler hints and the loop form are public: __launch_bounds__: NVIDIA, "CUDA C++ Best
// Practices Guide", "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; #pragma
// unroll: NVIDIA, "CUDA Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// the block-stride loop: shown in a listing on the same guide's page "Writing SIMT Kernels",
// "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void __launch_bounds__(256) k_state_copies(const StateCopy *table, int n) {
  int below = 0; // the number of entries e with table[e].first <= blockIdx.x
  for (int e0 = 0; e0 < n; e0 += 256)
    below += __syncthreads_count(e0 + (int)threadIdx.x < n && table[e0 + threadIdx.x].first <= blockIdx.x);
  const StateCopy c = table[below - 1];
  const size_t words = c.bytes / 16, from = (size_t)(blockIdx.x - c.first) * state_piece_words,
               to = from + state_piece_words < words ? from + state_piece_words : words;
  const uint4 *s = (const uint4 *)c.src;
  uint4 *d = (uint4 *)c.dst;
#pragma unroll 4
  for (size_t i = from + threadIdx.x; i < to; i += 256) d[i] = s[i];
  if (blockIdx.x == c.first && threadIdx.x < (c.bytes & 15))
    c.dst[words * 16 + threadIdx.x] = c.src[words * 16 + threadIdx.x];
}
// The live words, g_live_flags_device[0] (the failure word: a request with a bit set gets no proof)
// and [1] (for tests: which uses of an index found it outside its table), move with a request's
// state by atomicExch(): a copy would lose a bit set between a save and the next load. k_flags_load
// makes the words at `kept` live and ORs what it finds, set between two requests, into [2] and [3]:
// a bit in [2] fails the entire batch. Launch one thread in the request's stream, before its work.
// Public source for comparison: POSIX signals; see prover/common/slot_flags.h, g_live_flags_device.
static __global__ void k_flags_load(const unsigned *kept) {
  atomicOr(&g_live_flags_device[2], atomicExch(&g_live_flags_device[0], kept[0]));
  atomicOr(&g_live_flags_device[3], atomicExch(&g_live_flags_device[1], kept[1]));
}
// Moves the live words to `kept` and zeroes them. Launch it like k_flags_load, but after the work.
static __global__ void k_flags_save(unsigned *kept) {
  kept[0] = atomicExch(&g_live_flags_device[0], 0u);
  kept[1] = atomicExch(&g_live_flags_device[1], 0u);
}

// Stores each request's state in a separate GPU buffer, its bank. Before a turn, load() copies the
// turn's `reads` from the bank to the working buffers, at fixed addresses; afterwards save() copies
// its `writes` back. `member` is a request's index in its batch pass; only bank() checks its range.
struct MemberState {
  using Buffers = ForkMemory::Buffers;
  // A list of working buffers (address, bytes) and its part of the bank (`slot`; in other files a
  // slot is a request's place in its batch). Lists never stored at the same time may share a part.
  struct Item {
    int slot;
    Buffers list;
  };
  // What a turn needs in the working buffers at its start, and what it leaves there for later ones.
  struct Turn {
    std::vector<Item> reads, writes;
  };

  // Allocates one bank per request, each part sized for its largest list, and copies all turns'
  // copy tables to the GPU. Exits with code 4 if an entry's dst or src is not a multiple of 16.
  void build(int members, const std::vector<Turn> &turns) {
    n_turns = (int)turns.size();
    std::map<int, size_t> slot_bytes, offset;
    for (auto &t : turns)
      for (auto *items : {&t.reads, &t.writes})
        for (auto &it : *items) {
          size_t n = 0;
          for (auto &b : it.list) n += (b.second + 255) & ~(size_t)255;
          slot_bytes[it.slot] = std::max(slot_bytes[it.slot], n);
        }
    size_t bank_bytes = 0;
    for (auto &s : slot_bytes) {
      offset[s.first] = bank_bytes;
      bank_bytes += s.second;
    }
    bytes_a_member = bank_bytes;
    banks.resize(members);
    std::vector<StateCopy> table;
    in.assign((size_t)members * n_turns, {0, 0, 0});
    out.assign((size_t)members * n_turns, {0, 0, 0});
    for (int j = 0; j < members; j++) {
      CK(cudaMalloc(&banks[j], bank_bytes + 256));
      for (int t = 0; t < n_turns; t++)
        for (int save = 0; save < 2; save++) {
          const size_t first = table.size();
          unsigned blocks = 0;
          for (auto &it : save ? turns[t].writes : turns[t].reads) {
            char *at = banks[j] + offset[it.slot];
            for (auto &b : it.list) {
              if (b.second) {
                if (save)
                  table.push_back({at, (const char *)b.first, b.second, blocks});
                else
                  table.push_back({(char *)b.first, at, b.second, blocks});
                blocks += state_pieces(b.second);
              }
              at += (b.second + 255) & ~(size_t)255;
            }
          }
          (save ? out : in)[(size_t)j * n_turns + t] = {first, (unsigned)(table.size() - first), blocks};
        }
    }
    for (auto &c : table)
      if (((uintptr_t)c.dst | (uintptr_t)c.src) & 15) {
        printf("internal error: a buffer kept between a request's turns is not 16-byte aligned\n");
        exit(4);
      }
    CK(cudaMalloc(&table_dev, table.size() * sizeof(StateCopy) + 16));
    CK(cudaMemcpy(table_dev, table.data(), table.size() * sizeof(StateCopy), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&flags_dev, (size_t)members * FLAG_WORDS * sizeof(unsigned)));
  }
  // Call it before any work for a request. It makes the request's words the live ones, all zero.
  void begin(int member) const {
    CK(cudaMemsetAsync(kept_flags(member), 0, FLAG_WORDS * sizeof(unsigned), cudaStreamPerThread));
    k_flags_load<<<1, 1, 0, cudaStreamPerThread>>>(kept_flags(member));
  }
  void load(int member, int turn) const {
    launch(in[(size_t)member * n_turns + turn]);
    k_flags_load<<<1, 1, 0, cudaStreamPerThread>>>(kept_flags(member));
  }
  // Call it after a turn's work. Like load(), it only queues work; only stream order is sure to
  // separate it from the next load() or begin(), which may be the same request's. So the calling
  // thread's stream first waits for the side streams entered with side_streams::enter(), whether or
  // not a sub-proof has joined them. The caller's duty, not checked: any other stream that works
  // for a request has joined the calling thread's stream before the call. If one has not, a bit
  // that its kernel sets after the live words are saved counts as set between two requests or as
  // the next request's. A late write to the state misses the bank:
  // Prover::wipe_the_working_memory() zeroes or overwrites it before the next turn, the next load()
  // overwrites it, or the next turn's request finds it.
  void save(int member, int turn) const {
    CK(side_streams::join(cudaStreamPerThread));
    launch(out[(size_t)member * n_turns + turn]);
    k_flags_save<<<1, 1, 0, cudaStreamPerThread>>>(kept_flags(member));
  }
  // Call it after a request's last work, instead of save(). It joins the side streams like save(),
  // saves the live words, waits for the whole GPU and returns the saved failure word, without any
  // bit that a kernel sets after the save. A failed join, wait or copy to the host sets
  // FLAGS_UNREADABLE in the result. MemberState checks no kernel launch: call cudaGetLastError().
  unsigned end(int member) const {
    unsigned words[FLAG_WORDS] = {0};
    const bool joined = side_streams::join(cudaStreamPerThread) == cudaSuccess;
    k_flags_save<<<1, 1, 0, cudaStreamPerThread>>>(kept_flags(member));
    const bool read = cudaDeviceSynchronize() == cudaSuccess && joined &&
                      cudaMemcpy(words, kept_flags(member), sizeof words, cudaMemcpyDeviceToHost) == cudaSuccess;
    return words[0] | (read ? 0 : FLAGS_UNREADABLE);
  }

  // A request's entire bank, or {null, 0} if there is none.
  std::pair<void *, size_t> bank(int member) const {
    if (member < 0 || member >= (int)banks.size()) return {nullptr, 0};
    return {banks[member], bytes_a_member + 256};
  }

  // Every request's bank, and the buffer that keeps the requests' live words between their turns.
  Buffers all_banks() const {
    Buffers all;
    for (int j = 0; j < (int)banks.size(); j++) all.push_back(bank(j));
    if (flags_dev) all.push_back({flags_dev, banks.size() * FLAG_WORDS * sizeof(unsigned)});
    return all;
  }

  size_t bytes_a_member = 0;

private:
  int n_turns = 0;
  std::vector<char *> banks;
  StateCopy *table_dev = nullptr;
  static constexpr int FLAG_WORDS = 4; // two are in use; each request's words start at a multiple of 16 bytes
  unsigned *flags_dev = nullptr;       // each request's live words between its turns
  unsigned *kept_flags(int member) const { return flags_dev + (size_t)member * FLAG_WORDS; }
  struct Part {
    size_t first;             // its first entry in the table
    unsigned entries, blocks; // the numbers of its entries and of their pieces (the launch's blocks)
  };
  std::vector<Part> in, out;
  void launch(const Part &part) const {
    if (part.entries)
      k_state_copies<<<part.blocks, 256, 0, cudaStreamPerThread>>>(table_dev + part.first, (int)part.entries);
  }
};
