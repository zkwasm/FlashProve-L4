// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// hz::alloc() allocates a GPU buffer in one of four places, so that a process with many sub-proofs
// (parts of a request's proof; `fork` in identifiers) can reuse the same memory for them. A
// sub-proof is set up once and may run many times. By default alloc() calls cudaMalloc; the prover
// program cp_prove sets alloc_hook to call its allocator, fork_alloc() in
// prover/prover/fork_memory.h. This file enforces none of the rules stated at the places below.
#pragma once
namespace hz {
// Public source: TensorFlow Lite Micro's memory classes by lifetime, reused in turn:
// R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems", 2020,
// sections 4.4.1 and 4.5, https://arxiv.org/abs/2010.08678.
// Here the users are sub-proofs, and there are two more classes.
enum Place {
  /* Scratch memory of one run of a sub-proof; after the run its content has no meaning. Set-up may
     write it only with cudaMemset, cudaMemcpy or their Async forms: cp_prove's allocator records
     these calls and can repeat their writes in each run, but not a kernel's. If a `Late` object
     (prover/hiding/late.cuh) exists at the allocation, a run may use the buffer only in its second
     half. */
  // Public sources: GPU calls recorded once and repeated: T. Jain, G. Cooperman, "CRAC:
  // Checkpoint-Restart Architecture for CUDA with Streams and UVM", 2020, section 2.2,
  // https://arxiv.org/abs/2008.10596; CUDA Graphs (CUDA C++ Programming Guide); here what the
  // recorded calls wrote is written again for each run because other sub-proofs use the same memory
  // in between.
  // See also: NVIDIA, CUDA Programming Guide, "CUDA Graphs",
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html.
  FORK,
  // Small state and constants that sub-proofs of one shape may share in turn. Their set-ups may use
  // kernels, but must allocate the same SHAPE buffers in the same order and write the same content.
  // cp_prove's set-up code decides which sub-proofs have one shape; its allocator exits if a size
  // differs or a later sub-proof allocates fewer buffers, and does not compare the content.
  SHAPE,
  // Buffers that later sub-proofs read. cp_prove allocates none and creates no Lasting object.
  LASTING,
  // Never shared: for example, tables of addresses or sub-proof numbers, and process-wide tables.
  OWN
};
// If prover/prover/fork_memory.h is included first, cudaMalloc below is its macro, and this
// default hook requests its Place::Shape for every buffer.
static cudaError_t (*alloc_hook)(void **, size_t, Place) = [](void **p, size_t n, Place) {
  return cudaMalloc(p, n);
};
static int g_lasting = 0;
// While a Lasting object exists (g_lasting counts them), alloc() places every buffer except an OWN
// one in LASTING. Create one only around set-up code whose buffers later sub-proofs read. g_lasting
// and alloc_hook are static: a Lasting object or a new alloc_hook has effect only in its .cu file.
struct Lasting {
  Lasting() { g_lasting++; }
  ~Lasting() { g_lasting--; }
};
// The includer must define the macro CK() first; the one in prover/common/common.h exits on a CUDA
// error. The new buffer's content is unspecified. alloc() has no counterpart: callers that free
// a buffer use cudaFree.
template <class X> static void alloc(X *&p, size_t bytes, Place k) {
  CK(alloc_hook((void **)&p, bytes, g_lasting && k != OWN ? LASTING : k));
}
} // namespace hz
