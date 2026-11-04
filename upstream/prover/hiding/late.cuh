// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A sub-proof ("fork" in identifiers) covers one part of the forward pass. It is set up once and
// may run many times; each run has two halves. While a Late object exists, a program's allocation
// hooks may treat certain new GPU buffers (see g_late) as used only by the second half, and place
// them in memory that otherwise only the first half uses. The hooks are hz::alloc_hook
// (prover/hiding/place.cuh) and pt_alloc (prover/hiding/product_test.cuh). The prover, cp_prove,
// sets both to do this (prover/prover/inputs.cuh); the defaults ignore Late objects.
#pragma once
namespace hz {
// The number of Late objects. While it is nonzero, a program's hooks may treat a new buffer as
// Late if its place is FORK (a run's scratch memory, place.cuh) or it comes from pt_alloc()
// (product_test.cuh) with kind 1 or 2 (a run's data and scratch). Unchecked here: the first half
// must not use a Late buffer, and set-up may write it only with cudaMemset and cudaMemcpy. Static,
// like the hooks below: one copy per .cu file. Not atomic; a Late object also marks what other host
// threads allocate. Do not copy a Late: a copy only decrements g_late.
static int g_late = 0;
struct Late {
  Late() { g_late++; }
  ~Late() { g_late--; }
};
// Set-up may call region_hook() once, for a region of FORK memory for buffers that only the first
// half uses. In prover/attn/front.cuh these are a lookup forest's scratch memory and keys, held in
// a PtRegion (prover/hiding/product_test.cuh). A program's hooks may place the Late buffers that
// set-up allocates after this call in the same region, as far as they fit. The run calls
// late_hook() where the second half starts: from then on only Late buffers use the region. The
// default region_hook() returns memory from cudaMalloc, in which no Late buffer is placed. If the
// hook returns nullptr, there is no region, and the code that called it allocates the first half's
// buffers one by one.
static void *(*region_hook)(size_t bytes) = [](size_t bytes) -> void * {
  void *p = nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
};
static void (*late_hook)() = [] {
};
} // namespace hz
