// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For tests of the prover program cp_prove; a no-op unless SLOT_COUNT=1 (prover/common/setting.h).
// At the end of each sub-proof (the argument for one part of the forward pass), checks how often
// sc_door::leave() (prover/hiding/sumcheck_door.cuh) wrote each slot of the message array. Hiding
// requires each entry (`pad` in identifiers) of a sub-proof's mask row to hide at most one sent
// scalar. A scalar is hidden by the entry with its slot's number or, in a sumcheck, by entries
// with the numbers of that sumcheck's slots (in_the_clear(): by none of this row). So each slot
// must be written once, and a sumcheck's calls must fit its slots' layout: a call out of order or
// a round of the wrong degree is stopped (prover/hiding/psc_device.cuh); psc::Form and the number
// of end values are not recorded, so not checked. This file is a tripwire for the first condition,
// not an argument; a store that bypasses leave() is not counted. That write addresses depend only
// on the shape, not on a request's values, is known only from reading the code, not tested.
// Public sources: per-location access counts tested against a rule: M. Boyer, K. Skadron, W.
// Weimer, "Automated Dynamic Analysis of CUDA Programs", Third Workshop on Software Tools for
// MultiCore Systems, 2008, section 4.2, https://www.nvidia.com/docs/io/67190/stmcs08.pdf (there
// tested after each access; here writes only, tested at a sub-proof's end). Counts at a fixed
// offset, the direct mapping of shadow memory, as in "AddressSanitizer: A Fast Address Sanity
// Checker", USENIX ATC 2012, sections 2.1 and 3, by K. Serebryany, D. Bruening, A. Potapenko, D.
// Vyukov, https://www.usenix.org/conference/atc12/technical-sessions/presentation/serebryany.
#pragma once
#include "hiding/transcript.cuh"
#if SLOT_COUNT
static_assert(sizeof(T) == 24, "a slot's counts fill one element");
struct SlotTally {
  unsigned long long once; // slots written once, with a mask entry that is not zero
  // `bad`: slots not written exactly once or with a zero mask entry, and written indices above the
  // slots, where the array has no mask entry of this sub-proof. first_bad: the lowest such index.
  unsigned long long bad, first_bad;
};
// msg: 4 parts of PAD_STRIDE elements (PADOFF must equal it): slots, mask entries, scratch, counts.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_slot_tally(const T *msg, size_t slots, SlotTally *o) {
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < PAD_STRIDE; j += (size_t)gridDim.x * blockDim.x) {
    const unsigned long long written = *(const unsigned long long *)(msg + 3 * PAD_STRIDE + j);
    const bool good = j < slots ? written == 1 && !F::eq(msg[PAD_STRIDE + j], F::zero()) : !written;
    if (!good) {
      atomicAdd(&o->bad, 1ull);
      atomicMin(&o->first_bad, (unsigned long long)j);
    } else if (j < slots) {
      atomicAdd(&o->once, 1ull);
    }
  }
}
// A test that is not part of this repository sets g_slot_plant to inject ("plant") one fault just
// before the tally. 1: slot 0 is counted again. 2: the last slot's count is cleared, as if never
// written. 3: a write is counted at index `slots`. Modes 1 to 3 change only counts. 4: slot 0's
// mask entry is zeroed, after its last use on a scalar; the tally then ends the run. With a mode
// set, slot_counts_end() calls exit(2) unless 0 < slots < PAD_STRIDE, so these indices are valid.
// Public source: one injected fault per run, which the check must report: LLVM Project, libc++
// documentation, "Testing libc++", section "Testing hardening assertions",
// https://libcxx.llvm.org/TestingLibcxx.html (asks the same; there the assertion fires at the bad
// call, here the fault comes just before the tally).
static int g_slot_plant = 0;
__global__ void k_slot_plant(T *msg, int plant, size_t slots) {
  if (plant == 1) slot_count(msg, 0);
  if (plant == 2) *(unsigned long long *)(msg + 3 * PAD_STRIDE + slots - 1) = 0;
  if (plant == 3) slot_count(msg + slots, 0);
  if (plant == 4) msg[PAD_STRIDE] = F::zero();
}
#endif
// Clears msg's counts. Call it at the start of a sub-proof, where its mask entries are written.
static inline void slot_counts_begin(T *msg) {
#if SLOT_COUNT
  CK(cudaMemsetAsync(msg + 3 * PAD_STRIDE, 0, PAD_STRIDE * sizeof(T), cudaStreamPerThread));
  CK(cudaDeviceSynchronize()); // completes the clearing: the product tests' kernels run on streams of their own
#else
  (void)msg;
#endif
}
// Checks msg's counts; call it after the sub-proof's last scalar is written. Prints one line and,
// if an index breaks SlotTally's rule, a second before exit(3). slots <= PAD_STRIDE is not checked.
static inline void slot_counts_end(const char *kind, int number, T *msg, size_t slots) {
#if SLOT_COUNT
  static SlotTally *tally = nullptr;
  if (!tally) CK(cudaMallocManaged(&tally, sizeof *tally));
  CK(cudaDeviceSynchronize());
  *tally = {0, 0, ~0ull};
  if (g_slot_plant && (slots >= PAD_STRIDE || !slots)) {
    printf("slot count: %s %d: the slots fill the message array, or there are none, so the fault that the test mode asks for cannot be "
           "planted; the program ends\n", kind, number);
    exit(2);
  }
  if (g_slot_plant) k_slot_plant<<<1, 1, 0, cudaStreamPerThread>>>(msg, g_slot_plant, slots);
  k_slot_tally<<<256, 256, 0, cudaStreamPerThread>>>(msg, slots, tally);
  CK(cudaStreamSynchronize(cudaStreamPerThread));
  printf("slot count: %s %d: %zu slots: written once %llu; not as it must be %llu\n", kind, number, slots, tally->once, tally->bad);
  if (tally->bad) {
    unsigned long long written;
    T entry;
    CK(cudaMemcpy(&written, msg + 3 * PAD_STRIDE + tally->first_bad, sizeof written, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&entry, msg + PAD_STRIDE + tally->first_bad, sizeof entry, cudaMemcpyDeviceToHost));
    printf("slot count: %s %d: place %llu is the first that is not as it must be: written %llu times, its mask entry is %s; the program "
           "ends\n", kind, number, tally->first_bad, written, F::eq(entry, F::zero()) ? "zero" : "not zero");
    exit(3);
  }
#else
  (void)kind, (void)number, (void)msg, (void)slots;
#endif
}
