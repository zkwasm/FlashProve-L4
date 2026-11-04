// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The list (`Register`) of side streams: the CUDA streams, other than the proving thread's own
// (cudaStreamPerThread), on which a sub-proof (the proof of one part of the forward pass) queues
// work for a request. The prover program cp_prove interleaves requests. When it pauses or finishes
// one, kernels on the thread's stream store the request's failure bits (prover/common/slot_flags.h:
// a request with a bit set gets no proof) and, at a pause, its state. Side-stream work must finish
// before these kernels run: a failure bit set later counts for no request or for the next one, and
// a late write lands in memory that the next request uses. A sub-proof is meant to join its side
// streams itself. As a backstop, every side stream must be registered with enter(), and
// MemberState::save() and end() (prover/prover/member_state.cuh) call join() before they queue
// these kernels.
// Public source: a stream joined to another through an event: NVIDIA's CUDA Runtime API describes
// cudaEventRecord() and cudaStreamWaitEvent() ("Event Management", "Stream Management",
// https://docs.nvidia.com/cuda/cuda-runtime-api/).
#pragma once
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace side_streams {
struct Entered {
  cudaStream_t stream;
  cudaEvent_t reached; // only `join` records it
};
struct Register {
  std::mutex m;
  std::vector<Entered> all;
};
// The functions are static, so each translation unit has a separate list, and join() covers only
// the streams registered in the same unit. cp_prove is linked from several units (prover/build.sh);
// of these, only the unit of prover/prover/cp_prove.cu registers streams or calls join().
// The list is never destroyed, because exit() may be called while a thread uses it.
static inline Register &the_register() {
  static Register *const made = new Register;
  return *made;
}
// Call enter() once for each side stream, where the stream is created. A registered stream must
// never be destroyed: entries are never removed.
// Not registered:
// - the stream that loads weights ahead of their use (prover/prover/weight_stream.cuh): it only
//   copies weights, belongs to no request and must keep running across requests;
// - the zeroing stream (prover/prover/fork_memory.h): the host waits for it right after each use;
// - the copying streams of the committer program cp_commit (prover/hash_commitment/): it proves no
//   requests.
// prover/tests/guards_stand.py compares, as text, each call under prover/ that creates a stream
// (outside tests/, examples/, docs/ and build directories) with the test's list `STREAMS`, which
// records whether the next line registers the stream; a call missing from the list fails the test.
// Which streams carry a request's work is known only from reading the sources: no code checks that
// `STREAMS` marks every such stream as registered.
static inline cudaError_t enter(cudaStream_t stream) {
  cudaEvent_t reached;
  const cudaError_t e = cudaEventCreateWithFlags(&reached, cudaEventDisableTiming);
  if (e != cudaSuccess) return e;
  Register &r = the_register();
  std::lock_guard<std::mutex> l(r.m);
  r.all.push_back({stream, reached});
  return cudaSuccess;
}
// After a successful call, work queued later on `into` runs only when all work queued earlier on
// every registered stream, by any thread, has finished. The host does not wait, and unregistered
// streams are not waited for.
// Precondition, not checked: no stream capture is open on `into` or on a registered stream. In
// cp_prove the one proving thread's captures begin and end inside pt_graphed()
// (prover/hiding/product_test.cuh), so none is open when save() or end() runs.
static inline cudaError_t join(cudaStream_t into) {
  Register &r = the_register();
  std::lock_guard<std::mutex> l(r.m); // an event's recording and the wait for it stay together
  for (const Entered &s : r.all) {
    cudaError_t e = cudaEventRecord(s.reached, s.stream);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(into, s.reached, 0);
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
} // namespace side_streams
