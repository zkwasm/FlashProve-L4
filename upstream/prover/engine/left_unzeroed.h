// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Marks a GPU buffer that is released (returned to its memory pool or arena) without being zeroed,
// and states why its contents are public. The call follows the buffer's allocation:
//   left_unzeroed(buffer, "why what it holds is public");
// It is for three files of this directory: lkp_pieces.cuh, wf_heavy.cuh and member_heavy.cuh. Each
// promises, in the comment at its top, to zero some of its buffers before release: lkp_pieces.cuh
// and member_heavy.cuh all that have held a hidden value (in them, one that depends on the
// weights), wf_heavy.cuh those that have held lookup keys or their counts. A buffer that has held
// only functions of public values (the verifier's messages, public shapes) need not be zeroed. The
// call lets a reader, or a test, tell such a buffer from one whose zeroing was forgotten.
// Everywhere else in this directory, by device.cuh's rule, buffers with hidden values are released
// unzeroed and unmarked (products.cuh zeroes only cuBLAS's work space before release). There, a
// missing call shows nothing about what a buffer holds. That no such value reaches a proof rests on
// code writing every word of a new buffer that it later reads, a coding rule and no more
// (docs/limits-all-rows.tsv, row H6); the prover's own machine is trusted.
// The stated reason is a claim about hiding that only a reader can check: the test helper
// `LeftBehind` (below) does not count a hidden value left in a marked buffer.
#pragma once
#include <cstddef>
#ifdef BLK_LEFT_BEHIND_LOOK_FOR_A_TEST
#include <functional>
#endif

// Unless BLK_LEFT_BEHIND_LOOK_FOR_A_TEST is defined, the call is an empty inline function, and the
// reason serves only the reader of the source. Measured once: the committer program cp_commit, for
// the Goldilocks field, was built with its thirteen calls and again with those lines emptied. The
// calls added no function call, string or instruction, and the device code was the same (SASS, PTX
// and ELF, apart from the compiler's number for the translation unit). 15 bytes of host code
// differed, in four equality comparisons whose operands were swapped. The prover program cp_prove
// compiles five of the calls and was not compared.
// A test defines the macro before its first include; the call then goes to the calling thread's
// listener, if one is set. blocks/fence.h lists the macro among those that do not compile with
// SERVING_BUILD, which the production build (prover/build.sh) defines. `LeftBehind` in
// prover/engine/tests/left_behind.cuh sets the listener. It counts the non-zero 64-bit words that
// released buffers leave in its arena (prover/engine/arena.h), except those of marked buffers.
namespace blk {
#ifdef BLK_LEFT_BEHIND_LOOK_FOR_A_TEST
// The calling thread's listener, empty until a test sets it. Arguments: address, bytes, reason.
inline std::function<void(const void *, size_t, const char *)> &left_unzeroed_listener() {
  static thread_local std::function<void(const void *, size_t, const char *)> listener;
  return listener;
}
// buffer: a `Card` (device.cuh) or `Pooled` (pooled.cuh), already allocated; an empty one is
// ignored. `LeftBehind` marks that allocation only: call again after resize().
// ground: the reason. `LeftBehind` keeps the pointer, so pass a string literal.
template <class Buffer> inline void left_unzeroed(const Buffer &buffer, const char *ground) {
  if (const auto &listener = left_unzeroed_listener(); listener && buffer.size()) listener(buffer.get(), buffer.size() * sizeof(*buffer.get()), ground);
}
#else
template <class Buffer> inline void left_unzeroed(const Buffer &, const char *) {}
#endif
}  // namespace blk
