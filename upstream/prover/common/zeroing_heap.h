// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Replacements for the global operator new and operator delete that zero each block before freeing
// it. The programs cp_commit and cp_prove keep secret values, such as normalization weights and
// random masks, in std::vectors. Not covered: blocks allocated directly with malloc() or
// aligned_alloc(), what precompiled code of the C++ standard library allocates and frees, and
// blocks not yet freed when the process ends.
//
// Include this file in exactly one translation unit per program. cp_prove has two that include it:
// prover/build.sh makes the operators of the second, prover/hash_commitment/hf_keeper.cu, local
// symbols. Other operators may free a block allocated here, and the reverse, if they too use
// malloc() and free(), as the default ones do; those of a pool allocator must not.
#pragma once
#include <malloc.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
// Allocates a block or throws std::bad_alloc.
static void *zeroing_heap_take(size_t bytes, size_t alignment) {
  if (!bytes) bytes = 1;
  void *p = nullptr;
  if (!alignment)
    p = malloc(bytes);
  else if (bytes <= SIZE_MAX - (alignment - 1)) // above that the rounding would wrap to 0, and a small block would come back
    p = aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
  if (!p) throw std::bad_alloc();
  return p;
}
// Zeroes the entire usable size, because not every operator delete receives the requested size.
static void zeroing_heap_give(void *p) noexcept {
  if (!p) return;
  // The empty asm statement hides the origin of p from the compiler. Under _FORTIFY_SOURCE=3, which
  // Ubuntu's g++ 13 enables by default at -O1 and above, explicit_bzero() is checked against the
  // block size that the compiler knows. When a block's allocation and release are inlined into the
  // same function, that is the requested size. Without the asm statement the C library would then
  // abort the process ("*** buffer overflow detected ***") if the usable size is larger, and no
  // further block would be zeroed. The statement costs at most one register move (g++ 13.3).
  //
  // Limit: the statement only keeps the check from seeing the size. The C library still does not
  // support writing to a block's bytes beyond the requested size (malloc_usable_size(3), section
  // CAVEATS).
  __asm__ volatile("" : "+r"(p));
  explicit_bzero(p, malloc_usable_size(p));
  free(p);
}
void *operator new(size_t n) {
  return zeroing_heap_take(n, 0);
}
void *operator new[](size_t n) {
  return zeroing_heap_take(n, 0);
}
void *operator new(size_t n, std::align_val_t a) {
  return zeroing_heap_take(n, (size_t)a);
}
void *operator new[](size_t n, std::align_val_t a) {
  return zeroing_heap_take(n, (size_t)a);
}
void *operator new(size_t n, const std::nothrow_t &) noexcept {
  return malloc(n ? n : 1);
}
void *operator new[](size_t n, const std::nothrow_t &) noexcept {
  return malloc(n ? n : 1);
}
void operator delete(void *p, const std::nothrow_t &) noexcept {
  zeroing_heap_give(p);
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  zeroing_heap_give(p);
}
void operator delete(void *p) noexcept {
  zeroing_heap_give(p);
}
void operator delete[](void *p) noexcept {
  zeroing_heap_give(p);
}
void operator delete(void *p, size_t) noexcept {
  zeroing_heap_give(p);
}
void operator delete[](void *p, size_t) noexcept {
  zeroing_heap_give(p);
}
void operator delete(void *p, std::align_val_t) noexcept {
  zeroing_heap_give(p);
}
void operator delete[](void *p, std::align_val_t) noexcept {
  zeroing_heap_give(p);
}
void operator delete(void *p, size_t, std::align_val_t) noexcept {
  zeroing_heap_give(p);
}
void operator delete[](void *p, size_t, std::align_val_t) noexcept {
  zeroing_heap_give(p);
}
