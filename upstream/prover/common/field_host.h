// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Short names for fixed-width integer types.
// The macro HD marks a function that runs on both the host and the GPU. In plain C++ it expands to
// `static inline`, which would make a member function static: in code also compiled as plain C++,
// use it on free functions only.
#pragma once
#include <cstdint>
typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif
