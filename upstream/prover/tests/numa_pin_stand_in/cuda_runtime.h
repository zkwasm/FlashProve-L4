// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// A stand-in for CUDA's header of the same name, for prover/tests/numa_pin_test.cpp alone, so that
// the test builds with g++ on a machine without CUDA. It declares the one call of the CUDA runtime
// that prover/common/numa_pin.h makes; the test defines it. No program includes this file.
#pragma once
typedef int cudaError_t;
enum { cudaSuccess = 0 };
cudaError_t cudaDeviceGetPCIBusId(char *id, int length, int device);
