// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Compares f2b::F, BLAKE2b's compression function (prover/common/blake2b_f.h), with a list of calls
// computed by another implementation. Every call is made on the host and in a kernel: F's rotations
// are written differently for the GPU. Needs a GPU. Usage: blake2b_f_test <list> <number of lines>.
// The test also fails if a line is malformed or the list does not have <number of lines> lines;
// text after the answer on a line is ignored.
// verifier/tests/known_answers/blake2b_f.py <n> <seed> prints a list: n lines
// "h m t f -> F(h, m, t, f)" on arguments derived from the seed: state h (8 words), message block m
// (16), 128-bit counter t, last-block flag f (`last`: 0 or 1), new state (8). Lowercase
// hexadecimal, most significant digit first, 16 digits for a 64-bit word.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "common/blake2b_f.h"
struct Call {
  uint64_t h[8], m[16], t_low, t_high, answer[8];
  int last;
};
static bool words(const std::string &s, uint64_t *w, size_t n) {
  if (s.size() != 16 * n || s.find_first_not_of("0123456789abcdef") != std::string::npos) return false;
  for (size_t i = 0; i < n; i++) w[i] = strtoull(s.substr(16 * i, 16).c_str(), nullptr, 16);
  return true;
}
__global__ void k_calls(const Call *call, size_t n, uint64_t *out) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t h[8];
  for (int q = 0; q < 8; q++) h[q] = call[i].h[q];
  f2b::F(h, call[i].m, call[i].t_low, call[i].t_high, call[i].last != 0);
  for (int q = 0; q < 8; q++) out[8 * i + q] = h[q];
}
int main(int argc, char **argv) {
  if (argc != 3) return printf("usage: blake2b_f_test <the list> <the number of lines that it must have>\n"), 2;
  std::ifstream in(argv[1]);
  std::vector<Call> call;
  size_t ill = 0;
  for (std::string l; std::getline(in, l);) {
    std::istringstream s(l);
    std::string h, m, t, f, arrow, a;
    Call c;
    uint64_t tt[2];
    if (s >> h >> m >> t >> f >> arrow >> a && arrow == "->" && (f == "0" || f == "1") && words(h, c.h, 8) &&
        words(m, c.m, 16) && words(t, tt, 2) && words(a, c.answer, 8))
      c.t_high = tt[0], c.t_low = tt[1], c.last = f == "1", call.push_back(c);
    else
      ill++;
  }
  const size_t n = call.size();
  size_t host = 0, gpu = 0, bent = 0;
  for (const Call &c : call) {
    uint64_t h[8], b[8], m[16];
    for (int q = 0; q < 8; q++) h[q] = b[q] = c.h[q];
    f2b::F(h, c.m, c.t_low, c.t_high, c.last != 0);
    bool same = true;
    for (int q = 0; q < 8; q++) same = same && h[q] == c.answer[q];
    host += !same;
    // Negative control (`bent`), host only: with one bit of block m flipped the result must differ
    // from the listed answer.
    for (int q = 0; q < 16; q++) m[q] = c.m[q];
    m[5] ^= 1ull << 17;
    f2b::F(b, m, c.t_low, c.t_high, c.last != 0);
    same = true;
    for (int q = 0; q < 8; q++) same = same && b[q] == c.answer[q];
    bent += same;
  }
  Call *d = nullptr;
  uint64_t *o = nullptr;
  std::vector<uint64_t> got(8 * n + 1);
  bool ran = n && cudaMalloc(&d, n * sizeof(Call)) == cudaSuccess && cudaMalloc(&o, 64 * n) == cudaSuccess &&
             cudaMemset(o, 0, 64 * n) == cudaSuccess &&
             cudaMemcpy(d, call.data(), n * sizeof(Call), cudaMemcpyHostToDevice) == cudaSuccess;
  if (ran) {
    k_calls<<<(unsigned)((n + 127) / 128), 128>>>(d, n, o);
    ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(got.data(), o, 64 * n, cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  for (size_t i = 0; i < n; i++) {
    bool same = true;
    for (int q = 0; q < 8; q++) same = same && got[8 * i + q] == call[i].answer[q];
    gpu += !same;
  }
  const bool good = ran && !ill && n == strtoull(argv[2], nullptr, 10) && !host && !gpu && !bent;
  printf(
      "%zu calls read, %zu malformed lines; with a different answer: on the host %zu, on the GPU %zu%s; negative control, one "
      "bit of the block flipped: the same answer in %zu calls\n",
      n, ill, host, gpu, ran ? "" : " (the kernel did not run)", bent);
  printf("blake2b_f_test: %s\n", good ? "passes" : "FAILS");
  return good ? 0 : 1;
}
