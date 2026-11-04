#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
using u32 = uint32_t;
static void checked(cudaError_t e, int line) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%d: %s\n", line, cudaGetErrorString(e)); std::exit(1); }
}
#define CK(x) checked((x), __LINE__)
#define CK_ATTRIBUTE(k, a, v) CK(cudaFuncSetAttribute((k), (a), (v)))
#include "hiding/count_keys16.cuh"

int main() {
  constexpr int identities = 3, keys_per_identity = 32768;
  std::vector<uint16_t> input(identities * keys_per_identity);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint16_t>((i * 2654435761u) ^ (i >> 3));
  uint4 *device_keys = nullptr;
  CK(cudaMalloc(&device_keys, input.size() * sizeof(uint16_t)));
  CK(cudaMemcpy(device_keys, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  for (int bits : {8, 13, 15, 16}) {
    const int bins = 1 << bits;
    std::vector<u32> expected(identities * bins, 7), actual(expected.size(), 7);
    for (int b = 0; b < identities; ++b)
      for (int i = 0; i < keys_per_identity; ++i)
        expected[b * bins + (input[b * keys_per_identity + i] & (bins - 1))]++;
    u32 *device_counts = nullptr;
    CK(cudaMalloc(&device_counts, actual.size() * sizeof(u32)));
    CK(cudaMemcpy(device_counts, actual.data(), actual.size() * sizeof(u32), cudaMemcpyHostToDevice));
    if (bits == 16) count_keys16(device_keys, keys_per_identity / 8, 5, identities, device_counts, 0);
    else count_keys_l4(device_keys, keys_per_identity / 8, 5, identities, bits, device_counts, 0);
    CK(cudaMemcpy(actual.data(), device_counts, actual.size() * sizeof(u32), cudaMemcpyDeviceToHost));
    if (actual != expected) { std::fprintf(stderr, "histogram differs at %d bits\n", bits); return 1; }
    CK(cudaFree(device_counts));
  }
  CK(cudaFree(device_keys));
  std::puts("L4 histograms: four key widths, three identities, every bin matches CPU");
}
