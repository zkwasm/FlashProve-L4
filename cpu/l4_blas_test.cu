#include <cuda_runtime.h>
#include "common/blas.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
  cudaStream_t stream;
  ck(cudaStreamCreate(&stream));
  if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS) return 1;
  unsigned cases = 0;
  for (int m : {96, 360, 368}) for (int n : {8, 48}) for (int k : {4, 8, 16, 64}) {
    const int lda = k + 4, ldb = k + 8, ldc = m + 4;
    std::vector<int8_t> a(m * lda), b(n * ldb);
    for (size_t i = 0; i < a.size(); i++) a[i] = (int)(i * 17 % 255) - 128;
    for (size_t i = 0; i < b.size(); i++) b[i] = (int)(i * 31 % 255) - 128;
    int8_t *da, *db; int *dc;
    ck(cudaMalloc(&da, a.size())); ck(cudaMalloc(&db, b.size())); ck(cudaMalloc(&dc, n * ldc * sizeof(int)));
    ck(cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
    for (int scale : {-1, 1, 3}) for (bool add : {false, true}) for (bool captured : {false, true}) {
      std::vector<int> actual(n * ldc, 23), expected = actual;
      for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        int sum = 0;
        for (int q = 0; q < k; q++) sum += (int)a[i * lda + q] * b[j * ldb + q];
        expected[j * ldc + i] = scale * sum + (add ? 23 : 0);
      }
      ck(cudaMemcpy(dc, actual.data(), actual.size() * sizeof(int), cudaMemcpyHostToDevice));
      if (captured) ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      blas::gemm_tn(h, m, n, k, da, lda, db, ldb, dc, ldc, add, scale);
      if (captured) {
        cudaGraph_t graph; cudaGraphExec_t executable;
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        ck(cudaGraphLaunch(executable, stream)); ck(cudaStreamSynchronize(stream));
        ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));
      }
      ck(cudaMemcpy(actual.data(), dc, actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
      if (actual != expected) { std::fprintf(stderr, "product differs m=%d n=%d k=%d\n", m, n, k); return 1; }
      cases++;
    }
    ck(cudaFree(da)); ck(cudaFree(db)); ck(cudaFree(dc));
  }
  cublasDestroy(h);
  ck(cudaStreamDestroy(stream));
  std::printf("L4 int8 products: %u cases match CPU, including CUDA graphs, strides, signs, scale, and addition\n", cases);
}
