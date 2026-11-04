#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>
#include <vector>
int main() {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
  void *workspace; cudaMalloc(&workspace, 32u << 20);
  cublasSetStream(handle, cudaStreamPerThread);
  cublasSetWorkspace(handle, workspace, 32u << 20);
  for (int m : {96, 360, 368}) for (int n : {8, 48}) for (int k : {4, 8, 16, 32, 64, 256}) {
    int8_t *a, *b; int *c;
    cudaMalloc(&a, m * k); cudaMalloc(&b, n * k); cudaMalloc(&c, m * n * sizeof(int));
    cudaMemset(a, 1, m * k); cudaMemset(b, 1, n * k); cudaMemset(c, 0, m * n * sizeof(int));
    const int alpha = 1, beta = 0;
    auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
      a, CUDA_R_8I, k, b, CUDA_R_8I, k, &beta, c, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    std::vector<int> actual(m * n);
    auto copy = cudaMemcpy(actual.data(), c, actual.size() * sizeof(int), cudaMemcpyDeviceToHost);
    bool correct = copy == cudaSuccess;
    for (int x : actual) correct = correct && x == k;
    std::printf("m=%d n=%d k=%d cublas-status=%d cuda-status=%d all-products-correct=%d\n", m, n, k, int(status), int(copy), correct);
    cudaFree(a); cudaFree(b); cudaFree(c);
  }
  cublasDestroy(handle);
  cudaFree(workspace);
}
