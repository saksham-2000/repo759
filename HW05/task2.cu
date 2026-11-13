#include <cstdio>
#include <cuda_runtime.h>
#include <random>

__global__ void axpy(int *dA, int a) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  dA[blockIdx.x * blockDim.x + threadIdx.x] = a * x + y;
}

int main() {
  const int numBlocks = 2;
  const int threadsPerBlock = 8;
  const int n = numBlocks * threadsPerBlock;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_int_distribution<int> dist(1, 100);
  int a = dist(generator);

  int *dA;
  cudaMalloc(&dA, n * sizeof(int));
  axpy<<<numBlocks, threadsPerBlock>>>(dA, a);

  int hA[n];
  cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dA);

  for (int i = 0; i < n; i++) {
    std::printf(i == 0 ? "%d" : " %d", hA[i]);
  }
  std::printf("\n");
  return 0;
}
