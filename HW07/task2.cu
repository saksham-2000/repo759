#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "reduce.cuh"

// Usage: ./task2 N threads_per_block
int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s N threads_per_block\n", argv[0]);
    return 1;
  }
  unsigned int N = std::strtoul(argv[1], nullptr, 10);
  unsigned int threadsPerBlock = std::strtoul(argv[2], nullptr, 10);

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float *hIn = new float[N];
  for (unsigned int i = 0; i < N; i++) {
    hIn[i] = dist(generator);
  }

  const unsigned int firstBlocks = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);
  float *dIn, *dOut;
  cudaMalloc(&dIn, N * sizeof(float));
  cudaMalloc(&dOut, firstBlocks * sizeof(float));
  cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice);

  // Warm-up launch with n = 0 (writes only dOut[0], which is overwritten below):
  // the first launch pays a one-time kernel-loading cost that would otherwise be timed.
  reduce_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(dIn, dOut, 0);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduce(&dIn, &dOut, N, threadsPerBlock);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  float sum;
  cudaMemcpy(&sum, dIn, sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("%f\n%f\n", sum, ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dIn);
  cudaFree(dOut);
  delete[] hIn;
  return 0;
}
