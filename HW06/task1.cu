#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "matmul.cuh"

// Usage: ./task1 n threads_per_block
int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s n threads_per_block\n", argv[0]);
    return 1;
  }
  size_t n = std::strtoull(argv[1], nullptr, 10);
  unsigned int threadsPerBlock = std::strtoul(argv[2], nullptr, 10);
  size_t count = n * n;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  float *hA = new float[count];
  float *hB = new float[count];
  for (size_t i = 0; i < count; i++) {
    hA[i] = dist(generator);
    hB[i] = dist(generator);
  }

  float *dA, *dB, *dC;
  cudaMalloc(&dA, count * sizeof(float));
  cudaMalloc(&dB, count * sizeof(float));
  cudaMalloc(&dC, count * sizeof(float));
  cudaMemcpy(dA, hA, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, count * sizeof(float), cudaMemcpyHostToDevice);

  // Warm-up call on a 1x1 problem (overwritten below): the first launch in a
  // process pays a one-time module-loading cost that would otherwise be timed.
  matmul(dA, dB, dC, 1, threadsPerBlock);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul(dA, dB, dC, n, threadsPerBlock);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  float last;
  cudaMemcpy(&last, dC + count - 1, sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("%f\n%f\n", last, ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  delete[] hA;
  delete[] hB;
  return 0;
}
