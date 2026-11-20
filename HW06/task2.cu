#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "stencil.cuh"

// Usage: ./task2 n R threads_per_block
int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s n R threads_per_block\n", argv[0]);
    return 1;
  }
  unsigned int n = std::strtoul(argv[1], nullptr, 10);
  unsigned int R = std::strtoul(argv[2], nullptr, 10);
  unsigned int threadsPerBlock = std::strtoul(argv[3], nullptr, 10);
  unsigned int maskLen = 2 * R + 1;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  float *hImage = new float[n];
  float *hMask = new float[maskLen];
  for (unsigned int i = 0; i < n; i++) {
    hImage[i] = dist(generator);
  }
  for (unsigned int i = 0; i < maskLen; i++) {
    hMask[i] = dist(generator);
  }

  float *dImage, *dMask, *dOutput;
  cudaMalloc(&dImage, n * sizeof(float));
  cudaMalloc(&dMask, maskLen * sizeof(float));
  cudaMalloc(&dOutput, n * sizeof(float));
  cudaMemcpy(dImage, hImage, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dMask, hMask, maskLen * sizeof(float), cudaMemcpyHostToDevice);

  // Warm-up call on a 1-element problem (overwritten below): the first launch
  // in a process pays a one-time module-loading cost that would otherwise be timed.
  stencil(dImage, dMask, dOutput, 1, R, threadsPerBlock);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  stencil(dImage, dMask, dOutput, n, R, threadsPerBlock);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  float last;
  cudaMemcpy(&last, dOutput + n - 1, sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("%f\n%f\n", last, ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dImage);
  cudaFree(dMask);
  cudaFree(dOutput);
  delete[] hImage;
  delete[] hMask;
  return 0;
}
