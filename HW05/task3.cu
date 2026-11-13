#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "vscale.cuh"

// Usage: ./task3 n [threads_per_block]   (threads_per_block defaults to 512)
int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s n [threads_per_block]\n", argv[0]);
    return 1;
  }
  unsigned int n = std::strtoul(argv[1], nullptr, 10);
  unsigned int threadsPerBlock = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 512;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
  std::uniform_real_distribution<float> distB(0.0f, 1.0f);

  float *hA = new float[n];
  float *hB = new float[n];
  for (unsigned int i = 0; i < n; i++) {
    hA[i] = distA(generator);
    hB[i] = distB(generator);
  }

  float *dA, *dB;
  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));
  cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

  unsigned int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vscale<<<numBlocks, threadsPerBlock>>>(dA, dB, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

  std::printf("%f\n%f\n%f\n", ms, hB[0], hB[n - 1]);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  delete[] hA;
  delete[] hB;
  return 0;
}
