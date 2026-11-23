#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "matmul.cuh"

// CUDA 13 changed cudaMemPrefetchAsync to take a cudaMemLocation; this works with both.
static void prefetchToDevice(const void *ptr, size_t bytes) {
  int device;
  cudaGetDevice(&device);
#if CUDART_VERSION >= 13000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = device;
  cudaMemPrefetchAsync(ptr, bytes, location, 0);
#else
  cudaMemPrefetchAsync(ptr, bytes, device);
#endif
}

// Fills A and B with the same random integers in [-10, 10] for all three types,
// so the int, float and double results can be compared. Every partial sum stays
// below 2^24, so float and double hold these results exactly.
template <typename T>
static void run(void (*mm)(const T *, const T *, T *, unsigned int, unsigned int),
                const int *valsA, const int *valsB, unsigned int n, unsigned int block_dim,
                const char *fmt) {
  const size_t count = static_cast<size_t>(n) * n;
  const size_t bytes = count * sizeof(T);
  T *A, *B, *C;
  cudaMallocManaged(&A, bytes);
  cudaMallocManaged(&B, bytes);
  cudaMallocManaged(&C, bytes);
  for (size_t i = 0; i < count; i++) {
    A[i] = static_cast<T>(valsA[i]);
    B[i] = static_cast<T>(valsB[i]);
  }

  // Move the managed arrays to the GPU first, so page migration isn't timed.
  prefetchToDevice(A, bytes);
  prefetchToDevice(B, bytes);
  prefetchToDevice(C, bytes);
  cudaDeviceSynchronize();

  // Warm-up on a 1x1 problem (overwritten below): the first launch of each
  // kernel pays a one-time loading cost that would otherwise be timed.
  mm(A, B, C, 1, block_dim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  mm(A, B, C, n, block_dim);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::printf(fmt, C[0]);
  std::printf(fmt, C[count - 1]);
  std::printf("%f\n", ms);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

// Usage: ./task1 n block_dim
int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s n block_dim\n", argv[0]);
    return 1;
  }
  unsigned int n = std::strtoul(argv[1], nullptr, 10);
  unsigned int block_dim = std::strtoul(argv[2], nullptr, 10);
  const size_t count = static_cast<size_t>(n) * n;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_int_distribution<int> dist(-10, 10);
  int *valsA = new int[count];
  int *valsB = new int[count];
  for (size_t i = 0; i < count; i++) {
    valsA[i] = dist(generator);
    valsB[i] = dist(generator);
  }

  run<int>(matmul_1, valsA, valsB, n, block_dim, "%d\n");
  run<float>(matmul_2, valsA, valsB, n, block_dim, "%f\n");
  run<double>(matmul_3, valsA, valsB, n, block_dim, "%f\n");

  delete[] valsA;
  delete[] valsB;
  return 0;
}
