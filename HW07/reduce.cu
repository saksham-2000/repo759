#include "reduce.cuh"

#include <cuda_runtime.h>

// Reduction #4, "first add during load": each block covers 2 * blockDim.x
// elements, and each thread adds its two elements while loading them into shared
// memory. The block's partial sum is then reduced in shared memory and written
// to g_odata[blockIdx.x].
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata[];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

  float v = 0.0f;
  if (i < n) {
    v = g_idata[i];
  }
  if (i + blockDim.x < n) {
    v += g_idata[i + blockDim.x];
  }
  sdata[tid] = v;
  __syncthreads();

  // Fold the upper half onto the lower half until one value is left. Rounding
  // the half up handles block sizes that aren't powers of 2.
  for (unsigned int size = blockDim.x; size > 1;) {
    const unsigned int half = (size + 1) / 2;
    if (tid < size - half) {
      sdata[tid] += sdata[tid + half];
    }
    __syncthreads();
    size = half;
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

// Each pass shrinks the array to one partial sum per block. The input and
// output pointers are swapped after every pass, so the latest partial sums are
// always in *input; when one value is left, (*input)[0] is the total.
__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
  const size_t shmemBytes = threads_per_block * sizeof(float);
  while (N > 1) {
    const unsigned int blocks = (N + 2 * threads_per_block - 1) / (2 * threads_per_block);
    reduce_kernel<<<blocks, threads_per_block, shmemBytes>>>(*input, *output, N);
    float *tmp = *input;
    *input = *output;
    *output = tmp;
    N = blocks;
  }
  cudaDeviceSynchronize();
}
