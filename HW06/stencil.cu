#include "stencil.cuh"

#include <cuda_runtime.h>

// Shared memory layout (all floats, allocated dynamically at launch):
//   s_mask   [2R + 1]           the whole mask
//   s_image  [blockDim.x + 2R]  the image elements this block needs (with halo)
//   s_output [blockDim.x]       this block's output before it is written back
__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n,
                               unsigned int R) {
  extern __shared__ float shmem[];
  const unsigned int maskLen = 2 * R + 1;
  const unsigned int tileLen = blockDim.x + 2 * R;
  float *s_mask = shmem;
  float *s_image = s_mask + maskLen;
  float *s_output = s_image + tileLen;

  const unsigned int t = threadIdx.x;
  const long long blockStart = static_cast<long long>(blockIdx.x) * blockDim.x;

  // blockDim.x >= 2R + 1, so one pass loads the whole mask.
  if (t < maskLen) {
    s_mask[t] = mask[t];
  }
  // The tile is longer than the block, so each thread may load more than one element.
  // Elements outside [0, n) are taken to be 1.
  for (unsigned int j = t; j < tileLen; j += blockDim.x) {
    long long g = blockStart + j - static_cast<long long>(R);
    s_image[j] = (g >= 0 && g < static_cast<long long>(n)) ? image[g] : 1.0f;
  }
  __syncthreads();

  const long long i = blockStart + t;
  if (i < n) {
    // s_image[t + R] holds image[i], so image[i + j] is s_image[t + R + j].
    float sum = 0.0f;
    for (unsigned int k = 0; k < maskLen; k++) {
      sum += s_image[t + k] * s_mask[k];
    }
    s_output[t] = sum;
    output[i] = s_output[t];
  }
}

__host__ void stencil(const float *image, const float *mask, float *output, unsigned int n,
                      unsigned int R, unsigned int threads_per_block) {
  unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
  size_t shmemBytes = ((2 * R + 1) + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);
  stencil_kernel<<<blocks, threads_per_block, shmemBytes>>>(image, mask, output, n, R);
  cudaDeviceSynchronize();
}
