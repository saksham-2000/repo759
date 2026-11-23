#include "matmul.cuh"

#include <cuda_runtime.h>

// Tiled matrix multiplication: each block computes one block_dim x block_dim
// tile of C. For each step along k, the block loads one tile of A and one tile
// of B into shared memory, and every thread adds that tile's contribution to its
// element of C. Out-of-range elements are loaded as 0, so n need not be a
// multiple of block_dim.
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
  extern __shared__ __align__(sizeof(double)) unsigned char shmem[];
  const unsigned int bd = blockDim.x;
  T *As = reinterpret_cast<T *>(shmem);  // bd x bd tile of A
  T *Bs = As + bd * bd;                  // bd x bd tile of B

  const unsigned int tx = threadIdx.x, ty = threadIdx.y;
  const unsigned int row = blockIdx.y * bd + ty;
  const unsigned int col = blockIdx.x * bd + tx;

  T sum = 0;
  const unsigned int numTiles = (n + bd - 1) / bd;
  for (unsigned int t = 0; t < numTiles; t++) {
    const unsigned int aCol = t * bd + tx;
    const unsigned int bRow = t * bd + ty;
    As[ty * bd + tx] = (row < n && aCol < n) ? A[row * n + aCol] : T(0);
    Bs[ty * bd + tx] = (bRow < n && col < n) ? B[bRow * n + col] : T(0);
    __syncthreads();

    for (unsigned int k = 0; k < bd; k++) {
      sum += As[ty * bd + k] * Bs[k * bd + tx];
    }
    __syncthreads();  // finish with this tile before the next one overwrites it
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

template <typename T>
static void matmul(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
  const unsigned int gridDim1D = (n + block_dim - 1) / block_dim;
  dim3 block(block_dim, block_dim);
  dim3 grid(gridDim1D, gridDim1D);
  size_t shmemBytes = 2 * block_dim * block_dim * sizeof(T);
  matmul_kernel<T><<<grid, block, shmemBytes>>>(A, B, C, n);
  cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
  matmul<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
  matmul<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n,
                       unsigned int block_dim) {
  matmul<double>(A, B, C, n, block_dim);
}
