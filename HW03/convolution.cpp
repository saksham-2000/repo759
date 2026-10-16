#include "convolution.h"

// Value of the image at (i, j), with the boundary padding from HW02:
// inside the image -> f[i, j]; exactly one index out of range (edge) -> 1;
// both indices out of range (corner) -> 0.
static float pixel(const float *image, std::size_t n, long i, long j) {
    const long N = static_cast<long>(n);
    const bool i_in = (0 <= i && i < N);
    const bool j_in = (0 <= j && j < N);

    if (i_in && j_in) {
        return image[i * N + j];
    }
    if (i_in || j_in) {
        return 1.0f;
    }
    return 0.0f;
}

// Parallel version of convolve from HW02.
// Each thread handles a block of rows of the output, so no two threads write
// the same element and no synchronization is needed inside the loops.
void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    const long half = static_cast<long>((m - 1) / 2);

#pragma omp parallel for
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    const long fi = static_cast<long>(x + i) - half;
                    const long fj = static_cast<long>(y + j) - half;
                    sum += mask[i * m + j] * pixel(image, n, fi, fj);
                }
            }
            output[x * n + y] = sum;
        }
    }
}
