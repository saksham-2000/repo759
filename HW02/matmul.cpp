#include "matmul.h"

#include <algorithm>

// All functions compute C = A B for n x n matrices stored in row-major order.
// C is zeroed first since every variant accumulates into C[i][j].

// Loop order (i, j, k)
void mmul1(const double *A, const double *B, double *C, const unsigned int n) {
    std::fill_n(C, static_cast<std::size_t>(n) * n, 0.0);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Loop order (i, k, j)
void mmul2(const double *A, const double *B, double *C, const unsigned int n) {
    std::fill_n(C, static_cast<std::size_t>(n) * n, 0.0);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Loop order (j, k, i)
void mmul3(const double *A, const double *B, double *C, const unsigned int n) {
    std::fill_n(C, static_cast<std::size_t>(n) * n, 0.0);
    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int i = 0; i < n; i++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Loop order (i, j, k), with A and B as std::vector<double>
void mmul4(const std::vector<double> &A, const std::vector<double> &B,
           double *C, const unsigned int n) {
    std::fill_n(C, static_cast<std::size_t>(n) * n, 0.0);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
