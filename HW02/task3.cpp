#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main() {
    const unsigned int n = 1024;
    const std::size_t size = static_cast<std::size_t>(n) * n;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // A and B as raw arrays (mmul1-3) and as vectors (mmul4), same values
    double *A = new double[size];
    double *B = new double[size];
    double *C = new double[size];
    for (std::size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    const std::vector<double> A_vec(A, A + size);
    const std::vector<double> B_vec(B, B + size);

    std::cout << n << "\n";

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> elapsed;

    start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << elapsed.count() << "\n" << C[size - 1] << "\n";

    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << elapsed.count() << "\n" << C[size - 1] << "\n";

    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << elapsed.count() << "\n" << C[size - 1] << "\n";

    start = high_resolution_clock::now();
    mmul4(A_vec, B_vec, C, n);
    end = high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << elapsed.count() << "\n" << C[size - 1] << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
