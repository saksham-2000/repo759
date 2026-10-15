#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

#include "matmul.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n";
        return 1;
    }
    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const int t = std::atoi(argv[2]);
    const std::size_t size = n * n;

    // Same data range as HW02 task3: uniform in [-1, 1]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];
    for (std::size_t i = 0; i < size; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    omp_set_num_threads(t);

    const high_resolution_clock::time_point start = high_resolution_clock::now();
    mmul(A, B, C, n);
    const high_resolution_clock::time_point end = high_resolution_clock::now();
    const duration<double, std::milli> elapsed =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << C[0] << "\n";
    std::cout << C[size - 1] << "\n";
    std::cout << elapsed.count() << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
