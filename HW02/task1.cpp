#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

#include "scan.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " n\n";
        return 1;
    }

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    if (n == 0) {
        std::cerr << "n must be a positive integer\n";
        return 1;
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float *arr = new float[n];
    float *output = new float[n];
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }


    high_resolution_clock::time_point start = high_resolution_clock::now();
    scan(arr, output, n);
    high_resolution_clock::time_point end = high_resolution_clock::now();

    duration<double, std::milli> elapsed =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);


    std::cout << elapsed.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n - 1] << "\n";


    delete[] arr;
    delete[] output;

    return 0;
}
