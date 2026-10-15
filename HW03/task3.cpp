#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

#include "msort.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts\n";
        return 1;
    }
    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const int t = std::atoi(argv[2]);
    const std::size_t ts = std::strtoull(argv[3], nullptr, 10);

    // Random ints in [-1000, 1000]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    int *arr = new int[n];
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    omp_set_num_threads(t);

    const high_resolution_clock::time_point start = high_resolution_clock::now();
    msort(arr, n, ts);
    const high_resolution_clock::time_point end = high_resolution_clock::now();
    const duration<double, std::milli> elapsed =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << arr[0] << "\n";
    std::cout << arr[n - 1] << "\n";
    std::cout << elapsed.count() << "\n";

    delete[] arr;

    return 0;
}
