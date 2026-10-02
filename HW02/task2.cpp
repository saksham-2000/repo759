#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

#include "convolution.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n m\n";
        return 1;
    }

    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const std::size_t m = std::strtoull(argv[2], nullptr, 10);
    if (n == 0 || m == 0 || m % 2 == 0) {
        std::cerr << "n must be a positive integer and m a positive odd integer\n";
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> image_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);

    // i) n x n image in [-10.0, 10.0], row-major
    float *image = new float[n * n];
    for (std::size_t i = 0; i < n * n; i++) {
        image[i] = image_dist(gen);
    }

    // ii) m x m mask in [-1.0, 1.0], row-major
    float *mask = new float[m * m];
    for (std::size_t i = 0; i < m * m; i++) {
        mask[i] = mask_dist(gen);
    }

    float *output = new float[n * n];

    // iii) convolve, timing only the convolve call
    high_resolution_clock::time_point start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    high_resolution_clock::time_point end = high_resolution_clock::now();

    duration<double, std::milli> elapsed =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // iv) time in ms, v) first element, vi) last element
    std::cout << elapsed.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";

    // vii) deallocate
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
