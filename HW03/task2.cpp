#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

#include "convolution.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n";
        return 1;
    }
    const std::size_t n = std::strtoull(argv[1], nullptr, 10);
    const int t = std::atoi(argv[2]);
    const std::size_t m = 3;

    // Same data ranges as HW02 task2: image in [-10, 10], mask in [-1, 1]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> image_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);

    float *image = new float[n * n];
    for (std::size_t i = 0; i < n * n; i++) {
        image[i] = image_dist(gen);
    }

    float *mask = new float[m * m];
    for (std::size_t i = 0; i < m * m; i++) {
        mask[i] = mask_dist(gen);
    }

    float *output = new float[n * n];

    omp_set_num_threads(t);

    const high_resolution_clock::time_point start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    const high_resolution_clock::time_point end = high_resolution_clock::now();
    const duration<double, std::milli> elapsed =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";
    std::cout << elapsed.count() << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
