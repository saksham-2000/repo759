#include "msort.h"

#include <algorithm>

// Serial, non-recursive sort used below the threshold.
static void insertion_sort(int *arr, std::size_t n) {
    for (std::size_t i = 1; i < n; i++) {
        const int key = arr[i];
        std::size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

// Sorts arr[0, n) in place, using tmp[0, n) as scratch space for merging.
// The left half is sorted in a new task while the current thread sorts the right half
static void msort_rec(int *arr, int *tmp, std::size_t n, std::size_t threshold) {
    if (n < threshold || n < 2) {
        insertion_sort(arr, n);
        return;
    }

    const std::size_t half = n / 2;
#pragma omp task
    msort_rec(arr, tmp, half, threshold);
    msort_rec(arr + half, tmp + half, n - half, threshold);
#pragma omp taskwait

    std::merge(arr, arr + half, arr + half, arr + n, tmp);
    std::copy(tmp, tmp + n, arr);
}

void msort(int *arr, const std::size_t n, const std::size_t threshold) {
    int *tmp = new int[n];

    // One thread starts the recursion; the rest of the team runs the tasks.
#pragma omp parallel
#pragma omp single
    msort_rec(arr, tmp, n, threshold);

    delete[] tmp;
}
