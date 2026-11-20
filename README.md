# repo759

Homework code for ECE 759: High Performance Computing for Engineering Applications (UW–Madison, Fall 2026).

Each assignment lives in its own `HWXX/` folder with no subdirectories, as the course requires. Written answers and plots are submitted on Canvas, not here.

| Folder | Topic | Files |
|---|---|---|
| `HW01` | Euler and Slurm basics | `task4.sh` (Slurm script), `task6.cpp` (prints 0…N and N…0) |
| `HW02` | Sequential C++ and scaling | `scan` (inclusive prefix sum), `convolution` (2D, m×m mask), `matmul` (four versions, `mmul1`–`mmul4`) |
| `HW03` | OpenMP | parallel `matmul` and `convolution`, `msort` (merge sort with OpenMP tasks) |
| `HW04` | N-body simulation | `task2.cpp` (serial C++ port of `nbody.py`), `task3.cpp` (OpenMP, schedule chosen at runtime) |
| `HW05` | First CUDA programs | `task1.cu` (factorials in a kernel), `task2.cu` (`a·x + y` per thread), `vscale` (element-wise product) |
| `HW06` | CUDA matmul and stencil | `matmul` (one thread per element of C), `stencil` (1D convolution using shared memory) |

Headers marked as provided (for example `vscale.cuh`, `matmul.cuh`, `stencil.cuh`) come from the course resource repo and must not be changed; they are replaced with reference copies when grading.

## Building and running

All code is built and run on the Euler cluster. Programs run on compute nodes through Slurm (`sbatch`), never on the login node. The exact compile line for each task is in that assignment's handout. The usual forms are:

```sh
# C++ (HW02–HW04; add -fopenmp for OpenMP tasks)
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1

# CUDA (HW05 onwards; needs `module load nvidia/cuda`)
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
```

CPU timings use `std::chrono::high_resolution_clock` and GPU timings use CUDA events, both reported in milliseconds. GPU programs make one untimed warm-up launch first, so one-time startup cost is not counted.
