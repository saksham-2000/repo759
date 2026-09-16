#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    const int N = std::atoi(argv[1]);


    for (int i = 0; i <= N; i++) {
        std::printf(i == 0 ? "%d" : " %d", i);
    }
    std::printf("\n");

  
    for (int i = N; i >= 0; i--) {
        if (i != N) {
            std::cout << " ";
        }
        std::cout << i;
    }
    std::cout << "\n";

    return 0;
}
