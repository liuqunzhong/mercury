#include <iostream>
#ifdef USE_OPENMP
#include <omp.h>
#endif //end USE_OPENMP
int main() {
#pragma omp parallel
    {
        std::cout << "Hello, World!" << std::endl;
    }
    return 0;
}