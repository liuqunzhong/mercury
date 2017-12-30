#include <iostream>
#include <vector>
#include "tensor.h"
#include "tensor_ops.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif //end USE_OPENMP

using namespace mercury;
using namespace std;

void test_tensor(){
    Tensor<float> tin;
    vector<int> sh_in = {20, 20, 3, 1};
    tin.init(sh_in);
    fill_tensor_const(tin, 1.f);
    print_tensor(tin);
    fill_tensor_rand(tin, 0.f, 255.f);
    print_tensor(tin);
}

int main() {
#pragma omp parallel
    {
        printf("test multi-thread\n");
    }

    test_tensor();

    return 0;
}