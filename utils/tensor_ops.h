//
// Created by lxy on 17-12-30.
//

#ifndef MERCURY_UTILS_TENSOR_OPS_H
#define MERCURY_UTILS_TENSOR_OPS_H

#include <stdlib.h>
#include "tensor.h"

namespace mercury {

template <typename dtype>
void fill_tensor_const(Tensor<dtype>& tensor, dtype value){
    dtype* data = (dtype*)tensor.get_cpu_data_mutable();
    for (int i = 0; i < tensor.size(); ++i) {
        data[i] = value;
    }
}

template <typename dtype>
void fill_tensor_rand(Tensor<dtype>& tensor, dtype vstart, dtype vend){
    dtype* data = (dtype*)tensor.get_cpu_data_mutable();
    for (int i = 0; i < tensor.size(); ++i) {
        dtype rand_data = static_cast<dtype>(rand()) / RAND_MAX;
        data[i] = vstart + (vend - vstart) * rand_data;
    }
}

template <typename dtype>
void sum_tensor(Tensor<dtype>& tensor, float& sum){
    dtype* data = (dtype*)tensor.get_cpu_data();
    sum = static_cast<float >(0);
    for (int i = 0; i < tensor.size(); ++i) {
        sum += data[i];
    }
}


template <typename dtype>
void diff_tensor(Tensor<dtype>& tensor1, Tensor<dtype>& tensor2, float& sum, float& max_diff){
    CHECK_EQ(tensor1.size(), tensor2.size()) << "input tensor must be the same size";
    dtype* data1 = (dtype*)tensor1.get_cpu_data();
    dtype* data2 = (dtype*)tensor2.get_cpu_data();
    sum = static_cast<dtype>(0);
    max_diff = static_cast<dtype>(0);
    for (int i = 0; i < tensor1.size(); ++i) {
        float diff = fabsf(data1[i] - data2[i]);
        sum += diff;
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
}


template <typename dtype>
void print_tensor(Tensor<dtype>& tensor){
    dtype* data = (dtype*)tensor.get_cpu_data();
    printf("print tensor with size: %d\n", tensor.size());
    for (int i = 0; i < tensor.size(); ++i) {
        printf("%.2f ", static_cast<float>(data[i]));
        if((i + 1) % 10 == 0){
            printf("\n");
        }
    }
}

} // end namespace mercury
#endif //MERCURY_UTILS_TENSOR_OPS_H
