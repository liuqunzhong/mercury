#include "tensor.h"

namespace mercury {

#ifdef USE_CUDA
template<typename dtype>
void Tensor<dtype>::to_cpu() {
    size_t size_of_tensor = _size * _bytes;
    //LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    switch(_mem_head){
        case UNINITIALIZED:
            //LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_GPU:

            //LOG(INFO) << "copy to cpu from gpu, entire buffer";
            //cudaMemcpyAsync(_cpu_mem->get_data_mutable(), _gpu_mem->get_data(),\
                _cpu_mem->get_capacity(), cudaMemcpyDeviceToHost);
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
    }
}

template<typename dtype>
void Tensor<dtype>::to_gpu() {
    size_t size_of_tensor = _size * _bytes;
    //LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    switch(_mem_head){
        case UNINITIALIZED:
            //LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_CPU:
            //LOG(INFO) << "copy to gpu from cpu, entire buffer";
            //cudaMemcpyAsync(_gpu_mem->get_data_mutable(), _cpu_mem->get_data(), \
                _cpu_mem->get_capacity(), cudaMemcpyHostToDevice, stream);
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
}

template<typename dtype>
const dtype* Tensor<dtype>::get_cpu_data(int index) const{
    return (const dtype*)_cpu_mem->get_data() + index;
}

template<typename dtype>
dtype* Tensor<dtype>::get_cpu_data_mutable(int index) {
    _mem_head = HEAD_AT_CPU;
    return (dtype*)_cpu_mem->get_data_mutable() + index;
}

template<typename dtype>
const dtype* Tensor<dtype>::get_gpu_data(int index) const{
    return (const dtype*)_gpu_mem->get_data() + index;
}

template<typename dtype>
dtype* Tensor<dtype>::get_gpu_data_mutable(int index) {
    _mem_head = HEAD_AT_GPU;
    return (dtype*)_gpu_mem->get_data_mutable() + index;
}

#else

template<typename dtype>
void Tensor<dtype>::to_cpu(){
}

template<typename dtype>
void Tensor<dtype>::to_gpu(){
}

template<typename dtype>
const dtype* Tensor<dtype>::get_cpu_data(int index) const {
    return (const dtype*)_cpu_mem->get_data() + index;
}

template<typename dtype>
dtype* Tensor<dtype>::get_cpu_data_mutable(int index) {
    return (dtype*)_cpu_mem->get_data_mutable() + index;
}

template<typename dtype>
const dtype* Tensor<dtype>::get_gpu_data(int index) const{
    return (const dtype*)_cpu_mem->get_data() + index;
}

template<typename dtype>
dtype* Tensor<dtype>::get_gpu_data_mutable(int index) {
    return (dtype*)_cpu_mem->get_data_mutable() + index;
}

#endif

template void Tensor<float>::to_cpu();
template void Tensor<float>::to_gpu();

template int Tensor<float>::num() const;
template int Tensor<float>::channel() const;
template int Tensor<float>::height() const;
template int Tensor<float>::width() const;


template const float* Tensor<float>::get_cpu_data(int index) const;
template float* Tensor<float>::get_cpu_data_mutable(int index);
template const float* Tensor<float>::get_gpu_data(int index) const;
template float* Tensor<float>::get_gpu_data_mutable(int index);

} //namespace mercury
