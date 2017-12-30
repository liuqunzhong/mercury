

#include "devices.h"
#include "tensor.h"

namespace anakin {

namespace saber {

#ifdef USE_CUDA
// for layout NCHW
template<>
void Tensor<4, RTCUDA, float, NCHW>::to_cpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_GPU:
            //_cpu_mem->re_alloc(size_of_tensor);
            sync(false); // call event sync without blocking cpu

            if (_shape == _real_shape){
                LOG(INFO) << "copy to cpu from gpu, entire buffer";
                cudaMemcpyAsync(_cpu_mem->get_data_mutable(), _gpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyDeviceToHost, stream);
            } else{
                size_t len = sizeof(float) * _shape[3];
                LOG(INFO) << "copy to cpu from gpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float* ptr_batch_gpu = (float*)_gpu_mem->get_data() + _offset + \
                        i * _real_shape.count(1, 3);
                    float* ptr_batch_cpu = (float*)_cpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape.count(1, 3);

                    for (int j = 0; j < _shape[1]; ++j) {

                        float* ptr_channel_gpu = ptr_batch_gpu + j * _real_shape.count(2, 3);
                        float* ptr_channel_cpu = ptr_batch_cpu + j * _real_shape.count(2, 3);

                        for (int k = 0; k < _shape[2]; ++k) {

                            float* ptr_row_gpu = ptr_channel_gpu + k * _real_shape[3];
                            float* ptr_row_cpu = ptr_channel_cpu + k * _real_shape[3];
                            cudaMemcpyAsync(ptr_row_cpu, ptr_row_gpu, \
                                len, cudaMemcpyDeviceToHost, stream);
                        }
                    }
                }
            }
            _events._event.record_event();
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
    }
}

template<>
void Tensor<4, RTCUDA, float , NCHW>::to_gpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_CPU:
            //_gpu_mem->re_alloc(size_of_tensor);
            if (_shape == _real_shape){

                LOG(INFO) << "copy to gpu from cpu, entire buffer";
                cudaMemcpyAsync(_gpu_mem->get_data_mutable(), _cpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyHostToDevice, stream);

            } else{

                size_t len = sizeof(float) * _shape[3];
                LOG(INFO) << "copy to gpu from cpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float* ptr_batch_gpu = (float*)_gpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape.count(1, 3);
                    float* ptr_batch_cpu = (float*)_cpu_mem->get_data() + _offset + \
                        i * _real_shape.count(1, 3);

                    for (int j = 0; j < _shape[1]; ++j) {

                        float* ptr_channel_gpu = ptr_batch_gpu + j * _real_shape.count(2, 3);
                        float* ptr_channel_cpu = ptr_batch_cpu + j * _real_shape.count(2, 3);

                        for (int k = 0; k < _shape[2]; ++k) {

                            float* ptr_row_gpu = ptr_channel_gpu + k * _real_shape[3];
                            float* ptr_row_cpu = ptr_channel_cpu + k * _real_shape[3];
                            cudaMemcpyAsync(ptr_row_gpu, ptr_row_cpu, \
                                len, cudaMemcpyHostToDevice, stream);
                        }
                    }
                }
            }
            _events._event.record_event();
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
}

// for layout CHW
template<>
void Tensor<3, RTCUDA, float, CHW>::to_cpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_GPU:
            //_cpu_mem->re_alloc(size_of_tensor);
            sync(false);
            if (_shape == _real_shape){

                LOG(INFO) << "copy to cpu from gpu, entire buffer";
                cudaMemcpyAsync(_cpu_mem->get_data_mutable(), _gpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyDeviceToHost, stream);
            } else{
                size_t len = sizeof(float) * _shape[2];
                LOG(INFO) << "copy to cpu from cpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float *ptr_channel_gpu = (float *) _gpu_mem->get_data() + _offset + \
                        i * _real_shape.count(1, 2);
                    float *ptr_channel_cpu = (float *) _cpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape.count(1, 2);

                    for (int j = 0; j < _shape[1]; ++j) {

                        float *ptr_row_gpu = ptr_channel_gpu + j * _real_shape[2];
                        float *ptr_row_cpu = ptr_channel_cpu + j * _real_shape[2];

                        cudaMemcpyAsync(ptr_row_cpu, ptr_row_gpu, \
                                len, cudaMemcpyDeviceToHost, stream);
                    }
                }
            }
            _events._event.record_event();

            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
    }
}

template<>
void Tensor<3, RTCUDA, float , CHW>::to_gpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
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
            //_gpu_mem->re_alloc(size_of_tensor);
            if (_shape == _real_shape){

                LOG(INFO) << "copy to gpu from cpu, entire buffer";
                cudaMemcpyAsync(_gpu_mem->get_data_mutable(), _cpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyHostToDevice, stream);

            } else{

                size_t len = sizeof(float) * _shape[2];
                LOG(INFO) << "copy to gpu from cpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float* ptr_channel_gpu = (float*)_gpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape.count(1, 2);
                    float* ptr_channel_cpu = (float*)_cpu_mem->get_data() + _offset + \
                        i * _real_shape.count(1, 2);

                    for (int j = 0; j < _shape[1]; ++j) {

                        float* ptr_row_gpu = ptr_channel_gpu + j * _real_shape[2];
                        float* ptr_row_cpu = ptr_channel_cpu + j * _real_shape[2];

                        cudaMemcpyAsync(ptr_row_gpu, ptr_row_cpu, \
                                len, cudaMemcpyHostToDevice, stream);
                    }
                }
            }
            _events._event.record_event();
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
}

// for layout HW
template<>
void Tensor<2, RTCUDA, float, HW>::to_cpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_GPU:
            //_cpu_mem->re_alloc(size_of_tensor);
            sync(false);
            if (_shape == _real_shape){

                LOG(INFO) << "copy to cpu from gpu, entire buffer";
                cudaMemcpyAsync(_cpu_mem->get_data_mutable(), _gpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyDeviceToHost, stream);
            } else{

                size_t len = sizeof(float) * _shape[1];
                LOG(INFO) << "copy to cpu from gpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float *ptr_row_gpu = (float *) _gpu_mem->get_data() + _offset + \
                        i * _real_shape[1];
                    float *ptr_row_cpu = (float *) _cpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape[1];

                    cudaMemcpyAsync(ptr_row_cpu, ptr_row_gpu, \
                                len, cudaMemcpyDeviceToHost, stream);
                }
            }
            _events._event.record_event();
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
    }
}

template<>
void Tensor<2, RTCUDA, float , HW>::to_gpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
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
            //_gpu_mem->re_alloc(size_of_tensor);
            if (_shape == _real_shape){

                LOG(INFO) << "copy to gpu from cpu, entire buffer";
                cudaMemcpyAsync(_gpu_mem->get_data_mutable(), _cpu_mem->get_data(), \
                    _cpu_mem->get_capacity(), cudaMemcpyHostToDevice, stream);

            } else{

                size_t len = sizeof(float) * _shape[1];
                LOG(INFO) << "copy to gpu from cpu, block buffer, len: "<< len << \
                    ", offset: " << _offset;

                for (int i = 0; i < _shape[0]; ++i) {

                    float* ptr_row_gpu = (float*)_gpu_mem->get_data_mutable() + _offset + \
                        i * _real_shape[1];
                    float* ptr_row_cpu = (float*)_cpu_mem->get_data() + _offset + \
                        i * _real_shape[1];

                    cudaMemcpyAsync(ptr_row_gpu, ptr_row_cpu, \
                                len, cudaMemcpyHostToDevice, stream);

                }
            }
            _events._event.record_event();
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
}

// for layout W
template<>
void Tensor<1, RTCUDA, float, W>::to_cpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_GPU: {
            LOG(INFO) << "copy to gpu from cpu";
            //_cpu_mem->re_alloc(size_of_tensor);
            sync(false);
            size_t len = sizeof(float) * _shape[0];
            float *gpu_data = (float *) _gpu_mem->get_data() + _offset;
            float *cpu_data = (float *) _cpu_mem->get_data_mutable() + _offset;
            cudaMemcpyAsync(cpu_data, gpu_data, len, cudaMemcpyDeviceToHost, stream);
            _events._event.record_event();
            break;
        }
        case HEAD_AT_CPU:
        case SYNCED:
            break;
    }
}

template<>
void Tensor<1, RTCUDA, float , W>::to_gpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    cudaStream_t stream = _ctx->cuda_stream(_stream_id);
    switch(_mem_head){
        case UNINITIALIZED:
            //LOG(INFO) << "uninit, set mem to zero";
            _cpu_mem->re_alloc(size_of_tensor);
            _gpu_mem->re_alloc(size_of_tensor);
            _cpu_mem->mem_set(0, size_of_tensor);
            _gpu_mem->mem_set(0, size_of_tensor);
            _mem_head = SYNCED;
            break;
        case HEAD_AT_CPU: {
            LOG(INFO) << "copy to gpu from cpu";
            //_gpu_mem->re_alloc(size_of_tensor);
            size_t len = _shape[0] * sizeof(float);
            float *gpu_data = (float *) _gpu_mem->get_data_mutable() + _offset;
            float *cpu_data = (float *) _cpu_mem->get_data() + _offset;
            cudaMemcpyAsync(gpu_data, cpu_data, len, cudaMemcpyHostToDevice, stream);
            _events._event.record_event();
            break;
        }
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
}
/*
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void Tensor<Dim, target, dtype, layout>::to_cpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to cpu! size: "<<size_of_tensor;
    switch(_mem_head){
    case UNINITIALIZED:
        _cpu_mem->re_alloc(size_of_tensor);
        _gpu_mem->re_alloc(size_of_tensor);
        _cpu_mem->mem_set(0, size_of_tensor);
        _gpu_mem->mem_set(0, size_of_tensor);
        _mem_head = SYNCED;
        break;
    case HEAD_AT_GPU:
        _cpu_mem->re_alloc(size_of_tensor);
        cudaMemcpy(_cpu_mem->get_data_mutable(), _gpu_mem->get_data(),
                   size_of_tensor * _bytes, cudaMemcpyDeviceToHost);

        break;
    case HEAD_AT_CPU:
    case SYNCED:
        break;
    }
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
void Tensor<Dim, target, dtype, layout>::to_gpu() {
    size_t size_of_tensor = _shape.size() * _bytes;
    LOG(INFO)<<"copy to gpu! size: "<<size_of_tensor;
    switch(_mem_head){
    case UNINITIALIZED:
        _cpu_mem->re_alloc(size_of_tensor);
        _gpu_mem->re_alloc(size_of_tensor);
        _cpu_mem->mem_set(0, size_of_tensor);
        _gpu_mem->mem_set(0, size_of_tensor);
        _mem_head = SYNCED;
        break;
    case HEAD_AT_CPU:
        _gpu_mem->re_alloc(size_of_tensor);
        cudaMemcpy(_gpu_mem->get_data_mutable(), _cpu_mem->get_data(),
                size_of_tensor * _bytes, cudaMemcpyHostToDevice);
        break;
    case HEAD_AT_GPU:
    case SYNCED:
        break;
    }
}
*/

/*
 * \brief copy to cpu recursively, only copy shared tensor buffer
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void Tensor<Dim, target, dtype, layout>::recur_copy_to_cpu(){
    if (_is_leaf){
        to_cpu();
    } else {
        int size_child = 0;
        for (int i = 0; i < _children.size(); ++i) {
            _children[i]->recur_copy_to_cpu();
            size_child += _children[i]->size();
        }
        if(size_child < _real_shape.size()){
            LOG(WARNING) << "not entire data buffer copied to cpu";
        }
    }
}

/*
 * \brief copy to gpu recursively, only copy shared tensor buffer
 */
template<int Dim, TargetType target, typename dtype, LayoutType layout>
void Tensor<Dim, target, dtype, layout>::recur_copy_to_gpu(){
    if (_is_leaf){
        to_gpu();
    } else {
        int size_child = 0;
        for (int i = 0; i < _children.size(); ++i) {
            _children[i]->recur_copy_to_gpu();
            size_child += _children[i]->size();
        }
        if(size_child < _real_shape.size()){
            LOG(WARNING) << "not entire data buffer copied to gpu";
        }
    }
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
const void* Tensor<Dim, target, dtype, layout>::get_cpu_data(int index = 0) {
    CHECK_EQ(_ctx == nullptr, false) << "input tensor is not initialized";
    recur_copy_to_cpu();
    sync(true);
    return _cpu_mem->get_data() + (index + _offset) * _bytes;
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
void* Tensor<Dim, target, dtype, layout>::get_cpu_data_mutable(int index = 0) {
    CHECK_EQ(_ctx == nullptr, false) << "input tensor is not initialized";
    recur_copy_to_cpu();
    sync(true);
    _mem_head = HEAD_AT_CPU;
    return _cpu_mem->get_data_mutable() + (index + _offset) * _bytes;
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
const void* Tensor<Dim, target, dtype, layout>::get_gpu_data(int index = 0) {
    CHECK_EQ(_ctx == nullptr, false) << "input tensor is not initialized";
    recur_copy_to_gpu();
    sync(false);
    return _gpu_mem->get_data() + (index + _offset) * _bytes;
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
void* Tensor<Dim, target, dtype, layout>::get_gpu_data_mutable(int index = 0) {
    CHECK_EQ(_ctx == nullptr, false) << "input tensor is not initialized";
    recur_copy_to_gpu();
    sync(false);
    _mem_head = HEAD_AT_GPU;
    return _gpu_mem->get_data_mutable() + (index + _offset) * _bytes;
}

template <>
int Tensor<4, RTCUDA, float, NCHW>::num(){
    return _shape[0];
}
template <>
int Tensor<4, RTCUDA, float, NCHW>::channel(){
    return _shape[1];
}
template <>
int Tensor<4, RTCUDA, float, NCHW>::height(){
    return _shape[2];
}
template <>
int Tensor<4, RTCUDA, float, NCHW>::width(){
    return _shape[3];
}

template <>
int Tensor<3, RTCUDA, float, CHW>::num(){
    return 1;
}
template <>
int Tensor<3, RTCUDA, float, CHW>::channel(){
    return _shape[0];
}
template <>
int Tensor<3, RTCUDA, float, CHW>::height(){
    return _shape[1];
}
template <>
int Tensor<3, RTCUDA, float, CHW>::width(){
    return _shape[2];
}

template <>
int Tensor<2, RTCUDA, float, HW>::num(){
    return 1;
}
template <>
int Tensor<2, RTCUDA, float, HW>::channel(){
    return 1;
}
template <>
int Tensor<2, RTCUDA, float, HW>::height(){
    return _shape[0];
}
template <>
int Tensor<2, RTCUDA, float, HW>::width(){
    return _shape[1];
}

template <>
int Tensor<1, RTCUDA, float, W>::num(){
    return 1;
}
template <>
int Tensor<1, RTCUDA, float, W>::channel(){
    return 1;
}
template <>
int Tensor<1, RTCUDA, float, W>::height(){
    return 1;
}
template <>
int Tensor<1, RTCUDA, float, W>::width(){
    return _shape[0];
}

template void Tensor<4, RTCUDA, float, NCHW>::to_cpu();
template void Tensor<4, RTCUDA, float, NCHW>::to_gpu();

template void Tensor<3, RTCUDA, float, CHW>::to_cpu();
template void Tensor<3, RTCUDA, float, CHW>::to_gpu();

template void Tensor<2, RTCUDA, float, HW>::to_cpu();
template void Tensor<2, RTCUDA, float, HW>::to_gpu();

template void Tensor<1, RTCUDA, float, W>::to_cpu();
template void Tensor<1, RTCUDA, float, W>::to_gpu();

template int Tensor<4, RTCUDA, float, NCHW>::num();
template int Tensor<4, RTCUDA, float, NCHW>::channel();
template int Tensor<4, RTCUDA, float, NCHW>::height();
template int Tensor<4, RTCUDA, float, NCHW>::width();

template int Tensor<3, RTCUDA, float, CHW>::num();
template int Tensor<3, RTCUDA, float, CHW>::channel();
template int Tensor<3, RTCUDA, float, CHW>::height();
template int Tensor<3, RTCUDA, float, CHW>::width();

template int Tensor<2, RTCUDA, float, HW>::num();
template int Tensor<2, RTCUDA, float, HW>::channel();
template int Tensor<2, RTCUDA, float, HW>::height();
template int Tensor<2, RTCUDA, float, HW>::width();

template int Tensor<1, RTCUDA, float, W>::num();
template int Tensor<1, RTCUDA, float, W>::channel();
template int Tensor<1, RTCUDA, float, W>::height();
template int Tensor<1, RTCUDA, float, W>::width();

template const void* Tensor<4, RTCUDA, float, NCHW>::get_cpu_data(int index = 0);
template void* Tensor<4, RTCUDA, float, NCHW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<4, RTCUDA, float, NCHW>::get_gpu_data(int index = 0);
template void* Tensor<4, RTCUDA, float, NCHW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<3, RTCUDA, float, CHW>::get_cpu_data(int index = 0);
template void* Tensor<3, RTCUDA, float, CHW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<3, RTCUDA, float, CHW>::get_gpu_data(int index = 0);
template void* Tensor<3, RTCUDA, float, CHW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<2, RTCUDA, float, HW>::get_cpu_data(int index = 0);
template void* Tensor<2, RTCUDA, float, HW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<2, RTCUDA, float, HW>::get_gpu_data(int index = 0);
template void* Tensor<2, RTCUDA, float, HW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<1, RTCUDA, float, W>::get_cpu_data(int index = 0);
template void* Tensor<1, RTCUDA, float, W>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<1, RTCUDA, float, W>::get_gpu_data(int index = 0);
template void* Tensor<1, RTCUDA, float, W>::get_gpu_data_mutable(int index = 0);

#endif

#ifdef USE_ARM_PLACE
template<int Dim, TargetType target, typename dtype, LayoutType layout>
const void* Tensor<Dim, target, dtype, layout>::get_cpu_data(int index) {
        return _cpu_mem->get_data() + (_offset + index) * _bytes;
    }

template<int Dim, TargetType target, typename dtype, LayoutType layout>
void* Tensor<Dim, target, dtype, layout>::get_cpu_data_mutable(int index) {
        return _cpu_mem->get_data_mutable() + (index + _offset) * _bytes;
    }

template<int Dim, TargetType target, typename dtype, LayoutType layout>
const void* Tensor<Dim, target, dtype, layout>::get_gpu_data(int index) {
    return _cpu_mem->get_data() + (index + _offset) * _bytes;
}

template<int Dim, TargetType target, typename dtype, LayoutType layout>
void* Tensor<Dim, target, dtype, layout>::get_gpu_data_mutable(int index) {
    return _cpu_mem->get_data_mutable() + (index + _offset) * _bytes;
}


template <>
int Tensor<4, ARM, float, NCHW>::num(){
    return _shape[0];
}
template <>
int Tensor<4, ARM, float, NCHW>::channel(){
    return _shape[1];
}
template <>
int Tensor<4, ARM, float, NCHW>::height(){
    return _shape[2];
}
template <>
int Tensor<4, ARM, float, NCHW>::width(){
    return _shape[3];
}

template <>
int Tensor<3, ARM, float, CHW>::num(){
    return 1;
}
template <>
int Tensor<3, ARM, float, CHW>::channel(){
    return _shape[0];
}
template <>
int Tensor<3, ARM, float, CHW>::height(){
    return _shape[1];
}
template <>
int Tensor<3, ARM, float, CHW>::width(){
    return _shape[2];
}

template <>
int Tensor<2, ARM, float, HW>::num(){
    return 1;
}
template <>
int Tensor<2, ARM, float, HW>::channel(){
    return 1;
}
template <>
int Tensor<2, ARM, float, HW>::height(){
    return _shape[0];
}
template <>
int Tensor<2, ARM, float, HW>::width(){
    return _shape[1];
}

template <>
int Tensor<1, ARM, float, W>::num(){
    return 1;
}
template <>
int Tensor<1, ARM, float, W>::channel(){
    return 1;
}
template <>
int Tensor<1, ARM, float, W>::height(){
    return 1;
}
template <>
int Tensor<1, ARM, float, W>::width(){
    return _shape[0];
 }

template int Tensor<4, ARM, float, NCHW>::num();
template int Tensor<4, ARM, float, NCHW>::channel();
template int Tensor<4, ARM, float, NCHW>::height();
template int Tensor<4, ARM, float, NCHW>::width();

template int Tensor<3, ARM, float, CHW>::num();
template int Tensor<3, ARM, float, CHW>::channel();
template int Tensor<3, ARM, float, CHW>::height();
template int Tensor<3, ARM, float, CHW>::width();

template int Tensor<2, ARM, float, HW>::num();
template int Tensor<2, ARM, float, HW>::channel();
template int Tensor<2, ARM, float, HW>::height();
template int Tensor<2, ARM, float, HW>::width();

template int Tensor<1, ARM, float, W>::num();
template int Tensor<1, ARM, float, W>::channel();
template int Tensor<1, ARM, float, W>::height();
template int Tensor<1, ARM, float, W>::width();

template const void* Tensor<4, ARM, float, NCHW>::get_cpu_data(int index = 0);
template void* Tensor<4, ARM, float, NCHW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<4, ARM, float, NCHW>::get_gpu_data(int index = 0);
template void* Tensor<4, ARM, float, NCHW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<3, ARM, float, CHW>::get_cpu_data(int index = 0);
template void* Tensor<3, ARM, float, CHW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<3, ARM, float, CHW>::get_gpu_data(int index = 0);
template void* Tensor<3, ARM, float, CHW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<2, ARM, float, HW>::get_cpu_data(int index = 0);
template void* Tensor<2, ARM, float, HW>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<2, ARM, float, HW>::get_gpu_data(int index = 0);
template void* Tensor<2, ARM, float, HW>::get_gpu_data_mutable(int index = 0);

template const void* Tensor<1, ARM, float, W>::get_cpu_data(int index = 0);
template void* Tensor<1, ARM, float, W>::get_cpu_data_mutable(int index = 0);
template const void* Tensor<1, ARM, float, W>::get_gpu_data(int index = 0);
template void* Tensor<1, ARM, float, W>::get_gpu_data_mutable(int index = 0);

#endif //USE_ARM_PLACE

} //namespace saber

} //namespace anakin
