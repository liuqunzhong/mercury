#include "memory.h"
#include "type.h"
#include <cstring>
namespace mercury{

// the alignment of all the allocated buffers
const int MALLOC_ALIGN = 16;

static inline void* fast_malloc(size_t size)
{
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p = static_cast<char*>(malloc(offset + size));
    if (!p) {
        return nullptr;
    }
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
}

static inline void fast_free(void* ptr)
{
    if (ptr){
        free(static_cast<void**>(ptr)[-1]);
    }
}

CpuMemory::CpuMemory() : \
    Memory(){}

CpuMemory::CpuMemory(size_t size) : Memory(size) {
    _own_data = true;
#ifdef USE_CUDA
    cudaMallocHost(&_data, _capacity);
#else
    _data = fast_malloc(_capacity);
#endif
}
CpuMemory::CpuMemory(void *data, size_t size) : Memory(size) {
    _own_data = false;
    _data = data;
}
void CpuMemory::re_alloc(size_t size) {
    if(_own_data && size < _capacity) {
        return;
    } else {
        clean();
        _capacity = size;
        _own_data = true;
#ifdef USE_CUDA
        cudaMallocHost(&_data, _capacity);
#else
        _data = fast_malloc(_capacity);
#endif
    }
}

void CpuMemory::copyto(Memory& buf) {
	memcpy(buf.get_data_mutable(), _data, _capacity);
}

const void* CpuMemory::get_data() {
    return _data;
}
void* CpuMemory::get_data_mutable() {
    return _data;
}
void CpuMemory::mem_set(int c, size_t size) {
    //CHECK_LE(size, _capacity) << "memset size must equal to or less than buffer size! ";
    memset(_data, c, size);
}

void CpuMemory::clean(){
    if (_own_data){
#ifdef USE_CUDA
        cudaFreeHost(&_data);
#else
        fast_free(_data);
#endif
    }
    _own_data = false;
    _data = nullptr;
    _capacity = 0;
}
CpuMemory::~CpuMemory() {
    if (_own_data){
#ifdef USE_CUDA
        cudaFreeHost(&_data);
#else
        fast_free(_data);
#endif
    }
    _data = nullptr;
    _capacity = 0;
}

GpuMemory::GpuMemory() : Memory() {}

GpuMemory::GpuMemory(size_t size) : Memory(size) {
    _own_data = true;
#ifdef USE_CUDA
    //LOG(INFO)<<"allocate gpu memory! "<<_capacity;
    cudaMalloc(&_data, _capacity);
#endif
}

GpuMemory::GpuMemory(void *data, size_t size) : Memory(size) {
    _own_data = false;
    _capacity = size;
    _data = data;
}

void GpuMemory::re_alloc(size_t size) {
    if (_own_data && _capacity >= size) {
        return;
    } else {
        clean();
        _capacity = size;
        _own_data = true;
#ifdef USE_CUDA
        cudaMalloc(&_data, _capacity);
#endif
    }
}

void GpuMemory::copyto(Memory& buf) {
#ifdef USE_CUDA
	cudaMemcpy(buf.get_data_mutable(), _data, _capacity, cudaMemcpyDeviceToDevice);
#endif
}

void GpuMemory::mem_set(int c, size_t size) {
    //CHECK_LE(size, _capacity) << "memset size must equal to or less than buffer size! ";
    if (_own_data == false || _capacity < size) {
#ifdef USE_CUDA
        if (_own_data) {
                cudaFree(_data);
            }
            _capacity = size;
            _own_data = true;
            cudaMalloc(&_data, _capacity);
#endif
    }

#ifdef USE_CUDA
    cudaMemset((void*)_data, c, size);
#endif
}

const void* GpuMemory::get_data() {
    return _data;
}

void* GpuMemory::get_data_mutable() {
    return _data;
}

void GpuMemory::clean() {
    if (_own_data){
#ifdef USE_CUDA
        cudaFree(_data);
#endif
    }
    _own_data = false;
    _capacity = 0;
}
GpuMemory::~GpuMemory() {
    if (_own_data){
#ifdef USE_CUDA
        cudaFree(_data);
#endif
    }
    _own_data = false;
    _capacity = 0;
}

} //namespace mercury
