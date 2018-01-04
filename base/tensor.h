// create by lxy890123

#ifndef MERCURY_BASE_TENSOR_H
#define MERCURY_BASE_TENSOR_H

#include <vector>
#include <list>
#include <memory>
#include "type.h"
#include "memory.h"
#if defined _WIN32 || defined _WIN64
#include <assert.h>
#include <windows.h>
#else
#include "utils/logger/logger.h"
#endif

namespace mercury {

	static inline size_t alignSize(size_t sz, int n)
	{
		return (sz + n - 1) & -n;
	}

//! Tensor class
/*!
    wrapper of memory, hold dim and layout info
 */
template <typename dtype>
class Tensor{
public:

    /*
     * \brief constructor with layout, target and data type, memory not allocated
     */
    explicit Tensor() : _shape(){

        _size = 0;
        _bytes = sizeof(dtype);
        _cpu_mem = std::make_shared<CpuMemory>();
        _mem_head = UNINITIALIZED;
#ifdef USE_CUDA
        _gpu_mem = std::make_shared<GpuMemory>();
#endif
    }

    /*
     * \brief constructor, memory allocated
     */
    explicit Tensor(std::vector<int>& shape) : _shape(shape) {
#if defined _WIN32 || defined _WIN64
		assert(shape.size() > 0);
#else
        CHECK_GT(shape.size(), 0) << "error: input shape is empty!";
#endif
        _bytes = sizeof(dtype);
        _size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            _size *= shape[i];
        }
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
        CHECK_GT(_size, 0) << "error: input shape is wrong!";
#endif
        _cpu_mem = std::make_shared<CpuMemory>(_size * _bytes);
        _mem_head = SYNCED;
#ifdef USE_CUDA
        _gpu_mem = std::make_shared<GpuMemory>(_size * _bytes);
#endif
    }

    /*
     * \brief constructor, hold data pointer
     */
    explicit Tensor(void* cpu_data, void* gpu_data, \
        std::vector<int>& shape, size_t capacity) :  _shape(shape){

#if defined _WIN32 || defined _WIN64
		assert(shape.size() > 0);
#else
        CHECK_GT(shape.size(), 0) << "error: input shape is empty!";
#endif
        _size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            _size *= shape[i];
        }
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "error: input shape is wrong!";
#endif 
        _bytes = sizeof(dtype);
        if (cpu_data != nullptr){
            _cpu_mem = std::make_shared<CpuMemory>(cpu_data, capacity);
            _mem_head = HEAD_AT_CPU;
        } else{
            //LOG(INFO) << "cpu data uninit";
            _cpu_mem = std::make_shared<CpuMemory>(capacity);
            _mem_head = HEAD_AT_GPU;
        }
#ifdef USE_CUDA
        if (gpu_data != nullptr){
            if (_mem_head == HEAD_AT_CPU){
                _mem_head = SYNCED;
            }
            _gpu_mem = std::make_shared<GpuMemory>(gpu_data, capacity);
        } else {
            //LOG(INFO) << "gpu data uninit";
            _gpu_mem = std::make_shared<GpuMemory>(capacity);
            if (_mem_head == HEAD_AT_GPU) {
                _mem_head = UNINITIALIZED;
            }
        }
#endif

    }

    /*
     * \brief copy constructor
     */
	/*
    Tensor(Tensor<dtype>& tensor) {

#if defined _WIN32 || defined _WIN64
		assert(tensor._size > 0);
#else
		CHECK_GT(tensor._size, 0) << "input tensor is not initialized";
#endif 

        _shape = tensor._shape;
        _size = tensor._size;
        _bytes = tensor._bytes;
        _mem_head = tensor._mem_head;
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;
    }
	*/

    /*
     * \brief copy constructor const
     */
    Tensor(const Tensor<dtype>& tensor) {
#if defined _WIN32 || defined _WIN64
		assert(tensor._size > 0);
#else
		CHECK_GT(tensor._size, 0) << "input tensor is not initialized";
#endif 
        _shape = tensor._shape;
        _size = tensor._size;
        _bytes = tensor._bytes;
        _mem_head = tensor._mem_head;
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;
    }


    inline Tensor& operator=(const Tensor<dtype>& tensor){
		if (this == &tensor){
			return *this;
		}
#if defined _WIN32 || defined _WIN64
		assert(tensor._size > 0);
#else
		CHECK_GT(tensor._size, 0) << "input tensor is not initialized";
#endif 
        this->_shape = tensor._shape;
        this->_size = tensor._size;
        this->_bytes = tensor._bytes;
        this->_mem_head = tensor._mem_head;
        this->_cpu_mem = tensor._cpu_mem;
        this->_gpu_mem = tensor._gpu_mem;
        return *this;
    }

    /*
     * \brief initialized tensor with shape
     */
    void init(std::vector<int>& shape){

#if defined _WIN32 || defined _WIN64
		assert(shape.size() > 0);
#else
		CHECK_GT(shape.size(), 0) << "error: input shape is empty!";
#endif 
       
        _bytes = sizeof(dtype);
        _size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            _size *= shape[i];
        }
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "error: input shape is wrong!";
#endif 

        
        _cpu_mem = std::make_shared<CpuMemory>(_size * _bytes);
        _mem_head = SYNCED;
#ifdef USE_CUDA
        _gpu_mem = std::make_shared<GpuMemory>(_size * _bytes);
#endif
    }
	/*
	* \brief initialized tensor with width
	*/
	void init(int w) {
		std::vector<int> shape = {w};
		init(shape);
	}

	/*
	* \brief initialized tensor with w, h
	*/
	void init(int w, int h) {
		std::vector<int> shape = { w, h };
		init(shape);
	}

	/*
	* \brief initialized tensor with w, h, c
	*/
	void init(int w, int h, int c) {
		std::vector<int> shape = { w, h, c };
		init(shape);
	}

	/*
	* \brief initialized tensor with w, h, c, n
	*/
	void init(int w, int h, int c, int n) {
		std::vector<int> shape = { w, h, c, n };
		init(shape);
	}

    /*
     * \brief initialized tensor with shape and data pointer
     */
    void init_with_data(std::vector<int>& shape, void* cpu_data, void* gpu_data, size_t capacity){

#if defined _WIN32 || defined _WIN64
		assert(shape.size() > 0);
#else
		CHECK_GT(shape.size(), 0) << "error: input shape is empty!";
#endif 
        _size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            _size *= shape[i];
        }
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "error: input shape is wrong!";
#endif 
        _bytes = sizeof(dtype);
        if (cpu_data != nullptr){
            _cpu_mem = std::make_shared<CpuMemory>(cpu_data, capacity);
            _mem_head = HEAD_AT_CPU;
        } else{
            //LOG(INFO) << "cpu data uninit";
            _cpu_mem = std::make_shared<CpuMemory>(capacity);
            _mem_head = HEAD_AT_GPU;
        }
#ifdef USE_CUDA
        if (gpu_data != nullptr){
            if (_mem_head == HEAD_AT_CPU){
                _mem_head = SYNCED;
            }
            _gpu_mem = std::make_shared<GpuMemory>(gpu_data, capacity);
        } else {
            //LOG(INFO) << "gpu data uninit";
            _gpu_mem = std::make_shared<GpuMemory>(capacity);
            if (_mem_head == HEAD_AT_GPU) {
                _mem_head = UNINITIALIZED;
            }
        }
#endif
    }

    /*
     * \brief destructor
     */
    ~Tensor(){}

	/*
	* \brief check if tensor is empty
	*/
	inline bool empty() const {
		bool flag_gpu_init = false;
#ifdef USE_CUDA
		flag_gpu_init = _gpu_mem == nullptr;
#endif
		return _size == 0 || _cpu_mem == nullptr || flag_gpu_init;
	}

    /*
     * \brief re-allocate memory, too costly, not recommended.
     */
    inline void re_alloc(std::vector<int> shape){

#if defined _WIN32 || defined _WIN64
		assert(shape.size() > 0);
#else
		CHECK_GT(shape.size(), 0) << "error: input shape is empty!";
#endif 
        _shape = shape;
        int _size = 1;
        for (int i = 0; i < shape.size(); ++i) {
            _size *= shape[i];
        }
        _cpu_mem->re_alloc(_size * _bytes);
#ifdef USE_CUDA
        _gpu_mem->re_alloc(_size * _bytes);
#endif
    }

    /*
     * \brief share memory from operand tensor, only hold pointer.
     */
    void share_mem_from(const Tensor<dtype>& tensor){

        _shape = tensor._shape;
        _size = tensor._size;
        _mem_head = tensor.get_mem_head();
        _bytes = tensor._bytes;
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;
    }

	/*
	* \brief share memory from operand tensor, only hold pointer.
	*/
	void copyto(Tensor<dtype>& tensor) const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "error: tensor is not initialized!";
#endif
		tensor._shape = _shape;
		tensor._size = _size;
		tensor._mem_head = get_mem_head();
		tensor._bytes = _bytes;
		tensor._cpu_mem->re_alloc(_bytes * _size);
		_cpu_mem->copyto(*tensor._cpu_mem);
#ifdef USE_CUDA
		tensor._gpu_mem->re_alloc(_bytes * _size);
		_gpu_mem->copyto(*tensor._gpu_mem);
#endif
	}

    /*
     * \brief get const data from index pos of tensor, gpu memory
     * \if target only has cpu memory, return cpu data
     */
    const inline dtype* get_gpu_data(int index = 0) const;

    /*
     * \brief get const data from index pos of tensor, cpu memory
     */
    const inline dtype* get_cpu_data(int index = 0) const;

    /*
     * \brief get data from index pos of tensor, gpu memory
     * \if target only has cpu memory, return cpu data
     */
    inline dtype* get_gpu_data_mutable(int index = 0);

    /*
     * \brief get data from index pos of tensor, cpu memory
     */
    inline dtype* get_cpu_data_mutable(int index = 0);

    /*
     * \brief size of real buffer the memory held, in bytes.
     */
    inline size_t capacity() const{
        return _cpu_mem->get_capacity();
    }

    /*
     * \brief size of virtual shape size
     */
    inline int size() const{
        return _size;
    }

    /*
     * \brief return shape of current tensor block
     */
    const inline std::vector<int> get_shape() const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "input tensor is not initialized";
#endif 
        return _shape;
    }

    /*
     * \brief get num for NCHW
     */
    inline int num() const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "input tensor is not initialized";
#endif 
        if (_shape.size() == 4){
            return _shape[3];
        } else {
            return 1;
        }
    }
    /*
     * \brief get channel for NCHW
     */
    inline int channel() const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "input tensor is not initialized";
#endif 
        if (_shape.size() >= 3) {
            return _shape[2];
        } else{
            return 1;
        }
    }

    /*
     * \brief get height for NCHW
     */
    inline int height() const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "input tensor is not initialized";
#endif 
        if (_shape.size() >= 2){
            return _shape[1];
        } else{
            return 1;
        }
    }

    /*
     * \brief get width for NCHW
     */
    inline int width() const{
#if defined _WIN32 || defined _WIN64
		assert(_size > 0);
#else
		CHECK_GT(_size, 0) << "input tensor is not initialized";
#endif 
        return _shape[0];
    }

    /*
     * \brief get_mem_head
     */
    inline SyncHead get_mem_head() const {
        return _mem_head;
    }

	/*
	* \brief get_dims
	*/
	inline int dims() const {
		return _shape.size();
	}

	/*
	* \brief get_mem_head
	*/
	inline void release(){
		_shape.clear();
		_size = 0;
		_mem_head = UNINITIALIZED;
		_cpu_mem->clean();
		_gpu_mem->clean();
		_cpu_mem = nullptr;
		_gpu_mem = nullptr;
	}

	inline int refcount() const{
		return _cpu_mem.use_count();
	}

private:
    SyncHead _mem_head;
    void to_cpu();
    void to_gpu();
    std::vector<int> _shape;
    int _bytes;
    int _size;
    std::shared_ptr<CpuMemory> _cpu_mem{nullptr};
    std::shared_ptr<GpuMemory> _gpu_mem{nullptr};
};

} //namespace mercury
#endif //MERCURY_BASE_TENSOR_H
