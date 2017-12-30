/**
 * Copyright 2017 Baidu Inc. All Rights Reserved.
 *
 * \brief basic data wrapper: tensor
 * Detailed description
 * \file tensor.h
 * \author lixiaoyang(lixiaoyang05@baidu.com)
 * \date 2017-10-31
 */

#ifndef ANAKIN2_SABER_TENSOR_H
#define ANAKIN2_SABER_TENSOR_H

#include <vector>
#include <list>
#include <memory>
#include "type.h"
#include "memory.h"

namespace mercury {

//! Tensor class
/*!
    wrapper of memory, hold dim and layout info
 */
template <typename dtype>
class Tensor{
public:

    /*
     * \brief default constructor
     */

    Tensor(): _shape(), _real_shape(), \
        _offset(0), _mem_head(UNINITIALIZED), _stream_id(0){
        _ctx = nullptr;
        _bytes = sizeof(dtype);
        _cpu_mem = std::make_shared<CpuMemory>();
        _gpu_mem = std::make_shared<GpuMemory>();
        _is_leaf = true;
        _children.clear();
    }

    /*
     * \brief constructor with layout, target and data type, memory not allocated
     */
    explicit Tensor(Context<target>* ctx, int id = 0) : _events(ctx, id), \
        _shape(), _real_shape() {

        _stream_id = id;
        _ctx = ctx;
        _offset = 0;
        _bytes = sizeof(dtype);
        _cpu_mem = std::make_shared<CpuMemory>();
        _mem_head = UNINITIALIZED;
        if (target == GPU) {
            _gpu_mem = std::make_shared<GpuMemory>();
        }
        else if (target == RTCUDA) {
            _gpu_mem = std::make_shared<GpuMemory>();
        }
        _is_leaf = true;
        _children.clear();
    }

    /*
     * \brief constructor, memory allocated
     */
    explicit Tensor(Context<target> *ctx, Shape<Dim>& shape, int id = 0) : _shape(shape), \
        _events(ctx, id), _real_shape(shape){

        _stream_id = id;
        _ctx = ctx;
        _bytes = sizeof(dtype);
        _offset = 0;
        _cpu_mem = std::make_shared<CpuMemory>(_real_shape.size() * _bytes);
        _mem_head = SYNCED;
        if (target == GPU) {
            _gpu_mem = std::make_shared<GpuMemory>(_real_shape.size() * _bytes);
        }
        else if (target == RTCUDA) {
            _gpu_mem = std::make_shared<GpuMemory>(shape.size() * _bytes);
        }
        _is_leaf = true;
        _children.clear();
    }

    /*
     * \brief constructor, hold data pointer
     */
    explicit Tensor(Context<target> *ctx, void* cpu_data, void* gpu_data, \
        Shape<Dim>& shape, size_t capacity, int id = 0) :  _shape(shape), \
        _events(ctx, id), _real_shape(shape){

        _stream_id = id;
        CHECK_LE(shape.size() * sizeof(dtype), capacity) << \
            "ERROR, tensor size must smaller or equal to buffer capacity!";
        _ctx = ctx;
        _bytes = sizeof(dtype);
        _offset = 0;
        if (cpu_data != nullptr){
            _cpu_mem = std::make_shared<CpuMemory>(cpu_data, capacity);
            _mem_head = HEAD_AT_CPU;
        } else{
            LOG(INFO) << "cpu data uninit";
            _cpu_mem = std::make_shared<CpuMemory>(capacity);
            _mem_head = HEAD_AT_GPU;
        }

        if (target == GPU) {
            if (gpu_data != nullptr){
                if (_mem_head == HEAD_AT_CPU){
                    _mem_head = SYNCED;
                }
                _gpu_mem = std::make_shared<GpuMemory>(gpu_data, capacity);
            } else {
                LOG(INFO) << "gpu data uninit";
                _gpu_mem = std::make_shared<GpuMemory>(capacity);
                if (_mem_head == HEAD_AT_GPU) {
                    _mem_head = UNINITIALIZED;
                }
            }
        }
        else if (target == RTCUDA) {
            if (gpu_data != nullptr){
                if (_mem_head == HEAD_AT_CPU){
                    _mem_head = SYNCED;
                }
                _gpu_mem = std::make_shared<GpuMemory>(gpu_data, capacity);
            } else {
                LOG(INFO) << "gpu data uninit";
                _gpu_mem = std::make_shared<GpuMemory>(capacity);
                if (_mem_head == HEAD_AT_GPU) {
                    _mem_head = UNINITIALIZED;
                }
            }
        }
        _is_leaf = true;
        _children.clear();
    }

    /*
     * \brief copy constructor
     */
    Tensor(Tensor<Dim, target, dtype, layout>& tensor) {

        CHECK_EQ(tensor._ctx == nullptr, false) << "input tensor is not initialized";

        _ctx = tensor._ctx;
        _stream_id = tensor.get_streamid();
        _events.init(_ctx, _stream_id);
        _real_shape = tensor._real_shape;
        _shape = tensor._shape;
        _offset = tensor._offset;
        _bytes = tensor._bytes;
        _mem_head = tensor._mem_head;
        tensor.add_events(&_events);
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;
        _is_leaf = true;
        _children.clear();
        tensor.add_leaf(this);
    }


    /*
     * \brief copy constructor const, event tree and tensor tree are not copied
     */
    Tensor(const Tensor<Dim, target, dtype, layout>& tensor) {

        CHECK_EQ(tensor._ctx == nullptr, false) << "input tensor is not initialized";
        _ctx = tensor._ctx;
        _stream_id = tensor.get_streamid();
        _events.init(_ctx, _stream_id);
        _real_shape = tensor._real_shape;
        _shape = tensor._shape;
        _offset = tensor._offset;
        _bytes = tensor._bytes;
        _mem_head = tensor._mem_head;
        //tensor.add_events(&_events); // sync problem
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;
        _is_leaf = true;
        _children.clear();
    }

    /*
     * \brief transfer constructor
     */
    //explicit Tensor(Tensor<Dim, target, dtype, layout>&& tensor) {
    //    _ctx = tensor._ctx;
    //    _real_shape = tensor._real_shape;
    //    _shape = tensor._shape;
    //    _offset = tensor._offset;
    //    _bytes = tensor._bytes;
    //    //_cpu_mem = tensor._cpu_mem;
    //    //_gpu_mem = tensor._gpu_mem;
    //    _mem_head = UNINITIALIZED;
    //    _cpu_mem.swap(tensor._cpu_mem);
    //    _gpu_mem.swap(tensor._gpu_mem);
    //}

    Tensor& operator=(Tensor<Dim, target, dtype, layout>& tensor){

        CHECK_EQ(tensor._ctx == nullptr, false) << "input tensor is not initialized";
        if (this->_ctx == nullptr){
            _events.init(tensor._ctx, tensor.get_streamid());
        }
        this->_ctx = tensor._ctx;
        tensor.add_events(&this->_events);
        this->_real_shape = tensor._real_shape;
        this->_shape = tensor._shape;
        this->_offset = tensor._offset;
        this->_bytes = tensor._bytes;
        this->_mem_head = tensor._mem_head;
        this->_cpu_mem = tensor._cpu_mem;
        this->_gpu_mem = tensor._gpu_mem;
        tensor.add_leaf(this);
        return *this;
    }

    /*
     * \brief destructor
     */
    ~Tensor(){}


    /*
     * \brief init context, change nothing else
     */
    inline void accept_ctx(Context<target>* ctx){
        CHECK_EQ(ctx == nullptr, false) << "context is not initialized";
        _ctx = ctx;
    }

    /*
     * \brief re-allocate memory, too costly, not recommended.
     */
    inline void re_alloc(Shape<Dim> shape){
       CHECK_EQ(_ctx == nullptr, false) << "input tensor is not initialized";
        _shape = shape;
        _real_shape = shape;
        _cpu_mem->re_alloc(_real_shape.size() * _bytes);
        if (target == GPU) {
            _gpu_mem->re_alloc(_real_shape.size() * _bytes);
        }
        else if (target == RTCUDA) {
            _gpu_mem->re_alloc(_real_shape.size() * _bytes);
        }
    }

    /*
     * \brief share memory from operand tensor, only hold pointer.
     */
    void share_mem_from(Tensor<Dim, target, dtype, layout>& tensor){

        CHECK_EQ(tensor._ctx == nullptr, false) << "input tensor is not initialized";
        if (_ctx == nullptr){
            _ctx = tensor._ctx;
            _events.init(_ctx, tensor.get_streamid());
            _shape = tensor._shape;
            _offset = tensor._offset;
        }
        CHECK_EQ(tensor._ctx->get_dev_id(), _ctx->get_dev_id()) << \
            "shared tensors must be in the same device";
        tensor.add_events(&_events);

        _mem_head = tensor.get_mem_head();
        _real_shape = tensor._real_shape;
        _bytes = tensor._bytes;
        _cpu_mem = tensor._cpu_mem;
        _gpu_mem = tensor._gpu_mem;

        tensor.add_leaf(this);
    }

    /*
     * \brief get const data from index pos of tensor, gpu memory
     * \if target only has cpu memory, return cpu data
     */
    const inline void* get_gpu_data(int index = 0);

    /*
     * \brief get const data from index pos of tensor, cpu memory
     */
    const inline void* get_cpu_data(int index = 0);

    /*
     * \brief get data from index pos of tensor, gpu memory
     * \if target only has cpu memory, return cpu data
     */
    inline void* get_gpu_data_mutable(int index = 0);

    /*
     * \brief get data from index pos of tensor, cpu memory
     */
    inline void* get_cpu_data_mutable(int index = 0);

    /*
     * \brief size of real buffer the memory held, in bytes.
     */
    inline size_t capacity() {
        return _cpu_mem->get_capacity();
    }

    /*
     * \brief size of virtual shape size
     */
    inline int size() {
        return _shape.size();
    }

    /*
     * \brief return shape of current tensor block
     */
    const inline Shape<Dim> get_shape() {
        return _shape;
    }

    /*
     * \brief return real shape of tensor
     */
    const inline Shape<Dim> get_real_shape() {
        return _real_shape;
    }

    /*
     * \brief register pinned memory cpu memory.
     */
    inline void reg_page_lock(int size);

    /*
     * \brief get num for NCHW
     */
    inline int num();
    /*
     * \brief get channel for NCHW
     */
    inline int channel();
    /*
     * \brief get height for NCHW
     */
    inline int height();
    /*
     * \brief get width for NCHW
     */
    inline int width();

    /*
     * \brief set data offset
     */
    inline void set_offset(int offset){
        _offset = offset;
    }

    /*
     * \brief set data offset
     */
    inline int get_offset(){
        return _offset;
    }

    /*
     * \brief set new data shape
     */
    inline void set_shape(Shape<Dim> shape){
        CHECK_GE(_real_shape.size(), shape.size()) << \
            "new shape exceed the buffer size";
        _shape = shape;
    }

    /*
     * \brief sync tensor
     */
    inline void sync(bool block_host){
        _events.sync_tree(block_host);
    }

    /*
     * \brief add_events
     */
    inline void add_events(EventsTree<target>* events){
        _events.insert_children(events);
    }

    /*
     * \brief set_streamid
     */
    inline int set_streamid(int id = 0){
        _stream_id = id;
        _events._event._id = id;
    }

    /*
     * \brief get_streamid
     */
    inline int get_streamid() const {
        return _stream_id;
    }

    /*
     * \brief get_mem_head
     */
    inline SyncHead get_mem_head() const {
        return _mem_head;
    }

    /*
     * \brief delete leaf node
     */
    inline void delete_leaf() {
        _children.clear();
        _is_leaf = true;
    }

    /*
     * \brief add leaf
     */
    inline void add_leaf(Tensor<Dim, target, dtype, layout>* tensor) {
        _children.push_back(tensor);
        _is_leaf = false;
    }

public:
    Context<target> *_ctx;

private:
    int _stream_id;
    SyncHead _mem_head;
    void to_cpu();
    void to_gpu();
    Shape<Dim> _shape;
    Shape<Dim> _real_shape;
    int _offset;
    int _bytes;
    std::shared_ptr<CpuMemory> _cpu_mem{nullptr};
    std::shared_ptr<GpuMemory> _gpu_mem{nullptr};
    EventsTree<target> _events;

    bool _is_leaf;
    std::vector<Tensor<Dim, target, dtype, layout>*> _children;
    void recur_copy_to_cpu();
    void recur_copy_to_gpu();
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN2_SABER_TENSOR_H
