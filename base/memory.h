// create by lxy890123

#ifndef MERCURY_BASE_MEMORY_H
#define MERCURY_BASE_MEMORY_H

#include <stdlib.h>
#include "config.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mercury{

class Memory {
public:

    /*
     * \brief constructor
     */
    Memory(){
        _capacity = 0;
        _data = nullptr;
        _own_data = false;
    }
    /*
     * \brief constructor, allocate data
     */
    explicit Memory(size_t size){
        _capacity = size;
    }
    /*
     * \brief assigned function
     */
    Memory& operator = (Memory& buf){
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
    /*
     * \brief destructor
     */
    ~Memory(){}

    /*
     * \brief set _data to (c) with length of (size)
     */
    virtual void mem_set(int c, size_t size) = 0;

    /*
     * \brief re-alloc memory
     */
    virtual void re_alloc(size_t size) = 0;

    /*
     * \brief free memory
     */
    virtual void clean() =0;

    /*
     * \brief return const data pointer
     */
    virtual const void* get_data() = 0;

    /*
     * \brief return mutable data pointer
     */
    virtual void* get_data_mutable() = 0;

    /*
     * \brief return total size of memory, in bytes
     */
    inline size_t get_capacity() { return _capacity;}


protected:
    void* _data;
    bool _own_data;
    size_t _capacity;

};

class GpuMemory : public Memory {
public:
    explicit GpuMemory();
    ~GpuMemory();
    explicit GpuMemory(size_t size);
    explicit GpuMemory(void* data, size_t size);
    GpuMemory& operator = (GpuMemory& buf) {
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
    virtual void re_alloc(size_t size);
    virtual void clean();
    virtual void mem_set(int c, size_t size);
    virtual const void* get_data();
    virtual void* get_data_mutable();

};

class CpuMemory : public Memory {
public:
    explicit  CpuMemory();
    ~CpuMemory();
    explicit CpuMemory(size_t size);
    explicit CpuMemory(void* data, size_t size);
    CpuMemory& operator = (CpuMemory& buf) {
        this->_capacity = buf._capacity;
        this->_own_data = false;
        this->_data = buf._data;
        return *this;
    }
    virtual void re_alloc(size_t size);
    virtual void clean();
    virtual void mem_set(int c, size_t size);
    virtual const void* get_data();
    virtual void* get_data_mutable();

    // register pinned memory.
    inline void reg_page_lock(size_t size);

private:
    bool page_lock;  // whether the memory is pageable
    bool num_page_aligned;
};

} //namespace mercury
#endif //MERCURY_BASE_MEMORY_H
