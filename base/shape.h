/**
 * Copyright 2017 Baidu Inc. All Rights Reserved.
 *
 * \brief tensor shape
 * Detailed description
 * \file shape.h
 * \author lixiaoyang(lixiaoyang05@baidu.com)
 * \date 2017-10-31
 */

#ifndef ANAKIN2_SABER_SHAPE_H
#define ANAKIN2_SABER_SHAPE_H

#include <string>
#include <vector>
#include "type.h"
#include "utils/logger/logger.h"

namespace anakin{

namespace saber{

template<int Dim>
class Shape {
public:
    Shape():_head(1), _tail(), _size(Dim){}
    template<typename ...Types>
    Shape(int head , Types ...dims):_head(head),_tail(dims...), _size(Dim) {
        static_assert(sizeof...(dims) == Dim - 1,
                      "Shape initialized with error number of dims pack ! ");
    }
    Shape(std::vector<int> listDim):_size(listDim.size()) { init(listDim); }
    Shape(const Shape<Dim>& operand) { *this = operand; }
    Shape& operator=(const Shape<Dim>& shape) {
        this->_head = shape._head;
        this->_tail = shape._tail;
        this->_size = shape._size;
        return *this;
    }
    ~Shape(){}
    inline bool operator==(const Shape& shape) {
        return _head == shape._head && _tail == shape._tail;
    }
    inline bool operator!=(const Shape& shape) {
        return !(*this == shape);
    }
    inline int& operator[](int index) {
        if (index == 0) return _head;
        return _tail[index-1];
    }
    // dim of shape
    inline int dim(){
        //add implementation
        return Dim;
    }
    inline int count(int start, int end){
        //add implementation
        CHECK_GT(end, start) << "start index should small than end index";
        CHECK_LT(end, _size) << "end index exceed the shape size";
        int ct = 1;
        for (int i = start; i <= end; ++i) {
            if (i == 0){
                ct *= _head;
            }else {
               ct *= _tail[i -1];
            }
        }
        return ct;
    }
    // total size of buffer the shape represents
    inline int size(){
        //add implementation
        //LOG(INFO)<<"return head * tail = "<<_head * _tail.size();
        return _head * _tail.size();
    }
public:
    inline void init(std::vector<int> listDim) {
        //LOG(INFO)<<"init get head: "<<listDim[0];
        _head = listDim[0];
        _tail.init(std::vector<int>(listDim.begin() + 1, listDim.end()));
    }
private:
    // dims of shape
    int _head;
    Shape<Dim-1> _tail;
    int _size;
    // true if shape stand for data is sequence data.
    bool is_sequence{false};
};
// specialization in case of recursive instantiation。
template<>
class Shape<1> {
public:
    Shape():_head(0), _size(0) {}
    ~Shape(){}
    Shape(int head):_head(head), _size(head) {}
    Shape(const Shape<1>& operand) {
        _head = operand._head;
        _size = _head;
    }
    Shape(std::vector<int>& listDim):_head(listDim[0]) {}
    inline int& operator[](int index) {
        //CHECK_EQ(index, 0) << "Overflow: index in shape 1D should equal to 0 ! ";
        return _head;
    }
    inline bool operator==(const Shape& shape) {
        return _head == shape._head;
    }
    inline bool operator!=(const Shape& shape) {
        return !(*this == shape);
    }
    // dim of shape
    inline int dim(){
        //add implementation
        return 1;
    }
    inline int count(int start, int end){
        //add implementation
        return _head;
    }
    int size(){ return _head; }
    void init(std::vector<int> listDim) {_head = listDim[0];}
private:
    // dims of shape
    int _head;
    int _size;
    // true if shape stand for data is sequence data.
    bool is_sequence{false};
};
typedef Shape<5>  Shape5D;
typedef Shape<4>  Shape4D;
typedef Shape<3>  Shape3D;
typedef Shape<2>  Shape2D;
typedef Shape<1>  Shape1D;

} //namespace saber

} //namespace anakin
#endif //ANAKIN2_SABER_SHAPE_H
