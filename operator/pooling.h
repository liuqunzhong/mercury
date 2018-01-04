#ifndef MERCURY_OPERATOR_POOLING_H
#define MERCURY_OPERATOR_POOLING_H

#include "layer.h"

namespace mercury {

class Pooling : public Layer
{
public:
    Pooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

    enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

public:
    // param
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int global_pooling;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_POOLING_H
