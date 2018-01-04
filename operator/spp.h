#ifndef MERCURY_OPERATOR_SPP_H
#define MERCURY_OPERATOR_SPP_H

#include "layer.h"

namespace mercury {

class SPP : public Layer
{
public:
    SPP();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

    enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

public:
    // param
    int pooling_type;
    int pyramid_height;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SPP_H
