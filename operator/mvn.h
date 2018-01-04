// Tencent is pleased to support the open source community by making ncnn available.
//
#ifndef MERCURY_OPERATOR_MVN_H
#define MERCURY_OPERATOR_MVN_H

#include "layer.h"

namespace mercury {

class MVN : public Layer
{
public:
    MVN();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int normalize_variance;
    int across_channels;
    float eps;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_MVN_H
