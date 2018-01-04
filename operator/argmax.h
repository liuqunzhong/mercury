#ifndef MERCURY_OPERATOR_ARGMAX_H
#define MERCURY_OPERATOR_ARGMAX_H

#include "layer.h"

namespace mercury {

class ArgMax : public Layer
{
public:
    ArgMax();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int out_max_val;
    int topk;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_ARGMAX_H
