#ifndef MERCURY_OPERATOR_PERMUTE_H
#define MERCURY_OPERATOR_PERMUTE_H

#include "layer.h"

namespace mercury {

class Permute : public Layer
{
public:
    Permute();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int order_type;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_PERMUTE_H
