#ifndef MERCURY_OPERATOR_SOFTMAX_H
#define MERCURY_OPERATOR_SOFTMAX_H

#include "layer.h"

namespace mercury {

class Softmax : public Layer
{
public:
    Softmax();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    int axis;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SOFTMAX_H
