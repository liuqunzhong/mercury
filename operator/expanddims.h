#ifndef MERCURY_OPERATOR_EXPANDDIMS_H
#define MERCURY_OPERATOR_EXPANDDIMS_H

#include "layer.h"

namespace mercury {

class ExpandDims : public Layer
{
public:
    ExpandDims();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int expand_w;
    int expand_h;
    int expand_c;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_EXPANDDIMS_H
