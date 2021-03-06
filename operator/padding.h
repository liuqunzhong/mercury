#ifndef MERCURY_OPERATOR_PADDING_H
#define MERCURY_OPERATOR_PADDING_H

#include "layer.h"

namespace mercury {

class Padding : public Layer
{
public:
    Padding();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int top;
    int bottom;
    int left;
    int right;
    int type;
    float value;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_PADDING_H
