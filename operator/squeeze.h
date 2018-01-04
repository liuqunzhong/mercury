#ifndef MERCURY_OPERATOR_SQUEEZE_H
#define MERCURY_OPERATOR_SQUEEZE_H

#include "layer.h"

namespace mercury {

class Squeeze : public Layer
{
public:
    Squeeze();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int squeeze_w;
    int squeeze_h;
    int squeeze_c;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SQUEEZE_H
