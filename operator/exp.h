
#ifndef MERCURY_OPERATOR_EXP_H
#define MERCURY_OPERATOR_EXP_H

#include "layer.h"

namespace mercury {

class Exp : public Layer
{
public:
    Exp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float base;
    float scale;
    float shift;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_EXP_H
