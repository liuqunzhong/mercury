#ifndef MERCURY_OPERATOR_POWER_H
#define MERCURY_OPERATOR_POWER_H

#include "layer.h"

namespace mercury {

class Power : public Layer
{
public:
    Power();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float power;
    float scale;
    float shift;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_POWER_H
