#ifndef MERCURY_OPERATOR_THRESHOLD_H
#define MERCURY_OPERATOR_THRESHOLD_H

#include "layer.h"

namespace mercury {

class Threshold : public Layer
{
public:
    Threshold();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float threshold;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_THRESHOLD_H
