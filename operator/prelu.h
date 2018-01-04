#ifndef MERCURY_OPERATOR_PRELU_H
#define MERCURY_OPERATOR_PRELU_H

#include "layer.h"

namespace mercury {

class PReLU : public Layer
{
public:
    PReLU();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    int num_slope;
    Tensor<float> slope_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_PRELU_H
