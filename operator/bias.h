
#ifndef MERCURY_OPERATOR_BIAS_H
#define MERCURY_OPERATOR_BIAS_H

#include "layer.h"

namespace mercury {

class Bias : public Layer
{
public:
    Bias();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    // param
    int bias_data_size;

    // model
    Tensor<float> bias_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_BIAS_H
