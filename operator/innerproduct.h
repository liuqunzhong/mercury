#ifndef MERCURY_OPERATOR_INNERPRODUCT_H
#define MERCURY_OPERATOR_INNERPRODUCT_H

#include "layer.h"

namespace mercury {

class InnerProduct : public Layer
{
public:
    InnerProduct();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    // param
    int num_output;
    int bias_term;

    int weight_data_size;

    // model
    Tensor<float> weight_data;
    Tensor<float> bias_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_INNERPRODUCT_H
