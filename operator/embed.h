#ifndef MERCURY_OPERATOR_EMBED_H
#define MERCURY_OPERATOR_EMBED_H

#include "layer.h"

namespace mercury {

class Embed : public Layer
{
public:
    Embed();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    // param
    int num_output;
    int input_dim;
    int bias_term;

    int weight_data_size;

    // model
    Tensor<float> weight_data;
    Tensor<float> bias_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_EMBED_H
