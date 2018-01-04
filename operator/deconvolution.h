#ifndef MERCURY_OPERATOR_DECONVOLUTION_H
#define MERCURY_OPERATOR_DECONVOLUTION_H

#include "layer.h"

namespace mercury {

class Deconvolution : public Layer
{
public:
    Deconvolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;

    int weight_data_size;

    // model
    Tensor<float> weight_data;
    Tensor<float> bias_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_DECONVOLUTION_H
