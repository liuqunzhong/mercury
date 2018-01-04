#ifndef MERCURY_OPERATOR_DECONVOLUTIONDEPTHWISE_H
#define MERCURY_OPERATOR_DECONVOLUTIONDEPTHWISE_H

#include "layer.h"
#include "deconvolution.h"

namespace mercury {

class DeconvolutionDepthWise : public Deconvolution
{
public:
    DeconvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int group;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_DECONVOLUTIONDEPTHWISE_H
