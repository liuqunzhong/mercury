#ifndef MERCURY_OPERATOR_CONVOLUTIONDEPTHWISE_H
#define MERCURY_OPERATOR_CONVOLUTIONDEPTHWISE_H

#include "layer.h"
#include "convolution.h"

namespace mercury {

class ConvolutionDepthWise : public Convolution
{
public:
    ConvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int group;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_CONVOLUTIONDEPTHWISE_H
