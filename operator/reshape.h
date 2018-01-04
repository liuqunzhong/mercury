#ifndef MERCURY_OPERATOR_RESHAPE_H
#define MERCURY_OPERATOR_RESHAPE_H

#include "layer.h"

namespace mercury {

class Reshape : public Layer
{
public:
    Reshape();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

private:
    int w;
    int h;
    int c;
    int permute;
    int ndim;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_RESHAPE_H
