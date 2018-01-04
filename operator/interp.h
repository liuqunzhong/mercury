#ifndef MERCURY_OPERATOR_INTERP_H
#define MERCURY_OPERATOR_INTERP_H

#include "layer.h"

namespace mercury {

class Interp : public Layer
{
public:
    Interp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float> &bottom_blob, Tensor<float> &top_blob) const;

public:
    // param
    float width_scale;
    float height_scale;
    int output_width;
    int output_height;
    int resize_type;//1:near 2: bilinear
};

} // namespace mercury

#endif // MERCURY_OPERATOR_INTERP_H
