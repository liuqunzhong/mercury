#ifndef MERCURY_OPERATOR_CROP_H
#define MERCURY_OPERATOR_CROP_H

#include "layer.h"

namespace mercury {

class Crop : public Layer
{
public:
    Crop();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    int woffset;
    int hoffset;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_CROP_H
