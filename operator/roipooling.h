#ifndef MERCURY_OPERATOR_ROIPOOLING_H
#define MERCURY_OPERATOR_ROIPOOLING_H

#include "layer.h"

namespace mercury {

class ROIPooling : public Layer
{
public:
    ROIPooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    int pooled_width;
    int pooled_height;
    float spatial_scale;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_ROIPOOLING_H
