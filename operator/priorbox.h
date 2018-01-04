#ifndef MERCURY_OPERATOR_PRIORBOX_H
#define MERCURY_OPERATOR_PRIORBOX_H

#include "layer.h"

namespace mercury {

class PriorBox : public Layer
{
public:
    PriorBox();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    Tensor<float> min_sizes;
	Tensor<float> max_sizes;
	Tensor<float> aspect_ratios;
    float variances[4];
    int flip;
    int clip;
    int image_width;
    int image_height;
    float step_width;
    float step_height;
    float offset;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_PRIORBOX_H
