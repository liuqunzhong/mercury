#ifndef MERCURY_OPERATOR_DETECTIONOUTPUT_H
#define MERCURY_OPERATOR_DETECTIONOUTPUT_H

#include "layer.h"

namespace mercury {

class DetectionOutput : public Layer
{
public:
    DetectionOutput();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    int num_class;
    float nms_threshold;
    int nms_top_k;
    int keep_top_k;
    float confidence_threshold;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_DETECTIONOUTPUT_H
