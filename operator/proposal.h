#ifndef MERCURY_OPERATOR_PROPOSAL_H
#define MERCURY_OPERATOR_PROPOSAL_H

#include "layer.h"

namespace mercury {

class Proposal : public Layer
{
public:
    Proposal();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    // param
    int feat_stride;
    int base_size;
    int pre_nms_topN;
    int after_nms_topN;
    float nms_thresh;
    int min_size;

	Tensor<float> ratios;
	Tensor<float> scales;

	Tensor<float> anchors;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_PROPOSAL_H
