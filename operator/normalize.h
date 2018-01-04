#ifndef MERCURY_OPERATOR_NORMALIZE_H
#define MERCURY_OPERATOR_NORMALIZE_H

#include "layer.h"

namespace mercury {

class Normalize : public Layer
{
public:
    Normalize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    // param
    int across_spatial;
    int channel_shared;
    float eps;
    int scale_data_size;

    Tensor<float> scale_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_NORMALIZE_H
