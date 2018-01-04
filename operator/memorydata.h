#ifndef MERCURY_OPERATOR_MEMORYDATA_H
#define MERCURY_OPERATOR_MEMORYDATA_H

#include "layer.h"

namespace mercury {

class MemoryData : public Layer
{
public:
    MemoryData();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    int w;
    int h;
    int c;

    Tensor<float> data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_MEMORYDATA_H
