#ifndef MERCURY_OPERATOR_LRN_H
#define MERCURY_OPERATOR_LRN_H

#include "layer.h"

namespace mercury {

class LRN : public Layer
{
public:
    LRN();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

    enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };

public:
    // param
    int region_type;
    int local_size;
    float alpha;
    float beta;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_LRN_H
