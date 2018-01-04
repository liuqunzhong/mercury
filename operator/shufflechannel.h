#ifndef MERCURY_OPERATOR_SHUFFLECHANNEL_H
#define MERCURY_OPERATOR_SHUFFLECHANNEL_H

#include "layer.h"

namespace mercury {

class ShuffleChannel : public Layer
{
public:
    ShuffleChannel() {
        one_blob_only = true;
        support_inplace = false;
    }
    virtual int load_param(const ParamDict& pd) {
        group = pd.get(0, 1);
        return 0;
    }
    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int group;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SHUFFLECHANNEL_H
