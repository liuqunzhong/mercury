#ifndef MERCURY_OPERATOR_SPLIT_H
#define MERCURY_OPERATOR_SPLIT_H

#include "layer.h"

namespace mercury {

class Split : public Layer
{
public:
    Split();

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SPLIT_H
