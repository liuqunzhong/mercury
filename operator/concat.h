#ifndef MERCURY_OPERATOR_CONCAT_H
#define MERCURY_OPERATOR_CONCAT_H

#include "layer.h"

namespace mercury {

class Concat : public Layer
{
public:
    Concat();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

public:
    int axis;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_CONCAT_H
