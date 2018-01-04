#ifndef MERCURY_OPERATOR_FLATTEN_H
#define MERCURY_OPERATOR_FLATTEN_H

#include "layer.h"

namespace mercury {

class Flatten : public Layer
{
public:
    Flatten();

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_FLATTEN_H
