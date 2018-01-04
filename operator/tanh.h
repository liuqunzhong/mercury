#ifndef MERCURY_OPERATOR_TANH_H
#define MERCURY_OPERATOR_TANH_H

#include "layer.h"

namespace mercury {

class TanH : public Layer
{
public:
    TanH();

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
};

} // namespace mercury

#endif // MERCURY_OPERATOR_TANH_H
