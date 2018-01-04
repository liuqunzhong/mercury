#ifndef MERCURY_OPERATOR_SIGMOID_H
#define MERCURY_OPERATOR_SIGMOID_H

#include "layer.h"

namespace mercury {

class Sigmoid : public Layer
{
public:
    Sigmoid();

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SIGMOID_H
