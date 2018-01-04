#ifndef MERCURY_OPERATOR_DROPOUT_H
#define MERCURY_OPERATOR_DROPOUT_H

#include "layer.h"

namespace mercury {

class Dropout : public Layer
{
public:
    Dropout();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float scale;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_DROPOUT_H
