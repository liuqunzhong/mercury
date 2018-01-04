#ifndef MERCURY_OPERATOR_ELU_H
#define MERCURY_OPERATOR_ELU_H

#include "layer.h"

namespace mercury {

class ELU : public Layer
{
public:
    ELU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float alpha;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_ELU_H
