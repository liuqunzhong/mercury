#ifndef MERCURY_OPERATOR_BNLL_H
#define MERCURY_OPERATOR_BNLL_H

#include "layer.h"

namespace mercury {

class BNLL : public Layer
{
public:
    BNLL();

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
};

} // namespace mercury

#endif // MERCURY_OPERATOR_BNLL_H
