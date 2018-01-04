
#ifndef MERCURY_OPERATOR_ABSVAL_H
#define MERCURY_OPERATOR_ABSVAL_H

#include "layer.h"

namespace mercury {

class AbsVal : public Layer
{
public:
    AbsVal();

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
};

} // namespace mercury

#endif // MERCURY_OPERATOR_ABSVAL_H
