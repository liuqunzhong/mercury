#ifndef MERCURY_OPERATOR_REDUCTION_H
#define MERCURY_OPERATOR_REDUCTION_H

#include "layer.h"

namespace mercury {

class Reduction : public Layer
{
public:
    Reduction();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

    enum {
        ReductionOp_SUM     = 0,
        ReductionOp_ASUM    = 1,
        ReductionOp_SUMSQ   = 2,
        ReductionOp_MEAN    = 3,
        ReductionOp_MAX     = 4,
        ReductionOp_MIN     = 5,
        ReductionOp_PROD    = 6
    };

public:
    // param
    int operation;
    int dim;
    float coeff;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_REDUCTION_H
