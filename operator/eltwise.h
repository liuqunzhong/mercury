
#ifndef MERCURY_OPERATOR_ELTWISE_H
#define MERCURY_OPERATOR_ELTWISE_H

#include "layer.h"

namespace mercury {

class Eltwise : public Layer
{
public:
    Eltwise();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

    enum { Operation_PROD = 0, Operation_SUM = 1, Operation_MAX = 2 };

public:
    // param
    int op_type;
    Tensor<float> coeffs;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_ELTWISE_H
