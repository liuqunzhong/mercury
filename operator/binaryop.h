#ifndef MERCURY_OPERATOR_BINARYOP_H
#define MERCURY_OPERATOR_BINARYOP_H

#include "layer.h"

namespace mercury {

class BinaryOp : public Layer
{
public:
    BinaryOp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const;

    enum {
        Operation_ADD   = 0,
        Operation_SUB   = 1,
        Operation_MUL   = 2,
        Operation_DIV   = 3,
        Operation_MAX   = 4,
        Operation_MIN   = 5,
        Operation_POW   = 6
    };

public:
    // param
    int op_type;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_BINARYOP_H
