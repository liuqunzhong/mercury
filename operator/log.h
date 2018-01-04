#ifndef MERCURY_OPERATOR_LOG_H
#define MERCURY_OPERATOR_LOG_H

#include "layer.h"

namespace mercury {

class Log : public Layer
{
public:
    Log();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    float base;
    float scale;
    float shift;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_LOG_H
