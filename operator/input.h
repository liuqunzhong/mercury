#ifndef MERCURY_OPERATOR_INPUT_H
#define MERCURY_OPERATOR_INPUT_H

#include "layer.h"

namespace mercury {

class Input : public Layer
{
public:
    Input();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    int w;
    int h;
    int c;
	int n;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_INPUT_H
