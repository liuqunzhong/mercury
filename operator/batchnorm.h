#ifndef MERCURY_OPERATOR_BATCHNORM_H
#define MERCURY_OPERATOR_BATCHNORM_H

#include "layer.h"

namespace mercury {

class BatchNorm : public Layer
{
public:
    BatchNorm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    // param
    int channels;

    // model
    Tensor<float> slope_data;
    Tensor<float> mean_data;
    Tensor<float> var_data;
	Tensor<float> bias_data;

	Tensor<float> a_data;
	Tensor<float> b_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_BATCHNORM_H
