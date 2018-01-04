#ifndef MERCURY_OPERATOR_SCALE_H
#define MERCURY_OPERATOR_SCALE_H

#include "layer.h"

namespace mercury {

class Scale : public Layer
{
public:
    Scale();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(std::vector<Tensor<float>>& bottom_top_blobs) const;
    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
	Tensor<float> scale_data;
	Tensor<float> bias_data;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_SCALE_H
