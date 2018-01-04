#include "padding.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Padding)

Padding::Padding()
{
    one_blob_only = true;
    support_inplace = false;
}

int Padding::load_param(const ParamDict& pd)
{
    top = pd.get(0, 0);
    bottom = pd.get(1, 0);
    left = pd.get(2, 0);
    right = pd.get(3, 0);
    type = pd.get(4, 0);
    value = pd.get(5, 0.f);

    return 0;
}

int Padding::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    //copy_make_border(bottom_blob, top_blob, top, bottom, left, right, type, value);

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
