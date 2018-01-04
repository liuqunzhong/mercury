#include "input.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Input)

Input::Input()
{
    one_blob_only = true;
    support_inplace = true;
}

int Input::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int Input::forward_inplace(Tensor<float>& /*bottom_top_blob*/) const
{
    return 0;
}

} // namespace ncnn
