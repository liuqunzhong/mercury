#include "dropout.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Dropout)

Dropout::Dropout()
{
    one_blob_only = true;
    support_inplace = true;
}

int Dropout::load_param(const ParamDict& pd)
{
    scale = pd.get(0, 1.f);

    return 0;
}

int Dropout::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
    if (scale == 1.f)
    {
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            ptr[i] = ptr[i] * scale;
        }
    }
	*/
    return 0;
}

} // namespace ncnn
