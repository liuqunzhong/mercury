#include "threshold.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Threshold)

Threshold::Threshold()
{
    one_blob_only = true;
    support_inplace = true;
}

int Threshold::load_param(const ParamDict& pd)
{
    threshold = pd.get(0, 0.f);

    return 0;
}

int Threshold::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
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
            ptr[i] = ptr[i] > threshold ? 1.f : 0.f;
        }
    }
	*/
    return 0;
}

} // namespace ncnn
