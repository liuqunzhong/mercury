#include "elu.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(ELU)

ELU::ELU()
{
    one_blob_only = true;
    support_inplace = true;
}

int ELU::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 0.1f);

    return 0;
}

int ELU::forward_inplace(Tensor<float>& bottom_top_blob) const
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
            if (ptr[i] < 0.f)
                ptr[i] = alpha * (exp(ptr[i]) - 1.f);
        }
    }
	*/
    return 0;
}

} // namespace ncnn
