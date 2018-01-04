#include "log.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(Log)

Log::Log()
{
    one_blob_only = true;
    support_inplace = true;
}

int Log::load_param(const ParamDict& pd)
{
    base = pd.get(0, -1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Log::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (base == -1.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = log(shift + ptr[i] * scale);
            }
        }
    }
    else
    {
        float log_base_inv = 1.f / log(base);

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = log(shift + ptr[i] * scale) * log_base_inv;
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
