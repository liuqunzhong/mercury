#include "relu.h"

namespace mercury {

DEFINE_LAYER_CREATOR(ReLU)

ReLU::ReLU()
{
    one_blob_only = true;
    support_inplace = true;
}

int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

    return 0;
}

int ReLU::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
