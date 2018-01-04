#include "sigmoid.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(Sigmoid)

Sigmoid::Sigmoid()
{
    one_blob_only = true;
    support_inplace = true;
}

int Sigmoid::forward_inplace(Tensor<float>& bottom_top_blob) const
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
            ptr[i] = 1.f / (1.f + exp(-ptr[i]));
        }
    }
	*/
    return 0;
}

} // namespace ncnn
