#include "exp.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(Exp)

Exp::Exp()
{
    one_blob_only = true;
    support_inplace = true;
}

int Exp::load_param(const ParamDict& pd)
{
    base = pd.get(0, -1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Exp::forward_inplace(Tensor<float>& bottom_top_blob) const
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
                ptr[i] = exp(shift + ptr[i] * scale);
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
                ptr[i] = pow(base, (shift + ptr[i] * scale));
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
