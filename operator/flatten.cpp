#include "flatten.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Flatten)

Flatten::Flatten()
{
    one_blob_only = true;
    support_inplace = false;
}

int Flatten::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int channels = bottom_blob.channel();
    int size = w * h;

    top_blob.init(size * channels);
    if (top_blob.empty())
        return -100;
	/*
    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.data + size * q;

        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i];
        }
    }
	*/
    return 0;
}

} // namespace ncnn
