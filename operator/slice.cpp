#include "slice.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Slice)

Slice::Slice()
{
}

int Slice::load_param(const ParamDict& pd)
{
    slices = pd.get(0, Tensor<float>());

    return 0;
}

int Slice::forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const
{
	/*
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int q = 0;
    const int* slices_ptr = (const int*)slices.data;
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        int slice = slices_ptr[i];
        if (slice == -233)
        {
            slice = (channels - q) / (top_blobs.size() - i);
        }

        Mat& top_blob = top_blobs[i];
        top_blob.create(w, h, slice);
        if (top_blob.empty())
            return -100;

        int size = bottom_blob.cstep * slice;

        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.data;
        for (int j=0; j<size; j++)
        {
            outptr[j] = ptr[j];
        }

        q += slice;
    }
	*/
    return 0;
}

} // namespace ncnn
