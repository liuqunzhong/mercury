#include "split.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Split)

Split::Split()
{
}

int Split::forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const
{
	/*
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }
	*/
    return 0;
}

} // namespace ncnn
