#include "crop.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Crop)

Crop::Crop()
{
}

int Crop::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);

    return 0;
}

int Crop::forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const
{
    const Tensor<float>& bottom_blob = bottom_blobs[0];
    const Tensor<float>& reference_blob = bottom_blobs[1];
	/*
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    int top = hoffset;
    int bottom = h - outh - hoffset;
    int left = woffset;
    int right = w - outw - woffset;

    Mat& top_blob = top_blobs[0];

    copy_cut_border(bottom_blob, top_blob, top, bottom, left, right);
    if (top_blob.empty())
        return -100;
	*/
    return 0;
}

} // namespace mercury
