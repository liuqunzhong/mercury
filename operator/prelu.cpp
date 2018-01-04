#include "prelu.h"

namespace mercury {

DEFINE_LAYER_CREATOR(PReLU)

PReLU::PReLU()
{
    one_blob_only = true;
    support_inplace = true;
}

int PReLU::load_param(const ParamDict& pd)
{
    num_slope = pd.get(0, 0);

    return 0;
}

int PReLU::load_model(const ModelBin& mb)
{
    slope_data = mb.load(num_slope, 1);
    if (slope_data.empty())
        return -100;

    return 0;
}

int PReLU::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

        for (int i=0; i<size; i++)
        {
            if (ptr[i] < 0)
                ptr[i] *= slope;
        }
    }
	*/
    return 0;
}

} // namespace ncnn
