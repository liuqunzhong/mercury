#include "scale.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Scale)

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size != -233)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data.empty())
            return -100;
    }

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Scale::forward_inplace(std::vector<Tensor<float>>& bottom_top_blobs) const
{
	/*
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* bias_ptr = bias_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_blob.channel(q)[0];
            float bias = bias_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * s + bias;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_blob.channel(q)[0];

            for (int i=0; i<size; i++)
            {
                ptr[i] *= s;
            }
        }
    }
	*/
    return 0;
}

int Scale::forward_inplace(Tensor<float>& bottom_top_blob) const
{
	/*
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* scale_ptr = scale_data;
        const float* bias_ptr = bias_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];
            float bias = bias_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * s + bias;
            }
        }
    }
    else
    {
        const float* scale_ptr = scale_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] *= s;
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
