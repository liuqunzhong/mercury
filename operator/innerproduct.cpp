#include "innerproduct.h"

namespace mercury {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int InnerProduct::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int channels = bottom_blob.channel();
    int size = w * h;

    top_blob.init(1, 1, num_output);
    if (top_blob.empty())
        return -100;
	/*
    // num_output
    const float* weight_data_ptr = weight_data;
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);
        float sum = 0.f;

        if (bias_term)
            sum = bias_data.data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = weight_data_ptr + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        outptr[0] = sum;
    }
	*/
    return 0;
}

} // namespace ncnn
