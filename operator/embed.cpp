#include "embed.h"
#include <string.h>

namespace mercury {

DEFINE_LAYER_CREATOR(Embed)

Embed::Embed()
{
    one_blob_only = true;
    support_inplace = false;
}

int Embed::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    input_dim = pd.get(1, 0);
    bias_term = pd.get(2, 0);
    weight_data_size = pd.get(3, 0);

    return 0;
}

int Embed::load_model(const ModelBin& mb)
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

int Embed::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
	
    int words = bottom_blob.size();

    top_blob.init(num_output, words, 1);
    if (top_blob.empty())
        return MERC_ERR;
	/*
    // num_output
    const float* word_ptr = bottom_blob;
    const float* dict_ptr = weight_data;
    #pragma omp parallel for
    for (int q=0; q<words; q++)
    {
        float* outptr = top_blob.data + top_blob.w * q;

        int word_index = (int)word_ptr[q];

        // check word_index >= 0 && word_index < input_dim

        const float* em = dict_ptr + num_output * word_index;

        memcpy(outptr, em, num_output * sizeof(float));

        if (bias_term)
        {
            for (int p=0; p<num_output; p++)
            {
                outptr[p] += bias_data.data[p];
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
