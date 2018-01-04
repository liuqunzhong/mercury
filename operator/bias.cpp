#include "bias.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Bias)

Bias::Bias()
{
    one_blob_only = true;
    support_inplace = true;
}

int Bias::load_param(const ParamDict& pd)
{
    bias_data_size = pd.get(0, 0);

    return 0;
}

int Bias::load_model(const ModelBin& mb)
{
    bias_data = mb.load(bias_data_size, 1);
    if (bias_data.empty())
        return MERC_ERR;

    return 0;
}

int Bias::forward_inplace(Tensor<float>& bottom_top_blob) const
{
    int w = bottom_top_blob.width();
    int h = bottom_top_blob.height();
    int channels = bottom_top_blob.channel();
	int num = bottom_top_blob.num();
    int size = w * h;

    const float* bias_ptr = (const float*)bias_data.get_cpu_data();
	for (int i =0; i< num; i++)
	{
		float* data_batch = bottom_top_blob.get_cpu_data_mutable() + i * channels * size;
#pragma omp parallel for
		for (int q = 0; q < channels; q++)
		{
			float* ptr = data_batch + q * size;

			float bias = bias_ptr[q];

			for (int i = 0; i < size; i++)
			{
				ptr[i] += bias;
			}
		}

	}
    
    return 0;
}

} // namespace mercury
