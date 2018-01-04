#include "batchnorm.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(BatchNorm)

BatchNorm::BatchNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

int BatchNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);

    return 0;
}

int BatchNorm::load_model(const ModelBin& mb)
{
    slope_data = mb.load(channels, 1);
    if (slope_data.empty())
        return MERC_ERR;

    mean_data = mb.load(channels, 1);
    if (mean_data.empty())
        return MERC_ERR;

    var_data = mb.load(channels, 1);
    if (var_data.empty())
        return MERC_ERR;

    bias_data = mb.load(channels, 1);
    if (bias_data.empty())
        return MERC_ERR;

	std::vector<int> shape = { channels };
    a_data.init(shape);
    if (a_data.empty())
        return MERC_ERR;
    b_data.init(shape);
    if (b_data.empty())
        return MERC_ERR;
    const float* slope_data_ptr = (const float*)slope_data.get_cpu_data();
    const float* mean_data_ptr = (const float*)mean_data.get_cpu_data();
    const float* var_data_ptr = (const float*)var_data.get_cpu_data();
    const float* bias_data_ptr = (const float*)bias_data.get_cpu_data();
    float* a_data_ptr = (float*)a_data.get_cpu_data_mutable();
    float* b_data_ptr = (float*)b_data.get_cpu_data_mutable();
    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data_ptr[i]);
        a_data_ptr[i] = bias_data_ptr[i] - slope_data_ptr[i] * mean_data_ptr[i] / sqrt_var;
        b_data_ptr[i] = slope_data_ptr[i] / sqrt_var;
    }

    return 0;
}

int BatchNorm::forward_inplace(Tensor<float>& bottom_top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.width();
    int h = bottom_top_blob.height();
	int num = bottom_top_blob.num();
    int size = w * h;

    const float* a_data_ptr = (const float*)a_data.get_cpu_data();
    const float* b_data_ptr = (const float*)b_data.get_cpu_data();
	for (int i = 0; i< num; i++)
	{
		float* data_batch = (float*)bottom_top_blob.get_cpu_data_mutable() + i * size * channels;
#pragma omp parallel for
		for (int q = 0; q < channels; q++)
		{
			float* ptr = data_batch + q * size;

			float a = a_data_ptr[q];
			float b = b_data_ptr[q];

			for (int i = 0; i < size; i++)
			{
				ptr[i] = b * ptr[i] + a;
			}
		}
	}
    
    return 0;
}

} // namespace ncnn
