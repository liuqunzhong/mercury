#include "absval.h"

namespace mercury {

DEFINE_LAYER_CREATOR(AbsVal)

AbsVal::AbsVal()
{
    one_blob_only = true;
    support_inplace = true;
}

int AbsVal::forward_inplace(Tensor<float>& bottom_top_blob) const
{
    int w = bottom_top_blob.width();
    int h = bottom_top_blob.height();
    int channels = bottom_top_blob.channel();
	int num = bottom_top_blob.num();
    int size = w * h;
	
	for (int i = 0;i < num; i++)
	{
		float* data_batch = (float*)bottom_top_blob.get_cpu_data_mutable() + i * channels * size;
#pragma omp parallel for
		for (int q = 0; q < channels; q++)
		{
			float* ptr = data_batch + q * size;

			for (int i = 0; i < size; i++)
			{
				if (ptr[i] < 0)
					ptr[i] = -ptr[i];
			}
		}
	}
    
    return 0;
}

} // namespace mercury
