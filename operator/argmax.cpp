#include "argmax.h"
#include <algorithm>
#include <functional>

namespace mercury {

DEFINE_LAYER_CREATOR(ArgMax)

ArgMax::ArgMax()
{
}

int ArgMax::load_param(const ParamDict& pd)
{
    out_max_val = pd.get(0, 0);
    topk = pd.get(1, 1);

    return 0;
}

int ArgMax::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    int size = bottom_blob.size();

	if (out_max_val) {
		std::vector<int> shape = {topk, 2};
		top_blob.init(shape);
	} 
	else
	{
		std::vector<int> shape = { topk, 1 };
		top_blob.init(shape);
	} 
    if (top_blob.empty())
        return MERC_ERR;

    const float* ptr = bottom_blob.get_cpu_data();

    // partial sort topk with index
    // optional value
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                        std::greater< std::pair<float, int> >());

    float* outptr = (float*)top_blob.get_cpu_data_mutable();
    if (out_max_val)
    {
        float* valptr = outptr + topk;
        for (int i=0; i<topk; i++)
        {
            outptr[i] = vec[i].first;
            valptr[i] = vec[i].second;
        }
    }
    else
    {
        for (int i=0; i<topk; i++)
        {
            outptr[i] = vec[i].second;
        }
    }

    return 0;
}

} // namespace mercury
