#include "memorydata.h"

namespace mercury {

DEFINE_LAYER_CREATOR(MemoryData)

MemoryData::MemoryData()
{
    one_blob_only = false;
    support_inplace = false;
}

int MemoryData::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int MemoryData::load_model(const ModelBin& mb)
{
    if (c != 0)
    {
        data = mb.load(w, h, c, 1);
    }
    else if (h != 0)
    {
        data = mb.load(w, h, 1);
    }
    else if (w != 0)
    {
        data = mb.load(w, 1);
    }
    else // 0 0 0
    {
        data.init(1);
    }
    if (data.empty())
        return -100;

    return 0;
}

int MemoryData::forward(const std::vector<Tensor<float>>& /*bottom_blobs*/, std::vector<Tensor<float>>& top_blobs) const
{
    Tensor<float>& top_blob = top_blobs[0];

    data.copyto(top_blob);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
