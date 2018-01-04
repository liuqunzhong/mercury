#include "shufflechannel.h"

namespace mercury {

DEFINE_LAYER_CREATOR(ShuffleChannel)

int ShuffleChannel::forward(const Tensor<float> &bottom_blob, Tensor<float> &top_blob) const
{
    int c = bottom_blob.channel();
    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int chs_per_group = c / group;

    if (c != chs_per_group * group) {
//        cout << "Wrong group num";
//        error;
        // reject invalid group
        return -100;
    }
    top_blob.init(w, h, c);
    if (top_blob.empty())
        return -100;
	/*
    int dst_q;
    int src_q;
    // cstep * sizeof(float) if addr aligned needed
    size_t feature_sz = w * h * sizeof(float);
    for (int i = 0; i != group; ++i) {
        for (int j = 0; j != chs_per_group; ++j) {
            src_q = chs_per_group * i + j;
            dst_q = group * j + i;
            memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q),
                   feature_sz);
        }
    }
	*/
    return 0;
}

} // namespace ncnn
