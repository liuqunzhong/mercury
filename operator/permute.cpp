#include "permute.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Permute)

Permute::Permute()
{
    one_blob_only = true;
    support_inplace = false;
}

int Permute::load_param(const ParamDict& pd)
{
    order_type = pd.get(0, 0);

    return 0;
}

int Permute::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int channels = bottom_blob.channel();

    // order_type
    // 0 = w h c
    // 1 = h w c
    // 2 = w c h
    // 3 = c w h
    // 4 = h c w
    // 5 = c h w

    if (order_type == 0)
    {
        top_blob = bottom_blob;
    }
    else if (order_type == 1)
    {
        top_blob.init(h, w, channels);
        if (top_blob.empty())
            return -100;
		/*
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    outptr[i*h + j] = ptr[j*w + i];
                }
            }
        }
    }
    else if (order_type == 2)
    {
        top_blob.create(w, channels, h);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < channels; i++)
            {
                const float* ptr = bottom_blob.channel(i).row(q);

                for (int j = 0; j < w; j++)
                {
                    outptr[i*w + j] = ptr[j];
                }
            }
        }
    }
    else if (order_type == 3)
    {
        top_blob.create(channels, w, h);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    const float* ptr = bottom_blob.channel(j).row(q);

                    outptr[i*channels + j] = ptr[i];
                }
            }
        }
    }
    else if (order_type == 4)
    {
        top_blob.create(h, channels, w);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<w; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < channels; i++)
            {
                const float* ptr = bottom_blob.channel(i);

                for (int j = 0; j < h; j++)
                {
                    outptr[i*channels + j] = ptr[j*w + q];
                }
            }
        }
    }
    else if (order_type == 5)
    {
        top_blob.create(channels, h, w);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<w; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    const float* ptr = bottom_blob.channel(j);

                    outptr[i*channels + j] = ptr[i*w + q];
                }
            }
        }
		*/
    }
	return 0;
}

} // namespace mercury
