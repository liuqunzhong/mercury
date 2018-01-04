#include "normalize.h"
#include <math.h>

namespace mercury {

DEFINE_LAYER_CREATOR(Normalize)

Normalize::Normalize()
{
    one_blob_only = true;
    support_inplace = false;
}

int Normalize::load_param(const ParamDict& pd)
{
    across_spatial = pd.get(0, 0);
    channel_shared = pd.get(1, 0);
    eps = pd.get(2, 0.0001f);
    scale_data_size = pd.get(3, 0);

    return 0;
}

int Normalize::load_model(const ModelBin& mb)
{
    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    return 0;
}

int Normalize::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int channels = bottom_blob.channel();
    int size = w * h;

    top_blob.init(w, h, channels);
    if (top_blob.empty())
        return -100;
	/*
    if (across_spatial)
    {
        // square
        Mat square_sum_blob;
        square_sum_blob.create(channels);
        if (square_sum_blob.empty())
            return -100;

        float* square_sum_ptr = square_sum_blob;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);

            float ssum = 0.f;
            for (int i=0; i<size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            square_sum_ptr[q] = ssum;
        }

        // sum + eps
        float ssum = eps;
        for (int q=0; q<channels; q++)
        {
            ssum += square_sum_ptr[q];
        }

        // 1 / sqrt(ssum)
        float a = 1.f / sqrt(ssum);

        if (channel_shared)
        {
            float scale = a * scale_data.data[0];

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                float scale = a * scale_data.data[q];

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * scale;
                }
            }
        }
    }
    else
    {
        // square sum, 1 / sqrt(ssum)
        Mat square_sum_blob;
        square_sum_blob.create(w, h);
        if (square_sum_blob.empty())
            return -100;

        float* ssptr = square_sum_blob;

        if (channel_shared)
        {
            float scale = scale_data.data[0];

            #pragma omp parallel for
            for (int i=0; i<size; i++)
            {
                float ssum = eps;
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                ssptr[i] = 1.f / sqrt(ssum) * scale;
            }

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * ssptr[i];
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i=0; i<size; i++)
            {
                float ssum = eps;
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }

                ssptr[i] = 1.f / sqrt(ssum);
            }

            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                float scale = scale_data.data[q];

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * ssptr[i] * scale;
                }
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
