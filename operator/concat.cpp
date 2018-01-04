#include "concat.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Concat)

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Concat::forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const
{
    int dims = bottom_blobs[0].dims();

    if (dims == 1) // axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.width();
        }

        Tensor<float>& top_blob = top_blobs[0];
        top_blob.init(top_w);
        if (top_blob.empty())
            return MERC_ERR;

        float* outptr = top_blob.get_cpu_data_mutable();
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];

            int w = bottom_blob.width();

            const float* ptr = bottom_blob.get_cpu_data();
            memcpy(outptr, ptr, w * sizeof(float));

            outptr += w;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].width();

        // total height
        int top_h = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.height();
        }

        Tensor<float>& top_blob = top_blobs[0];
        top_blob.init(w, top_h);
        if (top_blob.empty())
            return -100;

        float* outptr = top_blob.get_cpu_data_mutable();
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];

            int size = w * bottom_blob.height();

            const float* ptr = bottom_blob.get_cpu_data();
            memcpy(outptr, ptr, size * sizeof(float));

            outptr += size;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].height();

        // total width
        int top_w = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.width();
        }

        Tensor<float>& top_blob = top_blobs[0];
        top_blob.init(top_w, h);
        if (top_blob.empty())
            return MERC_ERR;

        #pragma omp parallel for
        for (int i=0; i<h; i++)
        {
            float* outptr = top_blob.get_cpu_data_mutable(i * top_w);
            for (size_t b=0; b<bottom_blobs.size(); b++)
            {
                const Tensor<float>& bottom_blob = bottom_blobs[b];

                const float* ptr = bottom_blob.get_cpu_data(i * bottom_blob.width());
                memcpy(outptr, ptr, bottom_blob.width() * sizeof(float));

                outptr += bottom_blob.width();
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].width();
        int h = bottom_blobs[0].height();

        // total channels
        int top_channels = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.channel();
        }

        Tensor<float>& top_blob = top_blobs[0];
        top_blob.init(w, h, top_channels);
        if (top_blob.empty())
            return -100;

        int q = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Tensor<float>& bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.channel();
            int size = bottom_blob.width() * bottom_blob.height() * channels;

            const float* ptr = bottom_blob.get_cpu_data();
            float* outptr = top_blob.get_cpu_data_mutable(q * top_blob.width() * top_blob.height());
            memcpy(outptr, ptr, size * sizeof(float));

            q += channels;
        }

        return 0;
    }
	/*
    if (dims == 3 && axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (size_t b=0; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const float* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * sizeof(float));
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total height
        int top_w = 0;
        for (size_t b=0; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i=0; i<h; i++)
            {
                for (size_t b=0; b<bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    const float* ptr = bottom_blob.channel(q).row(i);
                    memcpy(outptr, ptr, bottom_blob.w * sizeof(float));

                    outptr += bottom_blob.w;
                }
            }
        }

        return 0;
    }
	*/
    return 0;
}

} // namespace mercury
