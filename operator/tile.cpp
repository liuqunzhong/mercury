#include "tile.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Tile)

Tile::Tile()
{
    one_blob_only = true;
    support_inplace = false;
}

int Tile::load_param(const ParamDict& pd)
{
    dim = pd.get(0, 0);
    tiles = pd.get(1, 1);

    return 0;
}

int Tile::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
/*
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    if (dim == 0)
    {
        top_blob.create(w, h, channels * tiles);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        int size = bottom_blob.cstep * channels;

        #pragma omp parallel for
        for (int p=0; p<tiles; p++)
        {
            float* outptr = top_blob.channel(p * channels);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i];
            }
        }
    }
    else if (dim == 1)
    {
        top_blob.create(w, h * tiles, channels);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int p=0; p<tiles; p++)
            {
                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i];
                }

                outptr += size;
            }
        }
    }
    else if (dim == 2)
    {
        top_blob.create(w * tiles, h, channels);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (int p=0; p<tiles; p++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        outptr[j] = ptr[j];
                    }

                    outptr += w;
                }

                ptr += w;
            }
        }
    }
	*/
    return 0;
}

} // namespace ncnn
