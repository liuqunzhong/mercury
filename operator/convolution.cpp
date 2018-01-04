#include "convolution.h"

namespace mercury {

DEFINE_LAYER_CREATOR(Convolution)

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

    return 0;
}

int Convolution::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Convolution::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.width();
    int h = bottom_blob.height();
    int channels = bottom_blob.channel();
	int n = bottom_blob.num();

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Tensor<float> bottom_blob_bordered = bottom_blob;
    if (pad_w > 0 || pad_h > 0)
    {
        //copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.width();
        h = bottom_blob_bordered.height();
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            //copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.width();
        h = bottom_blob_bordered.height();
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.init(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // num_output
    const float* weight_data_ptr = weight_data.get_cpu_data();
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.get_cpu_data_mutable(p * top_blob.width() * top_blob.height());

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = *bias_data.get_cpu_data(p);

                const float* kptr = weight_data_ptr + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const float* m = bottom_blob_bordered.get_cpu_data(q * bottom_blob_bordered.width() * bottom_blob_bordered.height());
                    const float* sptr = m + i * stride_h * stride_w + j*stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}

} // namespace ncnn
