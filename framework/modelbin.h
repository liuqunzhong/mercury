#ifndef MERCURY_FRAMEWORK_MODELBIN_H
#define MERCURY_FRAMEWORK_MODELBIN_H

#include <stdio.h>
#include "tensor.h"
#include "config.h"

namespace mercury {

class Net;
class ModelBin
{
public:
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = uint8
    // load vec
    Tensor<float> load(int w, int type) const;
    // load image
    Tensor<float> load(int w, int h, int type) const;
    // load dim
    Tensor<float> load(int w, int h, int c, int type) const;

    // construct from weight blob array
    ModelBin(const Tensor<float>* weights);

protected:
    mutable const Tensor<float>* weights;

    friend class Net;

#if USE_STDIO
    ModelBin(FILE* binfp);
    FILE* binfp;
#endif // USE_STDIO

    ModelBin(const unsigned char*& mem);
    const unsigned char*& mem;
};

} // namespace mercury

#endif // MERCURY_FRAMEWORK_MODELBIN_H
