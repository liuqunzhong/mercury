#ifndef MERCURY_OPERATOR_TILE_H
#define MERCURY_OPERATOR_TILE_H

#include "layer.h"

namespace mercury {

class Tile : public Layer
{
public:
    Tile();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

public:
    int dim;
    int tiles;
};

} // namespace mercury

#endif // MERCURY_OPERATOR_TILE_H
