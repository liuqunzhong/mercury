#include "layer.h"
#include <string.h>

namespace mercury {

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::forward(const std::vector<Tensor<float>>& bottom_blobs, std::vector<Tensor<float>>& top_blobs) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        bottom_blobs[i].copyto(top_blobs[i]);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs);
}

int Layer::forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const
{
    if (!support_inplace)
        return -1;

    bottom_blob.copyto(top_blob);
    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob);
}

int Layer::forward_inplace(std::vector<Tensor<float>>& /*bottom_top_blobs*/) const
{
    return -1;
}

int Layer::forward_inplace(Tensor<float>& /*bottom_top_blob*/) const
{
    return -1;
}

#include "layer_declaration.h"

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if USE_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // USE_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

} // namespace mercury
