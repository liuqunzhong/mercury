#ifndef MERCURY_FRAMEWORK_LAYER_H
#define MERCURY_FRAMEWORK_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "tensor.h"
#include "modelbin.h"
#include "paramdict.h"
#include "config.h"

namespace mercury {

class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

public:
    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Tensor<float>>& bottom_blobs, \
		std::vector<Tensor<float>>& top_blobs) const;
    virtual int forward(const Tensor<float>& bottom_blob, Tensor<float>& top_blob) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Tensor<float>>& bottom_top_blobs) const;
    virtual int forward_inplace(Tensor<float>& bottom_top_blob) const;

public:
#if USE_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // USE_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
};

// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if USE_STRING
    // layer type name
    const char* name;
#endif // USE_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if USE_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // USE_STRING
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name) \
    ::mercury::Layer* name##_layer_creator() { return new name; }

} // namespace mercury

#endif // MERCURY_FRAMEWORK_LAYER_H
