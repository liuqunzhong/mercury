#ifndef MERCURY_FRAMEWORK_NET_H
#define MERCURY_FRAMEWORK_NET_H

#include <stdio.h>
#include <vector>
#include "blob.h"
#include "layer.h"
#include "tensor.h"
#include "config.h"

namespace mercury {

class Extractor;
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

#if USE_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // USE_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if USE_STDIO
#if USE_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
#endif // USE_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // USE_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

protected:
    friend class Extractor;
#if USE_STRING
    int find_blob_index_by_name(const char* name) const;
    int find_layer_index_by_name(const char* name) const;
    int custom_layer_to_index(const char* type);
    Layer* create_custom_layer(const char* type);
#endif // USE_STRING
    Layer* create_custom_layer(int index);
    int forward_layer(int layer_index, std::vector<Tensor<float>>& blob_mats, bool lightmode) const;

protected:
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<layer_registry_entry> custom_layer_registry;
};

class Extractor
{
public:
    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

#if USE_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Tensor<float>& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Tensor<float>& feat);
#endif // USE_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Tensor<float>& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Tensor<float>& feat);

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, int blob_count);

private:
    const Net* net;
    std::vector<Tensor<float>> blob_mats;
    bool lightmode;
    int num_threads;
};

} // namespace mercury

#endif // MERCURY_FRAMEWORK_NET_H
