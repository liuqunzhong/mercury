#ifndef MERCURY_FRAMEWORK_PARAMDICT_H
#define MERCURY_FRAMEWORK_PARAMDICT_H

#include <stdio.h>
#include "tensor.h"
#include "config.h"

// at most 20 parameters
#define MAX_PARAM_COUNT 20

namespace mercury {

class Net;
class ParamDict
{
public:
    // empty
    ParamDict();

    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    Tensor<float> get(int id, const Tensor<float>& def) const;

    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Tensor<float>& v);

protected:
    friend class Net;

    void clear();

#if USE_STDIO
#if USE_STRING
    int load_param(FILE* fp);
#endif // USE_STRING
    int load_param_bin(FILE* fp);
#endif // USE_STDIO
    int load_param(const unsigned char*& mem);

protected:
    struct
    {
        int loaded;
        union { int i; float f; };
        Tensor<float> v;
    } params[MAX_PARAM_COUNT];
};

} // namespace mercury

#endif // MERCURY_FRAMEWORK_PARAMDICT_H
