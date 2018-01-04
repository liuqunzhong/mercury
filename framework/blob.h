#ifndef MERCURY_FRAMEWORK_BLOB_H
#define MERCURY_FRAMEWORK_BLOB_H

#include <string>
#include <vector>
#include "config.h"

namespace mercury {

class Blob
{
public:
    // empty
    Blob();

public:
#if USE_STRING
    // blob name
    std::string name;
#endif // USE_STRING
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    std::vector<int> consumers;
};

} // namespace mercury

#endif // MERCURY_FRAMEWORK_BLOB_H
