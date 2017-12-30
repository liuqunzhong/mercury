//
// Created by Li,Xiaoyang(SYS) on 17/12/30.
//

#ifndef MERCURY_BASE_TYPE_H
#define MERCURY_BASE_TYPE_H

struct int4d {
    int left;
    int right;
    int top;
    int bottom;
};

struct int2d {
    int x; // width
    int y; // height
};

enum POOL_TYPE{
    AK_POOL_MAX = 0,
    AK_POOL_AVE,
    AK_STOCHASTIC
};

enum BORDER_TYPE{
    AK_BORDER_CONSTANT = 0,
    AK_BORDER_REPLICATE
};

enum SyncHead{
    UNINITIALIZED = 0,
    HEAD_AT_CPU,
    HEAD_AT_GPU,
    SYNCED};

#endif //MERCURY_BASE_TYPE_H
