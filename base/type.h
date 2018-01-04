//
// Created by Li,Xiaoyang(SYS) on 17/12/30.
//

#ifndef MERCURY_BASE_TYPE_H
#define MERCURY_BASE_TYPE_H

enum MERC_STATUS
{
	MERC_ERR = -1,
	MERC_OK = 0,
	MERC_WARN = 1
	
};

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
    POOL_MAX = 0,
    POOL_AVE,
    STOCHASTIC
};

enum BORDER_TYPE{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE
};

enum SyncHead{
    UNINITIALIZED = 0,
    HEAD_AT_CPU,
    HEAD_AT_GPU,
    SYNCED};

#endif //MERCURY_BASE_TYPE_H
