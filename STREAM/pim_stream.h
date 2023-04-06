#ifndef __PIM_STREAM_H__
#define __PIM_STREAM_H__

#include "../common.h"

__global__ void add(row_t*, int);
__global__ void copy(row_t*, int);
__global__ void daxpy(row_t*, const float, int);
__global__ void scale(row_t*, const float, int);
__global__ void triad(row_t*, const float, int);

#endif
