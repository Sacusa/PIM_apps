#ifndef __PIM_REAL_H__
#define __PIM_REAL_H__

#include "../common.h"
#define KMEANS_NUM_CLUSTERS 5
#define HISTOGRAM_NUM_BINS 256

__global__ void bn_fwd(row_t*, int, row_t*, row_t*, row_t*, row_t*, row_t*,
        row_t*);
__global__ void bn_bwd(row_t*, int, row_t*, row_t*, row_t*, row_t*, row_t*,
        row_t*, row_t*, row_t*);
__global__ void kmeans(row_t*, int, int*, float*, int*, int, int, int, int,
        row_t*);
__global__ void histogram(uint32_t*, int, row_t*, int, uint32_t*, int);
__global__ void fully_connected(row_t*, row_t*, row_t*, int, int, int);
__global__ void grim(row_t*, int);

#endif
