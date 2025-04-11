#ifndef __PIM_REAL_H__
#define __PIM_REAL_H__

#include "../common.h"
#define KMEANS_NUM_CLUSTERS 5

__global__ void bn_fwd(row_t*, int, row_t*, row_t*, row_t*, row_t*, row_t*,
        row_t*);
__global__ void bn_bwd(row_t*, int, row_t*, row_t*, row_t*, row_t*, row_t*,
        row_t*, row_t*, row_t*);
__global__ void kmeans(row_t*, int, row_t*, int*, int, int, int, int, row_t*);
__global__ void fully_connected(row_t*, row_t*, row_t*, int, int);
__global__ void grim(row_t*, int);
__global__ void softmax(row_t *mem_rows, int num_rows);

#endif
