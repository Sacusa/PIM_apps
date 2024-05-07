#ifndef __REAL_H__
#define __REAL_H__

#include <stdint.h>

#define THREADS_PER_BLOCK 64
#define BN_BATCH_SIZE 32
#define KMEANS_NUM_CLUSTERS 5
#define HISTOGRAM_NUM_BINS 256
#define GRIM_NUM_BINS 8388608

__global__ void bn_fwd(float*, float*, int, int, float*, float*, float*,
        float*, float*);
__global__ void bn_bwd(float*, float*, float*, float*, float*, int, int,
        float*, float*, float*, float*, float*, float*, float*);
__global__ void kmeans(float*, int*, float*, int*, int, int, int, int);
__global__ void histogram(uint32_t*, int, uint32_t*, uint32_t*, int, int);
__global__ void fully_connected(float*, float*, float*, float*, int, int, int,
        int, int);
__global__ void grim(uint32_t*, uint32_t, uint32_t*);

#endif
