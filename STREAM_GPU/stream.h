#ifndef __STREAM_H__
#define __STREAM_H__

__global__ void add(float*, float*, float*);
__global__ void copy(float*, float*);
__global__ void daxpy(float*, float*, float);
__global__ void scale(float*, float);
__global__ void triad(float*, float*, float*, float);

#endif
