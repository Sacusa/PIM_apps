#include "stream.h"

__global__ void add(float *a, float *b, float *c) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void copy(float *a, float *b) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    b[i] = a[i];
}

__global__ void daxpy(float *a, float *b, float scalar) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    b[i] = b[i] + (scalar * a[i]);
}

__global__ void scale(float *a, float scalar) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] = scalar * a[i];
}

__global__ void triad(float *a, float *b, float *c, float scalar) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + (scalar * b[i]);
}
