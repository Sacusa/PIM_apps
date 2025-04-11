#include "real.h"
#include <stdio.h>

#define SQR(x) ((x)*(x))

__device__ float distance(float *p1, float *p2, int num_features)
{
    float d = 0;
    for (int i = 0; i < num_features; i++) {
        d += SQR(p1[i] - p2[i]);
    }
    return sqrtf(d);
}

__global__ void bn_fwd(float *x, float *z, int num_batches,
        int num_threads, float *mean, float *var, float *weight,
        float *bias, float *eps)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float invstd[BN_BATCH_SIZE];
    __shared__ float gamma[BN_BATCH_SIZE];

    if (thread_id < BN_BATCH_SIZE) {
        invstd[thread_id] = rsqrt(var[thread_id] + eps[thread_id]);
        gamma[thread_id] = fabsf(weight[thread_id] + eps[thread_id]);
    }

    __syncthreads();

    for (int batch = thread_id; batch < num_batches; batch += num_threads) {
        int batch_offset = batch * BN_BATCH_SIZE;

        for (int i = 0; i < BN_BATCH_SIZE; i++) {
            int batch_i = i + batch_offset;

            float xhat = (x[batch_i] - mean[i]) * invstd[i];
            z[batch_i] = (xhat * gamma[i]) + bias[i];
        }
    }
}

__global__ void bn_bwd(float *z, float *dz, float *w, float *dw, float *dx,
        int num_batches, int num_threads, float *mean, float *var,
        float *weight, float *bias, float *eps, float *edz, float *eydz)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float invstd[BN_BATCH_SIZE];
    __shared__ float gamma[BN_BATCH_SIZE];
    __shared__ float mul[BN_BATCH_SIZE];

    if (thread_id < BN_BATCH_SIZE) {
        invstd[thread_id] = rsqrt(var[thread_id] + eps[thread_id]);
        gamma[thread_id] = fabsf(weight[thread_id] + eps[thread_id]);
        mul[thread_id] = gamma[thread_id] * invstd[thread_id];
    }

    __syncthreads();

    for (int batch = thread_id; batch < num_batches; batch += num_threads) {
        int batch_offset = batch * BN_BATCH_SIZE;

        for (int i = 0; i < BN_BATCH_SIZE; i++) {
            int batch_i = i + batch_offset;

            float y = (z[batch_i] - bias[i]) / gamma[i];
            dx[batch_i] = (dz[batch_i] - w[batch]*edz[i] - \
                    w[batch]*y*eydz[i]) * mul[i];
            dw[batch] -= ((y + mean[i]*invstd[i])*edz[i] + 0.5*y*y*eydz[i]) * \
                         gamma[i];
        }
    }
}

__global__ void kmeans(float *datapoints, int *cluster_assignment,
        float *centroids, int *cluster_size, int num_datapoints,
        int num_features, int num_iters, int num_threads)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int iter = 0; iter < num_iters; iter++) {

        // Find closest centroid to each datapoint
        for (int p = thread_id; p < num_datapoints; p += num_threads) {
            float min_dist = INFINITY;
            int closest_centroid = 0;

            for(int c = 0; c < KMEANS_NUM_CLUSTERS; c++) {
                float dist = distance(&datapoints[p * num_features],
                        &centroids[c * num_features], num_features);

                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = c;
                }
            }

            //assign closest cluster id for this datapoint/thread
            cluster_assignment[p] = closest_centroid;
        }
    }
}

__global__ void fully_connected(float *input, float *weights,
        float *output, int input_size, int num_weights, int num_threads)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = thread_id; i < num_weights; i += num_threads) {
        float sum = 0;

        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[(i * input_size) + j];
        }

        output[i] = sum;
    }
}

__global__ void grim(uint32_t *bins, uint32_t threshold, uint32_t *mask)
{
    const int bin = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t bin_val = bins[bin] + bin;
    uint32_t sum = 0;

    while (bin_val > 0) {
        bin_val &= (bin_val - 1);
        sum++;
    }

    if (sum >= threshold) {
        uint32_t mask_val = 1 << (bin % 32);
        uint32_t mask_index = bin / 32;
        atomicAdd(&mask[mask_index], mask_val);
        //mask[mask_index] += mask_val;
    }
}
