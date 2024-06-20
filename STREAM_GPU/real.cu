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

        // Reset cluster sizes and centroids
        //if (thread_id < KMEANS_NUM_CLUSTERS) {
        //    cluster_size[thread_id] = 0;
        //    for (int f = 0; f < num_features; f++) {
        //        centroids[(thread_id * num_features) + f] = 0;
        //    }
        //}

        //__syncthreads();

        //if (threadIdx.x == 0) {
        //    int first_point = blockDim.x * blockIdx.x;
        //    int last_point = blockDim.x * (blockIdx.x + 1);

        //    // Compute cluster size
        //    float cluster_size_local[KMEANS_NUM_CLUSTERS] = {0};

        //    for (int p = first_point; p < last_point; p++) {
        //        cluster_size_local[cluster_assignment[p]]++;
        //    }

        //    for (int c = 0; c < KMEANS_NUM_CLUSTERS; c++) {
        //        atomicAdd(&cluster_size[c], cluster_size_local[c]);
        //    }

        //    // Compute cluster datapoint sums
        //    for (int f = 0; f < num_features; f++) {
        //        float cluster_datapoint_sums[KMEANS_NUM_CLUSTERS] = {0};

        //        for (int p = first_point; p < last_point; p++) {
        //            cluster_datapoint_sums[cluster_assignment[p]] += \
        //                    datapoints[(p * num_features) + f];
        //        }

        //        for (int c = 0; c < KMEANS_NUM_CLUSTERS; c++) {
        //            atomicAdd(&centroids[(c * num_features) + f],
        //                    cluster_datapoint_sums[c]);
        //        }
        //    }
        //}

        //__syncthreads();
        //
        //// Recompute centroids
        //if (thread_id < KMEANS_NUM_CLUSTERS) {
        //    for (int f = 0; f < num_features; f++) {
        //        centroids[(thread_id * num_features) + f] /= \
        //                cluster_size[thread_id];
        //    }
        //}

        //__syncthreads();
    }
}

__global__ void histogram(uint32_t *input, int num_elements,
        uint32_t *local_bins, uint32_t *bins, int num_threads, int num_blocks)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize local memory to store block-wise totals
    __shared__ uint32_t smem_bins[HISTOGRAM_NUM_BINS];
    for (int i = thread_id; i < HISTOGRAM_NUM_BINS; i += blockDim.x) {
        smem_bins[i] = 0;
    }
    __syncthreads();

    // Compute block-wise totals
    for (int i = thread_id; i < num_elements; i += num_threads) {
        atomicAdd(&smem_bins[input[i]], 1);
    }

    __syncthreads();

    // Write block-wise totals to global memory
    for (int i = thread_id; i < HISTOGRAM_NUM_BINS; i += blockDim.x) {
        local_bins[(blockIdx.x * HISTOGRAM_NUM_BINS) + i] = smem_bins[i];
    }

    __syncthreads();

    // Sum values from all blocks
    if (thread_id < HISTOGRAM_NUM_BINS) {
        uint32_t total = 0;
        for (int i = 0; i < num_blocks; i++) {
            total += local_bins[(i * HISTOGRAM_NUM_BINS) + thread_id];
        }
        bins[thread_id] = total;
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
