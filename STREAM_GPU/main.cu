#include <stdio.h>
#include <string.h>
#include "stream.h"
#include "real.h"

void checkCudaError(cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <kernel> <array length>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *kernel_name = argv[1];
    int array_length = atoi(argv[2]);

    float scalar = 0.5;

    int blocksPerGrid = (array_length + THREADS_PER_BLOCK - 1) / \
                        THREADS_PER_BLOCK;

    if (strcmp(kernel_name, "grim") != 0) {
        printf("CUDA kernel launch with %d blocks of %d threads\n",
                blocksPerGrid, THREADS_PER_BLOCK);
    }

    if (strcmp(kernel_name, "stream_add") == 0) {
        float *a, *b, *c;
        checkCudaError(cudaMalloc((void **)&a, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&b, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&c, sizeof(float) * array_length));

        add<<<blocksPerGrid, THREADS_PER_BLOCK>>>(a, b, c);

        checkCudaError(cudaFree(a));
        checkCudaError(cudaFree(b));
        checkCudaError(cudaFree(c));
    }

    else if (strcmp(kernel_name, "stream_copy") == 0) {
        float *a, *b;
        checkCudaError(cudaMalloc((void **)&a, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&b, sizeof(float) * array_length));

        copy<<<blocksPerGrid, THREADS_PER_BLOCK>>>(a, b);

        checkCudaError(cudaFree(a));
        checkCudaError(cudaFree(b));
    }

    else if (strcmp(kernel_name, "stream_daxpy") == 0) {
        float *a, *b;
        checkCudaError(cudaMalloc((void **)&a, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&b, sizeof(float) * array_length));

        daxpy<<<blocksPerGrid, THREADS_PER_BLOCK>>>(a, b, scalar);

        checkCudaError(cudaFree(a));
        checkCudaError(cudaFree(b));
    }

    else if (strcmp(kernel_name, "stream_scale") == 0) {
        float *a;
        checkCudaError(cudaMalloc((void **)&a, sizeof(float) * array_length));

        scale<<<blocksPerGrid, THREADS_PER_BLOCK>>>(a, scalar);

        checkCudaError(cudaFree(a));
    }

    else if (strcmp(kernel_name, "stream_triad") == 0) {
        float *a, *b, *c;
        checkCudaError(cudaMalloc((void **)&a, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&b, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&c, sizeof(float) * array_length));

        triad<<<blocksPerGrid, THREADS_PER_BLOCK>>>(a, b, c, scalar);

        checkCudaError(cudaFree(a));
        checkCudaError(cudaFree(b));
        checkCudaError(cudaFree(c));
    }

    else if (strcmp(kernel_name, "bn_fwd") == 0) {
        int num_batches = array_length / BN_BATCH_SIZE;
        int num_threads = blocksPerGrid * THREADS_PER_BLOCK;

        float *x, *z, *mean, *var, *weight, *bias, *eps;
        checkCudaError(cudaMalloc((void **)&x, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&z, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&mean,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&var,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&weight,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&bias,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&eps,
                    sizeof(float) * BN_BATCH_SIZE));

        bn_fwd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(x, z, num_batches,
                num_threads, mean, var, weight, bias, eps);

        checkCudaError(cudaFree(x));
        checkCudaError(cudaFree(z));
        checkCudaError(cudaFree(mean));
        checkCudaError(cudaFree(var));
        checkCudaError(cudaFree(weight));
        checkCudaError(cudaFree(bias));
        checkCudaError(cudaFree(eps));
    }

    else if (strcmp(kernel_name, "bn_bwd") == 0) {
        int num_batches = array_length / BN_BATCH_SIZE;
        int num_threads = blocksPerGrid * THREADS_PER_BLOCK;

        float *z, *dz, *w, *dw, *dx, *mean, *var, *weight, *bias, *eps, *edz,
              *eydz;
        checkCudaError(cudaMalloc((void **)&z, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&dz, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&w, sizeof(float) * num_batches));
        checkCudaError(cudaMalloc((void **)&dw, sizeof(float) * num_batches));
        checkCudaError(cudaMalloc((void **)&dx, sizeof(float) * array_length));
        checkCudaError(cudaMalloc((void **)&mean,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&var,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&weight,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&bias,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&eps,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&edz,
                    sizeof(float) * BN_BATCH_SIZE));
        checkCudaError(cudaMalloc((void **)&eydz,
                    sizeof(float) * BN_BATCH_SIZE));

        bn_bwd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(z, dz, w, dw, dx,
                num_batches, num_threads, mean, var, weight, bias, eps, edz,
                eydz);

        checkCudaError(cudaFree(z));
        checkCudaError(cudaFree(dz));
        checkCudaError(cudaFree(w));
        checkCudaError(cudaFree(dw));
        checkCudaError(cudaFree(dx));
        checkCudaError(cudaFree(mean));
        checkCudaError(cudaFree(var));
        checkCudaError(cudaFree(weight));
        checkCudaError(cudaFree(bias));
        checkCudaError(cudaFree(eps));
        checkCudaError(cudaFree(edz));
        checkCudaError(cudaFree(eydz));
    }

    else if (strcmp(kernel_name, "kmeans") == 0) {
        int num_datapoints = array_length;
        int num_features = 32;
        int num_iters = 1;
        int num_threads = blocksPerGrid * THREADS_PER_BLOCK;

        float *datapoints, *centroids;
        int *cluster_assignment, *cluster_size;

        checkCudaError(cudaMalloc((void **)&datapoints,
                    num_datapoints * num_features * sizeof(float)));
        checkCudaError(cudaMalloc((void **)&centroids,
                    KMEANS_NUM_CLUSTERS * num_features * sizeof(float)));
        checkCudaError(cudaMalloc((void **)&cluster_assignment,
                    num_datapoints * sizeof(int)));
        checkCudaError(cudaMalloc((void **)&cluster_size,
                    KMEANS_NUM_CLUSTERS * sizeof(int)));

        kmeans<<<blocksPerGrid, THREADS_PER_BLOCK>>>(datapoints,
                cluster_assignment, centroids, cluster_size, num_datapoints,
                num_features, num_iters, num_threads);

        checkCudaError(cudaFree(datapoints));
        checkCudaError(cudaFree(centroids));
        checkCudaError(cudaFree(cluster_assignment));
        checkCudaError(cudaFree(cluster_size));
    }

    else if (strcmp(kernel_name, "histogram") == 0) {
        uint32_t *input, *local_bins, *bins;
        int num_threads = blocksPerGrid * THREADS_PER_BLOCK;

        checkCudaError(cudaMalloc((void **)&input,
                    array_length * sizeof(uint32_t)));
        checkCudaError(cudaMalloc((void **)&local_bins,
                    blocksPerGrid * HISTOGRAM_NUM_BINS * sizeof(uint32_t)));
        checkCudaError(cudaMalloc((void **)&bins,
                    HISTOGRAM_NUM_BINS * sizeof(uint32_t)));

        histogram<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, array_length,
                local_bins, bins, num_threads, blocksPerGrid);

        checkCudaError(cudaFree(input));
        checkCudaError(cudaFree(local_bins));
        checkCudaError(cudaFree(bins));
    }

    else if (strcmp(kernel_name, "fc") == 0) {
        float *input, *weights, *output;

        int input_size = 256;
        int num_threads = blocksPerGrid * THREADS_PER_BLOCK;

        checkCudaError(cudaMalloc((void **)&input,
                    input_size * sizeof(float)));
        checkCudaError(cudaMalloc((void **)&weights,
                    array_length * input_size * sizeof(float)));
        checkCudaError(cudaMalloc((void **)&output,
                    array_length * sizeof(float)));

        fully_connected<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, weights,
                output, input_size, array_length, num_threads);

        checkCudaError(cudaFree(input));
        checkCudaError(cudaFree(weights));
        checkCudaError(cudaFree(output));
    }

    else if (strcmp(kernel_name, "grim") == 0) {
        // GRIM assumes bin size is 32
        blocksPerGrid = (GRIM_NUM_BINS + THREADS_PER_BLOCK - 1) / \
                            THREADS_PER_BLOCK;

        printf("CUDA kernel launch with %d blocks of %d threads\n",
                blocksPerGrid, THREADS_PER_BLOCK);

        uint32_t *bins, *mask;
        checkCudaError(cudaMalloc((void **)&bins, sizeof(uint32_t) * \
                    GRIM_NUM_BINS));
        checkCudaError(cudaMalloc((void **)&mask, sizeof(uint32_t) * \
                    GRIM_NUM_BINS));

        grim<<<blocksPerGrid, THREADS_PER_BLOCK>>>(bins, 5, mask);

        checkCudaError(cudaFree(bins));
        checkCudaError(cudaFree(mask));
    }

    else {
        printf("Invalid kernel name: %s\n", kernel_name);
        return EXIT_FAILURE;
    }

    return 0;
}
