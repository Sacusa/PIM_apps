#include <algorithm>
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "STREAM/pim_real.h"
#include "STREAM/pim_stream.h"

void checkCudaError(cudaError_t err)
{
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" pim_state_t *init_pim(pim_kernel_t pim_kernel, int array_length,
        int element_size)
{
    int row_factor = ROW_SIZE * NUM_BANKS * NUM_CHIPS;
    int array_size = array_length * element_size;

    if ((array_size % row_factor) != 0) {
        printf("<array length> * <element size> must be a factor of %d\n",
                row_factor);
        exit(EXIT_FAILURE);
    }

    int num_rows = array_size / row_factor;

    pim_state_t *pim_state = (pim_state_t*) malloc(sizeof(pim_state_t));
    pim_state->kernel = pim_kernel;
    pim_state->num_rows = num_rows;

    switch (pim_kernel) {
        case STREAM_ADD: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 3));
            pim_state->num_args = 0;
            break;
        }

        case STREAM_COPY: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 2));
            pim_state->num_args = 0;
            break;
        }

        case STREAM_DAXPY: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 2));
            pim_state->num_args = 0;
            break;
        }

        case STREAM_SCALE: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows));
            pim_state->num_args = 0;
            break;
        }

        case STREAM_TRIAD: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 3));
            pim_state->num_args = 0;
            break;
        }

        case BN_FWD: {
            row_t *mean, *var, *weight, *bias, *eps, *temp;

            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 2));
            checkCudaError(cudaMalloc((void **)&mean,   sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&var,    sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&weight, sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&bias,   sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&eps,    sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&temp,
                        sizeof(row_t) * num_rows * 2));

            pim_state->num_args = 6;
            pim_state->args = (void**) malloc(pim_state->num_args * \
                    sizeof(void*));
            pim_state->args[0] = mean;
            pim_state->args[1] = var;
            pim_state->args[2] = weight;
            pim_state->args[3] = bias;
            pim_state->args[4] = eps;
            pim_state->args[5] = temp;
            break;
        }

        case BN_BWD: {
            row_t *mean, *var, *weight, *bias, *eps, *edz, *eydz, *temp;

            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * 5));
            checkCudaError(cudaMalloc((void **)&mean,   sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&var,    sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&weight, sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&bias,   sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&eps,    sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&edz,    sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&eydz,   sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&temp,
                        sizeof(row_t) * num_rows * 6));

            pim_state->num_args = 8;
            pim_state->args = (void**) malloc(pim_state->num_args * \
                    sizeof(void*));
            pim_state->args[0] = mean;
            pim_state->args[1] = var;
            pim_state->args[2] = weight;
            pim_state->args[3] = bias;
            pim_state->args[4] = eps;
            pim_state->args[5] = edz;
            pim_state->args[6] = eydz;
            pim_state->args[7] = temp;
            break;
        }

        case KMEANS: {
            row_t *temp, *centroids;
            int *cluster_assignment;

            pim_state->num_datapoints = array_size;
            pim_state->num_features = 32;
            pim_state->num_iters = 1;

            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * num_rows * pim_state->num_features));
            checkCudaError(cudaMalloc((void **)&temp,
                        sizeof(row_t) * KMEANS_NUM_CLUSTERS * \
                        pim_state->num_features));
            checkCudaError(cudaMalloc((void **)&centroids,
                        (((KMEANS_NUM_CLUSTERS * pim_state->num_features * \
                        sizeof(float)) + ROW_SIZE) / ROW_SIZE) * \
                        sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&cluster_assignment,
                        KMEANS_NUM_CLUSTERS * sizeof(int)));

            pim_state->num_args = 3;
            pim_state->args = (void**) malloc(pim_state->num_args * \
                    sizeof(void*));
            pim_state->args[0] = cluster_assignment;
            pim_state->args[1] = centroids;
            pim_state->args[2] = temp;
            break;
        }

        case HISTOGRAM: {
            int num_threads = NUM_CHIPS * 32;
            num_rows = ((num_threads * HISTOGRAM_NUM_BINS * 4) / \
                    row_factor) + 1;
            pim_state->num_rows = num_rows;
            pim_state->num_datapoints = array_size;

            uint32_t *input, *bins;

            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        num_rows * sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&input,
                        array_length * sizeof(uint32_t)));
            checkCudaError(cudaMalloc((void **)&bins,
                        HISTOGRAM_NUM_BINS * sizeof(uint32_t)));

            pim_state->num_args = 2;
            pim_state->args = (void**) malloc(pim_state->num_args * \
                    sizeof(void*));
            pim_state->args[0] = input;
            pim_state->args[1] = bins;
            break;
        }

        case FULLY_CONNECTED:
        case FULLY_CONNECTED_128_ELEM: {
            row_t *input;
            row_t *output;

            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        pim_state->num_rows * sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&input, sizeof(row_t)));
            checkCudaError(cudaMalloc((void **)&output, sizeof(row_t)));

            pim_state->num_args = 2;
            pim_state->args = (void**) malloc(pim_state->num_args * \
                    sizeof(void*));
            pim_state->args[0] = input;
            pim_state->args[1] = output;
            break;
        }

        case GRIM: {
            pim_state->num_rows++;  // result row
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * pim_state->num_rows));
            pim_state->num_args = 0;
            break;
        }

        case SOFTMAX: {
            checkCudaError(cudaMalloc((void **)&pim_state->rows,
                        sizeof(row_t) * pim_state->num_rows));
            pim_state->num_args = 0;
            break;
        }
    }

    return pim_state;
}

extern "C" void launch_pim(pim_state_t *pim_state, cudaStream_t stream)
{
    const float scalar = 0.5;

    int numThreads = NUM_CHIPS * 32;
    int threadsPerBlock = std::max(numThreads / PIM_GRF_SIZE, 64);
    int blocksPerGrid = (numThreads + threadsPerBlock - 1) / \
                        threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

    switch (pim_state->kernel) {
        case STREAM_ADD:
            add<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(pim_state->rows,
                    pim_state->num_rows);
            break;
        case STREAM_COPY:
            copy<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows);
            break;
        case STREAM_DAXPY:
            daxpy<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, scalar, pim_state->num_rows);
            break;
        case STREAM_SCALE:
            scale<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, scalar, pim_state->num_rows);
            break;
        case STREAM_TRIAD:
            triad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, scalar, pim_state->num_rows);
            break;
        case BN_FWD:
            bn_fwd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows,
                    (row_t*) pim_state->args[0], (row_t*) pim_state->args[1],
                    (row_t*) pim_state->args[2], (row_t*) pim_state->args[3],
                    (row_t*) pim_state->args[4], (row_t*) pim_state->args[5]);
            break;
        case BN_BWD:
            bn_bwd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows,
                    (row_t*) pim_state->args[0], (row_t*) pim_state->args[1],
                    (row_t*) pim_state->args[2], (row_t*) pim_state->args[3],
                    (row_t*) pim_state->args[4], (row_t*) pim_state->args[5],
                    (row_t*) pim_state->args[6], (row_t*) pim_state->args[7]);
            break;
        case KMEANS:
            kmeans<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows,
                    (row_t*) pim_state->args[1], (int*) pim_state->args[0],
                    pim_state->num_datapoints, pim_state->num_features,
                    pim_state->num_iters, numThreads,
                    (row_t*) pim_state->args[2]);
            break;
        case HISTOGRAM:
            histogram<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    (uint32_t*) pim_state->args[0], pim_state->num_datapoints,
                    pim_state->rows, pim_state->num_rows,
                    (uint32_t*) pim_state->args[1], numThreads);
            break;
        case FULLY_CONNECTED:
            fully_connected<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    (row_t*) pim_state->args[0], pim_state->rows,
                    (row_t*) pim_state->args[1], pim_state->num_rows,
                    NUM_COLS);
            break;
        case FULLY_CONNECTED_128_ELEM:
            fully_connected<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    (row_t*) pim_state->args[0], pim_state->rows,
                    (row_t*) pim_state->args[1], pim_state->num_rows, 8);
            break;
        case GRIM:
            grim<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows);
            break;
        case SOFTMAX:
            softmax<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    pim_state->rows, pim_state->num_rows);
            break;
        case NOP:
            break;
    }

    // Wait for the kernel to start running
    //if (pim_state->kernel != NOP) {
    //    cudaStreamGetPriority(stream, NULL);
    //}

    checkCudaError(cudaGetLastError());
}

extern "C" void free_pim(pim_state_t *pim_state)
{
    if (pim_state->kernel != NOP) {
        checkCudaError(cudaFree(pim_state->rows));

        if (pim_state->num_args > 0) {
            for (int i = 0; i < pim_state->num_args; i++) {
                checkCudaError(cudaFree(pim_state->args[i]));
            }

            free(pim_state->args);
        }
    }

    free(pim_state);
}
