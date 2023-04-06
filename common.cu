#include <stdio.h>
#include <string.h>
#include "common.h"
#include "STREAM/pim_stream.h"

extern "C" pim_state_t *init_pim(pim_kernel_t pim_kernel, int array_length,
        int element_size)
{
    cudaError_t err = cudaSuccess;

    int row_factor = ROW_SIZE * NUM_BANKS * NUM_CHIPS;
    int array_size = array_length * element_size;

    if ((array_size % row_factor) != 0) {
        printf("<array length> * <element size> must be a factor of %d\n",
                row_factor);
        exit(EXIT_FAILURE);
    }

    int num_rows = array_size / row_factor;

    row_t *mem_rows = NULL;
    switch (pim_kernel) {
        case STREAM_ADD:
            err = cudaMalloc((void **)&mem_rows, sizeof(row_t) * num_rows * 3);
            break;
        case STREAM_COPY:
            err = cudaMalloc((void **)&mem_rows, sizeof(row_t) * num_rows * 2);
            break;
        case STREAM_DAXPY:
            err = cudaMalloc((void **)&mem_rows, sizeof(row_t) * num_rows * 2);
            break;
        case STREAM_SCALE:
            err = cudaMalloc((void **)&mem_rows, sizeof(row_t) * num_rows);
            break;
        case STREAM_TRIAD:
            err = cudaMalloc((void **)&mem_rows, sizeof(row_t) * num_rows * 3);
            break;
    }

    if (err != cudaSuccess) {
        printf("Failed to allocate device memory (error code: %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    pim_state_t *pim_state = (pim_state_t*) malloc(sizeof(pim_state_t));
    pim_state->kernel = pim_kernel;
    pim_state->rows = mem_rows;
    pim_state->num_rows = num_rows;

    return pim_state;
}

extern "C" void launch_pim(pim_state_t *pim_state, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    const float scalar = 0.5;

    int threadsPerBlock = 512;
    int blocksPerGrid = ((NUM_CHIPS * NUM_WARPS_PER_CHIP * \
                          NUM_THREADS_PER_WARP) + threadsPerBlock - 1) / \
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
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code: %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void free_pim(pim_state_t *pim_state)
{
    cudaError_t err = cudaSuccess;

    err = cudaFree(pim_state->rows);
    if (err != cudaSuccess) {
        printf("Failed to free device memory (error code: %s)\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(pim_state);
}
