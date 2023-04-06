#include <stdio.h>
#include <string.h>
#include "../common.h"
#include "pim_stream.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <kernel> <array length> <element size (in B)>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    char *kernel_name = argv[1];
    int array_length = atoi(argv[2]);
    int element_size = atoi(argv[3]);

    pim_kernel_t pim_kernel;

    if (strcmp(kernel_name, "stream_add") == 0) {
        pim_kernel = STREAM_ADD;
    }
    else if (strcmp(kernel_name, "stream_copy") == 0) {
        pim_kernel = STREAM_COPY;
    }
    else if (strcmp(kernel_name, "stream_daxpy") == 0) {
        pim_kernel = STREAM_DAXPY;
    }
    else if (strcmp(kernel_name, "stream_scale") == 0) {
        pim_kernel = STREAM_SCALE;
    }
    else if (strcmp(kernel_name, "stream_triad") == 0) {
        pim_kernel = STREAM_TRIAD;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    pim_state_t *pim_state = init_pim(pim_kernel, array_length, element_size);
    launch_pim(pim_state, stream);
    cudaDeviceSynchronize();
    free_pim(pim_state);

    return 0;
}
