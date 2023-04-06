#include "../common.h"
#include "pim_stream.h"

__global__ void add(row_t *mem_rows, int num_rows) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    const int chip = thread_index / (NUM_WARPS_PER_CHIP * \
                                     NUM_THREADS_PER_WARP);
    const int first_row = thread_index % \
                          (NUM_WARPS_PER_CHIP * NUM_THREADS_PER_WARP);
    const float dummy_val = 100;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = first_row; row_A < num_rows;
             row_A += NUM_THREADS_PER_WARP) {
            for (int col = 0; col < NUM_COLS; col++) {
                int row_B = row_A + num_rows;
                int row_C = row_B + num_rows;
                uint64_t mem_index = INDEX(chip, bank, col);

                // reg = a[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])),
                          "f"(dummy_val)
                        : /* no clobbers */);

                // reg = reg + b[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])),
                          "f"(dummy_val)
                        : /* no clobbers */);

                // c[i] = reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_C].val[mem_index])),
                          "f"(dummy_val)
                        : /* no clobbers */);
            }
        }
    }
}

__global__ void copy(row_t *mem_rows, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    const int chip = thread_index / (NUM_WARPS_PER_CHIP * \
                                     NUM_THREADS_PER_WARP);
    const int first_row = thread_index % \
                        (NUM_WARPS_PER_CHIP * NUM_THREADS_PER_WARP);
    const float dummy_val = 100;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = first_row; row_A < num_rows;
             row_A += NUM_THREADS_PER_WARP) {
            for (int col = 0; col < NUM_COLS; col++) {
                int row_B = row_A + num_rows;
                uint64_t mem_index = INDEX(chip, bank, col);

                // reg = a[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])),
                          "f"(dummy_val)
                        : /* no clobbers */);

                // b[i] = reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])),
                          "f"(dummy_val)
                        : /* no clobbers */);
            }
        }
    }
}

__global__ void daxpy(row_t *mem_rows, const float scalar, int num_rows) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    const int chip = thread_index / (NUM_WARPS_PER_CHIP * \
                                     NUM_THREADS_PER_WARP);
    const int first_row = thread_index % \
                          (NUM_WARPS_PER_CHIP * NUM_THREADS_PER_WARP);

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = first_row; row_A < num_rows;
             row_A += NUM_THREADS_PER_WARP) {
            for (int col = 0; col < NUM_COLS; col++) {
                int row_B = row_A + num_rows;
                uint64_t mem_index = INDEX(chip, bank, col);

                // reg = scalar * a[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);

                // reg = b[i] + reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);

                // b[i] = reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }
        }
    }
}

__global__ void scale(row_t *mem_rows, const float scalar, int num_rows) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    const int chip = thread_index / (NUM_WARPS_PER_CHIP * \
                                     NUM_THREADS_PER_WARP);
    const int first_row = thread_index % \
                          (NUM_WARPS_PER_CHIP * NUM_THREADS_PER_WARP);

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row = first_row; row < num_rows;
             row += NUM_THREADS_PER_WARP) {
            for (int col = 0; col < NUM_COLS; col++) {
                uint64_t mem_index = INDEX(chip, bank, col);

                // reg = scalar * a[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);

                // a[i] = reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }
        }
    }
}

__global__ void triad(row_t *mem_rows, const float scalar, int num_rows) {
    int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    const int chip = thread_index / (NUM_WARPS_PER_CHIP * \
                                     NUM_THREADS_PER_WARP);
    const int first_row = thread_index % \
                          (NUM_WARPS_PER_CHIP * NUM_THREADS_PER_WARP);

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = first_row; row_A < num_rows;
             row_A += NUM_THREADS_PER_WARP) {
            for (int col = 0; col < NUM_COLS; col++) {
                int row_B = row_A + num_rows;
                int row_C = row_B + num_rows;
                uint64_t mem_index = INDEX(chip, bank, col);

                // reg = scalar * b[i]
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);

                // reg = a[i] + reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);

                // c[i] = reg
                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_C].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }
        }
    }
}
