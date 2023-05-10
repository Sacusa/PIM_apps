#include "../common.h"
#include "pim_stream.h"

__global__ void add(row_t *mem_rows, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    const float dummy_val = 100;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = 0; row_A < num_rows; row_A++) {
            int row_B = row_A + num_rows;
            int row_C = row_B + num_rows;
            for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

                // reg = a[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_A].val[mem_index])),
                              "f"(dummy_val)
                            : /* no clobbers */);
                }

                __threadfence();

                // reg = reg + b[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_B].val[mem_index])),
                              "f"(dummy_val)
                            : /* no clobbers */);
                }

                __threadfence();

                // c[i] = reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_C].val[mem_index])),
                              "f"(dummy_val)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }
}

__global__ void copy(row_t *mem_rows, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    const float dummy_val = 100;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = 0; row_A < num_rows; row_A++) {
            int row_B = row_A + num_rows;
            for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

                // reg = a[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_A].val[mem_index])),
                              "f"(dummy_val)
                            : /* no clobbers */);
                }

                __threadfence();

                // b[i] = reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_B].val[mem_index])),
                              "f"(dummy_val)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }
}

__global__ void daxpy(row_t *mem_rows, const float scalar, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = 0; row_A < num_rows; row_A++) {
            int row_B = row_A + num_rows;
            for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

                // reg = scalar * a[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_A].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();

                // reg = b[i] + reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_B].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();

                // b[i] = reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_B].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }
}

__global__ void scale(row_t *mem_rows, const float scalar, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row = 0; row < num_rows; row++) {
            for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

                // reg = scalar * a[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row].val[mem_index])), "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();

                // a[i] = reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row].val[mem_index])), "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }
}

__global__ void triad(row_t *mem_rows, const float scalar, int num_rows) {
    const int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int bank = 0; bank < NUM_BANKS; bank += NUM_PIM_UNITS) {
        for (int row_A = 0; row_A < num_rows; row_A++) {
            int row_B = row_A + num_rows;
            int row_C = row_B + num_rows;
            for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

                // reg = scalar * b[i]
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_B].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();

                // reg = a[i] + reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_A].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();

                // c[i] = reg
                for (int i = 0; i < PIM_RF_SIZE; i++) {
                    uint64_t mem_index = INDEX(thread_index, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(mem_rows[row_C].val[mem_index])),
                              "f"(scalar)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }
}
