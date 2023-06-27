#include "../common.h"
#include "pim_stream.h"

__global__ void add(row_t *mem_rows, int num_rows) {
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    float dummy = 100;
    uint16_t store = 100;

    for (int row_A = 0; row_A < num_rows; row_A++) {
        int row_B = row_A + num_rows;
        int row_C = row_B + num_rows;
        for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

            // reg = a[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + b[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // c[i] = reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_C].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

__global__ void copy(row_t *mem_rows, int num_rows) {
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    float dummy = 100;
    uint16_t store = 100;

    for (int row_A = 0; row_A < num_rows; row_A++) {
        int row_B = row_A + num_rows;
        for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

            // reg = a[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // b[i] = reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

__global__ void daxpy(row_t *mem_rows, float scalar, int num_rows) {
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    uint16_t store = 100;

    for (int row_A = 0; row_A < num_rows; row_A++) {
        int row_B = row_A + num_rows;
        for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

            // reg = scalar * a[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + b[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }

            __threadfence();

            // b[i] = reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

__global__ void scale(row_t *mem_rows, float scalar, int num_rows) {
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    uint16_t store = 100;

    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

            // reg = scalar * a[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }

            __threadfence();

            // a[i] = reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

__global__ void triad(row_t *mem_rows, float scalar, int num_rows) {
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    uint16_t store = 100;

    for (int row_A = 0; row_A < num_rows; row_A++) {
        int row_B = row_A + num_rows;
        int row_C = row_B + num_rows;
        for (int col = 0; col < NUM_COLS; col += PIM_RF_SIZE) {

            // reg = scalar * b[i]
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_B].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = a[i] + reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_A].val[mem_index])), "f"(scalar)
                        : /* no clobbers */);
            }

            __threadfence();

            // c[i] = reg
            for (int i = thread_id_in_grp; i < PIM_RF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_C].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}
