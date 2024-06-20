#include <cstdio>
#include "../common.h"
#include "pim_real.h"

__global__ void bn_fwd(row_t *mem_rows, int num_rows, row_t *mean,
        row_t *var, row_t *weight, row_t *bias, row_t *eps, row_t *temp) {
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

    // Compute invstd (temp[0])
    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

        // reg = var
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = reg + eps
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(eps->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = sqrt(reg)
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = 1 / reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // temp[0] = reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.u16 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[0].val[mem_index])), "h"(store)
                    : /* no clobbers */);
        }

        __threadfence();
    }

    // Compute gamma (temp[1])
    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

        // reg = abs(weight)
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(weight->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = reg + eps
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(eps->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // temp[1] = reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.u16 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[1].val[mem_index])), "h"(store)
                    : /* no clobbers */);
        }

        __threadfence();
    }

    for (int row_X = 0; row_X < num_rows; row_X++) {
        int row_z = row_X + num_rows;
        for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

            // reg = a[i]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_X].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg - mean
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mean->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * invstd
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[0].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * gamma
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[1].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + beta
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(bias->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // b[i] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_z].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

// mem_rows[0] = z
// mem_rows[1] = dz
// mem_rows[2] = w
// mem_rows[3] = dw
// mem_rows[4] = dx
//
// temp[0] = invstd
// temp[1] = gamma
// temp[2] = mul
// temp[3] = (z - beta) / gamma
// temp[4] and temp[5] change values over time
__global__ void bn_bwd(row_t *mem_rows, int num_rows, row_t *mean,
        row_t *var, row_t *weight, row_t *bias, row_t *eps, row_t *edz,
        row_t *eydz, row_t *temp) {
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

    // Compute invstd (temp[0])
    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

        // reg = var
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = reg + eps
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(eps->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = sqrt(reg)
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = 1 / reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(var->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // temp[0] = reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.u16 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[0].val[mem_index])), "h"(store)
                    : /* no clobbers */);
        }

        __threadfence();
    }

    // Compute gamma (temp[1])
    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

        // reg = abs(weight)
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(weight->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = reg + eps
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(eps->val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // temp[1] = reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.u16 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[1].val[mem_index])), "h"(store)
                    : /* no clobbers */);
        }

        __threadfence();
    }

    // Compute mul (temp[2])
    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

        // reg = gamma
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[1].val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // reg = reg * invstd
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[0].val[mem_index])), "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        // mul = reg
        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank, col + i);

            asm volatile ("st.cs.global.u16 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(temp[2].val[mem_index])), "h"(store)
                    : /* no clobbers */);
        }

        __threadfence();
    }

    // Main loop
    for (int row_z = 0; row_z < num_rows; row_z++) {
        int row_dz = row_z  + num_rows;
        int row_w  = row_dz + num_rows;
        int row_dw = row_w  + num_rows;
        int row_dx = row_dw + num_rows;

        for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

            // reg = z[i]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_z].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg - bias
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(bias->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg / gamma
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[1].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // temp[3] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = w[i]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_w].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * edz
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(edz->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // temp[4] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[4].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = w[i]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_w].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * temp[3]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * eydz
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(eydz->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // temp[5] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[5].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = dz[i]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dz].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg - temp[4]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[4].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg - temp[5]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[5].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * temp[2]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[2].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // dx = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dx].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = mean
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mean->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * invstd (temp[0])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[0].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + y (temp[3])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * edz
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(edz->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // temp[4] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[4].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = y (temp[3])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * y (temp[3])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * 0.5
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[3].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * eydz
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(eydz->val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + temp[4]
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[4].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg + mulW (gamma, temp[1])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(temp[1].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // dw = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dw].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }

        // dw = -dw
        for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

            // reg = dw
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dw].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = reg * -1
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dw].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // mul = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row_dw].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

// The number of datapoints is a multiple of row size.
// number of rows = number of points * number of features
__global__ void kmeans(row_t *mem_rows, int num_rows, row_t *centroids,
        int *cluster_assignment, int num_datapoints, int num_features,
        int num_iters, int num_threads, row_t *temp)
{
    // This is the maximum number of threads that will issue requests for a
    // group of banks that map to a unique PIM unit
    int threads_per_pim_grp = NUM_THREADS_PER_WARP / \
                              (NUM_BANKS / NUM_PIM_UNITS);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id_in_grp = thread_id % threads_per_pim_grp;

    int chip = thread_id / NUM_THREADS_PER_WARP;
    int bank = ((thread_id % NUM_THREADS_PER_WARP) / threads_per_pim_grp) * \
               NUM_PIM_UNITS;

    int num_datapoint_rows = num_rows / num_features;
    int cols_per_centroid = (num_features * sizeof(float)) / COL_SIZE;
    int num_features_in_srf = (PIM_SRF_SIZE * COL_SIZE) / sizeof(float);

    float dummy = 100;
    uint16_t store = 100;

    for (int iter = 0; iter < num_iters; iter++) {

        for (int c = 0; c < KMEANS_NUM_CLUSTERS; c++) {
            for (int row = 0; row < num_rows; row += num_features) {
                int curr_centroid_col = c * cols_per_centroid;
                int c_row = (c * num_datapoint_rows) + row;

                for (int f = 0; f < num_features; f++) {
                    if ((f % num_features_in_srf) == 0) {
                        // Load centroid into SRF
                        for (int i = thread_id_in_grp; i < PIM_SRF_SIZE ;
                                i += threads_per_pim_grp) {
                            uint64_t mem_index = INDEX(chip, bank,
                                    curr_centroid_col + i);

                            asm volatile ("st.cs.global.f32 [%0], %1;"
                                    : /* no outputs */
                                    : "l"(&(centroids->val[mem_index])),
                                      "f"(dummy)
                                    : /* no clobbers */);
                        }

                        __threadfence();

                        curr_centroid_col += PIM_SRF_SIZE;
                    }

                    __syncthreads();

                    for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

                        // reg = datapoint[f] - centroid[f] (scalar memory)
                        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                                i += threads_per_pim_grp) {
                            uint64_t mem_index = INDEX(chip, bank, col + i);

                            asm volatile ("st.cs.global.f32 [%0], %1;"
                                    : /* no outputs */
                                    : "l"(&(mem_rows[row + f].val[mem_index])),
                                      "f"(dummy)
                                    : /* no clobbers */);
                        }

                        __threadfence();

                        // reg = reg * reg
                        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                                i += threads_per_pim_grp) {
                            uint64_t mem_index = INDEX(chip, bank, col + i);

                            asm volatile ("st.cs.global.f32 [%0], %1;"
                                    : /* no outputs */
                                    : "l"(&(mem_rows[row + f].val[mem_index])),
                                      "f"(dummy)
                                    : /* no clobbers */);
                        }

                        __threadfence();

                        // reg = reg + temp[c]
                        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                                i += threads_per_pim_grp) {
                            uint64_t mem_index = INDEX(chip, bank, col + i);

                            asm volatile ("st.cs.global.f32 [%0], %1;"
                                    : /* no outputs */
                                    : "l"(&(temp[c_row].val[mem_index])),
                                      "f"(dummy)
                                    : /* no clobbers */);
                        }

                        __threadfence();

                        // temp[c] = reg
                        for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                                i += threads_per_pim_grp) {
                            uint64_t mem_index = INDEX(chip, bank, col + i);

                            asm volatile ("st.cs.global.u16 [%0], %1;"
                                    : /* no outputs */
                                    : "l"(&(temp[c_row].val[mem_index])),
                                      "h"(store)
                                    : /* no clobbers */);
                        }

                        __threadfence();
                    }
                }

                // temp[c] = sqrt(temp[c])
                for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {
                    // reg = sqrt(temp[c])
                    for (int i = thread_id_in_grp; i < PIM_GRF_SIZE ;
                            i += threads_per_pim_grp) {
                        uint64_t mem_index = INDEX(chip, bank, col + i);

                        asm volatile ("st.cs.global.f32 [%0], %1;"
                                : /* no outputs */
                                : "l"(&(temp[c_row].val[mem_index])),
                                  "f"(dummy)
                                : /* no clobbers */);
                    }

                    __threadfence();

                    // temp[c] = reg
                    for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                            i += threads_per_pim_grp) {
                        uint64_t mem_index = INDEX(chip, bank, col + i);

                        asm volatile ("st.cs.global.u16 [%0], %1;"
                                : /* no outputs */
                                : "l"(&(temp[c_row].val[mem_index])),
                                  "h"(store)
                                : /* no clobbers */);
                    }

                    __threadfence();
                }
            }
        }

        // Find closest centroid to each datapoint
        //for (int p = thread_id; p < num_datapoints; p += num_threads) {
        //    float min_dist = INFINITY;
        //    int closest_centroid = 0;

        //    for(int c = 0; c < KMEANS_NUM_CLUSTERS; c++) {
        //        int row = (c * num_datapoint_rows) + \
        //                  (p / num_datapoints_per_row);
        //        int index = (p % num_datapoints_per_row) * 4;
        //        float dist = temp[row].val[index];

        //        if (dist < min_dist) {
        //            min_dist = dist;
        //            closest_centroid = c;
        //        }
        //    }

        //    //assign closest cluster id for this datapoint/thread
        //    cluster_assignment[p] = closest_centroid;
        //}
    }
}

__global__ void histogram(uint32_t *input, int num_elements, row_t *local_bins,
        int num_rows, uint32_t *bins, int num_threads)
{
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

    // Initialize local memory to store thread totals
    uint32_t thread_bins[HISTOGRAM_NUM_BINS];
    for (int i = 0; i < HISTOGRAM_NUM_BINS; i++) {
        thread_bins[i] = 0;
    }

    // Compute thread totals
    for (int i = thread_id; i < num_elements; i += num_threads) {
        thread_bins[input[i]]++;
    }

    __syncthreads();

    // Write block-wise totals to global memory
    int row_factor = ROW_SIZE * NUM_BANKS * NUM_CHIPS;
    int bin_row = (thread_id * HISTOGRAM_NUM_BINS * 4) / row_factor;
    int base_index = (thread_id * HISTOGRAM_NUM_BINS * 4) % row_factor;

    for (int i = 0; i < HISTOGRAM_NUM_BINS; i++) {
        local_bins[bin_row].val[base_index + (i * 4)] = thread_bins[i];
    }

    __syncthreads();

    // Sum values from all blocks
    int result_row = num_rows - 1;
    for (int row = 0; row < result_row; row++) {
        for (int bin = 0; bin < ((NUM_COLS * COL_SIZE) / 4);
                bin += HISTOGRAM_NUM_BINS) {
            for (int col = 0; col < HISTOGRAM_NUM_BINS; col += PIM_GRF_SIZE) {

                // reg = local_bin
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, bin + col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(local_bins[row].val[mem_index])),
                              "f"(dummy)
                            : /* no clobbers */);
                }

                __threadfence();

                // reg = reg + bin_total
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(local_bins[result_row].val[mem_index])),
                              "f"(dummy)
                            : /* no clobbers */);
                }

                __threadfence();

                // bin_total = reg
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, col + i);

                    asm volatile ("st.cs.global.u16 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(local_bins[result_row].val[mem_index])),
                              "h"(store)
                            : /* no clobbers */);
                }

                __threadfence();
            }
        }
    }

    if (thread_id < HISTOGRAM_NUM_BINS) {
        uint32_t total = 0;

        for (int c = 0; c < NUM_CHIPS; c++) {
            for (int b = 0; b < NUM_BANKS; b++) {
                for (int bin = 0; bin < ((NUM_COLS * COL_SIZE) / 4);
                        bin += HISTOGRAM_NUM_BINS) {
                    uint64_t mem_index = INDEX(c, b, bin + thread_id);
                    total += local_bins[result_row].val[mem_index];
                }
            }
        }

        bins[thread_id] = total;
    }
}

/*
 * Each row is the first element of all vectors within the batch.
 * This means that:
 * 1) num_rows = num elements in each vector
 * 2) row size = batch size
 */
__global__ void fully_connected(row_t *input, row_t *weights, row_t *output,
        int num_weight_rows, int num_weight_cols)
{
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

    // The number of input vector elements that can be stored in the SRF at
    // any time
    int num_inputs_in_srf = (PIM_SRF_SIZE * COL_SIZE) / 4;

    // Number of input vector elements, which is a function of how many
    // input vector elements can be stored in the SRF and the number of
    // elements
    int num_input_blocks = num_weight_rows / num_inputs_in_srf;

    // Global weights row counter
    int weight_row = 0;

    for (int input_block = 0; input_block < num_input_blocks;
            input_block++) {
        // Load the input vector into SRF
        for (int i = thread_id_in_grp; i < PIM_SRF_SIZE;
                i += threads_per_pim_grp) {
            uint64_t mem_index = INDEX(chip, bank,
                    (input_block * PIM_SRF_SIZE) + i);

            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(input->val[mem_index])),
                      "f"(dummy)
                    : /* no clobbers */);
        }

        __threadfence();

        for (int row = 0; row < num_inputs_in_srf; row++) {
            for (int col = 0; col < num_weight_cols; col += PIM_GRF_SIZE) {
                // reg = weights * input
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(weights[weight_row].val[mem_index])),
                              "f"(dummy)
                            : /* no clobbers */);
                }

                __threadfence();

                // reg += output
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(output->val[mem_index])),
                              "f"(dummy)
                            : /* no clobbers */);
                }

                __threadfence();

                // output = reg
                for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                        i += threads_per_pim_grp) {
                    uint64_t mem_index = INDEX(chip, bank, col + i);

                    asm volatile ("st.cs.global.f32 [%0], %1;"
                            : /* no outputs */
                            : "l"(&(output->val[mem_index])),
                              "f"(dummy)
                            : /* no clobbers */);
                }

                __threadfence();
            }

            weight_row++;
        }
    }
}

__global__ void grim(row_t *mem_rows, int num_rows)
{
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

    int mask_row = num_rows - 1;
    for (int row = 0; row < mask_row; row++) {
        for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {

            // reg = GRIM(input[i])
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg = result[i] OR mask
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[mask_row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            // mask[i] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.u16 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[mask_row].val[mem_index])), "h"(store)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}

__global__ void softmax(row_t *mem_rows, int num_rows)
{
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

    for (int row = 0; row < num_rows; row++) {

        for (int col = 0; col < NUM_COLS; col += (PIM_GRF_SIZE/2)) {
            // reg[0] = e^a[i]
            for (int i = thread_id_in_grp; i < (PIM_GRF_SIZE/2);
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // reg[1] += reg[0]
            for (int i = thread_id_in_grp; i < (PIM_GRF_SIZE/2);
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();
        }

        __syncthreads();

        if (thread_id_in_grp == 0) {
            // Accumulate reg[1]
            for (int col = 0; col < ((PIM_GRF_SIZE/2) - 1); col++) {
                uint64_t mem_index = INDEX(chip, bank, col);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // Reduce reg[1] to a single value
            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(mem_rows[row].val[INDEX(chip, bank, 0)])),
                      "f"(dummy)
                    : /* no clobbers */);

            __threadfence();

            // Store the reduced value in SRF
            asm volatile ("st.cs.global.f32 [%0], %1;"
                    : /* no outputs */
                    : "l"(&(mem_rows[row].val[INDEX(chip, bank, 0)])),
                      "f"(dummy)
                    : /* no clobbers */);

            __threadfence();
        }

        __syncthreads();

        for (int col = 0; col < NUM_COLS; col += PIM_GRF_SIZE) {
            // reg = a[i] / srf[0]
            for (int i = thread_id_in_grp; i < (PIM_GRF_SIZE/2);
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();

            // a[i] = reg
            for (int i = thread_id_in_grp; i < PIM_GRF_SIZE;
                    i += threads_per_pim_grp) {
                uint64_t mem_index = INDEX(chip, bank, col + i);

                asm volatile ("st.cs.global.f32 [%0], %1;"
                        : /* no outputs */
                        : "l"(&(mem_rows[row].val[mem_index])), "f"(dummy)
                        : /* no clobbers */);
            }

            __threadfence();
        }
    }
}
