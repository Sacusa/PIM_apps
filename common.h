#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda_runtime.h>
#include <stdint.h>

/**********************
 * DRAM configuration *
 **********************/

// Address mapping for QV100 from GPGPU-Sim:
// addr_dec_mask[CHIP]  = 0000000000001f00
// addr_dec_mask[BK]    = 00000000000e2000
// addr_dec_mask[ROW]   = 00000001fff00000
// addr_dec_mask[COL]   = 000000000001c0e0
// addr_dec_mask[BURST] = 000000000000001f

#define NUM_CHIPS 32
#define NUM_BANKS 16
#define NUM_ROWS  8192
#define NUM_COLS  64

#define INDEX(chip,bank,col) ((((chip) & 0x1f) << 8) + \
                              ((((bank) & 0xf) >> 1) << 17) + \
                              (((bank) & 0x1) << 13) + \
                              ((((col) & 0x3f) >> 3) << 14) + \
                              (((col) & 0x7) << 5))

#define ROW_SIZE 2048  // in bytes
#define COL_SIZE 32    // in bytes
#define ROW_OFFSET (1ULL<<20)

struct row_t {
    uint8_t val[ROW_OFFSET];
};
typedef struct row_t row_t;

/*********************
 * PIM configuration *
 *********************/

#define NUM_PIM_UNITS (NUM_BANKS/2)
#define PIM_RF_SIZE 8  // PIM register size in number of cols
                       // Column size is 32 bytes

enum pim_kernel_t {
    NOP,    // This allows template code to just not launch a kernel
    STREAM_ADD,
    STREAM_COPY,
    STREAM_DAXPY,
    STREAM_SCALE,
    STREAM_TRIAD,
    BN_FWD,
    BN_BWD,
    KMEANS,
    HISTOGRAM,
    FULLY_CONNECTED,
    GRIM
};
typedef enum pim_kernel_t pim_kernel_t;

struct pim_state_t {
    pim_kernel_t kernel;
    row_t *rows;
    int num_rows;

    // array of pointers for additional arguments
    void **args;
    int num_args;

    // Kmeans arguments
    // Other benchmarks reuse them too
    int num_datapoints;
    int num_features;
    int num_iters;
};
typedef struct pim_state_t pim_state_t;

/****************************
 * GPU shader configuration *
 ****************************/

#define NUM_THREADS_PER_WARP 32

/*************************
 * Function declarations *
 *************************/

#ifdef __cplusplus
extern "C" pim_state_t *init_pim(pim_kernel_t, int, int);
extern "C" void launch_pim(pim_state_t*, cudaStream_t);
extern "C" void free_pim(pim_state_t*);
#else
pim_state_t *init_pim(pim_kernel_t, int, int);
void launch_pim(pim_state_t*, cudaStream_t);
void free_pim(pim_state_t*);
#endif

#endif
