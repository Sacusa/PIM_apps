#ifndef _FIND_ELLIPSE_KERNEL_H_
#define _FIND_ELLIPSE_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
extern float *GICOV_CUDA(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y, cudaStream_t stream);
extern float *dilate_CUDA(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n, cudaStream_t stream);
extern void select_device();
extern void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel);
#ifdef __cplusplus
}
#endif


#endif
