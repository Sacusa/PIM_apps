#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER HEADER
//========================================================================================================================================================================================================200

void
kernel_gpu_cuda_wrapper_2(	knode *knodes,
							long knodes_elem,
							long knodes_mem,

							int order,
							long maxheight,
							int count,

							long *currKnode,
							long *offset,
							long *lastKnode,
							long *offset_2,
							int *start,
							int *end,
							int *recstart,
							int *reclength,

                            cudaStream_t stream);

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif