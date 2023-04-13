include ../../common/make.config

CUDA_SDK_PATH := $(SDK_DIR)

# Determine the correct version of the cutil library
CUTIL_LIB = # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif

NVCC_FLAGS=--gpu-architecture=compute_70 --gpu-code=compute_70 -I$(CUDA_SDK_PATH)/common/inc -L$(CUDA_SDK_PATH)/common/lib $(CUTIL_LIB) --cudart shared

PIM_DIR = /u/sgupta45/PIM_apps

PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double

euler3d: euler3d.cu $(PIM_DEPS)
	nvcc $(KERNEL_DIM) -O2 -Xptxas -v $(NVCC_FLAGS) euler3d.cu $(PIM_DEPS) -o euler3d

euler3d_double: euler3d_double.cu $(PIM_DEPS)
	nvcc -Xptxas -v -O3 $(NVCC_FLAGS) euler3d_double.cu $(PIM_DEPS) -o euler3d_double


pre_euler3d: pre_euler3d.cu $(PIM_DEPS)
	nvcc -Xptxas -v -O3 $(NVCC_FLAGS) pre_euler3d.cu $(PIM_DEPS) -o pre_euler3d

pre_euler3d_double: pre_euler3d_double.cu $(PIM_DEPS)
	nvcc -Xptxas -v -O3 $(NVCC_FLAGS) pre_euler3d_double.cu $(PIM_DEPS) -o pre_euler3d_double


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
