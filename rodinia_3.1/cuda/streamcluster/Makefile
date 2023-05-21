include ../../common/make.config

NVCC = $(CUDA_DIR)/bin/nvcc

NVCC_FLAGS = -I$(CUDA_DIR)/include -arch=sm_70 --cudart shared

TARGET_G = sc_gpu

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o


# make dbg=1 tells nvcc to add debugging symbols to the binary
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

DEPS = streamcluster_cuda_cpu.cpp streamcluster_cuda.cu streamcluster_header.cu $(PIM_DEPS)

$(TARGET_G): $(DEPS)
	$(NVCC) $(NVCC_FLAGS) $(DEPS) -o $(TARGET_G) -lcuda


clean:
	rm -f *.o *~ *.txt $(TARGET_G) *.linkinfo

