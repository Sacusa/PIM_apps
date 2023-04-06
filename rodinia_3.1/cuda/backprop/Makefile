include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g -O2 -lstdc++

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -arch=sm_70 -I$(CUDA_DIR)/include

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

PIM_DIR = /u/sgupta45/PIM_apps

DEPS = backprop.o facetrain.o imagenet.o backprop_cuda.o $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

backprop: $(DEPS)
	$(CC) $(CC_FLAGS) $(DEPS) -o backprop -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c

backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
