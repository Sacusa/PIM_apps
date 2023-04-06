include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

CC_FLAGS=-I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch=sm_70 --cudart shared

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

PIM_DIR = /u/sgupta45/PIM_apps

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CC_FLAGS) $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

clean:
	rm gaussian
