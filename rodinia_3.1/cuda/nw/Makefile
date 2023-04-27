include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

CC_FLAGS := -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch=sm_70 --cudart shared

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

release: $(SRC) $(PIM_DEPS)
	$(CC) ${KERNEL_DIM} $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

enum: $(SRC) $(PIM_DEPS)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

debug: $(SRC) $(PIM_DEPS)
	$(CC) ${KERNEL_DIM} -g $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

debugenum: $(SRC) $(PIM_DEPS)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
