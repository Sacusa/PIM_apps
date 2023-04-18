include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

CC_FLAGS := -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch=sm_70 --cudart shared

SRC = pathfinder.cu

EXE = pathfinder

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

release: $(SRC) $(PIM_DEPS)
	$(CC) $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

enum: $(SRC) $(PIM_DEPS)
	$(CC) -deviceemu $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

debug: $(SRC) $(PIM_DEPS)
	$(CC) -g $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

debugenum: $(SRC) $(PIM_DEPS)
	$(CC) -g -deviceemu $(SRC) $(PIM_DEPS) -o $(EXE) $(CC_FLAGS)

clean:
	rm -f pathfinder
