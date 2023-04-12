include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

CC_FLAGS=-I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch=sm_70 --cudart shared

INCLUDE := $(CUDA_DIR)/include

PIM_DIR = /u/sgupta45/PIM_apps

DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

SRC = bfs.cu

EXE = bfs

release: $(SRC) $(DEPS)
	$(CC) $(SRC) $(DEPS) -o $(EXE) $(CC_FLAGS)

enum: $(SRC)
	$(CC) -deviceemu $(SRC) $(DEPS) -o $(EXE) $(CC_FLAGS)

debug: $(SRC)
	$(CC) -g $(SRC) $(DEPS) -o $(EXE) $(CC_FLAGS)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) $(DEPS) -o $(EXE) $(CC_FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
