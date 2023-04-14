include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

CC_FLAGS := -arch=sm_70 --cudart shared

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu $(PIM_DEPS)
	$(CC) $(CC_FLAGS) -cuda nn_cuda.cu $(PIM_DEPS)
	$(CC) $(CC_FLAGS) -o nn nn_cuda.cu $(PIM_DEPS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
