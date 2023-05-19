include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.o $(PIM_DEPS)
	$(CC)	main.o \
				$(PIM_DEPS) \
				-I$(INCLUDE) \
				-L$(CUDA_LIB_DIR)  \
				-arch=sm_70 -lm -lcuda -lcudart -o srad

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	nvcc	main.cu \
				-c -O3 -arch sm_70

# delete all object files
clean:
	rm *.o srad
