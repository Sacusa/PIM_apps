
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o $(PIM_DEPS)
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o $(PIM_DEPS) -I/usr/local/cuda/include -lm -arch=sm_70 --cudart shared -o heartwall

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3 -arch=sm_70

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
