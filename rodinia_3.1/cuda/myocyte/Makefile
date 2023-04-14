# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

PIM_DIR = /u/sgupta45/PIM_apps
PIM_DEPS = $(PIM_DIR)/common.o $(PIM_DIR)/STREAM/pim_stream.o

# link objects(binaries) together
myocyte.out:		main.o $(PIM_DEPS)
	nvcc	main.o \
				$(PIM_DEPS) \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm -lcuda -lcudart \
				-arch=sm_70 --cudart shared \
                                -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	nvcc	main.cu \
				-arch=sm_70 --cudart shared -c -O3 -g

# delete all object files
clean:
	rm *.o *.out output.txt
