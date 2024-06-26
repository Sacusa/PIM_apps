BIN = main
DEPS = ../common.o pim_real.o pim_stream.o

CC=nvcc
CCFLAGS=-ccbin g++ -m64 -lcudart --std=c++11 -O3 -gencode arch=compute_70,code=compute_70

.SECONDARY: $(DEPS)

all: $(BIN)

%.o: %.cu
	$(CC) $(CCFLAGS) -c -o $@ $<

%: %.cu $(DEPS)
	$(CC) $(CCFLAGS) -o $@ $< $(DEPS)

clean:
	rm -f ../*.o *.o $(BIN)
