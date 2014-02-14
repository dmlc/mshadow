export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops
export LDFLAGS= -L/usr/local/cuda-5.5/lib64 -I/usr/local/cuda-5.5/include -lpthread -lm -lcudart -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5
export NVCCFLAGS = -O3 --use_fast_math -ccbin g++

# specify tensor path
BIN = test
OBJ =
CUOBJ = testcuda.o
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

test: testcompile2.cpp mshadow/*.h testcuda.o
testcuda.o: testcuda.cu mshadow/*.h mshadow/cuda/*.cuh
testmkl.o: testmkl.cpp mshadow/*.h mshadow/cuda/*.cuh

$(BIN) :
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
