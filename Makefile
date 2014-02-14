# set LD_LIBRARY_PATH

export CC  = gcc
export CXX = g++
export NVCC =nvcc

export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops
export LDFLAGS= -L$(LD_LIBRARY_PATH) -lpthread -lm -lcudart 
export NVCCFLAGS = -O3 --use_fast_math -ccbin g++

# specify tensor path
BIN = test
OBJ =
CUOBJ = testcuda.o 
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

test: testcompile.cpp mshadow/*.h testcuda.o
testcuda.o: testcuda.cu mshadow/*.h mshadow/cuda/*.cuh
#testmkl.o: testmkl.cpp mshadow/*.h mshadow/cuda/*.cuh

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
