export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse2 -Wno-unknown-pragmas -funroll-loops
export LDFLAGS= -lpthread -lm 
export NVCCFLAGS = -O3 --maxrregcount=20 

# specify tensor path
BIN = test
OBJ = 
CUOBJ = 
CUBIN = testcuda
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

test: testcompile.cpp tensor/*.h
testcuda: testcuda.cu tensor/*.h tensor/cuda/*.cuh

$(BIN) :
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
