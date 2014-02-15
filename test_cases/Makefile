# set LD_LIBRARY_PATH

export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CUDA_INCLUDE = /usr/local/cuda-5.5/include
export CUDA_LIBS = /usr/local/cuda-5.5/lib64
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops
export LDFLAGS= -L$(LD_LIBRARY_PATH) -L$(CUDA_LIBS) -I$(CUDA_INCLUDE) -lpthread -lm -lcudart -lcublas -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX)

# specify tensor path
BIN = test
OBJ =
CUOBJ = testcuda.o
CUBIN =
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

test: mkl_sgemm.cpp ../mshadow/*.h testcuda.o
testcuda.o: testcuda.cu ../mshadow/*.h ../mshadow/cuda/*.cuh
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
