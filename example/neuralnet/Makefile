# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../../


ifeq ($(blas),1)
	LDFLAGS= -lcblas -lm -lcudart -lcublas -lcurand 
	CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1
else
	LDFLAGS=  -lm -lcudart -lcublas -lcurand  -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread 
endif
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX)

# specify tensor path
BIN =
OBJ =
CUOBJ =
CUBIN = nnet convnet
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

nnet: nnet.cu
convnet: convnet.cu

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
