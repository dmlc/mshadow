# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
include config.mk
include ../../make/mshadow.mk
export CFLAGS = -Wall -O3 -fopenmp -I../../ $(MSHADOW_CFLAGS)
export LDFLAGS= -lm $(MSHADOW_LDFLAGS)
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX) $(MSHADOW_NVCCFLAGS)

# specify tensor path
BIN = local_sum.cpu 
OBJ =
CUOBJ =
CUBIN = local_sum.gpu
.PHONY: clean all

all: $(BIN) $(CUBIN) 

local_sum.cpu: local_sum.cpp
local_sum.gpu: local_sum.cu

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~
