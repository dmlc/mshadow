# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../ -DMSHADOW_STAND_ALONE=1 -std=c++11 -fopenmp
export LDFLAGS= -lm -lpthread
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX)

# specify tensor path
BIN = test
OBJ =
CUOBJ =
CUBIN = 
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

test: test.cpp *.h

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
