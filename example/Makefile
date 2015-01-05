# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = clang++
export NVCC =nvcc
export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../ -DMSHADOW_STAND_ALONE=1
export LDFLAGS= -lm 
export NVCCFLAGS = -O3 --use_fast_math -ccbin $(CXX)

# specify tensor path
BIN = basic defop concat
OBJ =
CUOBJ =
CUBIN = basicx
.PHONY: clean all

all: $(BIN) $(OBJ) $(CUBIN) $(CUOBJ)

basic: basic.cpp
concat: concat.cpp
defop: defop.cpp
basicx: basic.cu

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
