# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 
# specify tensor path
BIN = exp_lazy

.PHONY: clean all

all: $(BIN) 

exp_lazy: exp_lazy.cpp

$(BIN) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

clean:
	rm -rf $(BIN) *~
