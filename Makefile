export CC  = clang
export CXX = clang
export CFLAGS = -Wall -O3 -msse2

# specify tensor path
BIN = test
OBJ =
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm -Wunknown-pragmas
test: testcompile.cpp tensor/*.h

$(BIN) :
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
