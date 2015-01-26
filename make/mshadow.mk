#---------------------------------------------------------------------------------------
#  mshadow configuration script
#
#  include mshadow.mk after the variables are set
#  
#  Add MSHADOW_CFLAGS to the compile flags
#  Add MSHADOW_LDFLAGS to the linker flags
#  Add MSHADOW_NVCCFLAGS to the nvcc compile flags
#----------------------------------------------------------------------------------------

MSHADOW_CFLAGS = -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas
MSHADOW_LDFLAGS = -lm 
MSHADOW_NVCCFLAGS = 

ifeq ($(USE_CUDA), 0)
	MSHADOW_CFLAGS += -DMSHADOW_USE_CUDA=0
else
	MSHADOW_LDFLAGS += -lcudart -lcublas -lcurand
endif
ifneq ($(USE_CUDA_PATH), NONE)
	MSHADOW_CFLAGS += -I$(USE_CUDA_PATH)/include
	MSHADOW_LDFLAGS += -L$(USE_CUDA_PATH)/lib64
endif

ifeq ($(USE_BLAS), mkl)
ifneq ($(USE_INTEL_PATH), NONE)
	MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/mkl/lib/intel64
	MSHADOW_LDFLAGS += -L$(USE_INTEL_PATH)/lib/intel64
	MSHADOW_CFLAGS += -I$(USE_INTEL_PATH)/mkl/include
endif
	MSHADOW_LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
else
	MSHADOW_CFLAGS += -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0
endif
ifeq ($(USE_BLAS), openblas)
	MSHADOW_LDFLAGS += -lopenblas
else ifeq ($(USE_BLAS), atlas)
	MSHADOW_LDFLAGS += -lcblas
else ifeq ($(USE_BLAS), blas)
	MSHADOW_LDFLAGS += -lblas
endif

ifeq ($(PS_PATH), NONE)
PS_PATH = ..
endif
ifeq ($(PS_THIRD_PATH), NONE)
PS_THIRD_PATH = $(PS_PATH)/third_party
endif

ifeq ($(USE_DIST_PS),1)
MSHADOW_CFLAGS += -DMSHADOW_DIST_PS=1 -std=c++11 \
	-I$(PS_PATH)/src -I$(PS_THIRD_PATH)/include
PS_LIB = $(addprefix $(PS_PATH)/build/, libps.a libpsmain.a) \
	$(addprefix $(PS_THIRD_PATH)/lib/, libgflags.a libzmq.a libprotobuf.a \
	libglog.a libz.a libsnappy.a)
MSHADOW_NVCCFLAGS += --std=c++11
else
	MSHADOW_CFLAGS+= -DMSHADOW_DIST_PS=0
endif
