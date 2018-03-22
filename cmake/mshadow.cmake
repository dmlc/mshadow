set(mshadow_LINKER_LIBS "")

set(BLAS "Open" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

if(USE_MKL_IF_AVAILABLE)
  if(NOT MKL_FOUND)
    find_package(MKL)
  endif()
  if(MKL_FOUND)
    if(USE_MKLML_MKL)
      set(BLAS "open")
    else()
      set(BLAS "MKL")
    endif()
  endif()
endif()

if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${Atlas_LIBRARIES})
  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)
elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${OpenBLAS_LIB})
  add_definitions(-DMSHADOW_USE_CBLAS=1)
  add_definitions(-DMSHADOW_USE_MKL=0)
elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${MKL_LIBRARIES})
  add_definitions(-DMSHADOW_USE_CBLAS=0)
  add_definitions(-DMSHADOW_USE_MKL=1)
elseif(BLAS STREQUAL "apple")
  find_package(Accelerate REQUIRED)
  include_directories(SYSTEM ${Accelerate_INCLUDE_DIR})
  list(APPEND mshadow_LINKER_LIBS ${Accelerate_LIBRARIES})
  add_definitions(-DMSHADOW_USE_MKL=0)
  add_definitions(-DMSHADOW_USE_CBLAS=1)
endif()

if(SUPPORT_MSSE2)
	add_definitions(-DMSHADOW_USE_SSE=1)
else()
	add_definitions(-DMSHADOW_USE_SSE=0)
endif()

if(SUPPORT_MF16C)
    add_definitions(-DMSHADOW_USE_F16C=1)
else()
    add_definitions(-DMSHADOW_USE_F16C=0)
endif()

if(USE_CUDA)
	find_package(CUDA 5.5 QUIET)
	find_cuda_helper_libs(curand)
	if(NOT CUDA_FOUND)
		message(FATAL_ERROR "-- CUDA is disabled.")
	endif()
	add_definitions(-DMSHADOW_USE_CUDA=1)
	add_definitions(-DMSHADOW_FORCE_STREAM)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND mshadow_LINKER_LIBS ${CUDA_CUDART_LIBRARY}
                              ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
else()
  add_definitions(-DMSHADOW_USE_CUDA=0)
endif()
