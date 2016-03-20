/*!
 *  Copyright (c) 2014 by Contributors
 * \file base.h
 * \brief definitions of base types, operators, macros functions
 *
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_BASE_H_
#define MSHADOW_BASE_H_
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#define NOMINMAX
#endif
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <sstream>
#include <string>

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#include <inttypes.h>
#endif
// macro defintiions
/*!
 * \brief if this macro is define to be 1,
 * mshadow should compile without any of other libs
 */
#ifndef MSHADOW_STAND_ALONE
#define MSHADOW_STAND_ALONE 0
#endif
/*! \brief whether do padding during allocation */
#ifndef MSHADOW_ALLOC_PAD
#define MSHADOW_ALLOC_PAD true
#endif
/*!
 * \brief
 *  x dimension of data must be bigger pad_size * ratio to be alloced padded memory,
 *  otherwise use tide allocation
 *  for example, if pad_ratio=2, GPU memory alignement size is 32,
 *  then we will only allocate padded memory if x dimension > 64
 *  set it to 0 then we will always allocate padded memory
 */
#ifndef MSHADOW_MIN_PAD_RATIO
  #define MSHADOW_MIN_PAD_RATIO 2
#endif

#if MSHADOW_STAND_ALONE
  #define MSHADOW_USE_CBLAS 0
  #define MSHADOW_USE_MKL   0
  #define MSHADOW_USE_CUDA  0
#endif

/*!
 * \brief force user to use GPU stream during computation
 *  error will be shot when default stream NULL is used
 */
#ifndef MSHADOW_FORCE_STREAM
#define MSHADOW_FORCE_STREAM 1
#endif

/*! \brief use CBLAS for CBLAS */
#ifndef MSHADOW_USE_CBLAS
  #define MSHADOW_USE_CBLAS 0
#endif
/*! \brief use MKL for BLAS */
#ifndef MSHADOW_USE_MKL
  #define MSHADOW_USE_MKL   1
#endif

/*!
 * \brief use CUDA support, must ensure that the cuda include path is correct,
 * or directly compile using nvcc
 */
#ifndef MSHADOW_USE_CUDA
  #define MSHADOW_USE_CUDA   1
#endif

/*!
 * \brief use CUDNN support, must ensure that the cudnn include path is correct
 */
#ifndef MSHADOW_USE_CUDNN
  #define MSHADOW_USE_CUDNN 0
#endif

/*!
 * \brief seems CUDAARCH is deprecated in future NVCC
 * set this to 1 if you want to use CUDA version smaller than 2.0
 */
#ifndef MSHADOW_OLD_CUDA
#define MSHADOW_OLD_CUDA 0
#endif

/*!
 * \brief macro to decide existence of c++11 compiler
 */
#ifndef MSHADOW_IN_CXX11
#define MSHADOW_IN_CXX11 (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
                          __cplusplus >= 201103L || defined(_MSC_VER))
#endif

/*! \brief whether use SSE */
#ifndef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 1
#endif
/*! \brief whether use NVML to get dynamic info */
#ifndef MSHADOW_USE_NVML
  #define MSHADOW_USE_NVML 0
#endif
// SSE is conflict with cudacc
#ifdef __CUDACC__
  #undef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 0
#endif

#if MSHADOW_USE_CBLAS
extern "C" {
    #include <cblas.h>
}
#elif MSHADOW_USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
  #include <mkl_vsl.h>
  #include <mkl_vsl_functions.h>
#endif

#if MSHADOW_USE_CUDA
  #include <cuda.h>
  #include <cublas_v2.h>
  #include <curand.h>
#endif

#if MSHADOW_USE_CUDNN == 1
  #include <cudnn.h>
#endif

#if MSHADOW_USE_NVML
  #include <nvml.h>
#endif

// --------------------------------
// MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code
#ifdef MSHADOW_XINLINE
  #error "MSHADOW_XINLINE must not be defined"
#endif
#ifdef _MSC_VER
#define MSHADOW_FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define MSHADOW_FORCE_INLINE inline __attribute__((always_inline))
#endif
#ifdef __CUDACC__
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE __device__ __host__
#else
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE
#endif
/*! \brief cpu force inline */
#define MSHADOW_CINLINE MSHADOW_FORCE_INLINE

#if defined(__GXX_EXPERIMENTAL_CXX0X) ||\
    defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
  #define MSHADOW_CONSTEXPR constexpr
#else
  #define MSHADOW_CONSTEXPR const
#endif

/*!
 * \brief default data type for tensor string
 *  in code release, change it to default_real_t
 *  during development, change it to empty string so that missing
 *  template arguments can be detected
 */
#ifndef MSHADOW_DEFAULT_DTYPE
#define MSHADOW_DEFAULT_DTYPE = default_real_t
#endif

/*!
 * \brief DMLC marco for logging
 */
#ifndef MSHADOW_USE_GLOG
#define MSHADOW_USE_GLOG DMLC_USE_GLOG
#endif  // MSHADOW_USE_GLOG

/*!
 * \brief Protected cuda call in mshadow
 * \param func Expression to call.
 * It checks for CUDA errors after invocation of the expression.
 */
#define MSHADOW_CUDA_CALL(func)                                    \
  {                                                                \
    cudaError_t e = (func);                                        \
    if (e == cudaErrorCudartUnloading) {                           \
      throw dmlc::Error(cudaGetErrorString(e));                    \
    }                                                              \
    CHECK(e == cudaSuccess)                                        \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Run function and catch error, log unknown error.
 * \param func Expression to call.
 */
#define MSHADOW_CATCH_ERROR(func)                                     \
  {                                                                   \
    try {                                                             \
      (func);                                                         \
    } catch (const dmlc::Error &e) {                                    \
      std::string what = e.what();                                      \
      if (what.find("driver shutting down") == std::string::npos) {     \
        LOG(ERROR) << "Ignore CUDA Error " << what;                     \
      }                                                                 \
    }                                                                   \
  }

#include "./half.h"

/*! \brief namespace for mshadow */
namespace mshadow {
/*! \brief buffer size for each random number generator */
const unsigned kRandBufferSize = 1000000;
/*! \brief pi  */
const float kPi = 3.1415926f;
/*! \brief type that will be used for index */
typedef unsigned index_t;
/*! \brief float point type that will be used in default by mshadow */
typedef float default_real_t;

/*! \brief data type flag */
enum TypeFlag {
  kFloat32,
  kFloat64,
  kFloat16,
  kUint8,
  kInt32
};

template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = kFloat32;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1)
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_FLOAT;
  typedef float ScaleType;
#endif
};
template<>
struct DataType<double> {
  static const int kFlag = kFloat64;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1)
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_DOUBLE;
  typedef double ScaleType;
#endif
};
template<>
struct DataType<half::half_t> {
  static const int kFlag = kFloat16;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1)
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_HALF;
  typedef float ScaleType;
#endif
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = kUint8;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = kInt32;
};

/*! \brief type enum value for default real type */
const int default_type_flag = DataType<default_real_t>::kFlag;

/*! \brief namespace for operators */
namespace op {
// binary operator
/*! \brief mul operator */
struct mul{
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a * b;
  }
};
/*! \brief plus operator */
struct plus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a + b;
  }
};
/*! \brief minus operator */
struct minus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a - b;
  }
};
/*! \brief divide operator */
struct div {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a / b;
  }
};
/*! \brief get rhs */
struct right {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return b;
  }
};
// unary operator/ function: example
// these operators can be defined by user,
// in the same style as binary and unary operator
// to use, simply write F<op::identity>( src )
/*! \brief identity function that maps a real number to it self */
struct identity{
  /*! \brief map a to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a;
  }
};
}  // namespace op
/*! \brief namespace for savers */
namespace sv {
/*! \brief save to saver: = */
struct saveto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a = b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 0.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::right OPType;
};
/*! \brief save to saver: += */
struct plusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a += b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::plus OPType;
};
/*! \brief minus to saver: -= */
struct minusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a -= b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return -1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::minus OPType;
};
/*! \brief multiply to saver: *= */
struct multo {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a *= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::mul OPType;
};
/*! \brief divide to saver: /= */
struct divto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType& a, DType b) { // NOLINT(*)
    a /= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::div OPType;
};
}  // namespace sv
/*! \brief namespace for potential reducer operations */
namespace red {
namespace limits {
/*!
 * \brief minimum value of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType MinValue(void);
/*! \brief minimum value of float */
template<>
MSHADOW_XINLINE float MinValue<float>(void) {
  return -FLT_MAX;
}
/*! \brief minimum value of double */
template<>
MSHADOW_XINLINE double MinValue<double>(void) {
  return -DBL_MAX;
}
/*! \brief minimum value of half */
template<>
MSHADOW_XINLINE half::half_t MinValue<half::half_t>(void) {
  return MSHADOW_HALF_MIN;
}
/*! \brief minimum value of int */
template<>
MSHADOW_XINLINE int MinValue<int>(void) {
  return INT_MIN;
}
}  // namespace limits

/*! \brief sum reducer */
struct sum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    dst += src;
  }
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 0;
  }
};
/*! \brief maximum reducer */
struct maximum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    using namespace std;
#ifdef __CUDACC__
    dst = ::max(dst, src);
#else
    dst = max(dst, src);
#endif  // __CUDACC__
  }
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = limits::MinValue<DType>();
  }
};
/*! \brief minimum reducer */
struct minimum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    using namespace std;
#ifdef __CUDACC__
    dst = ::min(dst, src);
#else
    dst = min(dst, src);
#endif  // __CUDACC__
  }
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = -limits::MinValue<DType>();
  }
};
}  // namespace red
}  // namespace mshadow
#endif  // MSHADOW_BASE_H_
