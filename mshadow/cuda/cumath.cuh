/*!
 * Copyright (c) 2018 by Contributors
 * \file cumath.cuh
 * \brief generic interface for cuda arithmetic and math intrinsics
 * \author Deokjae Lee
 */
#ifndef MSHADOW_CUDA_CUMATH_CUH_
#define MSHADOW_CUDA_CUMATH_CUH_

#ifdef __CUDACC__

namespace mshadow {
namespace cuda {

template<typename DType>
struct CuMath;

template<>
struct CuMath<float> {
  using DType = float;
  MSHADOW_FORCE_INLINE __device__ static DType Log(DType f) {
    return ::__logf(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Sqrt(DType f) {
    return ::__fsqrt_rn(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Cos(DType f) {
    return ::__cosf(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Sin(DType f) {
    return ::__sinf(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Abs(DType f) {
    return ::fabsf(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Fma(DType f1, DType f2, DType f3) {
    return ::__fmaf_rn(f1, f2, f3);
  }
  MSHADOW_FORCE_INLINE __device__ static DType FmaRD(DType f1, DType f2, DType f3) {
    return ::__fmaf_rd(f1, f2, f3);
  }
};

template<>
struct CuMath<double> {
  using DType = double;
  MSHADOW_FORCE_INLINE __device__ static DType Log(DType f) {
    return ::log(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Sqrt(DType f) {
    return __dsqrt_rn(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Cos(DType f) {
    return ::cos(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Sin(DType f) {
    return ::sin(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Abs(DType f) {
    return ::fabs(f);
  }
  MSHADOW_FORCE_INLINE __device__ static DType Fma(DType f1, DType f2, DType f3) {
    return ::__fma_rn(f1, f2, f3);
  }
  MSHADOW_FORCE_INLINE __device__ static DType FmaRD(DType f1, DType f2, DType f3) {
    return ::__fma_rd(f1, f2, f3);
  }
};

}  // namespace cuda
}  // namespace mshadow

#endif  // __CUDACC__

#endif  // MSHADOW_CUDA_CUMATH_CUH_