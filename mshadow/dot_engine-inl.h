/*!
 *  Copyright (c) 2014 by Contributors
 * \file dot_engine-inl.h
 * \brief definitions of how Matrix Multiplications can be evaluated
 * \author Tianqi Chen
 */
#ifndef MSHADOW_DOT_ENGINE_INL_H_
#define MSHADOW_DOT_ENGINE_INL_H_

#include "./base.h"
#include "./extension/implicit_gemm.h"

namespace mshadow {
namespace expr {
//---------------------------------------------------------------------
// Matrix Multiplications, depends on BLAS Engine
//---------------------------------------------------------------------
template<typename SV, typename Device, int ddim, int ldim,
         int rdim, bool ltrans, bool rtrans, typename DType>
struct DotEngine {
  inline static void Eval(Tensor<Device, ddim, DType> *p_dst,
                          const Tensor<Device, ldim, DType> &lhs,
                          const Tensor<Device, rdim, DType> &rhs,
                          DType scale);
};
// handles the dot, use CblasColMajor
template<typename Device, typename DType = default_real_t>
struct BLASEngine {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<Device> *stream) {
  }
  inline static void gemm(Stream<Device> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, DType alpha,
                          const DType *A, int lda, const DType *B, int ldb,
                          DType beta, DType *C, int ldc) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemm(Stream<Device> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, DType alpha,
                                  const DType *A, int lda, const DType *B, int ldb,
                                  DType beta, DType *C, int ldc, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void gemv(Stream<Device> *stream,
                          bool trans, int m, int n,
                          DType alpha, const DType *A, int lda,
                          const DType *X, int incX,
                          DType beta, DType *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<Device> *stream,
                                  bool trans, int m, int n,
                                  DType alpha, const DType *A, int lda,
                                  const DType *X, int incX,
                                  DType beta, DType *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<Device> *stream,
                         int m, int n, DType alpha,
                         const DType *X, int incX,
                         const DType *Y, int incY, DType *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<Device> *stream,
                         int n,
                         const DType* X, int incX,
                         const DType* Y, int incY,
                         DType* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#if MSHADOW_STAND_ALONE
template<>
struct BLASEngine<cpu, float> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, float> lhs((float*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, float> rhs((float*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, float> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, DType alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static bool GetT(bool t) {
    return t ? true : false;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    if (alpha == 1.0f && beta == 0.0f) {
      bool transpose_left = transb;
      bool transpose_right = transa;
      Tensor<cpu, 2, double> lhs((double*)B, Shape2(transpose_left ? k : n, transpose_left ? n : k));  // NOLINT(*)
      Tensor<cpu, 2, double> rhs((double*)A, Shape2(transpose_right ? m : k, transpose_right ? k : m));  // NOLINT(*)
      Tensor<cpu, 2, double> dst(C, Shape2(m, n));
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      } else {
        LOG(FATAL) << "Not implmented!";
      }
    } else {
      LOG(FATAL) << "Not implmented!";
    }
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, DType alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          double alpha, const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

#elif (MSHADOW_USE_MKL || MSHADOW_USE_CBLAS)  // NOLINT(*)
template<>
struct BLASEngine<cpu, float> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n,
                          float alpha, const float *A, int lda,
                          const float *X, int incX,
                          float beta, float *Y, int incY) {
    cblas_sgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cblas_sger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float* ret) {
    *ret = cblas_sdot(n, X, incX, Y, incY);
  }
};

template<>
struct BLASEngine<cpu, double> {
  inline static CBLAS_TRANSPOSE GetT(bool t) {
    return t ? CblasTrans : CblasNoTrans;
  }
  inline static void SetStream(Stream<cpu> *stream) {
  }
  inline static void gemm(Stream<cpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline static void batched_gemm(Stream<cpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<cpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cblas_dgemv(CblasColMajor, GetT(trans), m, n, alpha,
                A, lda, X, incX, beta, Y, incY);
  }
  inline static void batched_gemv(Stream<cpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cblas_dger(CblasColMajor, m, n, alpha, X, incX, Y, incY, A, lda);
  }
  inline static void batched_ger(Stream<cpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<cpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double* ret) {
    *ret = cblas_ddot(n, X, incX, Y, incY);
  }
};
#endif  // MSHADOW_USE_CBLAS || MSHADOW_USE_MKL || MSHADOW_STAND_ALONE
// CuBLAS redirect code
#if MSHADOW_USE_CUDA
// All CuBLAS goes to here, use legacy API: not threadsafe
template<>
struct BLASEngine<gpu, half::half_t> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, half::half_t alpha,
                          const half::half_t *A, int lda,
                          const half::half_t *B, int ldb, half::half_t beta,
                          half::half_t *C, int ldc) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 7050
#if MSHADOW_USE_PASCAL == 1
    cublasStatus_t err = cublasHgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha.cuhalf_,
                A, lda, B, ldb, &beta.cuhalf_, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas Hgemm fail";
#else
    float alpha_f = float(alpha);  // NOLINT(*)
    float beta_f = float(beta);  // NOLINT(*)
#if CUDA_VERSION >= 8000
    cublasStatus_t err = cublasSgemmEx(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha_f,
                A, CUDA_R_16F, lda, B, CUDA_R_16F,
                ldb, &beta_f, C, CUDA_R_16F, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas SgemmEx fail";
#else
    cublasStatus_t err = cublasSgemmEx(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha_f,
                A, CUBLAS_DATA_HALF, lda, B, CUBLAS_DATA_HALF,
                ldb, &beta_f, C, CUBLAS_DATA_HALF, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas SgemmEx fail";
#endif  // CUDA_VERSION >= 8000
#endif  // MSHADOW_USE_PASCAL == 1
#else
    LOG(FATAL) << "Require CUDA version >= 7.5!";
#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 7050
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, half::half_t alpha,
                                  const half::half_t *A, int lda, const half::half_t *B, int ldb,
                                  half::half_t beta, half::half_t *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, half::half_t alpha,
                          const half::half_t *A, int lda,
                          const half::half_t *X, int incX, half::half_t beta,
                          half::half_t *Y, int incY) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  half::half_t alpha, const half::half_t *A, int lda,
                                  const half::half_t *X, int incX,
                                  half::half_t beta, half::half_t *Y, int incY, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, half::half_t alpha,
                         const half::half_t *X, int incX,
                         const half::half_t *Y, int incY, half::half_t *A, int lda) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, half::half_t alpha,
                         const half::half_t *X, int incX, const half::half_t *Y, int incY,
                         half::half_t *A, int lda, int batch_count) {
    LOG(FATAL) << "Not implmented!";
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const half::half_t* X, int incX,
                         const half::half_t* Y, int incY,
                         half::half_t *ret) {
    LOG(FATAL) << "Not implmented!";
  }
};

template<>
struct BLASEngine<gpu, float> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, float alpha,
                          const float *A, int lda,
                          const float *B, int ldb, float beta,
                          float *C, int ldc) {
    cublasStatus_t err = cublasSgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sgemm fail";
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb,
                                  float beta, float *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, float alpha,
                          const float *A, int lda,
                          const float *X, int incX, float beta,
                          float *Y, int incY) {
    cublasStatus_t err = cublasSgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sgemv fail";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  float alpha, const float *A, int lda,
                                  const float *X, int incX,
                                  float beta, float *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda) {
    cublasStatus_t err = cublasSger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Sger fail";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, float alpha,
                         const float *X, int incX,
                         const float *Y, int incY, float *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const float* X, int incX,
                         const float* Y, int incY,
                         float *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasSdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};

template<>
struct BLASEngine<gpu, double> {
  inline static cublasOperation_t GetT(bool t) {
    return t ? CUBLAS_OP_T : CUBLAS_OP_N;
  }
  inline static void SetStream(Stream<gpu> *stream) {
    cublasStatus_t err = cublasSetStream(Stream<gpu>::GetBlasHandle(stream),
                    Stream<gpu>::GetStream(stream));
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: set stream fail";
  }
  inline static void gemm(Stream<gpu> *stream,
                          bool transa, bool transb,
                          int m, int n, int k, double alpha,
                          const double *A, int lda,
                          const double *B, int ldb,
                          double beta, double *C, int ldc) {
    cublasStatus_t err = cublasDgemm(Stream<gpu>::GetBlasHandle(stream),
                GetT(transa), GetT(transb), m, n, k, &alpha,
                A, lda, B, ldb, &beta, C, ldc);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemm fail";
  }
  inline static void batched_gemm(Stream<gpu> *stream,
                                  bool transa, bool transb,
                                  int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  double beta, double *C, int ldc, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemm(stream, transa, transb, m, n, k, alpha,
           A + i * m * k, lda, B + i * k * n, ldb,
           beta, C + i * m * n, ldc);
    }
  }
  inline static void gemv(Stream<gpu> *stream,
                          bool trans, int m, int n, double alpha,
                          const double *A, int lda,
                          const double *X, int incX,
                          double beta, double *Y, int incY) {
    cublasStatus_t err = cublasDgemv(Stream<gpu>::GetBlasHandle(stream),
                GetT(trans), m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dgemv fail";
  }
  inline static void batched_gemv(Stream<gpu> *stream,
                                  bool trans, int m, int n,
                                  double alpha, const double *A, int lda,
                                  const double *X, int incX,
                                  double beta, double *Y, int incY, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      gemv(stream, trans, m, n, alpha, A + i * m * n, lda,
           X + i * (trans ? m : n) * incX, incX,
           beta, Y + i * (trans ? n : m) * incY, incY);
    }
  }
  inline static void ger(Stream<gpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda) {
    cublasStatus_t err = cublasDger(Stream<gpu>::GetBlasHandle(stream),
                                    m, n, &alpha, X, incX, Y, incY, A, lda);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dger fail";
  }
  inline static void batched_ger(Stream<gpu> *stream,
                         int m, int n, double alpha,
                         const double *X, int incX,
                         const double *Y, int incY, double *A, int lda, int batch_count) {
    for (int i = 0; i < batch_count; ++i) {
      ger(stream, m, n, alpha, X + i * m * incX, incX, Y + i * n * incY, incY,
          A + i * lda * n, lda);
    }
  }
  inline static void dot(Stream<gpu> *stream,
                         int n,
                         const double* X, int incX,
                         const double* Y, int incY,
                         double *ret) {
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t err = cublasDdot(Stream<gpu>::GetBlasHandle(stream),
                                    n, X, incX, Y, incY, ret);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas: Dot fail";
    cublasSetPointerMode(Stream<gpu>::GetBlasHandle(stream),
                         CUBLAS_POINTER_MODE_HOST);
  }
};
#endif  // MSHADOW_USE_CUDA
// helper function to decide which shape we are in
inline static Shape<2> GetShape(const Shape<2> &shape, bool transpose) {
  return transpose ? Shape2(shape[1], shape[0]) : shape;
}
inline static Shape<3> GetBatchedShape(const Shape<3> &shape, bool transpose) {
  return transpose ? Shape3(shape[0], shape[2], shape[1]) : shape;
}
// dst = dot(lhs[.T], rhs[.T])
template<typename SV, typename xpu,
         bool transpose_left, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 2, 2, 2, transpose_left, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 2, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
#if MSHADOW_STAND_ALONE
    if (xpu::kDevMask == cpu::kDevMask && scale == 1.0f) {
      if (!transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs, rhs); return;
      } else if (!transpose_left && transpose_right) {
        dst = expr::implicit_dot(lhs, rhs.T()); return;
      } else if (transpose_left && !transpose_right) {
        dst = expr::implicit_dot(lhs.T(), rhs); return;
      }
    }
#endif
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sleft = GetShape(lhs.shape_, transpose_left);
    Shape<2> sright = GetShape(rhs.shape_, transpose_right);
    CHECK(dst.size(0) == sleft[0] && dst.size(1) == sright[1] && sleft[1] == sright[0])
      << "dot-gemm: matrix shape mismatch";
    // use column major argument to compatible with most BLAS
    BLASEngine<xpu, DType>::gemm
        (dst.stream_,
         transpose_right , transpose_left,
         transpose_right ? rhs.size(0) : rhs.size(1),
         transpose_left  ? lhs.size(1) : lhs.size(0),
         transpose_right ? rhs.size(1) : rhs.size(0),
         DType(scale * SV::AlphaBLAS()),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, lhs.stride_,
         DType(SV::BetaBLAS()),
         dst.dptr_, dst.stride_);
  }
};
template<typename SV, typename xpu, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 1, 1, 2, false, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 1, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 2, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 1, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<2> sright = GetShape(rhs.shape, transpose_right);
    CHECK(dst.size(0) == sright[1] && lhs.size(0) == sright[0])
      << "dot-gemv: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << sright << "\n";
    BLASEngine<xpu, DType>::gemv
        (dst.stream_,
         transpose_right,
         rhs.size(1), rhs.size(0), scale * SV::AlphaBLAS(),
         rhs.dptr_, rhs.stride_,
         lhs.dptr_, 1, SV::BetaBLAS(),
         dst.dptr_, 1);
  }
};
template<typename SV, typename xpu, typename DType>
struct DotEngine<SV, xpu, 2, 1, 1, true, false, DType> {
  inline static void Eval(Tensor<xpu, 2, DType> *p_dst,
                          const Tensor<xpu, 1, DType> &lhs,
                          const Tensor<xpu, 1, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 2, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    CHECK(dst.size(0) == lhs.size(0) && dst.size(1) == rhs.size(0))
      << "dot-ger: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << lhs.shape_ << "\n"
      << "rhs: " << rhs.shape_;
    if (SV::BetaBLAS() == 0.0f) {
      BLASEngine<xpu, DType>::ger
          (dst.stream_, rhs.size(0), lhs.size(0), scale * SV::AlphaBLAS(),
           rhs.dptr_, 1, lhs.dptr_, 1, dst.dptr_, dst.stride_);
    } else {
      DotEngine<SV, xpu, 2, 2, 2, true, false,
                DType>::Eval(dst, lhs.FlatTo2D(), rhs.FlatTo2D(), scale);
    }
  }
};
// dst = batched_dot(lhs[.T], rhs[.T])
template<typename SV, typename xpu,
  bool transpose_left, bool transpose_right, typename DType>
struct DotEngine<SV, xpu, 3, 3, 3, transpose_left, transpose_right, DType> {
  inline static void Eval(Tensor<xpu, 3, DType> *p_dst,
                          const Tensor<xpu, 3, DType> &lhs,
                          const Tensor<xpu, 3, DType> &rhs,
                          DType scale) {
    Tensor<xpu, 3, DType> &dst = *p_dst;
    // set kernel stream
    // if there is no stream, crush
    BLASEngine<xpu, DType>::SetStream(dst.stream_);
    Shape<3> sleft = GetBatchedShape(lhs.shape_, transpose_left);
    Shape<3> sright = GetBatchedShape(rhs.shape_, transpose_right);
    CHECK(dst.size(0) == sleft[0] && dst.size(0) == sright[0])
      << "batch_dot-gemm: batchsize must be equal."
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << sleft << "\n"
      << "rhs: " << sright << "\n";
    CHECK(dst.size(1) == sleft[1] && dst.size(2) == sright[2] && sleft[2] == sright[1])
      << "batch_dot-gemm: matrix shape mismatch"
      << "dst: " << dst.shape_ << "\n"
      << "lhs: " << sleft << "\n"
      << "rhs: " << sright << "\n";
    // use column major argument to compatible with most BLAS
    if (sleft[1] == 1) {
      // For (batch, 1, K) gemm (batch, K, N), we can use (batch, N, K) gemv (batch, K)
      BLASEngine<xpu, DType>::batched_gemv
        (dst.stream_,
        transpose_right,
        rhs.size(2), rhs.size(1), scale * SV::AlphaBLAS(),
        rhs.dptr_, rhs.stride_,
        lhs.dptr_, 1, SV::BetaBLAS(),
        dst.dptr_, 1, dst.size(0));
    } else if (sleft[2] == 1 && (SV::BetaBLAS() == 0.0f || SV::BetaBLAS() == 1.0f)) {
      // For (batch, M, 1) gemm (batch, 1, N) + Beta = 0, we can use (batch, M) ger (batch, N)
      if (SV::BetaBLAS() == 0.0f) {
        dst = DType(0);
      }
      BLASEngine<xpu, DType>::batched_ger
        (dst.stream_, sright[2], sleft[1], scale * SV::AlphaBLAS(),
        rhs.dptr_, 1, lhs.dptr_, 1, dst.dptr_, dst.stride_, dst.size(0));
    } else if (sright[2] == 1) {
      // For (batch, M, K) gemm (batch, K, 1), we can use (batch, M, K) gemv (batch, K)
      BLASEngine<xpu, DType>::batched_gemv
        (dst.stream_,
        !transpose_left,
        lhs.size(2), lhs.size(1), scale * SV::AlphaBLAS(),
        lhs.dptr_, lhs.stride_,
        rhs.dptr_, 1, SV::BetaBLAS(),
        dst.dptr_, 1, dst.size(0));
    } else {
      // For general case, use gemm
      BLASEngine<xpu, DType>::batched_gemm
        (dst.stream_,
        transpose_right, transpose_left,
        transpose_right ? rhs.size(1) : rhs.size(2),
        transpose_left ? lhs.size(2) : lhs.size(1),
        transpose_right ? rhs.size(2) : rhs.size(1),
        DType(scale * SV::AlphaBLAS()),
        rhs.dptr_, rhs.stride_,
        lhs.dptr_, lhs.stride_,
        DType(SV::BetaBLAS()),
        dst.dptr_, dst.stride_, dst.size(0));
    }
  }
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_DOT_ENGINE_INL_H_
