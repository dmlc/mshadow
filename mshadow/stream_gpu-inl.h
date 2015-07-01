/*!
 *  Copyright (c) 2014 by Contributors
 * \file stream_gpu-inl.h
 * \brief implementation of GPU code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_STREAM_GPU_INL_H_
#define MSHADOW_STREAM_GPU_INL_H_
#include "./base.h"
#include "./tensor.h"
#include "./utils.h"

namespace mshadow {
#if MSHADOW_USE_CUDA==1
// Stream alocation
// actual implementation of GPU stream in CUDA
template<>
struct Stream<gpu> {
  /*! \brief handle state */
  enum HandleState {
    NoHandle = 0,
    OwnHandle = 1,
    BorrowHandle = 2
  };
  /*! \brief cudaStream */
  cudaStream_t stream_;
  /*! \brief cublas handle */
  cublasHandle_t blas_handle_;
  /*!\brief cublas handle ownership */
  HandleState blas_handle_ownership_;

  Stream(void) : stream_(0), blas_handle_ownership_(NoHandle) {}
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {
    cudaError_t err = cudaStreamSynchronize(stream_);
    utils::Check(err == cudaSuccess, cudaGetErrorString(err));
  }
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    utils::Error(cudaGetErrorString(err));
    return false;
  }
  /*!
   * \brief returns actual cudaStream_t given an input GPU stream pointer
   * \param stream pointer to GPU stream
   */
  inline static cudaStream_t GetStream(Stream<gpu> *stream) {
    if (stream == NULL) {
#if MSHADOW_FORCE_STREAM
      utils::Error("Default GPU stream was used when MSHADOW_FORCE_STREAM was on");
#endif
      return 0;
    }
    else return stream->stream_;
  }
  /*!
   * \brief return actual cublasHandle
   * \param pointer to GPU stream
   */
  inline static cublasHandle_t GetBlasHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      utils::Check(stream->blas_handle_ownership_ != NoHandle,
                   "No handle exist in source stream");
      return stream->blas_handle_;
    }
  }
  /*! \brief Destory cublas handle if own it */
  inline void DestoryBlasHandle() {
    if (blas_handle_ownership_ == OwnHandle) {
      cublasStatus_t err = cublasDestroy(blas_handle_);
      blas_handle_ownership_ = NoHandle;
      utils::Check(err == CUBLAS_STATUS_SUCCESS, "Destory cublas handle failed");
    }
  }
  /*! \brief Set cublas handle from other stream,
             if there this stream originally own a handle, destory it,
             and borrow a handle from source stream
   *  \param pointer to GPU stream
   */
  inline void SetBlasHandle(Stream<gpu> *stream) {
    this->DestoryBlasHandle();
    utils::Check(stream->blas_handle_ownership_ != NoHandle,
                 "No handle exist in source stream");
    blas_handle_ = GetBlasHandle(stream);
    blas_handle_ownership_ = BorrowHandle;
  }
  /*! \brief Destory original blas handle and create a new one */
  inline void CreateBlasHandle() {
    this->DestoryBlasHandle();
    cublasStatus_t err = cublasCreate(&blas_handle_);
    utils::Check(err == CUBLAS_STATUS_SUCCESS, "Create cublas handle failed");
  }
};
template<>
inline Stream<gpu> *NewStream<gpu>(void) {
  Stream<gpu> *st = new Stream<gpu>();
  cudaError_t err = cudaStreamCreate(&st->stream_);
  utils::Check(err == cudaSuccess, cudaGetErrorString(err));
  return st;
}
template<>
inline void DeleteStream<gpu>(Stream<gpu> *stream) {
  cudaError_t err = cudaStreamDestroy(stream->stream_);
  utils::Check(err == cudaSuccess, cudaGetErrorString(err));
  stream->DestoryBlasHandle();
  delete stream;
}
#endif
}
#endif  // MSHADOW_STREAM_GPU_INL_H_
