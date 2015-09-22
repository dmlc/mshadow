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
#include "./logging.h"

namespace mshadow {
#if MSHADOW_USE_CUDA == 1
// Stream alocation
// actual implementation of GPU stream in CUDA
template<>
struct Stream<gpu> {
  /*! \brief handle state */
  enum HandleState {
    NoHandle = 0,
    OwnHandle = 1,
  };
  /*! \brief cudaStream */
  cudaStream_t stream_;
  /*! \brief cublas handle */
  cublasHandle_t blas_handle_;
  /*! \brief cudnn handle */
  #if MSHADOW_USE_CUDNN == 1
  cudnnHandle_t dnn_handle_;
  #endif
  /*! \brief cublas handle ownership */
  HandleState blas_handle_ownership_;
  /*! \brief cudnn handle ownership */
  HandleState dnn_handle_ownership_;

  Stream(void) : stream_(0),
                 blas_handle_ownership_(NoHandle),
                 dnn_handle_ownership_(NoHandle) {}
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {
    MSHADOW_CUDA_CALL(cudaStreamSynchronize(stream_));
  }
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    LOG(FATAL) << cudaGetErrorString(err);
    return false;
  }
  /*!
   * \brief returns actual cudaStream_t given an input GPU stream pointer
   * \param stream pointer to GPU stream
   */
  inline static cudaStream_t GetStream(Stream<gpu> *stream) {
    if (stream == NULL) {
#if MSHADOW_FORCE_STREAM
      LOG(FATAL) << "Default GPU stream was used when MSHADOW_FORCE_STREAM was on";
#endif
      return 0;
    } else {
      return stream->stream_;
    }
  }
  /*!
   * \brief return actual cublasHandle
   * \param pointer to GPU stream
   */
  inline static cublasHandle_t GetBlasHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->blas_handle_ownership_, NoHandle)
        << "No handle exist in source stream";
      return stream->blas_handle_;
    }
  }
  /*! \brief Destory cublas handle if own it */
  inline void DestoryBlasHandle() {
    if (blas_handle_ownership_ == OwnHandle) {
      cublasStatus_t err = cublasDestroy(blas_handle_);
      blas_handle_ownership_ = NoHandle;
      CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Destory cublas handle failed";
    }
  }
  /*! \brief Destory original blas handle and create a new one */
  inline void CreateBlasHandle() {
    this->DestoryBlasHandle();
    cublasStatus_t err = cublasCreate(&blas_handle_);
    blas_handle_ownership_ = OwnHandle;
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Create cublas handle failed";
  }
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
  inline static cudnnHandle_t GetDnnHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->dnn_handle_ownership_, NoHandle) << "No handle exist in source stream";
      return stream->dnn_handle_;
    }
  }
#endif
  inline void DestroyDnnHandle() {
// #if MSHADOW_USE_CUDNN && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    if (dnn_handle_ownership_ == OwnHandle) {
      cudnnStatus_t err = cudnnDestroy(dnn_handle_);
      CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    }
#endif
  }
  inline void CreateDnnHandle() {
// #if MSHADOW_USE_CUDNN == 1 && defined(__CUDACC__)
#if MSHADOW_USE_CUDNN == 1
    this->DestroyDnnHandle();
    cudnnStatus_t err = cudnnCreate(&dnn_handle_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    err = cudnnSetStream(dnn_handle_, stream_);
    CHECK_EQ(err, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(err);
    this->dnn_handle_ownership_ = OwnHandle;
#endif
  }
};
template<>
inline Stream<gpu> *NewStream<gpu>(bool create_blas_handle,
                                   bool create_dnn_handle) {
  Stream<gpu> *st = new Stream<gpu>();
  MSHADOW_CUDA_CALL(cudaStreamCreate(&st->stream_));
  if (create_blas_handle) {
    st->CreateBlasHandle();
  }
  if (create_dnn_handle) {
    st->CreateDnnHandle();
  }
  return st;
}
template<>
inline void DeleteStream<gpu>(Stream<gpu> *stream) {
  MSHADOW_CUDA_CALL(cudaStreamDestroy(stream->stream_));
  stream->DestoryBlasHandle();
  stream->DestroyDnnHandle();
  delete stream;
}
#endif
}  // namespace mshadow
#endif  // MSHADOW_STREAM_GPU_INL_H_
