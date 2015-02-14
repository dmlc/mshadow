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
  /*! \brief cudaStream */
  cudaStream_t stream_;
  Stream(void) : stream_(0) {}
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
  delete stream;
}
#endif 
}
#endif  // MSHADOW_STREAM_GPU_INL_H_
