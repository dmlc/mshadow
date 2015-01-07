/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_local-inl.h
 * \brief local multi-threading implementation of PS abstraction
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_DIST_INL_H_
#define MSHADOW_PS_DIST_INL_H_
#include "./ps_local-inl.h"

namespace mshadow {
namespace ps {
#if MSHADOW_DIST_PS_
template<typename xpu, typename DType>
class DistServer : public LocalServer<xpu, DType> {
 public:
  // parent type
  typedef LocalServer<xpu, DType> Parent;
  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    Parent::Init(devices);
  }
  virtual ~DistServer(void) {
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data, int key) {
    for (index_t i = 1; i < data.size(0); ++i) {
      data[0] += data[i];
    }    
    // something like
    //auto callback = [&]() {
    // receive data into dptr
    // call pullready to notify the module
    //this->PullReady(recvdata, key);
    //}
    // push(key, data[0].dptr_, data.MSize(), callback);
  }    
};
#endif
}  // namespace ps
}  // namespace msahdow
#endif

