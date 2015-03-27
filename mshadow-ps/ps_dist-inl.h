/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_local-inl.h
 * \brief local multi-threading implementation of PS abstraction
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_DIST_INL_H_
#define MSHADOW_PS_DIST_INL_H_
#include "./ps.h"
#include "./ps_local-inl.h"

#if MSHADOW_DIST_PS
#include "parameter/kv_layer.h"
namespace mshadow {
namespace ps {
template<typename xpu, typename DType>
class DistModel : public LocalModel<xpu, DType> {
 public:
  // parent type
  typedef LocalModel<xpu, DType> Parent;

  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    Parent::Init(devices);
    if (this->custom_server != NULL) {
      delete this->custom_server;
      this->custom_server = NULL;
    }
  }
  virtual ~DistModel(void) {
  }

 protected:
  // do nothing
  virtual void InitCustomerServer(void) {
  }
  virtual void ServerInitKey(Tensor<cpu, 2> weight, int key) {
    // this is called when key get initialized for the first time
    // weight can be used to hold the model that pulled back
    // use this to initialize the key on serverside
    shared_model_.Pull(
        PS::Parameter::Request(key), weight.dptr_, weight.MSize(),
        [this, weight, key]() {
          // call PullReady to notify LocalServer pulling is ready
          this->PullReady(weight, key);
        });
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // summation the data fron all devices
    this->ReduceSum(data);

    // push and pull
    Tensor<cpu, 2> sendrecv = data[0];
    utils::Assert(data[0].CheckContiguous(), "data must be contiguous");

    int ts = shared_model_.Push(
        PS::Parameter::Request(key), data[0].dptr_, data[0].MSize(), false);

    // let this pull request wait the push finish at the server node
    shared_model_.Pull(
        PS::Parameter::Request(key, -1, {ts}), data[0].dptr_, data[0].MSize(),
        [this, weight, key]() {
          // call PullReady to notify LocalServer pulling is ready
          this->PullReady(weight, key);
        });
  }

 private:
  PS::KVLayer<DType, IModelUpdater<DType>> shared_model_;
};

/**
 * @brief bridge IModelUpdater to KVLayerUpdater
 */

template<typename DType>
class UpdaterWrapper {
 public:
  UpdaterWrapper(IModelUpdater<DType> * updater)
      : updater_(updater) { }
  ~UpdaterWrapper() { delete updater_; }

  /// @brief initialize the data
  void Init(int id, size_t size, V* data) {
    updater->InitModel(id, data, size);
  }

  /// @brief update the model by using received data
  void Update(int id, size_t size, const V* recv_data, V* data) {
    updater->Update(id, recv_data, size);
  }
 private:
  IModelUpdater<DType> *updater_;
};

template<typename DType>
class MShadowServerNode : public PS::App {
 public:
  // conf: get from the flag -app_conf
  MShadowServerNode(const std::string &conf) : App() {
    IModelUpdater<DType> *updater = CreateModelUpdater<DType>();
    updater->InitUpdater(MyRank(), conf);

    UpdaterWrapper<DType> *wrapper = new UpdaterWrapper(updater);
    shared_model_.set_updater(wrapper);
  }
  virtual ~MShadowServerNode() { }
 private:
  PS::KVLayer<DType, UpdaterWrapper<DType> > shared_model_;
};

// NOTE: do not add PS::CreateServer here add it in the program that uses
// mshadow-ps

}  // namespace ps
}  // namespace msahdow
#endif
#endif
