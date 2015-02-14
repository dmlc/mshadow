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
#include "./kv_array.h"
#include "system/app.h"
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
    shared_model_ = new PS::KVArray<DType>();
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
    using namespace PS;
    MessagePtr pull_msg(new Message(kServerGroup));
    pull_msg->task.set_key_channel(key);
    Range<Key>(0, weight.MSize()).to(pull_msg->task.mutable_key_range());
    shared_model_->setArray(key, weight.dptr_, weight.MSize());
    pull_msg->fin_handle = [this, weight, key]() {
      // call PullReady to notify LocalServer pulling is ready
      this->PullReady(weight, key);
    };
    shared_model_->pull(pull_msg);
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // here we only use sum reduction, can change to others
    for (index_t i = 1; i < data.size(0); ++i) {
      data[0] += data[i];
    }

    // push
    Tensor<cpu, 2> sendrecv = data[0];
    using namespace PS;
    utils::Assert(data[0].CheckContiguous(), "data must be contiguous");
    SArray<DType> val; val.copyFrom(sendrecv.dptr_, sendrecv.MSize());
    MessagePtr push_msg(new Message(kServerGroup));
    push_msg->addValue(val);
    // LL << val;
    push_msg->task.set_key_channel(key);
    Range<Key>(0, val.size()).to(push_msg->task.mutable_key_range());
    int push_time = CHECK_NOTNULL(shared_model_)->push(push_msg);

    // pull
    MessagePtr pull_msg(new Message(kServerGroup, -1, push_time));
    pull_msg->task.set_key_channel(key);
    Range<Key>(0, sendrecv.MSize()).to(pull_msg->task.mutable_key_range());
    shared_model_->setArray(key, sendrecv.dptr_, sendrecv.MSize());
    pull_msg->fin_handle = [this, sendrecv, key]() {
      // call PullReady to notify LocalServer pulling is ready
      this->PullReady(sendrecv, key);
    };
    shared_model_->pull(pull_msg);
  }

 private:
  PS::KVArray<DType>* shared_model_ = nullptr;
};

template<typename DType>
class MShadowServerNode : public PS::App {
 public:
  // conf: get from the flag -app_conf
  MShadowServerNode(const std::string &conf) : App() {
    updater_ = CreateModelUpdater<DType>();

    updater_->InitUpdater(myRank(), conf);
    shared_model_ = new PS::KVArray<DType>();
    shared_model_->setUpdater(updater_);
  }
  virtual ~MShadowServerNode() {
    delete updater_;
    delete shared_model_;
  }
 private:
  IModelUpdater<DType> *updater_;
  PS::KVArray<DType>* shared_model_;
};

// NOTE: do not add PS::CreateServer here add it in the program that uses
// mshadow-ps

}  // namespace ps
}  // namespace msahdow
#endif
#endif
