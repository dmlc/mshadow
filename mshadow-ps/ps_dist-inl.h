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
#include "./kv_array.h"

namespace mshadow {
namespace ps {
#if MSHADOW_DIST_PS_
template<typename xpu, typename DType>
class DistServer : public LocalServer<xpu, DType> {
 public:
  // parent type
  typedef LocalServer<xpu, DType> Parent;
  virtual void SetParam(const char *name, const char *val) {
    Parent::SetParam(name, val);
    if (!strcmp(name, "name")) name_ = val;
    if (!strcmp(name, "parent_name")) parent_name_ = val;
  }  
  // initialize the parameter server
  virtual void Init(const std::vector<int> &devices) {
    Parent::Init(devices);
    CHECK(!name_.empty());
    CHECK(!parent_name_.empty());
    shared_model_ = new PS::KVArray<DType>(name_, parent_name_);    
  }
  virtual ~DistServer(void) {    
  }
  // override this function, to use parameter server
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                int key) {
    // here we only use sum reduction, can change to others
    for (index_t i = 1; i < data.size(0); ++i) {
      data[0] += data[i];
    }
    // things to send and recv
    Tensor<cpu, 2> sendrecv = data[0];
    using namespace PS;
    utils::Assert(data[0].CheckContiguous(),
                  "data must be contiguous");
    // TODO the zero copy version
    // SArray<DType> val(data.dptr_, data.MSize(), false);
    SArray<DType> val; val.copyFrom(sendrecv.dptr_, sendrecv.MSize());
    MessagePtr msg(new Message(kServerGroup));
    msg->addValue(val);
    msg->task.set_key_channel(key);
    Range<Key>(0, val.size()).to(msg->task.mutable_key_range());
    records_[key].push = CHECK_NOTNULL(shared_model_)->push(msg);
    // setup callback
    auto& rec = records_[key];
    MessagePtr msg(new Message(kServerGroup, -1, rec.push));
    msg->task.set_key_channel(key);
    Range<Key>(0, sendrecv.MSize()).to(msg->task.mutable_key_range());
    
    msg->fin_handle = [this, sendrecv, key]() {
      const auto& recv = shared_model_->array(key);
      CHECK_EQ(sendrecv.MSize(), recv.size());      
      memcpy(CHECK_NOTNULL(sendrecv.dptr_), recv.data(), recv.size() * sizeof(DType));
      // call PullReady to notify LocalServer pulling is ready
      this->PullReady(sendrecv, key);      
    };
    rec.pull = CHECK_NOTNULL(shared_model_)->pull(msg);    
  }

 private:
  struct Record {
    int push = -1;
    int pull = -1;
    DType* data = nullptr;
  };
  std::string name_;
  std::string parent_name_;
  PS::KVArray<DType>* shared_model_;
};
#endif
}  // namespace ps
}  // namespace msahdow
#endif

