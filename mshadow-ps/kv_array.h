#pragma once
#include "parameter/shared_parameter.h"
#include "ps.h"
namespace PS {

DECLARE_string(app_name);

template <typename V>
class KVArray : public SharedParameter<Key> {
 public:
  KVArray(const string& my_name = FLAGS_app_name,
          const string& parent_name = FLAGS_app_name + "_model") :
      SharedParameter<Key>(my_name, parent_name) { }
  virtual ~KVArray() { }

  void setArray(int key, V* data, size_t size) {
    val_[key] = SArray<V>(data, size, false);
  }
  void setUpdater(ICustomServer<V>* updater) {
    updater_ = updater;
  }

  // SArray<V>& array(int key) { return val_[key]; }

  // funcs will be called by the system
  MessagePtrList slice(const MessagePtr& msg, const KeyRangeList& krs);
  void getValue(const MessagePtr& msg);
  void setValue(const MessagePtr& msg);
 protected:
  std::unordered_map<int, SArray<V>> val_;
  // an array is place into multiple servers only if its length > min_slice_size
  size_t min_slice_size_ = 1000;

  ICustomServer<V>* updater_ = nullptr;
 private:
};


template <typename V>
void KVArray<V>::setValue(const MessagePtr& msg) {
  CHECK_EQ(msg->value.size(), 1);
  SArray<V> recv_data(msg->value[0]);
  Range<Key> kr(msg->task.key_range());
  CHECK_EQ(kr.size(), recv_data.size());
  int key = msg->task.key_channel();
  auto& my_val = val_[key];

  if (isWorker()) {
    if (my_val.empty()) my_val.resize(kr.size(), 0);
    CHECK_GE(my_val.size(), kr.end());
    my_val.segment(kr).copyFrom(recv_data);
  } else if (isServer()) {
    // TODO this server can do flexible consistency control here

    if (my_val.empty()) {
      // initialize weight
      my_val.resize(kr.size(), 0);
      CHECK_NOTNULL(updater_)->InitKey(key, my_val.data(), my_val.size());
    }

    // update weight
    CHECK_GE(my_val.size(), kr.size());
    CHECK_NOTNULL(updater_)->Update(key, recv_data.data(), recv_data.size());
  }
}

// only be called at servers, namely a worker pull data from this server
template <typename V>
void KVArray<V>::getValue(const MessagePtr& msg) {
  auto& my_val = val_[msg->task.key_channel()];
  Range<Key> kr(msg->task.key_range());
  CHECK_GE(my_val.size(), kr.end());
  SArray<V> send_data(kr.size());
  send_data.copyFrom(my_val.segment(kr));
  msg->addValue(send_data);
}

// divide a message into n part, where part i goes to server i. it's a zero-copy
// implementation
template <typename V>
MessagePtrList KVArray<V>::slice(const MessagePtr& msg, const KeyRangeList& krs) {
  // divide the key range
  size_t n = krs.size();
  MessagePtrList ret(n);
  Range<Key> kr(msg->task.key_range());
  for (size_t i = 0; i < n; ++i) {
    ret[i] = MessagePtr(new Message());
    ret[i]->miniCopyFrom(*msg);
    ret[i]->valid = true;
    auto mut_kr = ret[i]->task.mutable_key_range();
    if (kr.size() < min_slice_size_) {
      if (i == 0) {
        // server 0 get all data
        kr.to(mut_kr);
      } else {
        Range<Key>(0,0).to(mut_kr);
        // do not sent to server 1 - n
        ret[i]->valid = false;
      }
    } else {
      kr.evenDivide(n, i).to(mut_kr);
    }
  }

  // divide the data
  for (size_t i = 0; i < msg->value.size(); ++i) {
    SArray<V> data(msg->value[i]);
    CHECK_EQ(data.size(), kr.size());
    for (size_t j = 0; j < n; ++j) {
      if (ret[j]->valid) {
        Range<Key> kr(ret[i]->task.key_range());
        ret[i]->addValue(data.segment(kr));
      }
    }
  }
  return ret;
}


} // namespace PS
