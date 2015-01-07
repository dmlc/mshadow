/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_local-inl.h
 * \brief local multi-threading implementation of PS abstraction
 *
 * \author Tianqi Chen
 */
#ifndef MSHADOW_PS_LOCAL_INL_H_
#define MSHADOW_PS_LOCAL_INL_H_
#include <map>
#include <utility>
#include "./thread.h"
#include "./thread_util.h"
#include "./ps.h"

namespace mshadow {
namespace ps {
// multi-threaded implementation of 
template<typename xpu, typename DType>
class LocalServer : public IParamServer<xpu, DType> {
 public:
  // redefine callback function
  typedef typename IParamServer<xpu, DType>::CallbackFunction
  CallbackFunction;
  // destructor
  virtual ~LocalServer(void) {
    destroy_signal = true;
    push_queue.Abort(1);
    pull_queue.Abort(1);
    thread_push_handler.Join();
    thread_pull_handler.Join();
    push_queue.Destroy();
    pull_map.Destroy();
  }
  virtual void PullWait(int key, int devid) {
    
  }
  virtual void Init(const std::vector<int> &devices) {
    utils::Check(devices.size() != 0,
                 "LocalServer.Init: must at least contain 1 devices");
    push_queue.Init();
    this->devices = devices;
    // initialize device id to local index
    dev2index.clear();
    for (size_t i = 0; i < devices.size(); ++i) {
      int devid = devices[i];
      utils::Assert(devid >= 0, "device id must be bigger than 0");
      if (devid >= static_cast<int>(dev2index.size())) {
        dev2index.resize(devid + 1, -1);
      }
      dev2index[devid] = static_cast<int>(i);
    }
    // initialize the thread
    thread_push_handler.Start(PushHandlerThread, this);
  }
 protected:
  virtual void Push_(Tensor<xpu, 2, DType> data,
                     int key, int devid, int priority) {
    this->InitPullMap(key);    
    push_queue.Push(PullTask(data, key, devid), priority);
  }
  virtual void PullReq_(Tensor<xpu, 2, DType> data,
                        int key, int devid, int priority,
                        CallbackFunction callback,
                        void *callback_arg) {
    PullEntry &e = pull_map.GetRef(key);
    utils::Assert(e.req.size() == devices.size(),
                  "must initialize the key");
    const int wid = GetWorkIndex(devid);
    PullReqRecord &r = e.req[wid];
    r.dest = data;
    r.priority = priority;
    r.callback = callback;
    r.callback_arg = callback_arg;    
    request_lock.Lock();
    utils::Check(!r.pending,
                 "cannot send duplicate pull request before it finishes");
    if (e.ready) {
      pull_queue.Push(std::make_pair(key, devid));
    } else {
      r.pending = true;
    }
    request_lock.Unlock();
  }
  /*!
   * \brief called to notify that the data is ready for pull
   * \param data the data that can be pulled back
   * \param the key of the data
   */
  virtual void PullReady(Tensor<cpu, 2> data, int key) {
    PullEntry &e = pull_map.GetRef(key);
    utils::Assert(e.req.size() == devices.size(),
                  "must initialize the key");
    request_lock.Lock();
    e.ready = true;
    for (int i = 0; i < e.req.size(); ++i) {
      if (e.req[i].pending) {
        pull_queue.Push(std::make_pair(key, devices[i]));
        e.req[i].pending = false;
      }
    }
    request_lock.Unlock();
  }
  /*!
   * \brief event handler for push finish
   *  called when all the data with same key comes in
   * \param data the buffer holds the data in all devices
   * \param key the key of the data
   */  
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data, int key) {
    for (index_t i = 1; i < data.size(0); ++i) {
      data[0] += data[i];
    }
    this->PullReady(data[0], key);
  }
  
 private:
  /*! \brief task running */    
  struct PullTask {
    /*! \brief the task data source */
    Tensor<xpu, 2, DType> data;
    /*! \brief the key to the tensor */
    int key;
    /*!
     * \brief the device id, (key,devid),
     * uniquely identifies a mem location
     */
    int devid;
    PullTask(void) {}
    PullTask(Tensor<xpu, 2, DType> data, int key, int devid)
        : data(data), key(key), devid(devid) {}
  };
  /*! \brief data structure to hold temporal push result */
  struct PushEntry {
    // temporal space to hold input data
    TensorContainer<cpu, 3, DType> data;
    // indicator whether the certain devices is already copied in
    std::vector<bool> copied;
    // number of data copied in
    int num_copied;
    // constructor
    explicit PushEntry(int ndevice, Shape<2> shape)
        : data(false) {
      data.Resize(Shape3(ndevice, shape[0], shape[1]));
      num_copied = 0;
      copied.resize(ndevice, false);
    }
  };
  // a record to remember things related to pull request
  struct PullReqRecord {
    // whether this record contains a pending request
    // waiting for pull ready
    bool pending;
    // the destination to pull data into
    Tensor<xpu, 2, DType> dest;
    // the priority of the 
    int priority;
    // callback function
    CallbackFunction *callback;
    // argument for callback
    void *callback_arg;
    PullReqRecord(void) : pending(false) {
    }
  };
  /*! \brief data structure to hold pull request */
  struct PullEntry {
    // data to be pulled back
    Tensor<cpu, 2, DType> data;      
    // whether the data is ready
    bool ready;
    // pullrequest record
    std::vector<PullReqRecord> req;
    // whether there is thread waiting on this event
    std::vector<bool> wait;
    PullEntry(void)
        : ready(false) {
    }
  };
  // signal to notify all the thread about class destruction
  bool destroy_signal;
  // vector of devices
  std::vector<int> devices;
  // device index to local index
  std::vector<int> dev2index;
  //----- data structure used to support push ----
  // stream used by push thread each device for memcpy
  std::vector<Stream<xpu>*> push_stream;
  // the queue used for push task
  utils::ThreadPQueue<PullTask> push_queue;
  // thread to handle push task
  utils::Thread thread_push_handler;
  // the map of push buffer
  std::map<int, PushEntry*> push_buffer;
  //----- data structure used to support pull ----
  // the queue used for pull task
  utils::ThreadPQueue<std::pair<int, int> > pull_queue;
  // stream used by pull thread each device for memcpy
  std::vector<Stream<xpu>*> pull_stream;
  // the map to store pull status
  utils::ThreadSafeMap<PullEntry> pull_map;
  // thread to handle pull task
  utils::Thread thread_pull_handler;
  // lock to lock request field
  utils::Mutex request_lock;
  // lock to lock wait field
  utils::Mutex wait_lock;  
  // push handler
  inline void PushHandler(void) {
    // allocate stream resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      push_stream.push_back(NewStream<xpu>());
    }
    while (!destroy_signal) {
      PullTask tsk;
      if (push_queue.Pop(&tsk)) {
        if (push_buffer.count(tsk.key) == 0) {
          push_buffer[tsk.key] = new PushEntry(devices.size(), tsk.data.shape_);
        }
        const int wid = GetWorkIndex(tsk.devid);
        PushEntry &e = *push_buffer[tsk.key];
        utils::Check(e.data[0].shape_ == tsk.data.shape_,
                     "Tensor with same key must share same shape");
        utils::Assert(!e.copied[wid], "data inconsistency");
        // start copy
        SetDevice<xpu>(tsk.devid);
        Copy(e.data[wid], tsk.data, push_stream[wid]);
        // mark copied
        e.copied[wid] = true;
        e.num_copied += 1;
        if (e.num_copied >= static_cast<int>(devices.size())) {
          this->HandlePushFinish(e.data, tsk.key);
          std::fill(e.copied.begin(), e.copied.end(), false);
          e.num_copied = 0;
        }
      } else {
        utils::Assert(destroy_signal, "abort but not destroy");
      }
    }
    // free resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      DeleteStream(push_stream[dev2index[devices[i]]]);
    }
    for (typename std::map<int, PushEntry*>::iterator
             it = push_buffer.begin(); it != push_buffer.end(); ++it) {
      delete it->second;
    }
    push_buffer.clear();
  }
  /*!\brief entry point of loader thread */
  inline static MSHADOW_THREAD_PREFIX PushHandlerThread(void *pthread) {
    static_cast<LocalServer*>(pthread)->PushHandler();
    utils::ThreadExit(NULL);
    return NULL;
  }
  // get internal index of device
  inline int GetWorkIndex(int devid) const {
    utils::Check(devid >= 0 &&
                 devid < static_cast<int>(dev2index.size()) &&
                 dev2index[devid] >= 0,
                 "Push: invalid devid");
    return dev2index[devid];
  }  
  // functions to handle pull
  inline void InitPullMap(int key) {
    pull_map.Init(key);
    PullEntry &e = pull_map.GetRef(key);
    if (e.req.size() == 0) {
      request_lock.Lock();
      // must recheck after lock
      if (e.req.size() == 0) {
        e.req.resize(devices.size(), PullReqRecord());
      }
      request_lock.Unlock();      
    }
    if (e.wait.size() == 0) {
      wait_lock.Lock();
      // must recheck after lock
      if (e.wait.size() == 0) {
        e.wait.resize(devices.size(), false);
      }
      wait_lock.Unlock();      
    }
  }
};
}  // namespace ps
}  // namespace mshadow
#endif // MSHADOW_PS_LOCAL_INL_H_
