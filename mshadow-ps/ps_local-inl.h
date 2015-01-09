/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps_local-inl.h
 * \brief local multi-threading implementation of PS abstraction
 *
 * \author Tianqi Chen, Mu Li
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
    pull_queue.Destroy();
    pull_map.Destroy();
    request_lock.Destroy();
    wait_lock.Destroy();
    wait_cond.Destroy();
  }
  virtual void PullWait(int key, int devid) {
    const int wid = GetWorkIndex(devid);
    PullEntry *p = pull_map.Get(key);
    if (p == NULL || p->wait.size() == 0) return;
    PullEntry &e = *p;
    // wake up waiters if any
    utils::Assert(e.wait.size() == devices.size(),
                  "PullWait: must initialize the wait");
    PullWaitRecord &w = e.wait[wid];
    if (!w.finished) {
      wait_lock.Lock();
      w.nwait += 1;
      while (!w.finished) {
        wait_cond.Wait(&wait_lock);
      }
      w.nwait -= 1;
      utils::Assert(w.nwait >= 0, "boundary check");
      wait_lock.Unlock();
    }
  }
  virtual void Init(const std::vector<int> &devices) {
    utils::Check(devices.size() != 0,
                 "LocalServer.Init: must at least contain 1 devices");
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
    // initialize all the thread related things
    push_queue.Init();
    pull_queue.Init();
    pull_map.Init();
    request_lock.Init();
    wait_lock.Init();
    wait_cond.Init();
    // initialize the thread
    thread_push_handler.Start(PushHandlerThread, this);
    thread_pull_handler.Start(PullHandlerThread, this);
  }
 protected:
  virtual void Push_(Tensor<xpu, 2, DType> data,
                     int key, int devid, int priority) {
    this->InitPullMap(key, devid);
    push_queue.Push(PullTask(data, key, devid), priority);
  }
  virtual void PullReq_(Tensor<xpu, 2, DType> data,
                        int key, int devid, int priority,
                        CallbackFunction callback,
                        void *callback_arg) {
    PullEntry &e = pull_map.GetRef(key);
    utils::Assert(e.req.size() == devices.size(),
                  "PullReq: must initialize the key, req");
    utils::Assert(e.wait.size() == devices.size(),
                  "PullReq: must initialize the key, wait");
    const int wid = GetWorkIndex(devid);
    PullReqRecord &r = e.req[wid];
    r.dest = data;
    r.priority = priority;
    r.callback = callback;
    r.callback_arg = callback_arg;
    // reset pull request finish mark
    wait_lock.Lock();
    e.wait[wid].finished = false;
    wait_lock.Unlock();
    // check ready event
    request_lock.Lock();
    utils::Check(!r.pending,
                 "key = %d, cannot send duplicate pull request before it finishes",
                 key);
    if (e.req[wid].ready) {
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
                  "PullReady: must initialize the key, req");
    request_lock.Lock();
    e.src = data;
    for (index_t i = 0; i < e.req.size(); ++i) {
      e.req[i].ready = true;
      if (e.req[i].pending) {
        pull_queue.Push(std::make_pair(key, devices[i]));
        e.req[i].pending = false;
      }
    }
    request_lock.Unlock();
  }
  /*!
   * \brief event handler for push finish
   *  called when all the data with same key comes int
   * \param data the buffer holds the data in all devices
   * \param result_buffer temporal buffer to hold the reduction result
   * \param key the key of the data
   */
  virtual void HandlePushFinish(Tensor<cpu, 3, DType> data,
                                Tensor<cpu, 2, DType> result_buffer,
                                int key) {
    Copy(result_buffer, data[0]);
    for (index_t i = 1; i < data.size(0); ++i) {
      result_buffer += data[i];
    }
    this->PullReady(result_buffer, key);
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
    // temporal space to hold to copy back
    TensorContainer<cpu, 2, DType> result_buffer;
    // indicator whether the certain devices is already copied in
    std::vector<bool> copied;
    // number of data copied in
    int num_copied;
    // constructor
    explicit PushEntry(int ndevice, Shape<2> shape)
        : data(false), result_buffer(false) {
      data.Resize(Shape3(ndevice, shape[0], shape[1]));
      result_buffer.Resize(shape);
      num_copied = 0;
      copied.resize(ndevice, false);
    }
  };
  // a record to remember things related to pull request
  struct PullReqRecord {
    // whether this record contains a pending request
    // whether pull is ready to go
    bool ready;
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
    PullReqRecord(void) : ready(false), pending(false) {
    }
  };
  // a record to help handle pullwait
  struct PullWaitRecord {
    // number of thread that waits for the request to finish
    int nwait;
    // the request was finished
    bool finished;
    PullWaitRecord(void) : nwait(0), finished(true) {
      // set finished to true so pull without pull request returns
    }
  };
  /*! \brief data structure to hold pull request */
  struct PullEntry {
    // data to be pulled back
    Tensor<cpu, 2, DType> src;
    // pullrequest record
    std::vector<PullReqRecord> req;
    // whether there is thread waiting on this event
    std::vector<PullWaitRecord> wait;
    PullEntry(void) {
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
  // conditional variable to do waiting
  utils::ConditionVariable wait_cond;
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
        // wait till the copy finishes
        push_stream[wid]->Wait();
        // mark copied
        e.copied[wid] = true;
        e.num_copied += 1;
        if (e.num_copied >= static_cast<int>(devices.size())) {
          this->HandlePushFinish(e.data, e.result_buffer, tsk.key);
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
      DeleteStream(push_stream[i]);
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

  // push handler
  inline void PullHandler(void) {
    // allocate stream resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      pull_stream.push_back(NewStream<xpu>());
    }
    while (!destroy_signal) {
      std::pair<int, int> tsk;
      if (pull_queue.Pop(&tsk)) {
        const int key = tsk.first;
        const int devid = tsk.second;
        const int wid = GetWorkIndex(devid);
        PullEntry &e = pull_map.GetRef(key);
        {
          // handle request
          utils::Assert(e.req.size() == devices.size(),
                        "PullHandler: must initialize the key, req");
          PullReqRecord &r = e.req[wid];
          SetDevice<xpu>(devid);
          Copy(r.dest, e.src, pull_stream[wid]);
          // callback, if any
          if (r.callback != NULL) {
            (*r.callback)(pull_stream[wid], r.callback_arg);
          }
          // wait till the operation finishes
          pull_stream[wid]->Wait();
        }
        {
          // wake up waiters if any
          utils::Assert(e.wait.size() == devices.size(),
                        "PullHandler, must initialize the key, req");
          PullWaitRecord &w = e.wait[wid];
          wait_lock.Lock();
          w.finished = true;
          if(w.nwait != 0) {
            wait_cond.Broadcast();
          }
          wait_lock.Unlock();
        }
      } else {
        utils::Assert(destroy_signal, "abort but not destroy");
      }
    }
    // free resources
    for (size_t i = 0; i < devices.size(); ++i) {
      SetDevice<xpu>(devices[i]);
      DeleteStream(pull_stream[i]);
    }
  }
  /*!\brief entry point of loader thread */
  inline static MSHADOW_THREAD_PREFIX PullHandlerThread(void *pthread) {
    static_cast<LocalServer*>(pthread)->PullHandler();
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
  inline void InitPullMap(int key, int devid) {
    pull_map.Init(key);
    PullEntry &e = pull_map.GetRef(key);
    request_lock.Lock();
    // must recheck after lock
    if (e.req.size() == 0) {
      e.req.resize(devices.size(), PullReqRecord());
    }
    request_lock.Unlock();
    e.req[GetWorkIndex(devid)].ready = false;
    // check wait map
    if (e.wait.size() == 0) {
      wait_lock.Lock();
      // must recheck after lock
      if (e.wait.size() == 0) {
        e.wait.resize(devices.size(), PullWaitRecord());
      }
      wait_lock.Unlock();
    }
  }
};
}  // namespace ps
}  // namespace mshadow
#endif // MSHADOW_PS_LOCAL_INL_H_
