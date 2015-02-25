#ifndef MSHADOW_PS_THREAD_UTIL_H_
#define MSHADOW_PS_THREAD_UTIL_H_
/*!
 * \file thread_util.h
 * \brief data structures for multi-threading communication
 * \author Tianqi Chen
 */
#include <utility>
#include <queue>
#include "./thread.h"
namespace mshadow {
namespace utils {
/*!
 * \brief thread safe queue that can be used for customer consumer model
 * in the future, it will support priority scheduling
 * \tparam DType the content of the queue
 */
template<typename DType>
class ThreadPQueue {
 public:
  /*! \brief intitialize the queue, must call this before use */
  inline void Init(void) {
    lock_.Init();
    counter_.Init(0);
  }
  /*! \brief destroy the resources on the queue */
  inline void Destroy(void) {
    lock_.Destroy();
    counter_.Destroy();
  }
  /*!
   * \brief Destroy the queue
   *        wake up all the threads waits on pop
   *  this is usually used in class destructor
   * \param max_nthread the maximum number of thread that
   *  could be waiting on the queue
   */
  inline void Abort(int max_nthread = 1) {
    for (int i = 0; i < max_nthread; ++ i) {
      counter_.Post();
    }   
  }
  /*!
   * \brief push an element to the queue
   * \param data the data to be puhed into queue
   * \param optionally priority level to hint which
   *        element should be poped first
   */
  inline void Push(const DType &data, int priority = 0) {
    lock_.Lock();
    queue_.push(Entry(data, priority));
    lock_.Unlock();
    counter_.Post();    
  }
  /*!
   * \brief pop an element from the queue
   * this will block the thread if the queue is empty
   * \param data_out the address to put output of the queue
   * \return true if a correct element is returned
   *  false if abort is called and no element was left in queue
   */
  inline bool Pop(DType *data_out) {
    counter_.Wait();
    lock_.Lock();
    if (queue_.size() == 0) {
      lock_.Unlock(); return false;
    }
    utils::Assert(queue_.size() != 0, "Queue.Pop");
    *data_out = queue_.top().data;
    queue_.pop();
    lock_.Unlock();
    return true;
  }

 private:
  // entry in the queue
  struct Entry {
    DType data;
    int priority;
    Entry(const DType &data, int priority)
        : data(data), priority(priority) {}
    inline bool operator<(const Entry &b) const {
      return priority < b.priority;
    }
  };

  // the queue to push
  std::priority_queue<Entry> queue_;
  // lock for accessing the queue
  utils::Mutex lock_;
  // counter to count number of push tasks
  utils::Semaphore counter_;
};

// naive implementation of threadsafe map
template<typename TValue>
class ThreadSafeMap {
 public:
  inline void Init(void) {
    lock_.Init();
  }
  inline void Destroy(void) {
    for (typename std::map<int, TValue*>::iterator
             it = map_.begin(); it != map_.end(); ++it) {
      delete it->second;
    }
    lock_.Destroy();
  }
  inline TValue *Get(int key) {
    TValue *ret;
    lock_.Lock();
    typename std::map<int, TValue*>::const_iterator 
        it = map_.find(key);
    if (it == map_.end() || it->first != key) {
      ret = NULL;
    } else {
      ret = it->second;
    }
    lock_.Unlock();
    return ret;
  }
  inline TValue &GetRef(int key) {
    TValue *ret = this->Get(key);
    utils::Assert(ret != NULL, "key=%d does not exist", key);
    return *ret;
  }
  inline void Init(int key) {
    lock_.Lock();
    if (map_.count(key) == 0) {
      map_[key] = new TValue();
    }
    lock_.Unlock();
  }

 private:
  // lock for accessing the queue
  utils::Mutex lock_;
  std::map<int, TValue*> map_;
};

}  // namespace utils
}  // namespace mshadow
#endif  // MSHADOW_PS_THREAD_UTIL_H_
