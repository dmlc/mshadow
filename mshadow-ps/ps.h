/*!
 *  Copyright (c) 2014 by Contributors
 * \file ps.h
 * \brief parameter server abstraction for mshadow tensor
 *  this is a plugin of mshadow that can be used to syncrhonize
 *  parameters across device and machines
 *
 * \author Tianqi Chen, Mu Li
 */
#ifndef MSHADOW_PS_H_
#define MSHADOW_PS_H_
#include <vector>
// optionally support of lambda function in C++11, if available
#if __cplusplus >= 201103L
#include <functional>
#endif  // C++11
#include "../mshadow/tensor.h"

/*! \brief whether to adapt distributed PS from parameter-server */
#ifndef MSHADOW_DIST_PS_
#define MSHADOW_DIST_PS_ 1
#endif

namespace mshadow {
namespace ps {
/*!
 * \brief interface of parameter server
 * \tparam xpu the device of the data lies
 * \tparam DType the type of element in the tensor
 */
template<typename xpu,
         typename DType MSHADOW_DEFAULT_DTYPE>
class IParamServer {
 public:
  /*!
   * \brief callback function that will be executed when pull request finishes
   *        before calling the callback, the thread context is already switched
   *        to the device of pullrequest
   * \param stream the stream of callback thread, it is recommended to operate using this stream
   * \param arg the argument of callback function
   */
  typedef void (CallbackFunction) (Stream<xpu> *stream, void *arg);
  /*! \brief virtual destructor */
  virtual ~IParamServer(void) {}
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*!
   * \brief initialize the paramerver server client
   * \param devices specifies the possible device id
   *   to be input from Push and Pull,
   */
  virtual void Init(const std::vector<int> &devices) {}
  /*!
   * \brief initialize the paramerver server client
   * without specifying the devices, only device 0 is allowed
   */
  inline void Init(void) {
    std::vector<int> dev;
    dev.push_back(0);
    this->Init(dev);
  }
  /*!
   * \brief initialize a key with certain shape
   * \param shape the shape content of the key
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  template<int dim>
  inline void InitKey(Shape<dim> shape,
                      int key, int devid) {
    this->InitKey_(shape.FlatTo2D(), key, devid);
  }
  /*!
   * \brief wait until the pull event finishes
   * if there was no pull request, wait will directly returns
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void PullWait(int key, int devid = 0) = 0;
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   */
  template<int dim>
  inline void Push(Tensor<xpu, dim, DType> data,
                   int key,
                   int devid = 0,
                   int priority = 0) {
    this->Push_(data.FlatTo2D(), key, devid, priority);
  }
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   * \param callback the callback function that will
   *                 be invoked when the request finishes
   * \param callback_arg the argument to pass to callback
   */
  template<int dim>
  inline void PullReq(Tensor<xpu, dim, DType> data,
                      int key,
                      int devid = 0,
                      int priority = 0,
                      CallbackFunction callback = NULL,
                      void *callback_arg = NULL) {
    this->PullReq_(data.FlatTo2D(), key,
                   devid, priority, callback, callback_arg);
  }
#if __cplusplus >= 201103L
  template<int dim>
  inline void PullReq(Tensor<xpu, dim, DType> data,
                      int key,
                      int devid,
                      int priority,
                      std::function<void(Stream<xpu> *stream)> callback) {
    // need to allocate space, because callback can happen latter..
    auto calbk = new std::function<void(Stream<xpu> *stream)>();
    *calbk = callback;
    this->PullReq(data, key, devid, priority, InvokeLambda_, calbk);
  }
#endif  // C++11
 protected:
  /*!
   * \brief initialize a key with certain shape
   * \param shape the shape content of the key
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void InitKey_(Shape<2> shape,
                        int key, int devid) = 0;
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   */
  virtual void Push_(Tensor<xpu, 2, DType> data,
                     int key,
                     int devid = 0,
                     int priority = 0) = 0;
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   * \param priority the priority of this operation,
   *   the bigger the number is the higher the priority will be
   * \param callback the callback function that will
   *                 be invoked when the request finishes
   * \param callback_arg the argument to pass to callback
   */
  virtual void PullReq_(Tensor<xpu, 2, DType> data,
                        int key,
                        int devid,
                        int priority,
                        CallbackFunction callback,
                        void *callback_arg) = 0;

 private:
// C++11 support for lambda prepare function
#if __cplusplus >= 201103L
  /*! \brief hack function to convert lambda to callback function */
  inline static void InvokeLambda_(Stream<xpu> *stream, void *fun) {
    auto *fp = static_cast<std::function<void(Stream<xpu> *stream)>*>(fun);
    (*fp)(stream);
    delete fp;
  }
#endif  // C++11
};
/*! \brief interface for customized mshadow server */
template<typename DType>
class ICustomServer {
 public:
  virtual ~ICustomServer(void) {}
  /*!
   * \brief set parameters from outside
   * \param name name of parameter
   * \param val value of parameter
   */
  virtual void SetParam(const char *name, const char *val) = 0;
  /*!
   * \brief init the server
   * \param rank the rank of the node
   * \param conf configuration
   */
  virtual void Init(int rank, const std::string &conf) = 0;
  /*!
   * \brief initialize the key
   * \param key the key of data we point to
   * \param dptr the data pointer
   * \param size size of the parameter key
   */
  virtual void InitKey(int key, DType *dptr, size_t size) = 0;
  /*!
   * \param key the key of data we point to
   * \param dptr the data pointer
   * \param size size of the parameter key
   */
  virtual void Update(int key, DType *dptr, size_t size) = 0;
};
/*!
 * \brief create customized server
 * this is a server defined by user
 * \return new server
 */
template<typename DType>
ICustomServer<DType> *CreateServer(void);
}  // namespace ps
}  // namespace mshadow

#include "./ps_local-inl.h"
#include "./ps_dist-inl.h"
namespace mshadow {
namespace ps {
/*!
 * \brief create a parameter server implementation
 * \param type the type of paramerver server
 */
template<typename xpu, typename DType>
inline IParamServer<xpu, DType> *Create(const char *type) {
  if (!strcmp("local", type)) return new LocalServer<xpu, DType>();
#if MSHADOW_DIST_PS_
  if (!strcmp("dist", type)) return new DistServer<xpu, DType>();
#endif
  utils::Error("unknown server type %s\n", type);
  return NULL;
}
}  // namespace ps
}  // namespace mshadow
#endif
