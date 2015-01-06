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
#include "../mshadow/tensor.h"

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
   * \param num_device number of parallel device
   *        we want to support in current process
   *        in the future, the device id must be in [0, num_device)
   */
  virtual void Init(int num_device = 1) {}
  /*!
   * \brief wait until the pull event finishes
   *
   * \param devid the device id this tensor lies in
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param data the data 
   */
  virtual void PullWait(int devid, int key) = 0;
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  template<int dim>
  inline void Push(mshadow::Tensor<xpu, dim, DType> data, 
                   int key, int devid = 0) {
    this->Push_(data.FlatTo2D(), key, devid);
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
   */
  template<int dim>
  inline void PullReq(mshadow::Tensor<xpu, dim, DType> data, 
                   int key, int devid = 0) {
    this->PullReq_(data, key, devid);
  }

 protected:
  /*!
   * \brief push out a tensor to parameter server
   *  this call is asynchronize and returns immediately
   *
   * \param data the data
   * \param key the unique key to indicate the tensor
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void Push_(mshadow::Tensor<xpu, 2, DType> data,
                     int key, int devid = 0) = 0;                    
  /*!
   * \brief send a pull request, to pull parameter into data
   *  this call is asynchronize and returns immediately
   *  use PullWait to wait the event of copy finish
   *
   * \param data the data
   * \param key the unique key to indicate the tensor,
   *        this is unique per device
   * \param devid the device id this tensor lies in
   */
  virtual void PullReq_(mshadow::Tensor<xpu, 2, DType> data,
                        int key, int devid = 0) = 0;
};
/*! 
 * \brief create a parameter server implementation
 * \param type the type of paramerver server
 */
template<typename xpu, typename DType>
inline IParamServer<xpu, DType> *Create(const char *type);
}  // namespace ps
}  // namespace mshadow
#endif
