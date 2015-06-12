/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_blob.h
 * \brief TBlob class that holds common representation of 
 *  arbirary dimension tensor, can be used to transformed
 *  to normal fixed dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_BLOB_H_
#define MSHADOW_TENSOR_BLOB_H_
#include <vector>
#include "./tensor.h"

namespace mshadow {
/*!
 * \brief dynamic shape class that can hold shape
 *   of arbirary dimension
 */
struct TShape {
  /*! \brief shape of the tensor */
  std::vector<index_t> shape_;  
  /*! \brief return number of dimension of the tensor inside */
  inline index_t ndim(void) const {
    return static_cast<index_t>(shape_.size());
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */  
  inline index_t &operator[](index_t i) {
    return shape_[i];
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */  
  inline const index_t &operator[](index_t i) const {
    return shape_[i];
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  inline Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    if (shape_.size() == 0) {
      return Shape2(0, 0);
    } 
    s.shape_[1] = this->shape_[shape_.size()- 1];
    index_t ymax = 1;
    for (size_t i = 1; i < shape_.size(); ++i) {
      ymax *= this->shape_[i - 1];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*!
   * \brief get the shape of tensor specifying dim
   * \return the shape requested
   * \tparam dim dimension of the tensor
   */
  template<int dim>
  inline Shape<dim> get(void) const {
    utils::Check(dim == this->shape_.size(),
                 "dimension do not match target dimension");
    Shape<dim> s;
    for (int i = 0; i < dim; ++i) {
      s[i] = shape_[i];
    }
    return s;
  }
  /*!
   * \brief assignment from shape
   * \param src source shape
   * \tparam dim shape dimension
   */
  template<int dim>
  inline TShape &operator=(const Shape<dim> &shape) {
    shape_.resize(dim);
    for (int i = 0; i < dim; ++i) {
      this->shape_[i] = shape[i];
    }    
    return *this;
  }
  /*!
   * \return whether two shape equals 
   * \param s the shape to compare against
   */
  inline bool operator==(const TShape &s) const {
    if (shape_.size() != s.shape_.size()) return false;
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (shape_[i] != s.shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   * \param the shape to compare against
   */
  template<int dim>
  inline bool operator==(const Shape<dim> &s) const {
    if (shape_.size() != dim) return false;
    for (index_t i = 0; i < dim; ++i) {
      if (shape_[i] != s.shape_[i]) return false;
    }
    return true;
  }
};

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself do not involve any arithmentic operations,
 *  but it can be converted to tensor of fixed dimension for further operations, with fixdim
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 */
class TBlob {
 public:
  /*! \brief pointer to the data */
  void *dptr_;
  /*! \brief shape of the tensor */
  TShape shape_;
  /*!
   * \brief storing the stride information in x dimension
   */
  index_t stride_;
  /*! \brief device mask of the corresponding device */
  int dev_mask_;
  
  /*! \brief default constructor, default copy assign will work */
  TBlob(void) : dptr_(NULL), dev_mask_(cpu::kDevMask) {}
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  TBlob(const Tensor<Device, dim, DType> &src) {
    *this = src; 
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline TBlob
  &operator=(const Tensor<Device, dim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    stride_ = src.stride_;
    dev_mask_ = Device::kDevMask;
    return *this;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return shape_[shape_.shape_.size() - 1] == stride_;
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam DType the type of elements in the tensor
   * \return tensor after flatten
   */  
  template<typename Device, typename DType>
  inline Tensor<Device, 2, DType> FlatTo2D(Stream<Device> *stream = NULL) const {
    utils::Check(Device::kDevMask == dev_mask_,
                 "TBlob.get: device type do not match specified type");
    return Tensor<Device, 2, DType>(static_cast<DType*>(dptr_),
                                    shape_.FlatTo2D(), stride_, stream);
  }
  /*! \brief return number of dimension of the tensor inside */
  inline int ndim(void) const {
    return shape_.ndim();
  }
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param idx the dimension count from the highest dimensin
   * \return the size
   */  
  inline index_t size(index_t idx) const {
    return shape_[idx];
  }
  /*!
   * \brief fetch the tensor, with respect to specific dimension
   * if dim do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline Tensor<Device, dim, DType> get(Stream<Device> *stream = NULL) const {
    utils::Check(Device::kDevMask == dev_mask_,
                 "TBlob.get: device type do not match specified type");
    return Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                       shape_.get<dim>(),
                                       stride_, stream);
  }
};
} // namespace mshadow
#endif  // MSHADOW_TENSOR_BLOB_H_
