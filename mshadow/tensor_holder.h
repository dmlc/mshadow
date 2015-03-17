/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_holder.h
 * \brief tensor holder that holds common representation of 
 *  arbirary dimension tensor, can be used to transformed to normal fixed
 *  dimenson tensor
 * \author Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_HOLDER_H_
#define MSHADOW_TENSOR_HOLDER_H_
#include <vector>
#include "./tensor.h"

namespace mshadow {
/*!
 * \brief holder class that can be used to hold tensor of any dimension
 *  holder itself do not involve any arithmentic operations,
 *  can be converted to tensor of fixed dimension for further operations
 */
struct ShapeHolder {
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
   * \brief get the shape of tensor specifying ndim
   * \return the shape requested
   * \tparam ndim dimension of the tensor
   */
  template<int ndim>
  inline mshadow::Shape<ndim> get(void) const {
    utils::Check(ndim == shape_.size(),
                 "dimension do not match target dimension");
    Shape<ndim> s;
    for (index_t i = 0; i < ndim; ++i) {
      s[i] = shape_[i];
    }
    return s;
  }
  /*!
   * \brief assignment from shape
   * \param src source shape
   * \tparam ndim shape dimension
   */
  template<int ndim>
  inline ShapeHolder &operator=(const Shape<ndim> &shape) {
    shape_.resize(ndim);
    for (index_t i = 0; i < ndim; ++i) {
      shape_[i] = shape[i];
    }    
    return *this;
  }
  /*!
   * \return whether two shape equals 
   * \param s the shape to compare against
   */
  inline bool operator==(const ShapeHolder &s) const {
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
  template<int ndim>
  inline bool operator==(const Shape<ndim> &s) const {
    if (shape_.size() != ndim) return false;
    for (index_t i = 0; i < ndim; ++i) {
      if (shape_[i] != s.shape_[i]) return false;
    }
    return true;
  }
};

/*!
 * \brief holder class that can be used to hold tensor of any dimension
 *  holder itself do not involve any arithmentic operations,
 *  can be converted to tensor of fixed dimension for further operations, with fixdim
 *
 *  Like tensor, this data structure is like a pointer class and do not
 *  implicit allocated, de-allocate space.
 *  This data structure can be helpful to hold tensors of different dimensions
 *  and wait for further processing
 * \tparam Device which device the tensor is on
 * \tparam DType the type of elements in the tensor
 */
template<typename Device, typename DType = default_real_t>
class TensorHolder {
 public:
  /*! \brief pointer to the data */
  DType *dptr_;
  /*! \brief shape of the tensor */
  ShapeHolder shape_;
  /*!
   * \brief storing the stride information in x dimension
   */
  index_t stride_;
  /*!
   * \brief stream where the computation lies
   * stream is a device dependency concept where each computation
   */
  Stream<Device> *stream_;
  /*! \brief default constructor, default copy assign will work */
  TensorHolder(void) : dptr_(NULL), stream_(NULL) {
  }
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam ndim tensor dimension
   */
  template<int ndim>
  TensorHolder(const Tensor<Device, ndim, DType> &src) {
    *this = src; 
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam ndim tensor dimension
   */
  template<int ndim>
  inline TensorHolder<Device, DType>
  &operator=(const Tensor<Device, ndim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    stride_ = src.stride_;
    stream_ = src.stream_;
    return *this;
  }
  /*!
   * \brief set the stream to do computation of current tensor
   * \param stream the computation stream
   */
  inline void set_stream(Stream<Device> *stream) {
    this->stream_ = stream;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return shape_[shape_.shape_.size() - 1] == stride_;
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \return tensor after flatten
   */  
  inline Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr_, shape_.FlatTo2D(), stride_, stream_);    
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
   * if ndim do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \tparam ndim dimension of the tensor
   */
  template<int ndim>
  inline Tensor<Device, ndim, DType> get(void) const {
    return Tensor<Device, ndim, DType>(dptr_, shape_.get<ndim>(),
                                       stride_, stream_);
  }
  /*!
   * \brief allocate space for the tensor holder
   * \param pad whether padding is requred
   */
  inline void AllocSpace(bool pad = MSHADOW_ALLOC_PAD) {
    if (this->ndim() == 0) return;
    Tensor<Device, 2, DType> ts = this->FlatTo2D();
    mshadow::AllocSpace(&ts, pad);
    dptr_ = ts.dptr_;
    stride_ = ts.stride_;    
  }
  /*! \brief free space holded by this tensor holder */
  inline void FreeSpace(void) {
    if (this->ndim() == 0) return;
    Tensor<Device, 2, DType> ts = this->FlatTo2D();
    mshadow::FreeSpace(&ts);
  }
};
/*!
 * \brief fetch the tensor, with respect to specific dimension
 * if ndim do not match the stored dimension, an error will be issued
 * \param src the source tensor holder
 * \return the tensor requested
 * \tparam ndim dimension of the tensor
 * \tparam Device the device where the tensor lies
 * \tparam DType the data type of the tensor
 */
template<int ndim, typename Device, typename DType>
inline Tensor<Device, ndim, DType> fixdim(const TensorHolder<Device, DType> &src) {
  const ShapeHolder &s = src.shape_;
  return Tensor<Device, ndim, DType>(src.dptr_, s.get<ndim>(),
                                     src.stride_, src.stream_);  
}
} // namespace mshadow
#endif  // MSHADOW_TENSOR_HOLDER_H_
