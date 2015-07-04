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
#include <algorithm>
#include "./tensor.h"

namespace mshadow {
/*!
 * \brief dynamic shape class that can hold shape
 *   of arbirary dimension
 */
struct TShape {
 public:
  /*! \brief constructor */
  TShape()
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {}
  /*!
   * \brief constructor from TShape
   * \param s the source shape
   */
  TShape(const TShape &s)
      : ndim_(s.ndim_) {
    if (ndim_ <= kStackCache) {
      data_heap_ = NULL;
      num_heap_allocated_ = 0;
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    } else {
      data_heap_ = new index_t[ndim_];
      num_heap_allocated_ = ndim_;
      std::copy(s.data_heap_, s.data_heap_ + ndim_, data_heap_);
    }
  }
  /*!
   * \brief construct the TShape from content of iterator
   * \param begin the beginning of iterator
   * \param end end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  TShape(RandomAccessIterator begin,
         RandomAccessIterator end)
      : ndim_(0),
        num_heap_allocated_(0),
        data_heap_(NULL) {
    this->CopyFrom(begin, end);
  }
#if MSHADOW_IN_CXX11
  /*!
   * \brief move constructor from TShape
   * \param s the source shape
   */
  TShape(TShape &&s)
      : ndim_(s.ndim_),
        num_heap_allocated_(s.num_heap_allocated_),
        data_heap_(s.data_heap_) {
    if (ndim_ <= kStackCache) {
      std::copy(s.data_stack_, s.data_stack_ + ndim_, data_stack_);
    }
    // remove data heap space from s
    s.data_heap_ = NULL;
  }
#endif
  /*! \brief destructor */
  ~TShape() {
    // data_heap_ can be NULL
    delete [] data_heap_;
  }
  /*!
   * \brief copy shape from content betwen two iterators
   * \param begin the beginning of iterator
   * \param end the end of the iterator
   * \tparam RandomAccessIterator iterator type
   */
  template<typename RandomAccessIterator>
  inline void CopyFrom(RandomAccessIterator begin,
                       RandomAccessIterator end) {
    this->SetDim(end - begin);
    std::copy(begin, end, data());
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const TShape &shape) {
    this->SetDim(shape.ndim_);
    const index_t *src = shape.data();
    std::copy(src, src + ndim_, data());
    return *this;
  }
  /*!
   * \brief assignment from vector
   * \param shape source shape
   * \return reference of self
   */
  inline TShape &operator=(const std::vector<index_t> &shape) {
    this->CopyFrom(shape.begin(), shape.end());
    return *this;
  }
  /*!
   * \brief assignment from shape
   * \param shape source shape
   * \tparam dim shape dimension
   * \return reference of self
   */
  template<int dim>
  inline TShape &operator=(const Shape<dim> &shape) {
    this->SetDim(dim);
    index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (int i = 0; i < dim; ++i) {
      d[i] = shape[i];
    }
    return *this;
  }
  /*! \return the data content of the shape */
  inline const index_t *data() const {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \return the data content of the shape */
  inline index_t *data() {
    return ndim_ <= kStackCache ? data_stack_ : data_heap_;
  }
  /*! \brief return number of dimension of the tensor inside */
  inline index_t ndim(void) const {
    return ndim_;
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline index_t &operator[](index_t i) {
    return data()[i];
  }
  /*!
   * \brief get corresponding index
   * \param i dimension index
   * \return the corresponding dimension size
   */
  inline const index_t &operator[](index_t i) const {
    return data()[i];
  }
  /*! \brief total number of elements in the tensor */
  inline size_t Size(void) const {
    size_t size = 1;
    const index_t *d = this->data();
    for (index_t i = 0; i < ndim_; ++i) {
      size *= d[i];
    }
    return size;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  inline Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    if (ndim_ == 0) return Shape2(0, 0);
    const index_t *d = this->data();
    s.shape_[1] = d[ndim_ - 1];
    index_t ymax = 1;
    for (index_t i = 1; i < ndim_; ++i) {
      ymax *= d[i - 1];
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
    utils::Check(dim == ndim_,
                 "dimension do not match target dimension");
    const index_t *d = this->data();
    Shape<dim> s;
    for (int i = 0; i < dim; ++i) {
      s[i] = d[i];
    }
    return s;
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   */
  inline bool operator==(const TShape &s) const {
    if (ndim_ != s.ndim_) return false;
    if (ndim_ <= kStackCache) {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_stack_[i] != s.data_stack_[i]) return false;
      }
    } else {
      for (index_t i = 0; i < ndim_; ++i) {
        if (data_heap_[i] != s.data_heap_[i]) return false;
      }
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   */
  inline bool operator!=(const TShape &s) const {
    return !(*this == s);
  }
  /*!
   * \return whether two shape equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator==(const Shape<dim> &s) const {
    if (ndim_ != dim) return false;
    const index_t *d = dim <= kStackCache ? data_stack_ : data_heap_;
    for (index_t i = 0; i < dim; ++i) {
      if (d[i] != s.shape_[i]) return false;
    }
    return true;
  }
  /*!
   * \return whether two shape not equals
   * \param s the shape to compare against
   * \tparam dim dimension of the shape
   */
  template<int dim>
  inline bool operator!=(const Shape<dim> &s) const {
    return !(*this == s);
  }

 private:
  // the shape will be stored in data_stack_
  // when dimension is smaller than kStackCache
  // when it is bigger, it will be stored in data_heap_;
  /*! \brief size of in stack space */
  static const index_t kStackCache = 4;
  /*! \brief number of dimnsion of the shape */
  index_t ndim_;
  /*! \brief number of cells allocated in data_heap_ */
  index_t num_heap_allocated_;
  /*! \brief in stack space used to store shape when it is small */
  index_t data_stack_[kStackCache];
  /*! \brief space to store shape when dimension is big*/
  index_t *data_heap_;
  /*!
   * \brief internal function to set the dimension
   * \param dim the dimension of the shape
   */
  inline void SetDim(index_t dim) {
    if (dim > kStackCache &&
        dim > num_heap_allocated_) {
      // data_heap_ can be NULL
      delete [] data_heap_;
      data_heap_ = new index_t[dim];
      num_heap_allocated_ = dim;
    }
    ndim_ = dim;
  }
};

/*! \brief data type flag */
template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = 0;
};
template<>
struct DataType<double> {
  static const int kFlag = 1;
};

/*!
 * \brief tensor blob class that can be used to hold tensor of any dimension,
 *  any device and any data type,
 *  This is a weak type that can be used to transfer data through interface
 *  TBlob itself do not involve any arithmentic operations,
 *  but it can be converted to tensor of fixed dimension for further operations
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
  /*! \brief type flag of the tensor blob */
  int type_flag_;
  /*! \brief default constructor, default copy assign will work */
  TBlob(void)
      : dptr_(NULL), dev_mask_(cpu::kDevMask),
        type_flag_(DataType<default_real_t>::kFlag) {}
  /*!
   * \brief constructor that construct TBlob from contiguous memory
   * \param dptr the pointer to the memory
   * \param shape the shape of the data
   * \param dev_mask the device mask, can be cpu::kDevMask or gpu::kDevMask
   */
  template<typename DType>
  TBlob(DType *dptr,
        const TShape &shape,
        int dev_mask)
      : dptr_(dptr), shape_(shape),
        stride_(shape[shape.ndim() - 1]),
        dev_mask_(dev_mask),
        type_flag_(DataType<DType>::kFlag) {}
  /*!
   * \brief constructor from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  TBlob(const Tensor<Device, dim, DType> &src) {  // NOLINT(*)
    *this = src;
  }
  /*!
   * \brief assignment from tensor
   * \param src source tensor
   * \tparam Device which device the tensor is on
   * \tparam dim tensor dimension
   * \tparam DType the type of elements in the tensor
   * \return reference of self
   */
  template<typename Device, int dim, typename DType>
  inline TBlob
  &operator=(const Tensor<Device, dim, DType> &src) {
    dptr_ = src.dptr_;
    shape_ = src.shape_;
    stride_ = src.stride_;
    dev_mask_ = Device::kDevMask;
    type_flag_ = DataType<DType>::kFlag;
    return *this;
  }
  /*!
   * \return whether the tensor's memory is continuous
   */
  inline bool CheckContiguous(void) const {
    return shape_[shape_.ndim() - 1] == stride_;
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
    utils::Check(Device::kDevMask == dev_mask_ && DataType<DType>::kFlag == type_flag_,
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
    utils::Check(Device::kDevMask == dev_mask_ && DataType<DType>::kFlag == type_flag_,
                 "TBlob.get: device type do not match specified type");
    return Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                       shape_.get<dim>(),
                                       stride_, stream);
  }
  /*!
   * \brief fetch a tensor in given shape
   * if size do not match the stored dimension, an error will be issued
   * \return the tensor requested
   * \param shape the shape required
   * \param stream the possible stream target tensor should reside on
   * \tparam Device which device the tensor is on
   * \tparam dim dimension of the tensor
   * \tparam DType the type of elements in the tensor
   */
  template<typename Device, int dim, typename DType>
  inline Tensor<Device, dim, DType> get_with_shape(const TShape &shape,
                                                   Stream<Device> *stream = NULL) const {
    utils::Check(Device::kDevMask == dev_mask_ && DataType<DType>::kFlag == type_flag_,
                 "TBlob.get_with_shape: device type do not match specified type");
    utils::Check(this->CheckContiguous(), "TBlob.get_reshape: must be contiguous");
    utils::Check(this->shape_.Size() == shape.Size(),
                 "TBlob.get_with_shape: new and old shape do not match total elements");
    return Tensor<Device, dim, DType>(static_cast<DType*>(dptr_),
                                      shape.get<dim>(),
                                      shape[dim - 1],
                                      stream);
  }
};
}  // namespace mshadow
#endif  // MSHADOW_TENSOR_BLOB_H_
