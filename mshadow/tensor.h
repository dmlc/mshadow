#ifndef MSHADOW_TENSOR_H_
#define MSHADOW_TENSOR_H_
/*!
 * \file tensor.h
 * \brief header file of tensor data structure and functions
 *  This lib requires explicit memory allocation and de-allocation
 *  all the data structure Tensor<cpu,1>, Tensor<gpu,1> are like handles(pointers),
 *  no memory allocation is happening during calculation
 *   
 *  For STL style tensor, see tensor_container.h
 * \author Bing Xu, Tianqi Chen
 */
#include "./base.h"
#include "./expression.h"

namespace mshadow {
/*! \brief device name CPU */
struct cpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = true;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1<<0;
};
/*! \brief device name CPU */
struct gpu {
  /*! \brief whether this device is CPU or not */
  static const bool kDevCPU = false;
  /*! \brief device flag number, identifies this device */
  static const int kDevMask = 1<<1;
};
/*!
 * \brief shape of a tensor
 *       IMPORTANT NOTE: this shape is different from numpy.shape
 *       shape[0] gives the lowest dimension, shape[dimension-1] gives the highest dimension
 *       shape[k] corresponds to k-th dimension of tensor
 * \tparam dimension dimension of tensor
 */
template<int dimension>
struct Shape {
  /*! \brief dimension of current shape */
  static const int kDimension = dimension;
  /*! \brief dimension of current shape minus one */
  static const int kSubdim = dimension - 1;
  /*! \brief storing the dimension information */
  index_t shape_[kDimension];
  /*!
   * \brief storing the stride information in x dimension
   *    this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
   */
  index_t stride_;
  /*! \brief default constructor, do nothing */
  MSHADOW_XINLINE Shape(void) {}
  /*! \brief constuctor */
  MSHADOW_XINLINE Shape(const Shape<kDimension> &s) {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      this->shape_[i] = s[i];
    }
    this->stride_ = s.stride_;
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE index_t &operator[](index_t idx) {
    return shape_[idx];
  }
  /*!
   * \brief get corresponding index
   * \param idx dimension index
   * \return the corresponding dimension size
   */
  MSHADOW_XINLINE const index_t &operator[](index_t idx) const {
    return shape_[idx];
  }
  /*! \return whether two shape equals */
  MSHADOW_XINLINE bool operator==(const Shape<kDimension> &s) const {
    #pragma unroll
    for (int i = 0; i < kDimension; ++i) {
      if (s.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  /*!
   * flatten the higher dimension to second dimension, return a 2D shape
   * \return the flat 2d shape
   */
  MSHADOW_XINLINE Shape<2> FlatTo2D(void) const {
    Shape<2> s;
    s.stride_ = this->stride_;
    s.shape_[1] = this->shape_[kDimension - 1];
    index_t ymax = 1;    
    #pragma unroll
    for (int i = 0; i < kDimension - 1; ++i) {
      ymax *= this->shape_[i];
    }
    s.shape_[0] = ymax;
    return s;
  }
  /*! \return number of valid elements */
  MSHADOW_XINLINE size_t Size(void) const {
    size_t size = this->shape_[0];
    #pragma unroll
    for (int i = 1; i < kDimension; ++i) {
      size *= this->shape_[i];
    }
    return size;
  }
  /*! \return memory size, including the aligned x dimension */
  MSHADOW_XINLINE size_t MSize(void) const {
    size_t memsz = this->stride_;
    #pragma unroll
    for (int i = 0; i < kDimension - 1; ++i) {
      memsz *= this->shape_[i];
    }
    return memsz;
  }
  /*!
   * \return product shape in [dimstart,dimend)
   * \param dimstart start dimension
   * \param dimend end dimension
   */
  MSHADOW_XINLINE index_t ProdShape(int dimstart, int dimend) const {
    index_t num = 1;
    #pragma unroll
    for (int i = dimstart; i < dimend; ++i) {
      num *= this->shape_[i];
    }
    return num;
  }
  /*!
   * \brief get subshape that takes off largest dimension
v   * \return subshape
   */
  MSHADOW_XINLINE Shape<kSubdim> SubShape(void) const {
    Shape<kSubdim> s;
    s.stride_ = this->stride_;
    // for cuda
    #pragma unroll
    for (int i = 0; i < kSubdim; ++i) {
      s.shape_[i] = this->shape_[i + 1];
    }
    return s;
  }
};  // Shape
//------------------------------------------------
// useful construction functions to generate shape
//-------------------------------------------------
/*!
 * \brief construct a one dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<1> Shape1(index_t s0) {
  Shape<1> s; s[0] = s0; s.stride_ = s0;
  return s;
}
/*!
 * \brief construct a two dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
  Shape<2> s; s[0] = s0; s[1] = s1; s.stride_ = s1;
  return s;
}
/*!
 * \brief construct a three dimension shape, stride will equal s0
 * \param s0 size of dimension 0
 * \param s1 size of dimension 1
 * \param s2 size of dimension 2
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<3> Shape3(index_t s0, index_t s1, index_t s2) {
  Shape<3> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s.stride_ = s2;
  return s;
}
/*!
 * \brief construct a four dimension shape, stride will equal s0
 * \param s3 size of dimension 3
 * \param s2 size of dimension 2
 * \param s1 size of dimension 1
 * \param s0 size of dimension 0
 * \return the shape construction
 */
MSHADOW_XINLINE Shape<4> Shape4(index_t s3, index_t s2, index_t s1, index_t s0) {
  Shape<4> s;
  s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; s.stride_ = s3;
  return s;
}
/*!
 * \brief computaion stream structure, used for asynchronize computation
 */
template<typename Device>
struct Stream {
};
/*!
 * \brief Tensor RValue, this is the super type of all kinds of possible tensors
 * \tparam Container the tensor type
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Container, typename Device, int dimension, typename DType>
struct TRValue: public expr::RValueExp<Container, DType> {
};
// more compact template
/*!
 * \brief general tensor
 * \tparam Device which device the tensor is on
 * \tparam dimension dimension of the tensor
 * \tparam DType the type of elements in the tensor
 */
template<typename Device, int dimension, typename DType =  default_real_t>
struct Tensor: public TRValue<Tensor<Device, dimension, DType>, Device, dimension, DType> {
 public:
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief whether current type lies in cpu */
  static const bool kDevCPU = Device::kDevCPU;
  /*! \brief dimension of subtype */
  static const int  kSubdim = dimension - 1;
  //--------------------------------
  // struct memembers
  //--------------------------------
  /*! \brief pointer to the data */
  DType *dptr;
  /*! \brief shape of the tensor */
  Shape<dimension> shape;
  /*! 
   * \brief stream where the computation lies 
   * stream is a device dependency concept where each computation
   */
  Stream<Device> *stream;
  //--------------------------------
  // functions
  //-------------------------------- 
  /*! \brief default constructor */
  MSHADOW_XINLINE Tensor(void) : stream(NULL) {}
  /*! \brief constructor from shape  */
  MSHADOW_XINLINE Tensor(const Shape<dimension> &shape) : shape(shape), stream(NULL) {}
  /*! \brief constructor from data pointer and shape  */
  MSHADOW_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape)
      : dptr(dptr), shape(shape), stream(NULL) {}
  /*!
   * \brief return size of i-th dimension, start counting from highest dimension
   * \param the dimension count from the highest dimensin
   * \return the size
   */
  MSHADOW_XINLINE index_t size(index_t i) const {
    return shape[i];
  }
  /*!
   * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
   * \return tensor after flatten
   */
  MSHADOW_XINLINE Tensor<Device, 2, DType> FlatTo2D(void) const {
    return Tensor<Device, 2, DType>(dptr, shape.FlatTo2D());
  }
  /*!
   * \brief get a element of dimension - 1
   * \param idx index
   * \return the result tensor
   */
  MSHADOW_XINLINE Tensor<Device, kSubdim, DType> operator[](index_t idx) const {
    Shape<kSubdim> s = shape.SubShape();
    return Tensor<Device, kSubdim, DType>(dptr + s.MSize() * idx, s);
  }
  /*!
   * \brief slice the tensor in highest dimension [begin,end)
   * \param begin begin position of slice
   * \param end end position of slice
   * \return tensor after slice
   */
  MSHADOW_XINLINE Tensor<Device, dimension, DType> Slice(index_t begin, index_t end) const {
    Shape<dimension> s = this->shape;
    s[0] = end - begin;
    return Tensor<Device, dimension, DType>(dptr + s.SubShape().MSize() * begin, s);
  }
  /*!\brief functions to fit expression template */
  inline Tensor<Device, dimension, DType> &operator=(default_real_t s) {
    return this->__assign(s);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &operator=(const expr::Exp<E,expr::type::kMapper> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &operator=(const expr::Exp<E,expr::type::kChainer> &exp) {
    return this->__assign(exp);
  }
  /*!\brief functions to fit expression template */
  template<typename E>
  inline Tensor<Device, dimension, DType> &operator=(const expr::Exp<E,expr::type::kComplex> &exp) {
    return this->__assign(exp);
  }
};
/*
 *  respecialized class Tensor1D, thei is due to different implementation in operator[]
 */
template<typename Device, typename DType>
struct Tensor<Device, 1, DType>: public expr::RValueExp<Tensor<Device, 1, DType>, DType> {
 public:
  DType *dptr;
  Shape<1> shape;
  Stream<Device> *stream;
  // constructor
  MSHADOW_XINLINE Tensor(void) : stream(NULL) {}
  MSHADOW_XINLINE Tensor(const Shape<1> &shape): shape(shape), stream(NULL) {}
  MSHADOW_XINLINE Tensor(DType *dptr, Shape<1> shape) 
      : dptr(dptr), shape(shape), stream(NULL) {}  
  MSHADOW_XINLINE Tensor<Device, 2> FlatTo2D(void) const {
    return Tensor<Device, 2>(dptr, shape.FlatTo2D());
  }
  MSHADOW_XINLINE Tensor<Device, 1> Slice(index_t begin, index_t end) const {
    Shape<1> s;
    s[0] = s.stride_ = end  - begin;
    return Tensor<Device, 1>(dptr + begin, s);
  }
  MSHADOW_XINLINE index_t size(index_t i) const {
    return shape[0];
  }
  MSHADOW_XINLINE DType &operator[](index_t idx) { return dptr[idx]; }
  MSHADOW_XINLINE const DType &operator[](index_t idx)const { return dptr[idx]; }
  // functions to fit expression template
  inline Tensor<Device, 1, DType> &operator=(double s) {
    return this->__assign(s);
  }
  template<typename E>
  inline Tensor<Device, 1, DType> &operator=(const expr::Exp<E,expr::type::kMapper> &exp) {
    return this->__assign(exp);
  }
  template<typename E>
  inline Tensor<Device, 1, DType> &operator=(const expr::Exp<E,expr::type::kChainer> &exp) {
    return this->__assign(exp);
  }
  template<typename E>
  inline Tensor<Device, 1, DType> &operator=(const expr::Exp<E,expr::type::kComplex> &exp) {
    return this->__assign(exp);
  }
};
//------------------------
// Function Declarations
//-----------------------
/*!
 * \brief initialize tensor engine, used to call intialization functions of dependent libs
 *        this function should be called before all GPU tensor operations,
 *        for using tensors in CPU, this call is actually not needed
 * \param device_id GPU device id to be choosed
 */
inline void InitTensorEngine(int device_id = 0);
/*!
 * \brief Shutdown tensor engine,
 *        this function should be called after all GPU tensor operations,
 *        for using tensors in CPU, this call is actually not needed
 */inline void ShutdownTensorEngine(void);

/*!
 * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
 *        this function is responsible to set the stride_ in each obj.shape
 * \param obj the tensor object, with shape specified
 * \param pad whether padding dimension 0, to make last dimension aligned,
 *            padding may help improve efficiency of matrix multiplications
 *            if true, will allocate space with stride_ that may not equals shape[0]
 *            if false, will allocate continuous space
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void AllocSpace(Tensor<cpu, dim, DType> &obj, bool pad = MSHADOW_ALLOC_PAD);
/*! \brief refer to comment of cpu ver \sa AllocSpace */
template<int dim, typename DType>
inline void AllocSpace(Tensor<gpu, dim, DType> &obj, bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
 * \param obj the tensor object
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> &obj);
/*! \brief refer to comment of cpu ver \sa FreeSpace */
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> &obj);
/*!
 * \brief CPU/GPU: short cut to allocate and initialize a Tensor
 * \param shape: shape of tensor
 * \param initv: initialization value
 * \param pad : padding option
 * \tparam Device device of tensor
 * \tparam DType type of element in tensor
 * \tparam dim dimention of tensor
 * \sa AllocSpace
 */
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType> NewTensor(const Shape<dim> &shape, DType initv, bool pad = MSHADOW_ALLOC_PAD);
/*!
 * \brief copy data from one tensor to another, with same shape
 * \param dst target tensor
 * \param src source tensor
 * \tparam dim specify the dim of tensor
 * \tparam DType type of element in tensor
 */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst, const Tensor<cpu, dim, DType> &src);
/*! \brief refer to comment of cpu ver \sa Copy */
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst, const Tensor<gpu, dim, DType> &src);
/*! \brief refer to comment of cpu ver \sa Copy */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst, const Tensor<cpu, dim, DType> &src);
/*! \brief refer to comment of cpu ver \sa Copy */
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst, const Tensor<gpu, dim, DType> &src);
/*!
 * \brief CPU/GPU: normalize softmax: dst[i][j] = exp(energy[i][j]) /(sum_j exp(energy[i][j]))
 * \param dst destination
 * \param energy input energy
 */
inline void Softmax(Tensor<cpu, 2> dst, const Tensor<cpu, 2> &energy);
/*! \brief refer to comment of cpu ver \sa Softmax */
inline void Softmax(Tensor<gpu, 2> dst, const Tensor<gpu, 2> &energy);
// function declarations to support expression, no need to understand them
// these functions do not need to be directly used
/*!
 * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
 * \tparam Saver specify storage method
 * \tparam R specifies the storage type of the tensor
 * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
 */
template<typename Saver, typename R, int dim, typename DType, typename E, int etype>
inline void MapExp(TRValue<R, cpu, dim, DType> dst, const expr::Exp<E, etype> &exp);
/*! \brief refer to comment of cpu ver \sa MapExp */
template<typename Saver, typename R, int dim, typename DType, typename E, int etype>
inline void MapExp(TRValue<R, gpu, dim, DType> dst, const expr::Exp<E, etype> &exp);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, cpu, 1, DType> dst, const expr::Exp<E, etype> &exp, DType scale = 1);
/*! \brief refer to comment of cpu ver \sa MapReduceKeepLowest */
template<typename Saver, typename Reducer, typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, gpu, 1, DType> dst, const expr::Exp<E, etype> &exp, DType scale = 1);
/*!
 * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
 * \tparam Saver specify storage method
 * \tparam Reducer specify a reducer method
 * \tparam R specifies the storage type of the tensor
 * \tparam DType the type of elements in the tensor
 * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
 * \tparam E specifies the expression type, not need to specify this parameter during usage
 * \tparam etype expression type
 * \param dst destination
 * \param exp expression
 * \param scale scale the result before save
 * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
 */
template<typename Saver, typename Reducer, int dimkeep, typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, cpu, 1, DType> dst, const expr::Exp<E, etype> &exp, DType scale = 1);
/*! \brief refer to comment of cpu ver \sa MapReduceKeepHighDim */
template<typename Saver, typename Reducer, int dimkeep, typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, gpu, 1, DType> dst, const expr::Exp<E, etype> &exp, DType scale = 1);
}  // namespace mshadow
#endif // TENSOR_H
