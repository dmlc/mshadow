/*!
 * Copyright (c) 2015 by Contributors
 * \file tensor_dot.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_TENSOR_DOT_GRAD_H_
#define MSHADOW_EXTENSION_TENSOR_DOT_GRAD_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Backward for tensor dot
 *  \tparam DataExp type of left expression
 *  \tparam TopExp type of right expression
 *  \tparam DType data type
 */
template<typename DataExp, typename TopExp, typename DType>
struct TensorDotGradExp: public Exp<TensorDotGradExp<DataExp, TopExp, DType>,
                                DType, type::kChainer> {
  /*! \brief data oprand */
  const DataExp &data_;
  /*! \brief top grad oprand */
  const TopExp &top_;
  /*! \brief shape of output */
  Shape<3> shape_;
  /*! \brief size of middle dimension */
  index_t size_;
  /*! constructor */
  TensorDotGradExp(const DataExp &data, const TopExp &top)
    : data_(data), top_(top) {
    Shape<3> dshape = ShapeCheck<3, DataExp>::Check(data_);
    Shape<2> tshape = ShapeCheck<2, TopExp>::Check(top_);
    CHECK_EQ(dshape[0], tshape[0]) << "Shape of two oprand doesn't match";
    CHECK_EQ(dshape[2], tshape[1]) << "Shape of two oprand doesn't match";
    this->shape_ = dshape;
    this->size_ = dshape[1];
  }
};  // struct TensorDotGradExp

/*!
 * \brief pooling subregion results together
 * \param data data oprand
 * \param top top grad oprand
 * \tparam DataExp left expression
 * \tparam TopExp right expression
 * \tparam DType the content data type
 */
template<typename DataExp, typename TopExp, typename DType, int detype, int tetype>
inline TensorDotGradExp<DataExp, TopExp, DType>
tensor_dot_grad(const Exp<DataExp, DType, detype> &data,
           const Exp<TopExp, DType, tetype> &top) {
  TypeCheckPass<ExpInfo<DataExp>::kDim == 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  TypeCheckPass<ExpInfo<TopExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return TensorDotGradExp<DataExp, TopExp, DType>(data.self(), top.self());
}
//----------------------
// Execution plan
//----------------------
template<typename DataExp, typename TopExp, typename DType>
struct Plan<TensorDotGradExp<DataExp, TopExp, DType>, DType> {
 public:
  explicit Plan(const TensorDotGradExp<DataExp, TopExp, DType> &e)
      : data_(MakePlan(e.data_)), top_(MakePlan(e.top_)), size_(e.size_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return data_.Eval(i, j) * top_.Eval(i/size_, j);
  }

 private:
  Plan<DataExp, DType> data_;
  Plan<TopExp, DType> top_;
  const index_t size_;
};

template<typename DataExp, typename TopExp, typename DType>
inline Plan<TensorDotGradExp<DataExp, TopExp, DType>, DType>
MakePlan(const TensorDotGradExp<DataExp, TopExp, DType> &exp) {
  return Plan<TensorDotGradExp<DataExp, TopExp, DType>, DType>(exp);
}

template<int dim, typename DataExp, typename TopExp, typename DType>
struct ShapeCheck<dim, TensorDotGradExp<DataExp, TopExp, DType> > {
  inline static Shape<dim>
  Check(const TensorDotGradExp<DataExp, TopExp, DType> &t) {
    CHECK(dim == 3)
      << "TakeExp only support 3D output";
    ShapeCheck<3, DataExp>::Check(t.data_);
    ShapeCheck<2, TopExp>::Check(t.top_);
    return t.shape_;
  }
};

template<typename DataExp, typename TopExp, typename DType>
struct ExpInfo<TensorDotGradExp<DataExp, TopExp, DType> > {
  static const int kDim = 3;
  static const int kDevMask = ExpInfo<DataExp>::kDevMask;
};
}  //namespace expr
}  //namespace mshadow
#endif  // MSHADOW_EXTENSION_TENSOR_DOT_GRAD_H_
