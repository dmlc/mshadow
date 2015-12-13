/*!
 * Copyright (c) 2015 by Contributors
 * \file tensor_dot.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_TENSOR_DOT_H_
#define MSHADOW_EXTENSION_TENSOR_DOT_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Do dot product long the second dimension of two
 *         3 dimensional tensors. Output a dimensional tensor
 *  \tparam LhsExp type of left expression
 *  \tparam RhsExp type of right expression
 *  \tparam DType data type
 */
template<typename LhsExp, typename RhsExp, typename DType>
struct TensorDotExp: public Exp<TensorDotExp<LhsExp, RhsExp, DType>,
                                DType, type::kChainer> {
  /*! \brief left oprand */
  const LhsExp &lhs_;
  /*! \brief right oprand */
  const RhsExp &rhs_;
  /*! \brief shape of output*/
  Shape<2> shape_;
  /*! \brief size of middle dimension */
  index_t size_;
  /*! constructor */
  TensorDotExp(const LhsExp &lhs, const RhsExp &rhs)
    : lhs_(lhs), rhs_(rhs) {
    Shape<3> lshape = ShapeCheck<3, LhsExp>::Check(lhs_);
    Shape<3> rshape = ShapeCheck<3, RhsExp>::Check(rhs_);
    CHECK_EQ(lshape, rshape) << "Shape of two oprand must be the same.";
    this->shape_ = Shape2(lshape[0], lshape[2]);
    this->size_ = lshape[1];
  }
};  // struct TensorDotExp

/*!
 * \brief pooling subregion results together
 * \param lhs left oprand
 * \param rhs right oprand
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the content data type
 */
template<typename LhsExp, typename RhsExp, typename DType, int letype, int retype>
inline TensorDotExp<LhsExp, RhsExp, DType>
tensor_dot(const Exp<LhsExp, DType, letype> &lhs,
           const Exp<RhsExp, DType, retype> &rhs) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  TypeCheckPass<ExpInfo<RhsExp>::kDim == 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return TensorDotExp<LhsExp, RhsExp, DType>(lhs.self(), rhs.self());
}
//----------------------
// Execution plan
//----------------------
template<typename LhsExp, typename RhsExp, typename DType>
struct Plan<TensorDotExp<LhsExp, RhsExp, DType>, DType> {
 public:
  explicit Plan(const TensorDotExp<LhsExp, RhsExp, DType> &e)
      : lhs_(MakePlan(e.lhs_)), rhs_(MakePlan(e.rhs_)), size_(e.size_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    DType res = 0;
    for (index_t k = 0; k < size_; ++k) {
      res += lhs_.Eval(i+k*size_, j) *
             rhs_.Eval(i+k*size_, j);
    }
    return res;
  }

 private:
  Plan<LhsExp, DType> lhs_;
  Plan<RhsExp, DType> rhs_;
  const index_t size_;
};

template<typename LhsExp, typename RhsExp, typename DType>
inline Plan<TensorDotExp<LhsExp, RhsExp, DType>, DType>
MakePlan(const TensorDotExp<LhsExp, RhsExp, DType> &exp) {
  return Plan<TensorDotExp<LhsExp, RhsExp, DType>, DType>(exp);
}

template<int dim, typename LhsExp, typename RhsExp, typename DType>
struct ShapeCheck<dim, TensorDotExp<LhsExp, RhsExp, DType> > {
  inline static Shape<dim>
  Check(const TensorDotExp<LhsExp, RhsExp, DType> &t) {
    CHECK(dim == 2)
      << "TakeExp only support 2D output";
    ShapeCheck<3, LhsExp>::Check(t.lhs_);
    ShapeCheck<3, RhsExp>::Check(t.rhs_);
    return t.shape_;
  }
};

template<typename LhsExp, typename RhsExp, typename DType>
struct ExpInfo<TensorDotExp<LhsExp, RhsExp, DType> > {
  static const int kDim = 2;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask;
};
}  //namespace expr
}  //namespace mshadow
#endif  // MSHADOW_EXTENSION_TENSOR_DOT_H_
