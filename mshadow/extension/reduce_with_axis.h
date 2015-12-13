/*!
 * Copyright (c) 2015 by Contributors
 * \file reduce_with_axis.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
#define MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief reduce out the dimension of src labeled by axis.
 *  \tparam Reducer type of reducer
 *  \tparam SrcExp type of source expression
 *  \tparam DType data type
 */
template<typename Reducer, int axis, typename SrcExp, typename DType, int srcdim>
struct ReduceWithAxisExp:
    public MakeTensorExp<ReduceWithAxisExp<Reducer, axis, SrcExp, DType, srcdim>,
                         SrcExp, srcdim-1, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief size of leading dimensions */
  index_t leading_;
  /*! \brief size of trailing dimensions */
  index_t trailing_;
  /*! \brief size of axis dimension */
  index_t size_;
  /*! \brief size of last src dimension */
  index_t last_;
  /*! constructor */
  explicit ReduceWithAxisExp(const SrcExp &src)
    : src_(src) {
    CHECK(srcdim > axis) << "reduce axis out of bound";
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->leading_ = 1;
    for (index_t i = 0; i < axis; ++i) {
      this->leading_ *= src_shape[i];
      this->shape_[i] = src_shape[i];
    }
    this->size_ = src_shape[axis];
    this->trailing_ = 1;
    for (index_t i = axis + 1; i < srcdim; ++i) {
      this->trailing_ *= src_shape[i];
      this->shape_[i-1] = src_shape[i];
    }
    this->last_ = src_shape[srcdim-1];
  }
};  // struct ReduceWithAxisExp

/*!
 * \brief pooling subregion results together
 * \param lhs left oprand
 * \param rhs right oprand
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the content data type
 */
template<typename Reducer, int axis, typename SrcExp, typename DType, int etype>
inline ReduceWithAxisExp<Reducer, axis, SrcExp, DType, ExpInfo<SrcExp>::kDim>
reduce_with_axis(const Exp<SrcExp, DType, etype> &src) {
  return ReduceWithAxisExp<Reducer, axis, SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self());
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, int axis, typename SrcExp, typename DType, int srcdim>
struct Plan<ReduceWithAxisExp<Reducer, axis, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ReduceWithAxisExp<Reducer, axis, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), leading_(e.leading_), trailing_(e.trailing_),
        size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_ + j)/trailing_;
    index_t y = (i*last_ + j)%trailing_;

    DType res; Reducer::SetInitValue(res);
    for (index_t k = 0; k < size_; ++k) {
      index_t z = (x*size_+k)*trailing_+y;
      Reducer::Reduce(res, src_.Eval(z/last_, z%last_));
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t leading_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_REDUCE_WITH_AXIS_H_
