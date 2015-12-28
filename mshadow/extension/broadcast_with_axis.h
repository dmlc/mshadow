/*!
 * Copyright (c) 2015 by Contributors
 * \file tensor_dot.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
#define MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_

#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief Backward for tensor dot
 *  \tparam DataExp type of left expression
 *  \tparam TopExp type of right expression
 *  \tparam DType data type
 */
template<typename SrcExp, typename DType, int srcdim>
struct BroadcastWithAxisExp:
    public MakeTensorExp<BroadcastWithAxisExp<SrcExp, DType, srcdim>,
                         SrcExp, srcdim+1, DType> {
  /*! \brief data oprand */
  const SrcExp &src_;
  /*! \brief size of middle dimension */
  index_t leading_;
  /*! \brief size of middle dimension */
  index_t trailing_;
  /*! \brief size of middle dimension */
  index_t size_;
  /*! \brief size of middle dimension */
  index_t last_;
  /*! constructor */
  BroadcastWithAxisExp(const SrcExp &src, const int axis, const index_t size)
    : src_(src), size_(size) {
    CHECK(srcdim > axis) << "broadcast axis out of bound";
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->leading_ = 1;
    for (index_t i = 0; i <= axis; ++i) {
      this->leading_ *= src_shape[i];
      this->shape_[i] = src_shape[i];
    }
    this->shape_[axis+1] = size_;
    this->trailing_ = 1;
    for (index_t i = axis+1; i < srcdim; ++i) {
      this->trailing_ *= src_shape[i];
      this->shape_[i+1] = src_shape[i];
    }
    this->last_ = src_shape[srcdim-1];
  }
};  // struct BroadcastWithAxisExp

/*!
 * \brief pooling subregion results together
 * \param data data oprand
 * \param top top grad oprand
 * \tparam DataExp left expression
 * \tparam TopExp right expression
 * \tparam DType the content data type
 */
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
broadcast_with_axis(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axis, size);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<BroadcastWithAxisExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const BroadcastWithAxisExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), leading_(e.leading_),
        trailing_(e.trailing_), size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i*last_+j)/trailing_/size_;
    index_t y = (i*last_+j)%trailing_;
    index_t z = x*trailing_ + y;
    return src_.Eval(z/last_, z%last_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t leading_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
