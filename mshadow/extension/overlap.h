/*!
 * Copyright (c) 2015 by Contributors
 * \file overlap.h
 * \brief Judge bounding box overlap
 *        ref: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/utils/bbox.pyx
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_OVERLAP_H_
#define MSHADOW_EXTENSION_OVERLAP_H_
#include <algorithm>
#include "../extension.h"

namespace mshadow {
namespace expr {

/*!
 * \brief 2D overlap expression
 * \tparam SrcExp source expression to be calculated
 * \tparam DType data type
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct OverlapExp: public MakeTensorExp<OverlapExp<SrcExp, DType, srcdim>,
                                        SrcExp, srcdim, DType> {
  /*! \brief boxes */
  const SrcExp &lhs_;
  /*! \brief query_boxes */
  const SrcExp &rhs_;
  /*! \brief constructor */
  explicit OverlapExp(const SrcExp &lhs, const SrcExp &rhs)
    : lhs_(lhs), rhs_(rhs) {
    // lhs shape: (N, 4)
    // rhs shape: (K, 4)
    // output : (N, K)
    CHECK_EQ(srcdim, 2) << "Input must be 2D Tensor";
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(lhs_);
    Shape<2> rhs_shape = ShapeCheck<srcdim, SrcExp>::Check(rhs_);
    CHECK_EQ(this->shape_[1], 4) << "boxes must be in shape (N, 4)";
    CHECK_EQ(rhs_shape[1], 4) << "query box must be in shape (K, 4)";
    this->shape_[1] = rhs_shape[0];
  }
};  // struct OverlapExp

/*!
 * \brief calcuate overlaps between boxes and query boxes
 * \param lhs boxes
 * \param rhs query boxes
 * \tparam SrcExp source expression
 * \tparam DType data type
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int etype>
inline OverlapExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
bbox_overlaps(const Exp<SrcExp, DType, etype> &lhs,
             const Exp<SrcExp, DType, etype> &rhs) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2>::Error_Expression_Does_Not_Meet_Dimension_Req();
  return OverlapExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
    (lhs.self(), rhs.self());
}

//----------------------
// Execution plan
//----------------------

template<typename SrcExp, typename DType, int srcdim>
struct Plan<OverlapExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const OverlapExp<SrcExp, DType, srcdim> &e)
    : lhs_(MakePlan(e.lhs_)),
      rhs_(MakePlan(e.rhs_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    DType box_area =
      (rhs_.Eval(j, 2) - rhs_.Eval(j, 0) + 1) *
      (rhs_.Eval(j, 3) - rhs_.Eval(j, 1) + 1);
    DType iw =
      (lhs_.Eval(i, 2) < rhs_.Eval(j, 2) ? lhs_.Eval(i, 2) : rhs_.Eval(j, 2)) -
      (lhs_.Eval(i, 0) > rhs_.Eval(j, 0) ? lhs_.Eval(i, 0) : rhs_.Eval(j, 0)) + 1;
    if (iw < 0.0f) {
      return DType(0.0f);
    } else {
      DType ih =
        (lhs_.Eval(i, 3) < rhs_.Eval(j, 3) ? lhs_.Eval(i, 3) : rhs_.Eval(j, 3)) -
        (lhs_.Eval(i, 1) > rhs_.Eval(j, 1) ? lhs_.Eval(i, 1) : rhs_.Eval(j, 1)) + 1;
      if (ih < 0.0f) {
        return DType(0.0f);
      } else {
        DType ua =
          (lhs_.Eval(i, 2) - lhs_.Eval(i, 0) + 1) *
          (lhs_.Eval(i, 3) - lhs_.Eval(i, 1) + 1) +
          box_area - iw * ih;
        return DType(iw * ih / ua);
      }
    }
  }

 private:
  Plan<SrcExp, DType> lhs_;
  Plan<SrcExp, DType> rhs_;
};  // struct Plan

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_OVERLAP_H_
