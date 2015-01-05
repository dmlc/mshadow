#ifndef MSHADOW_EXTENSION_CONCAT_H_
#define MSHADOW_EXTENSION_CONCAT_H_

#include "mshadow/extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief concat expression, concat two tensor's channel
 * \tparam LhsExp left expression
 * \tparam RhsExp right expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
struct ConcatExp : public TRValue<ConcatExp<LhsExp, RhsExp,
                                            Device, DType, srcdim>,
                                  Device, srcdim, DType> {
  const LhsExp &src1_;
  const RhsExp &src2_;
  index_t height_;
  index_t width_;
  index_t ch_src1_;
  index_t ch_src2_;
  Shape<4> shape_;
  ConcatExp(const LhsExp &src1, const RhsExp &src2) : src1_(src1), src2_(src2) {
    Shape<srcdim> sshape1 = ShapeCheck<srcdim, LhsExp>::Check(src1_);
    Shape<srcdim> sshape2 = ShapeCheck<srcdim, RhsExp>::Check(src2_);
    utils::Check(sshape1[srcdim - 2] == sshape2[srcdim - 2],
                 "ConcatExp: height requirement not met");
    utils::Check(sshape1[srcdim - 1] == sshape2[srcdim - 1],
                 "ConcatExp: width requirement not met");
    utils::Check(sshape1[0] == sshape2[0],
                 "ConcatExp: batch requirement not met");
    this->shape_ = sshape1;
    this->shape_[1] = sshape1[1] + sshape2[1];
    this->ch_src1_ = sshape1[1];
    this->ch_src2_ = sshape2[1];
    this->height_ = sshape1[2];
    this->width_ = sshape1[3];
  }
  template<typename E, int etype>
  inline void
  operator=(const expr::Exp<E, DType, etype> &exp) {
    this->__assign(exp);
  }
  inline void
  operator=(const DType &exp) {
    this->__assign(exp);
  }
}; // struct ConcatExp
/*!
 * \brief concat two 4D tensor
 * \param src1 source tensor1
 * \param src2 source tensor2
 * \return concated 4D tensor
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
inline ConcatExp<LhsExp, RhsExp, Device, DType, ExpInfo<LhsExp>::kDim>
concat(const TRValue<LhsExp, Device, srcdim, DType> &src1,
       const TRValue<RhsExp, Device, srcdim, DType> &src2) {
  TypeCheckPass<ExpInfo<LhsExp>::kDim == 4>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  TypeCheckPass<ExpInfo<RhsExp>::kDim == 4>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ConcatExp<LhsExp, RhsExp, Device, DType, ExpInfo<LhsExp>::kDim>
      (src1.self(), src2.self());
}
//------------------------
//  engine plugin
//------------------------
// runtime shapecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
struct ShapeCheck<srcdim, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> >{
  inline static Shape<srcdim> Check(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> &t) {
    return t.shape_;
  }
};
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
struct StreamInfo<Device, ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> >{
  inline static Stream<Device> *Get(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> &t) {
    Stream<Device> *lhs = StreamInfo<Device, LhsExp>::Get(t.src1_);
    Stream<Device> *rhs = StreamInfo<Device, RhsExp>::Get(t.src2_);
    if (lhs != rhs) return NULL;
    return lhs;
  }
};
// static typecheck
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
struct ExpInfo<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> >{
  static const int kDimLhs = ExpInfo<LhsExp>::kDim;
  static const int kDimRhs = ExpInfo<RhsExp>::kDim;
  // copy from binarymap
  static const int kDim = (kDimLhs >= 0 && kDimRhs >= 0) ?\
      (kDimLhs == 0 ?\
       kDimRhs :\
       ((kDimRhs == 0 || kDimLhs == kDimRhs) ? kDimLhs : -1)) : -1;
  static const int kDevMask = ExpInfo<LhsExp>::kDevMask & ExpInfo<RhsExp>::kDevMask;
};
//----------------------
// Execution plan
//---------------------
template<typename LhsExp, typename RhsExp,
         typename Device, typename DType, int srcdim>
struct Plan<ConcatExp<LhsExp, RhsExp, Device, DType, srcdim>, DType> {
 public:
  explicit Plan(const ConcatExp<LhsExp, RhsExp, Device, DType, srcdim> &e) :
      src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
      height_(e.height_), width_(e.width_),
      ch_src1_(e.ch_src1_), ch_src2_(e.ch_src2_), ch_(e.ch_src1_ + e.ch_src2_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) return src1_.Eval((b * ch_src1_ + c) * height_ + y, x);
    else return src2_.Eval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
  }
  MSHADOW_XINLINE DType &REval(index_t i, index_t j) {
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % ch_;
    const index_t b = i / ch_;
    const index_t x = j;
    if (c < ch_src1_) return src1_.REval((b * ch_src1_ + c) * height_ + y, x);
    else return src2_.REval((b * ch_src2_ + c - ch_src1_) * height_ + y, x);
  }
 private:
  Plan<LhsExp, DType> src1_;
  Plan<RhsExp, DType> src2_;
  const index_t height_, width_, ch_src1_, ch_src2_, ch_;
}; // struct Plan

}// namespace expr
} // namespace mshadow
#endif // MSHADOW_EXTENSION_CONCAT_H_
