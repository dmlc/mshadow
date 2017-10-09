/*!
 *  Copyright (c) 2014 by Contributors
 * \file pad.h
 * \brief support for pad
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_PAD_H_
#define MSHADOW_EXTENSION_PAD_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief padding expression, pad a image with zeros
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename SrcExp, typename DType, int srcdim>
struct PaddingExp:
      public MakeTensorExp<PaddingExp<SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief pad size in y */
  index_t pad_y_;
  /*! \brief pad size in x */
  index_t pad_x_;
  /*! \brief value to pad with */
  index_t value_;
  /*! \brief source tensor height */
  index_t src_height_;
  /*! \brief source tensor width */
  index_t src_width_;
  /*! \brief constructor */
  PaddingExp(const SrcExp &src, index_t pad_y, index_t pad_x, DType value)
      : src_(src), pad_y_(pad_y), pad_x_(pad_x), value_(value) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    src_height_ = this->shape_[srcdim - 2];
    src_width_  = this->shape_[srcdim - 1];
    this->shape_[srcdim - 2] += pad_y * 2;  // height
    this->shape_[srcdim - 1] += pad_x * 2;  // width
  }
};
/*!
 * \brief padding expression, pad an image on boundaries, padding affects shape[0], and shape[1]
 * \param src original image batches
 * \param pad padding size
 * \param value value to pad with
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
pad(const Exp<SrcExp, DType, etype> &src, index_t pad, DType value = static_cast<DType>(0)) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), pad, pad, value);
}
/*!
 * \brief padding expression, pad an image on boundaries, padding affects shape[0], and shape[1]
 * \param src original image batches
 * \param pad_y padding size in y
 * \param pad_x padding size in x
 * \param pad value to pad with
 * \return expression corresponding to padded result
 * \tparam SrcExp source expression
 * \tparam DType the content data type
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int etype>
inline PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
pad(const Exp<SrcExp, DType, etype> &src, index_t pad_y, index_t pad_x, DType value = static_cast<DType>(0)) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return PaddingExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
      (src.self(), pad_y, pad_x, value);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int srcdim>
struct Plan<PaddingExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const PaddingExp<SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)),
        pad_y_(e.pad_y_), pad_x_(e.pad_x_),
        value_(e.value_),
        new_height_(e.shape_[srcdim - 2]),
        src_height_(e.src_height_), src_width_(e.src_width_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    if (y < pad_y_ || x < pad_x_) return value_;
    const index_t h = y - pad_y_;
    const index_t w = x - pad_x_;
    if (h < src_height_ && w < src_width_) {
      return src_.Eval(c * src_height_ + h, w);
    } else {
      return value_;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t pad_y_;
  const index_t pad_x_;
  const DType value_;
  const index_t new_height_;
  const index_t src_height_;
  const index_t src_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_PAD_H_
