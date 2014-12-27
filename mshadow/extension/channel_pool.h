/*!
 *  Copyright (c) 2014 by Contributors
 * \file channel_pool.h
 * \brief support for chpool
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_CHANNEL_POOL_H_
#define MSHADOW_EXTENSION_CHANNEL_POOL_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief channel pooling expression, do reduction over (local nearby) channels,
 *        used to implement local response normalization
 * \tparam Reducer reduction method during pooling
 * \tparam SrcExp source expression to be pooled from
 * \tparam DType the type of elements
 * \tparam srcdim dimension of src
 */
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct ChannelPoolingExp:
      public MakeTensorExp<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>,
                           SrcExp, srcdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief neighbor size */
  index_t nsize_;
  /*! \brief constructor */
  ChannelPoolingExp(const SrcExp &src, index_t nsize)
      : src_(src), nsize_(nsize) {
    utils::Check(nsize % 2 == 1,
                 "chpool: local size must be odd");
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    utils::Check(this->shape_[srcdim - 3] >= nsize_,
                 "chpool: local size must be smaller than nchannels");
  }
};
/*!
 * \brief  channel pooling, do reduction over (local nearby) channels,
 *         used to implement local response normalization
 * \param src source data 
 * \param nsize neighbor size 
 * \return expression of pooled result
 * \tparam Reducer reducer type
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam etype type of expression
 */
template<typename Reducer, typename SrcExp, typename DType, int etype>
inline ChannelPoolingExp<Reducer, SrcExp, DType, ExpInfo<SrcExp>::kDim>
chpool(const Exp<SrcExp, DType, etype> &src, index_t nsize) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 3>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return ChannelPoolingExp<Reducer, SrcExp,
                           DType, ExpInfo<SrcExp>::kDim>(src.self(), nsize);
}
//----------------------
// Execution plan
//----------------------
template<typename Reducer, typename SrcExp, typename DType, int srcdim>
struct Plan<ChannelPoolingExp<Reducer, SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ChannelPoolingExp<Reducer, SrcExp, DType, srcdim> &e)
      : src_(MakePlan(e.src_)), channel_(e.shape_[srcdim - 3]),
        height_(e.shape_[srcdim - 2]), width_(e.shape_[srcdim - 1]),
        hnsize_(e.nsize_ / 2) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using std::min;
    const index_t y = i % height_;
    i /= height_;
    const index_t c = i % channel_;
    const index_t n = i / channel_;
    const index_t x = j;
    const index_t cstart = c < hnsize_ ? 0  : c - hnsize_;
    const index_t cend   = min(c + hnsize_ + 1, channel_);
    DType res; Reducer::SetInitValue(res);
    for (index_t cc = cstart; cc < cend; ++cc) {
      Reducer::Reduce(res, src_.Eval((n * channel_ + cc) * height_ + y, x));
    }
    return res;
  }
 private:
  Plan<SrcExp, DType> src_;
  const index_t channel_, height_, width_, hnsize_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CHANNEL_POOL_H_

