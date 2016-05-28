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

  /*!
  * \brief Broadcasting the tensor in the given axis. If keepdim is off, insert the broadcasting dim after axis. Otherwise broadcasting axis. 
  * \tparam SrcExp source expression
  * \tparam DType  data type
  * \tparam dimsrc source dimension
  * \tparam dimdst destination dimension
  */
template<typename SrcExp, typename DType, int dimsrc, int dimdst>
struct BroadcastWithAxisExp:
    public MakeTensorExp<BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst>,
                         SrcExp, dimdst, DType> {
  /*! \brief data oprand */
  const SrcExp &src_;
  /*! \brief size of the last dimension of dst */
  index_t dst_last_;
  /*! \brief product of the dimensions after the broadcasting axis */
  index_t trailing_;
  /*! \brief new dimension of the broadcasting axis*/
  index_t size_;
  /*! \brief size of the last dimension of src*/
  index_t last_;
  /*! constructor */
  BroadcastWithAxisExp(const SrcExp &src, const int axis, const index_t size)
    : src_(src), size_(size) {
    bool keepdim = (dimsrc == dimdst);
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
    this->trailing_ = 1;

    if (!keepdim) {
      CHECK(dimsrc > axis && axis >= -1) << "broadcast axis (no keepdim) out of bound, "  <<
        "axis must be between -1 and" << dimsrc - 1 << ", given=" << axis << ".";
      for (int i = 0; i <= axis; ++i) {
        this->shape_[i] = src_shape[i];
      }
      this->shape_[axis + 1] = size_;
      for (int i = axis + 1; i < dimsrc; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i + 1] = src_shape[i];
      }
    } else {
      CHECK(dimdst > axis && axis >= 0) << "broadcast axis (keepdim) out of bound, " <<
        "axis must be between 0 and" << dimdst - 1 << ", given=" << axis << ".";
      CHECK_EQ(src_shape[axis], 1) << "Size of the dimension of the broadcasting axis must be 1" <<
        " when keepdim is on, src_shape[" << axis << "]=" << src_shape[axis] << ".";
      for (int i = 0; i <= axis - 1; ++i) {
        this->shape_[i] = src_shape[i];
      }
      this->shape_[axis] = size_;
      for (int i = axis + 1; i < dimdst; ++i) {
        this->trailing_ *= src_shape[i];
        this->shape_[i] = src_shape[i];
      }
    }

    this->last_ = src_shape[dimsrc - 1];
    this->dst_last_ = this->shape_[dimdst - 1];
  }
};  // struct BroadcastWithAxisExp

/*!
 * \brief Broadcasting the tensor after given axis.
 * \param SrcExp source expression
 * \tparam DType data type
 * \tparam etype type of the expression
 */
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
  ExpInfo<SrcExp>::kDim + 1>
broadcast_with_axis(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
    ExpInfo<SrcExp>::kDim + 1>(src.self(), axis, size);
}

/*!
* \brief Broadcasting the tensor in the given axis (keepdim turned on)
* \param SrcExp source expression
* \tparam DType data type
* \tparam etype type of the expression
*/
template<typename SrcExp, typename DType, int etype>
inline BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
  ExpInfo<SrcExp>::kDim>
  broadcast_keepdim(const Exp<SrcExp, DType, etype> &src, const int axis, const index_t size) {
  return BroadcastWithAxisExp<SrcExp, DType, ExpInfo<SrcExp>::kDim,
    ExpInfo<SrcExp>::kDim>(src.self(), axis, size);
}

//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dimsrc, int dimdst>
struct Plan<BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst>, DType> {
 public:
  explicit Plan(const BroadcastWithAxisExp<SrcExp, DType, dimsrc, dimdst> &e)
       : src_(MakePlan(e.src_)), dst_last_(e.dst_last_),
         trailing_(e.trailing_), size_(e.size_), last_(e.last_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t x = (i * dst_last_ + j) / trailing_ / size_;
    index_t y = (i * dst_last_ + j) % trailing_;
    index_t z = x * trailing_ + y;
    return src_.Eval(z / last_, z % last_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t dst_last_, trailing_, size_, last_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_BROADCAST_WITH_AXIS_H_
