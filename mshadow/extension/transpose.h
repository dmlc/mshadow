/*!
 *  Copyright (c) 2016 by Contributors
 * \file transpose.h
 * \brief support for transpose
 * \author Junyuan Xie
 */
#ifndef MSHADOW_EXTENSION_TRANSPOSE_H_
#define MSHADOW_EXTENSION_TRANSPOSE_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief transpose axes of a tensor
 * input: Tensor<Device,dim>: ishape
 * output: Tensor<Device,dimdst> oshape[a1],oshape[a2] = ishape[a2],oshape[a1]
 *
 * \tparam SrcExp type of source expression
 * \tparam DType the type of elements 
 * \tparam dimsrc source dimension, assert a1 > a2
 * \tparam m_a1 one dimension to be swapped, encoded by dimsrc - a1 
 * \tparam a2 second dimension to be swapped, encoded by a2
 */
template<typename SrcExp, typename DType, int dimsrc>
struct TransposeExExp:
      public MakeTensorExp<TransposeExExp<SrcExp, DType, dimsrc>,
                           SrcExp, dimsrc, DType> {
  /*! \brief source expression */
  const SrcExp &src_;
  const Shape<dimsrc> axes_;
  Shape<dimsrc> dst_stride_;
  index_t src_stride_;
  /*! \brief constructor */
  explicit TransposeExExp(const SrcExp &src, Shape<dimsrc> axes) : src_(src), axes_(axes) {
    Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src);
    src_stride_ = src_shape[dimsrc-1];
    Shape<dimsrc> src_stride;
    src_stride[dimsrc-1] = 1;
    for (int i = dimsrc-2; i >= 0; --i) src_stride[i] = src_shape[i+1]*src_stride[i+1];
    for (int i = 0; i < dimsrc; ++i) {
      dst_stride_[i] = src_stride[axes[i]];
      this->shape_[i] = src_shape[axes[i]];
    }
  }
};
/*!
 * \brief a expression that reshapes a tensor to another shape
 * \param src Tensor<Device,dimsrc>:
 * \return a expresion with type Tensor<Device,dimdst>
 * \tparam a1 higher dimension to be swapped, assert a1 > a2
 * \tparam a2 lower dimension to be swapped
 * \tparam SrcExp source expression
 * \tparam DType the type of elements 
 * \tparam etype source expression type
 */
template<typename SrcExp, typename DType, int etype>
inline TransposeExExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
transpose(const Exp<SrcExp, DType, etype> &src, Shape<ExpInfo<SrcExp>::kDim> axes) {
  return TransposeExExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), axes);
}

template<typename SrcExp, typename DType, int dimsrc>
struct Plan<TransposeExExp<SrcExp, DType, dimsrc>, DType> {
 public:
  explicit Plan(const TransposeExExp<SrcExp, DType, dimsrc> &e)
      : src_(MakePlan(e.src_)),
        src_stride_(e.src_stride_),
        dst_stride_(e.dst_stride_),
        dst_shape_(e.shape_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    index_t idx = j*dst_stride_[dimsrc-1];
    #pragma unroll
    for (int k = dimsrc-2; k >= 0; --k) {
      idx += (i%dst_shape_[k])*dst_stride_[k];
      i /= dst_shape_[k];
    }
    return src_.Eval(idx/src_stride_, idx%src_stride_);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t src_stride_;
  const Shape<dimsrc> dst_stride_, dst_shape_;
};

}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_TRANSPOSE_H_
