/*!
 *  Copyright (c) 2014 by Contributors
 * \file pack_col2patch.h
 * \brief support for pack
 * \author Tianqi Chen
 */
#ifndef MSHADOW_EXTENSION_PACK_COL2PATCH_H_
#define MSHADOW_EXTENSION_PACK_COL2PATCH_H_
#include <algorithm>
#include "../extension.h"
namespace mshadow {
namespace expr {
/*!
 * \brief reverse operation of UnpackPatchToCol,
 *    used to backprop gradient back
 *    this is a version supporting multiple images
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam dstdim destination dimension
 */
template<typename SrcExp, typename DType, int dstdim>
struct PackColToPatchXExp:
      public MakeTensorExp<PackColToPatchXExp<SrcExp, DType, dstdim>,
                           SrcExp, dstdim, DType> {
  /*! \brief source operand */
  const SrcExp &src_;
  /*! \brief patch height */
  index_t psize_y_;
  /*! \brief patch height */
  index_t psize_x_;
  /*! \brief patch stride */
  index_t pstride_;
  /*! \brief constructor */
  PackColToPatchXExp(const SrcExp &src, Shape<dstdim> imshape,
                     index_t psize_y, index_t psize_x, index_t pstride)
      :src_(src), psize_y_(psize_y), psize_x_(psize_x), pstride_(pstride){
    this->shape_ = imshape;
    const index_t o_height = (imshape[dstdim - 2] - psize_y) / pstride + 1;
    const index_t o_width  = (imshape[dstdim - 1] - psize_x) / pstride + 1;
    Shape<2> sshape = ShapeCheck<2, SrcExp>::Check(src_);
    utils::Check(sshape[1] == o_height * o_width *
                 imshape.ProdShape(0, dstdim - 3),
                 "PackColToPatchExp: src.size(1) mismatch");
    utils::Check(sshape[0] == psize_y * psize_x * imshape[dstdim - 3],
                 "PackColToPatchExp: src.size(0) mismatch");
  }
};
/*!
 * \brief reverse operation of pack_col2patch, can be used to implement deconvolution
 * \return packed img expression
 * \param mat source matrix
 * \param imshape shape of target img
 * \param psize_y height of each patch
 * \param psize_x height of each patch
 * \param pstride stride of each patch
 * \tparam SrcExp source expression
 * \tparam DType the type of elements
 * \tparam dstdim destination dimension
 * \tparam etype type of expression
 */
template<typename SrcExp, typename DType, int dstdim, int etype>
inline PackColToPatchXExp<SrcExp, DType, dstdim>
pack_col2patch(const expr::Exp<SrcExp, DType, etype> &src,
               Shape<dstdim> imshape, index_t psize_y,
               index_t psize_x, index_t pstride) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim == 2>
      ::Error_Expression_Does_Not_Meet_Dimension_Req();
  utils::Check(imshape[dstdim - 1] >= psize_x &&
               imshape[dstdim - 2] >= psize_y,
               "PackColToPatch:image shape smaller than patch size");
  return PackColToPatchXExp<SrcExp, DType, dstdim>(src.self(), imshape,
                                                   psize_y, psize_x, pstride);
}
//----------------------
// Execution plan
//----------------------
template<typename SrcExp, typename DType, int dstdim>
struct Plan<PackColToPatchXExp<SrcExp, DType, dstdim>, DType> {
 public:
  explicit Plan(const PackColToPatchXExp<SrcExp, DType, dstdim> &e)
      :src_(MakePlan(e.src_)), psize_y_(e.psize_y_),
       psize_x_(e.psize_x_), pstride_(e.pstride_),
       i_channel_(e.shape_[dstdim - 3]), i_height_(e.shape_[dstdim - 2]),
       o_height_((e.shape_[dstdim - 2]  - psize_y_) / pstride_ + 1),
       o_width_((e.shape_[dstdim - 1]  - psize_x_) / pstride_ + 1) {
    // note: i/o convention are same as unpack
  }
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    using namespace std;
    const index_t y = i % i_height_;
    const index_t idivh = i / i_height_;
    const index_t c = idivh % i_channel_;
    const index_t n = idivh / i_channel_;
    const index_t x = j;
    const index_t py_min =
        y < psize_y_ ? 0 : (y-psize_y_ + pstride_) / pstride_;
    const index_t px_min =
        x < psize_x_ ? 0 : (x-psize_x_ + pstride_) / pstride_;
    const index_t py_max = min((y + pstride_) / pstride_, o_height_);
    const index_t px_max = min((x + pstride_) / pstride_, o_width_);
    DType res = static_cast<DType>(0);
    for (index_t py = py_min; py < py_max; ++py) {
      for (index_t px = px_min; px < px_max; ++px) {
        res += src_.Eval(((c * psize_y_ + y - py*pstride_) * psize_x_ +
                          x - px * pstride_),
                         (n * o_height_ + py) * o_width_ + px);
      }
    }
    return res;
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t psize_y_, psize_x_, pstride_, i_channel_;
  const index_t i_height_, o_height_, o_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_PACK_COL2PATCH_H_
