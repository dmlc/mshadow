
#ifndef MSHADOW_EXTENSION_CONCAT_H_
#define MSHADOW_EXTENSION_CONCAT_H_

#include "mshadow/extension.h"

namespace mshadow {
  namespace expr {
    /*!
     * \brief concat expression, concat two tensor's channel
      * \tparam SrcExp source expression
      * \tparam DType the type of elements
      * \tparam srcdim dimension of src
      */
    template<typename SrcExp, typename DType, int srcdim>
        struct ConcatExp : public MakeTensorExp<ConcatExp<SrcExp, DType, srcdim>,
        SrcExp, srcdim, DType> {
      const SrcExp &src1_;
      const SrcExp &src2_;
      index_t height_;
      index_t width_;
      index_t ch_src1_;
      index_t ch_src2_;
   ConcatExp(const SrcExp &src1, const SrcExp &src2) : src1_(src1), src2_(src2) {
        Shape<srcdim> sshape1 = ShapeCheck<srcdim, SrcExp>::Check(src1_);
        Shape<srcdim> sshape2 = ShapeCheck<srcdim, SrcExp>::Check(src2_);
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
    template<typename SrcExp, typename DType, int etype>
        inline ConcatExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
        concat(const Exp<SrcExp, DType, etype> &src1, const Exp<SrcExp, DType, etype> &src2) {
      TypeCheckPass<ExpInfo<SrcExp>::kDim == 4>
            ::Error_Expression_Does_Not_Meet_Dimension_Req();
      return ConcatExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src1.self(), src2.self());
    }

    //----------------------
    // Execution plan
    //---------------------
    template<typename SrcExp, typename DType, int srcdim>
        struct Plan<ConcatExp<SrcExp, DType, srcdim>, DType> {
   public:
      explicit Plan(const ConcatExp<SrcExp, DType, srcdim> &e) :
      src1_(MakePlan(e.src1_)), src2_(MakePlan(e.src2_)),
          height_(e.height_), width_(e.width_),
          ch_src1_(e.ch_src1_), ch_src2_(e.ch_src2_), ch_(e.ch_src1_ + e.ch_src2_) {}
      MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
        const index_t y = i % height_;
        i /= height_;
        const index_t c = i % ch_;
        const index_t x = j;
        if (c < ch_src1_) return src1_.Eval(c * height_ + y, x);
        else return src2_.Eval((c - ch_src1_) * height_ + y, x);
      }
   private:
      Plan<SrcExp, DType> src1_;
      Plan<SrcExp, DType> src2_;
      const index_t height_, width_, ch_src1_, ch_src2_, ch_;
    }; // struct Plan

  }// namespace expr
} // namespace mshadow




#endif // MSHADOW_EXTENSION_CONCAT_H_
