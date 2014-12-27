#ifndef MSHADOW_TENSOR_EXPR_EXT_H
#define MSHADOW_TENSOR_EXPR_EXT_H
/*!
 * \file tensor_expr_ext.h
 * \brief some extension of expressions, used to support something beyond elementwise op
 * \author Tianqi Chen, Bing Xu
 */
#include "tensor_expr_engine-inl.hpp"
namespace mshadow{
    // Declaration of expressions goes here
    namespace expr{


        /*!
         * \brief crop expression, cut off the boundary region, reverse operation of padding
         * \tparam SrcExp source expression to be pooled from
         * \tparam srcdim dimension of src
         */
        template<typename SrcExp, int srcdim>
        struct CroppingExp : public MakeTensorExp< CroppingExp<SrcExp, srcdim>, SrcExp, srcdim> {
            /*! \brief source operand */
            const SrcExp& src_;
            /*! \brief pad height */
            index_t pad_height_;
            /*! \brief pad height */
            index_t pad_width_;
            /*! \brief src height */
            index_t src_height_;
            /*! \brief constructor */
            CroppingExp(const SrcExp &src, Shape<2> cshape): src_(src) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check(src_);
                utils::Assert(this->shape_[1] >= cshape[1], "CroppingExp: height requirement not met");
                utils::Assert(this->shape_[0] >= cshape[0], "CroppingExp: width requirement not met");
                pad_height_ = (this->shape_[1] - cshape[1]) / 2;
                pad_width_ = (this->shape_[0] - cshape[0]) / 2;
                src_height_ = this->shape_[1];
                this->shape_[1] = cshape[1]; // width
                this->shape_[0] = cshape[0]; // height
            }
            /*! \brief constructor */
            CroppingExp(const SrcExp &src, Shape<2> cshape, index_t start_height, index_t start_width )
                : src_(src), pad_height_(start_height), pad_width_(start_width) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check(src_);
                utils::Assert(this->shape_[1] >= cshape[1]+start_height, "CroppingExp: height requirement not met");
                utils::Assert(this->shape_[0] >= cshape[0]+start_width, "CroppingExp: width requirement not met");
                src_height_ = this->shape_[1];
                this->shape_[1] = cshape[1]; // width
                this->shape_[0] = cshape[0]; // height
            }

        }; // struct CroppingExp


        /*!
         * \brief mirror expression, mirror a image in width
         * \tparam SrcExp source expression to be mirrored
         * \tparam srcdim dimension of src
         */
        template<typename SrcExp, int srcdim>
        struct MirroringExp : public MakeTensorExp<MirroringExp<SrcExp, srcdim>, SrcExp, srcdim> {
            /*! \brief source operand */
            const SrcExp& src_;
            /*! \brief constructor */
            MirroringExp(const SrcExp &src): src_(src) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check(src_);
            }
        };

    }; // namespace expr


    // Declaration of all functions go here
    namespace expr{



        /*!
         * \brief revserse operationg of padding, cut off boundaries, crop output from center of input
         * \param src original image batches
         * \param oshape output shape to be cropped
         * \return expression corresponding to padded result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline CroppingExp<SrcExp, ExpInfo<SrcExp>::kDim> crop(const Exp<SrcExp, etype> &src, Shape<2> oshape) {
             TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return CroppingExp<SrcExp, ExpInfo<SrcExp>::kDim>(src.self(), oshape);
         }
        /*!
         * \brief same as crop, but can specify starting position to do cropping
         * \param src original image batches
         * \param oshape output shape to be cropped
         * \param start_height start height position to do cropping
         * \param start_width  start width position to do cropping
         * \return expression corresponding to padded result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline CroppingExp<SrcExp, ExpInfo<SrcExp>::kDim> crop(const Exp<SrcExp, etype> &src, Shape<2> oshape, index_t start_height, index_t start_width) {
             TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return CroppingExp<SrcExp, ExpInfo<SrcExp>::kDim>(src.self(), oshape, start_height, start_width);
         }

        /*!
         * \brief mirroring expression, mirror images in width
         * \param src original image batches
         * \return expression corresponding to mirrored result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline MirroringExp<SrcExp, ExpInfo<SrcExp>::kDim> mirror(const Exp<SrcExp, etype> &src) {
             TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return MirroringExp<SrcExp, ExpInfo<SrcExp>::kDim>(src.self());
         }




    }; // namespace expr
}; // namespace mshadow

// ==================================================
//  implementations afterwards,
//  no need to read if only use the functions
// --------------------------------------------------
namespace mshadow{
    namespace expr{
    }; // namespace expr

    namespace expr{

    }; // namespace expr

    namespace expr{


    };
    namespace expr{
    };
    
    namespace expr{

    };

    namespace expr{
    }; // namespace expr

    namespace expr{

        template<typename SrcExp, int srcdim>
        struct Plan<CroppingExp<SrcExp, srcdim> > {
        public:
            Plan(const CroppingExp<SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), pad_height_(e.pad_height_),pad_width_(e.pad_width_), 
                  new_height_(e.shape_[1]), src_height_(e.src_height_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                const index_t h = y + pad_height_;
                const index_t w = x + pad_width_;
                return src_.Eval(c * src_height_ + h, w);
            }
        private:
            Plan<SrcExp> src_;
            const index_t pad_height_, pad_width_;
            const index_t new_height_;
            const index_t src_height_;
        };

        template<typename SrcExp, int srcdim>
        struct Plan< MirroringExp<SrcExp, srcdim> > {
        public:
            Plan(const MirroringExp<SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), width_(e.shape_[0]){}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                return src_.Eval(i, width_ - j - 1);
            }
        private:
            Plan<SrcExp> src_;
            const index_t width_;
        };
    }; // namespace expr

    namespace expr{

    };
}; // namespace mshadow



#endif

