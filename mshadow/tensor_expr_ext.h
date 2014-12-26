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
         * \brief pooling expression, do reduction over local patches of a image
         * \tparam Reducer reduction method during pooling
         * \tparam SrcExp source expression to be pooled from
         * \tparam srcdim dimension of src
         */
        template<typename Reducer, typename SrcExp, int srcdim>
        struct PoolingExp: public MakeTensorExp< PoolingExp<Reducer, SrcExp,srcdim>, SrcExp, srcdim> {
            /*! \brief source operand */
            const SrcExp& src_;
            /*! \brief kernel size in height */
            index_t ksize_y_;
            /*! \brief kernel size in width */
            index_t ksize_x_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief source height shape[1] */
            index_t src_height_;
            /*! \brief source width shape[0] */
            index_t src_width_;
            /*! \brief constructor */
            PoolingExp(const SrcExp &src, index_t ksize_y, index_t ksize_x, index_t kstride)
                : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
                Shape< srcdim > sshape = ShapeCheck< srcdim,SrcExp>::Check(src_);
                utils::Assert(sshape[0] >= ksize_x && sshape[1] >= ksize_y, "pool: kernel must be smaller than image");
                this->src_height_ = sshape[1];
                this->src_width_  = sshape[0];
                this->shape_ = sshape;
                this->shape_[1] =  (src_height_ - ksize_y) / kstride + 1;                
                this->shape_[0] =  (src_width_  - ksize_x) / kstride + 1;
            }
            /*! \brief constructor, specify shape */
            PoolingExp(const SrcExp &src, Shape<2> pshape, index_t ksize_y, index_t ksize_x, index_t kstride)
                : src_(src), ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
                Shape< srcdim > sshape = ShapeCheck< srcdim,SrcExp>::Check(src_);
                utils::Assert(sshape[0] >= ksize_x && sshape[1] >= ksize_y, "pool: kernel must be smaller than image");
                this->src_height_ = sshape[1];
                this->src_width_  = sshape[0];
                this->shape_    = sshape;
                this->shape_[1] = pshape[1];
                this->shape_[0] = pshape[0];
            } 
        };

        /*!
         * \brief unpooling expr reverse operation of pooling, used to pass gradient back
         * \tparam Reducer specifies reduction operation during pooling
         * \tparam Device which device it lies
         */
        template<typename Reducer, typename Device>
        struct UnPoolingExp: public MakeTensorExp< UnPoolingExp<Reducer, Device>, Tensor<Device,4>, 4> {
            /*! \brief source input, corresponds to src in pooling */
            const Tensor<Device, 4>& data_src_;
            /*! \brief result of pooled data, corresponds to result of pooling */
            const Tensor<Device, 4>& data_pooled_;
            /*! \brief gradient data of pooled part, to be propgate down */
            const Tensor<Device, 4>& grad_pooled_;
            /*! \brief kernel size in height */
            index_t ksize_y_;
            /*! \brief kernel size in width */
            index_t ksize_x_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief constructor */
            UnPoolingExp(const Tensor<Device,4> &data_src,  const Tensor<Device,4> &data_pooled,
                          const Tensor<Device,4> &grad_pooled, index_t ksize_y, index_t ksize_x, index_t kstride)
                : data_src_(data_src), data_pooled_(data_pooled), grad_pooled_(grad_pooled),
                  ksize_y_(ksize_y), ksize_x_(ksize_x), kstride_(kstride) {
                utils::Assert(grad_pooled.shape == data_pooled.shape, "UnPoolingExp: pooled shape mismatch");
                utils::Assert(grad_pooled.shape[2] == data_src.shape[2], "UnPoolingExp: pool and src shape mismatch");
                utils::Assert(grad_pooled.shape[3] == data_src.shape[3], "UnPoolingExp: pool and src shape mismatch");
                this->shape_ = data_src_.shape;
            }
        };

        /*!
         * \brief padding expression, pad a image with zeros
         * \tparam SrcExp source expression to be pooled from
         * \tparam srcdim dimension of src
         */
        template<typename SrcExp, int srcdim>
        struct PaddingExp : public MakeTensorExp<PaddingExp<SrcExp, srcdim>, SrcExp, srcdim> {
            /*! \brief source operand */
            const SrcExp& src_;
            /*! \brief pad size in y */
            index_t pad_y_;
            /*! \brief pad size in x */
            index_t pad_x_;
            /*! \brief source tensor height */
            index_t src_height_;
            /*! \brief source tensor width */
            index_t src_width_;
            /*! \brief constructor */
            PaddingExp(const SrcExp &src, index_t pad_y, index_t pad_x)
                : src_(src), pad_y_(pad_y), pad_x_(pad_x) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check(src_);
                src_height_ = this->shape_[1];
                src_width_  = this->shape_[0];
                this->shape_[1] += pad_y * 2; // height
                this->shape_[0] += pad_x * 2; // width
            }
        };

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

        /*!
         * \brief channel pooling expression, do reduction over (local nearby) channels, used to implement local response normalization
         * \tparam Reducer reduction method during pooling
         * \tparam SrcExp source expression to be pooled from
         * \tparam srcdim dimension of src
         */
        template<typename Reducer, typename SrcExp, int srcdim>
        struct ChannelPoolingExp: public MakeTensorExp< ChannelPoolingExp<Reducer, SrcExp,srcdim>, SrcExp, srcdim> {
            /*! \brief source operand */
            const SrcExp& src_;
            /*! \brief neighbor size */
            index_t nsize_;            
            /*! \brief constructor */
            ChannelPoolingExp(const SrcExp &src, index_t nsize): src_(src), nsize_(nsize){
                utils::Assert(nsize % 2 == 1, "ChannelPoolingExp: local size must be odd, to make it symmetric");
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check(src_);
                utils::Assert(this->shape_[2] >= nsize_, "ChannelPoolingExp: local size need to be smaller than number of channels");
            }
        };
    }; // namespace expr


    // Declaration of all functions go here
    namespace expr{


        /*!
         * \brief pooling subregion results together
         * \param src source image, shape[3]: batch, shape[2]: channel shape[1]: height shape[0]:width
         * \param ksize_y kernel size in height
         * \param ksize_x kernel size in width
         * \param kstride stride for each kernel
         * \return expression of pooled result
         * \tparam Reducer reducer type
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */        
        template<typename Reducer, typename SrcExp, int etype>
        inline PoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim > pool(const Exp<SrcExp,etype> &src, index_t ksize_y, index_t ksize_x, index_t kstride) {
            TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return PoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim >(src.self(), ksize_y, ksize_x, kstride);
        }

        /*! 
         * \brief same as pool, except the output shape is specified by pshape
         * \param src source image
         * \param pshape ouput shape 
         * \param ksize_y kernel size in y
         * \param ksize_x kernel size in x
         * \param kstride stride for each kernel
         * \return expression of pooled result
         * \tparam Reducer reducer type
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
        template<typename Reducer, typename SrcExp, int etype>
        inline PoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim > pool(const Exp<SrcExp,etype> &src, Shape<2> pshape, index_t ksize_y, index_t ksize_x, index_t kstride) {
            TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return PoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim >(src.self(), pshape, ksize_y, ksize_x, kstride);
        }

        /*!
         * \brief unpooling gradient for 4D, backprop gradient value back, revserse operation of pooling, same as unpooling, but allows unequal size of kernel
         * \param data_src  source input, corresponds to src in pooling
         * \param data_pooled result of pooled data, corresponds to result of pooling
         * \param grad_pooled gradient data of pooled part, to be propgate down
         * \param ksize_y kernel height
         * \param ksize_x kernel width
         * \param kstride stride for each kernel
         * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
         * \tparam Reducer reducer type
         * \tparam Device device where data lies
         */
         template<typename Reducer, typename Device>
         inline UnPoolingExp<Reducer, Device> unpool(const Tensor<Device,4>&data_src, const Tensor<Device,4> &data_pooled,
                                                      const Tensor<Device,4> &grad_pooled, index_t ksize_y, index_t ksize_x, index_t kstride) {
             return UnPoolingExp<Reducer, Device>(data_src, data_pooled, grad_pooled, ksize_y, ksize_x, kstride);
         }

        /*!
         * \brief padding expression, pad a image with zeros on boundaries, padding affects shape[0], and shape[1]
         * \param src original image batches
         * \param pad padding size
         * \return expression corresponding to padded result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline PaddingExp<SrcExp, ExpInfo<SrcExp>::kDim> pad(const Exp<SrcExp, etype> &src, index_t pad) {
             TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return PaddingExp<SrcExp, ExpInfo<SrcExp>::kDim>(src.self(), pad, pad);
         }
        
        /*!
         * \brief padding expression, pad a image with zeros on boundaries, padding affects shape[0], and shape[1]
         * \param src original image batches
         * \param pad_y padding size in y
         * \param pad_x padding size in x
         * \return expression corresponding to padded result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline PaddingExp<SrcExp, ExpInfo<SrcExp>::kDim> pad(const Exp<SrcExp, etype> &src, index_t pad_y, index_t pad_x) {
             TypeCheckPass< ExpInfo<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return PaddingExp<SrcExp, ExpInfo<SrcExp>::kDim>(src.self(), pad_y, pad_x);
         }


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

        /*!
         * \brief  channel pooling, do reduction over (local nearby) channels, used to implement local response normalization
         * \param src source data 
         * \param nsize neighbor size 
         * \return expression of pooled result
         * \tparam Reducer reducer type
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
        template<typename Reducer, typename SrcExp, int etype>
        inline ChannelPoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim > chpool(const Exp<SrcExp,etype> &src, index_t nsize) {
            TypeCheckPass< ExpInfo<SrcExp>::kDim >= 3 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return ChannelPoolingExp<Reducer,SrcExp, ExpInfo<SrcExp>::kDim >(src.self(),nsize);
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
        template<typename Reducer, typename SrcExp, int srcdim>
        struct Plan< PoolingExp< Reducer, SrcExp, srcdim> > {
        public:
            Plan(const PoolingExp<Reducer, SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_),
                  kstride_(e.kstride_),
                  src_height_(e.src_height_),src_width_(e.src_width_), new_height_(e.shape_[1]) {
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                using namespace std;
                const index_t py = i % new_height_;
                const index_t y_start = py * kstride_;
                const index_t y_end = min(y_start + ksize_y_, src_height_);
                const index_t px = j;
                const index_t x_start = px * kstride_;
                const index_t x_end = min(x_start + ksize_x_, src_width_);
                const index_t c = i / new_height_;
                
                real_t res = Reducer::kInitV;
                for (index_t y = y_start; y < y_end; ++y) {
                    for (index_t x = x_start; x < x_end; ++x) {
                        Reducer::Reduce(res, src_.Eval(c*src_height_+y, x));
                    }
                }
                return res;
            }
        private:
            Plan<SrcExp> src_;
            const index_t ksize_y_, ksize_x_, kstride_;
            const index_t src_height_, src_width_;
            const index_t new_height_;
        };

        template<typename Reducer, typename Device>
        struct Plan<UnPoolingExp<Reducer, Device> > {
        public:
            Plan(const UnPoolingExp<Reducer, Device> &e)
                : data_src_(e.data_src_), data_pooled_(e.data_pooled_), grad_pooled_(e.grad_pooled_),
                  ksize_y_(e.ksize_y_), ksize_x_(e.ksize_x_), kstride_(e.kstride_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                using namespace std;
                const index_t x = j;
                const index_t y = i % data_src_.shape[1];
                const index_t c = i / data_src_.shape[1];
                const real_t vsrc = data_src_[0][c][y][x];

                const index_t py_min = y < ksize_y_ ? 0 : (y-ksize_y_+kstride_)/kstride_;
                const index_t px_min = x < ksize_x_ ? 0 : (x-ksize_x_+kstride_)/kstride_;
                const index_t py_max = min((y+kstride_)/kstride_, data_pooled_.shape[1]);
                const index_t px_max = min((x+kstride_)/kstride_, data_pooled_.shape[0]);

                real_t val = 0;
                for(index_t py = py_min; py < py_max; ++py){
                    for(index_t px = px_min; px < px_max; ++px){
                        val += Reducer::PartialGrad(vsrc, data_pooled_[0][c][py][px]) * grad_pooled_[0][c][py][px];
                    }
                }
                return val;
            }
        private:
            Tensor<Device, 4> data_src_, data_pooled_, grad_pooled_;
            const index_t ksize_y_, ksize_x_;
            const index_t kstride_;
        };
    }; // namespace expr

    namespace expr{
        template<typename SrcExp, int srcdim>
        struct Plan< PaddingExp<SrcExp, srcdim> > {
        public:
            Plan(const PaddingExp<SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), pad_y_(e.pad_y_), pad_x_(e.pad_x_), 
                  new_height_(e.shape_[1]),
                  src_height_(e.src_height_), src_width_(e.src_width_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                if (y < pad_y_ || x < pad_x_) return 0.0f;
                const index_t h = y - pad_y_;
                const index_t w = x - pad_x_;
                if (h < src_height_ && w < src_width_) {
                    return src_.Eval(c * src_height_ + h, w);
                } else {
                    return 0.0f;
                }
            }
        private:
            Plan<SrcExp> src_;
            const index_t pad_y_;
            const index_t pad_x_;
            const index_t new_height_;
            const index_t src_height_;
            const index_t src_width_;
        };

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
        template<typename Reducer, typename SrcExp, int srcdim>
        struct Plan< ChannelPoolingExp< Reducer, SrcExp, srcdim> > {
        public:
            Plan(const ChannelPoolingExp<Reducer, SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), channel_(e.shape_[2]),
                  height_(e.shape_[1]),width_(e.shape_[0]), hnsize_(e.nsize_/2){
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                using namespace std;
                const index_t y = i % height_;
                i /= height_;
                const index_t c = i % channel_;
                const index_t n = i / channel_;
                const index_t x = j;
                const index_t cstart = c < hnsize_ ? 0  : c - hnsize_;
                const index_t cend   = min(c + hnsize_ + 1, channel_);
                real_t res = Reducer::kInitV;
                for(index_t cc = cstart; cc < cend; ++ cc){
                    Reducer::Reduce(res, src_.Eval((n*channel_+cc)*height_ + y, x));
                }
                return res;
            }
        private:
            Plan<SrcExp> src_;
            const index_t channel_, height_, width_, hnsize_;
        };
    };
}; // namespace mshadow



#endif

