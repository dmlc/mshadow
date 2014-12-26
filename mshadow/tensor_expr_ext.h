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
         * \brief unpack local (overlap) patches of image to column of mat, can be used to implement convolution, this expression allow unpack of a batch        
         *  this is a version support unpacking multiple images
         *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
         * \tparam SrcExp source expression
         * \tparam dstdim destination dimension
         */
        template<typename SrcExp, int srcdim>
        struct UnpackPatchToColXExp: public MakeTensorExp< UnpackPatchToColXExp<SrcExp,srcdim>, SrcExp, 2>{
            /*! \brief source operand */
            const SrcExp& img_;
            /*! \brief patch height */
            index_t psize_y_;
            /*! \brief patch width */
            index_t psize_x_;
            /*! \brief patch stride */
            index_t pstride_;
            /*! \brief number of input channel */
            index_t i_channel_;
            /*! \brief height of img */
            index_t i_height_;
            /*! \brief width of img */
            index_t i_width_;            
            /*! \brief constructor */
            UnpackPatchToColXExp(const SrcExp &img, index_t psize_y, index_t psize_x, index_t pstride)
                :img_(img), psize_y_(psize_y), psize_x_(psize_x), pstride_(pstride){
                Shape<srcdim> imshape = ShapeCheck<srcdim,SrcExp>::Check(img_);
                utils::Assert(imshape[0] >= psize_x && imshape[1] >= psize_y, "UnpackPatchToCol:image shape smaller than patch size");
                this->i_channel_ = imshape[2];
                this->i_height_  = imshape[1];
                this->i_width_   = imshape[0];
                // calculate number of batches 
                const index_t num = imshape.ProdShape(3, srcdim);
                const index_t o_height = (i_height_ - psize_y) / pstride + 1;
                const index_t o_width  = (i_width_  - psize_x) / pstride + 1;
                this->shape_[0] = o_height * o_width * num;
                this->shape_[1] = psize_y * psize_x * imshape[2];
            }
        };

        /*!
         * \brief reverse operation of UnpackPatchToCol, used to backprop gradient back
         *    this is a version supporting multiple images
         * \tparam Device which device it lies
         * \tparam dstdim destination dimension
         */
        template<typename Device, int dstdim>
        struct PackColToPatchXExp: public MakeTensorExp< PackColToPatchXExp<Device,dstdim>, Tensor<Device,2>, dstdim>{
            /*! \brief source operand */
            const Tensor<Device,2>& mat_;
            /*! \brief patch height */
            index_t psize_y_;
            /*! \brief patch height */
            index_t psize_x_;
            /*! \brief patch stride */
            index_t pstride_;
            /*! \brief constructor */
            PackColToPatchXExp(const Tensor<Device,2> &mat, Shape<dstdim> imshape, index_t psize_y, index_t psize_x, index_t pstride)
                :mat_(mat), psize_y_(psize_y), psize_x_(psize_x), pstride_(pstride){
                this->shape_ = imshape;
                const index_t o_height = (imshape[1]  - psize_y) / pstride + 1;                
                const index_t o_width  = (imshape[0]  - psize_x) / pstride + 1;                
                utils::Assert(mat.shape[0] == o_height * o_width * imshape.ProdShape(3,dstdim), "PackColToPatchExp: mat.shape[0] mismatch");
                utils::Assert(mat.shape[1] == psize_y * psize_x * imshape[2], "PackColToPatchExp: mat.shape[1] mismatch");
            }
        };

        /*!
         * \brief reshape the content to another shape
         * input: Tensor<Device,dimsrc>: ishape
         * output: Tensor<Device,dimdst> ishape.Size() == oshape.Size()
         * \tparam SrcExp source expression
         * \tparam dimdst target dimension
         * \tparam dimsrc source dimension
         */
        template<typename SrcExp, int dimdst, int dimsrc>
        struct ReshapeExp: public MakeTensorExp< ReshapeExp<SrcExp,dimdst,dimsrc>, SrcExp, dimdst>{
            /*! \brief source expression */
            const SrcExp& src_;
            /*! \brief smallest dimension of input */
            index_t ishape0_;
            /*! \brief constructor */
            ReshapeExp(const SrcExp &src, Shape<dimdst> shape):src_(src){
                Shape<dimsrc> ishape = ShapeCheck<dimsrc,SrcExp>::Check(src_);
                utils::Assert(ishape.Size() == shape.Size(), "reshape size must match");
                ishape0_ = ishape[0];
                this->shape_ = shape;
            }
        };

        /*!
         * \brief swap two axis of a tensor
         * input: Tensor<Device,dim>: ishape
         * output: Tensor<Device,dimdst> oshape[a1],oshape[a2] = ishape[a2],oshape[a1]
         *
         * \tparam SrcExp type of source expression
         * \tparam dimsrc source dimension
         * \tparam a1 smaller dimension to be swapped
         * \tparam a2 larger dimension to be swapped
         */
        template<typename SrcExp,int dimsrc, int a1, int a2>
        struct SwapAxisExp: public MakeTensorExp< SwapAxisExp<SrcExp,dimsrc,a1,a2>, SrcExp, dimsrc>{
            /*! \brief source expression */
            const SrcExp& src_;
            /*! \brief constructor */
            SwapAxisExp(const SrcExp &src):src_(src){                
                this->shape_ = ShapeCheck<dimsrc,SrcExp>::Check(src); 
                std::swap(this->shape_[a1], this->shape_[a2]);
            }
        };

        /*!
         * \brief reduction to 1 dimension tensor
         * input: Tensor<Device,k>: ishape
         * output: Tensor<Device,1> shape[0] = ishape[dimkeep];
         *
         * \tparam EType type of expression to be reduced
         * \tparam Reducer which reducer to use
         * \tparam srcdim dimension of source
         * \tparam dimkeep which dimension to be kept,
         */
        template<typename EType, typename Reducer,int dimkeep>
        struct ReduceTo1DExp: public Exp< ReduceTo1DExp<EType,Reducer, dimkeep>, type::kComplex >{
            /*! \brief source operand */
            const EType& src_;
            /*! \brief source operand, scale of the  */
            real_t scale_;
            /*! \brief construct a repmat expression from src and nrow */
            ReduceTo1DExp(const EType& src, real_t scale):src_(src),scale_(scale){}
        };

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
        /*! \brief operator overload */
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*(const ReduceTo1DExp<E,R,d> &e, real_t scale){
            return ReduceTo1DExp<E,R,d>(e.src_, e.scale_*scale);
        }
        /*! \brief operator overload */
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*(real_t scale, const ReduceTo1DExp<E,R,d> &e){
            return ReduceTo1DExp<E,R,d>(e.src_, e.scale_*scale);
        }

        /*!
         * \brief  unpack local (overlap) patches of image to column of mat, can be used to implement convolution
         *  after getting unpacked mat, we can use: output = dot(weight, mat) to get covolved results, the relations:
         *
         *  weight; shape[1]: out_channel, shape[0]: ichannel*psize_y*psize_x
         *  output; shape[1]: out_channel, shape[0]: out_height*out_width * num_of_images
         *  out_height = (in_height - psize_y) / pstride + 1, this means we pad inperfect patch with 0
         *  out_width  = (in_width - psize_x) / pstride + 1
         *
         * \return mat target matrix; shape[1]: in_channel*psize_y*psize_x  shape[0]: out_height*out_width * num_of_images
         * \param img source image; shape[2]:  in_channels, shape[1]: in_height, shape[0]: in_width, can be 3D or 4D tensor(multiple images)
         * \param psize_y height of each patch
         * \param psize_x width of each patch
         * \param pstride stride of each patch
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
        template<typename SrcExp, int etype>
        inline UnpackPatchToColXExp<SrcExp, ExpInfo<SrcExp>::kDim > unpack_patch2col(const Exp<SrcExp,etype> &img, index_t psize_y, index_t psize_x, index_t pstride){
            TypeCheckPass< ExpInfo<SrcExp>::kDim >= 3 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return UnpackPatchToColXExp<SrcExp, ExpInfo<SrcExp>::kDim >(img.self(), psize_y, psize_x, pstride);
        }

        /*!
         * \brief reverse operation of pack_col2patch, can be used to implement deconvolution
         * \return packed img expression
         * \param mat source matrix
         * \param imshape shape of target img
         * \param psize_y height of each patch
         * \param psize_x height of each patch
         * \param pstride stride of each patch
         * \tparam Device the Device where input data lies
         */
        template<typename Device, int dstdim>
        inline PackColToPatchXExp<Device,dstdim> pack_col2patch(const Tensor<Device,2> &mat, Shape<dstdim> imshape, index_t psize_y, index_t psize_x, index_t pstride){
            utils::Assert(imshape[0] >= psize_x && imshape[1] >= psize_y, "PackColToPatch:image shape smaller than patch size");
            return PackColToPatchXExp<Device,dstdim>(mat, imshape, psize_y, psize_x, pstride);
        }

        /*!
         * \brief a expression that reshapes a tensor to another shape
         * \param src Tensor<Device,dimsrc>:
         * \param oshape target shape
         * \return a expresion with type Tensor<Device,dimdst>
         * \tparam SrcExp source expression
         * \tparam etype source expression type
         * \tparam dimdst target dimension
         */
        template<typename SrcExp, int etype, int dimdst>
        inline ReshapeExp< SrcExp,dimdst, ExpInfo<SrcExp>::kDim > reshape(const Exp<SrcExp,etype> &src, Shape<dimdst> oshape){
            return ReshapeExp< SrcExp,dimdst, ExpInfo<SrcExp>::kDim >(src.self(), oshape);
        }

        /*!
         * \brief a expression that reshapes a tensor to another shape
         * \param src Tensor<Device,dimsrc>:
         * \return a expresion with type Tensor<Device,dimdst>
         * \tparam a1 smaller dimension to be swapped
         * \tparam a2 larger dimension to be swapped
         * \tparam SrcExp source expression
         * \tparam etype source expression type
         */
        template<int a1, int a2, typename SrcExp, int etype>
        inline SwapAxisExp< SrcExp, ExpInfo<SrcExp>::kDim, a1,a2> swapaxis(const Exp<SrcExp,etype> &src){ 
            typedef ExpInfo<SrcExp> Info;
            TypeCheckPass< Info::kDim>=a1+1 && Info::kDim >= a2+1 && a1+1 <= a2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return SwapAxisExp< SrcExp,Info::kDim,a1,a2>(src.self());
        }

        /*!
         * \brief a sum over all dimensions, except dimkeep
         * \param exp input expression that must be a matrix Tensor<?,2>
         * \return a expresion with type Tensor<Device,1>
         * \tparam dimkeep the dimension that will be kept
         * \tparam SrcExp expression
         * \tparam etype type of expression
         */
        template<int dimkeep,  typename SrcExp, int etype>
        inline ReduceTo1DExp<SrcExp, red::sum, dimkeep > sumall_except_dim(const Exp<SrcExp,etype> &exp){
            return ReduceTo1DExp<SrcExp,red::sum,dimkeep>(exp.self(), 1.0f);
        }

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

        /*!
         * \brief a expression that sum over rows of a matrix
         * \param exp input expression that must be a matrix Tensor<?,2>
         * \return a expresion with type Tensor<Device,1>
         * \tparam SrcExp expression
         * \tparam etype type of expression
         */
        template<typename SrcExp, int etype>
        inline ReduceTo1DExp<SrcExp, red::sum, 0 > sum_rows(const Exp<SrcExp,etype> &exp){
            return sumall_except_dim<0>(exp);
        }

    }; // namespace expr
}; // namespace mshadow

// ==================================================
//  implementations afterwards,
//  no need to read if only use the functions
// --------------------------------------------------
namespace mshadow{
    namespace expr{
        template<typename SV, typename Device, typename EType, typename Reducer, int dimkeep>
        struct ExpComplexEngine< SV, Device, 1, ReduceTo1DExp<EType,Reducer,dimkeep> >{
            inline static void Eval(Tensor<Device,1> &dst, const ReduceTo1DExp<EType,Reducer,dimkeep> &exp){
                TypeCheckPass< dimkeep!=0 >::Error_Expression_Does_Not_Meet_Dimension_Req();
                MapReduceKeepHighDim<SV,Reducer,dimkeep>(dst, exp.src_, exp.scale_);
            }
        };

        template<typename SV, typename Device, typename EType, typename Reducer>
        struct ExpComplexEngine< SV, Device, 1, ReduceTo1DExp<EType,Reducer,0> >{
            inline static void Eval(Tensor<Device,1> &dst, const ReduceTo1DExp<EType,Reducer,0> &exp){
                MapReduceKeepLowest<SV,Reducer>(dst, exp.src_, exp.scale_);
            }
        };
    }; // namespace expr

    namespace expr{

    }; // namespace expr

    namespace expr{
        template<typename SrcExp, int srcdim>
        struct Plan< UnpackPatchToColXExp<SrcExp,srcdim> >{
        public:
            Plan(const UnpackPatchToColXExp<SrcExp,srcdim> &e)
                :src_(MakePlan(e.img_)), psize_y_(e.psize_y_), psize_x_(e.psize_x_), pstride_(e.pstride_),
                 i_channel_(e.i_channel_), i_height_(e.i_height_), i_width_(e.i_width_),                 
                 o_height_((i_height_  - psize_y_) / pstride_ + 1),
                 o_width_ ((i_width_   - psize_x_) / pstride_ + 1){
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const{
                const index_t x_offset = i % psize_x_;
                const index_t idivp    = i / psize_x_;
                const index_t y_offset = idivp % psize_y_;
                const index_t c = idivp / psize_y_;                
                const index_t x = (j % o_width_) * pstride_ + x_offset;
                const index_t jdivw = j / o_width_;
                const index_t y = (jdivw % o_height_) * pstride_ + y_offset;
                const index_t n = jdivw / o_height_;

                if(x < i_width_ && y < i_height_){
                    return src_.Eval((n * i_channel_  + c) * i_height_ + y, x);
                }else{
                    return 0.0f;
                }
            }
        private:
            Plan<SrcExp> src_;
            const index_t psize_y_, psize_x_, pstride_, i_channel_, i_height_, i_width_, o_height_, o_width_;
        };

        template<typename Device, int dstdim>
        struct Plan< PackColToPatchXExp<Device, dstdim> >{
        public:
            Plan(const PackColToPatchXExp<Device, dstdim> &e)
                :mat_(e.mat_), psize_y_(e.psize_y_), psize_x_(e.psize_x_), pstride_(e.pstride_),
                 i_channel_(e.shape_[2]), i_height_(e.shape_[1]),
                 o_height_((e.shape_[1]  - psize_y_) / pstride_ + 1),
                 o_width_((e.shape_[0]  - psize_x_) / pstride_ + 1){
                // note: i/o convention are same as unpack
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const{
                using namespace std;
                const index_t y = i % i_height_;
                const index_t idivh = i / i_height_;                
                const index_t c = idivh % i_channel_;
                const index_t n = idivh / i_channel_; 
                const index_t x = j;
                const index_t py_min = y < psize_y_ ? 0 : (y-psize_y_+pstride_)/pstride_;
                const index_t px_min = x < psize_x_ ? 0 : (x-psize_x_+pstride_)/pstride_;
                const index_t py_max = min((y+pstride_)/pstride_, o_height_);
                const index_t px_max = min((x+pstride_)/pstride_, o_width_);
                real_t res = 0.0f;
                for(index_t py = py_min; py < py_max; ++py){
                    for(index_t px = px_min; px < px_max; ++px){
                        res += mat_[ (c * psize_y_ + y - py*pstride_) * psize_x_ + x - px*pstride_ ][ (n * o_height_ + py) * o_width_+px ];
                    }
                }
                return res;
            }
        private:
            Tensor<Device,2> mat_;
            const index_t psize_y_, psize_x_, pstride_, i_channel_, i_height_, o_height_, o_width_;
        };
    };

    namespace expr{
        template<typename SrcExp, int dimdst, int dimsrc>
        struct Plan< ReshapeExp<SrcExp,dimdst,dimsrc> >{
        public:
            Plan(const ReshapeExp<SrcExp,dimdst,dimsrc> &e)
                : src_(MakePlan(e.src_)), oshape0_(e.shape_[0]), ishape0_(e.ishape0_){
            }
            MSHADOW_XINLINE real_t Eval(index_t y, index_t x) const{
                const index_t idx = y * oshape0_ + x;
                return src_.Eval(idx / ishape0_, idx % ishape0_);
            }
        private:
            Plan<SrcExp> src_;
            const index_t oshape0_, ishape0_;
        };
        // special work plan for 1 dimensional data
        template<typename SrcExp,int dimdst>
        struct Plan< ReshapeExp<SrcExp,dimdst,1> >{
        public:
            Plan(const ReshapeExp<SrcExp,dimdst,1> &e)
                : src_(MakePlan(e.src_)), oshape0_(e.shape_[0]){
            }
            MSHADOW_XINLINE real_t Eval(index_t y, index_t x) const{
                return src_.Eval(0, y * oshape0_ + x);
            }
        private:
            Plan<SrcExp> src_;
            const index_t oshape0_;
        };
    };
    
    namespace expr{
        template<typename SrcExp,int dimsrc, int a1, int a2>
        struct Plan< SwapAxisExp<SrcExp,dimsrc,a1,a2> >{
        public:
            Plan(const SwapAxisExp<SrcExp,dimsrc,a1,a2> &e)
                : src_(MakePlan(e.src_)),
                  shape1_(e.shape_.ProdShape(1, a1)),
                  shape2_(e.shape_[a1]),
                  shape3_(e.shape_.ProdShape(a1+1, a2)),
                  shape4_(e.shape_[a2]){
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const{
                const index_t y = i % shape1_;
                i /= shape1_; 
                const index_t z = i % shape2_;
                i /= shape2_;
                const index_t c = i % shape3_;
                i /= shape3_;
                const index_t n = i % shape4_;
                // swap z and n
                return src_.Eval(((((i/shape4_)*shape2_ + z) * shape3_+c) * shape4_ + n) * shape1_ + y, j); 
            }
        private:
            Plan<SrcExp> src_;
            const index_t shape1_, shape2_, shape3_, shape4_;
        };

        template<typename SrcExp,int dimsrc, int a2>
        struct Plan< SwapAxisExp<SrcExp,dimsrc,0,a2> >{
        public:
            Plan(const SwapAxisExp<SrcExp,dimsrc,0,a2> &e)
                : src_(MakePlan(e.src_)),
                  shape0_(e.shape_[0]),
                  shape1_(e.shape_.ProdShape(1,a2)),
                  shape2_(e.shape_[a2]){
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t x) const{
                // swap x and z
                const index_t y = i % shape1_;
                i /= shape1_; 
                const index_t z = i % shape2_;
                const index_t n = i / shape2_;
                return src_.Eval( (n*shape0_ + x) * shape1_ + y , z); 
            }
        private:
            Plan<SrcExp> src_;
            const index_t shape0_, shape1_, shape2_;
        };
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

