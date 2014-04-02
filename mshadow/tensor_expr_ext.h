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
         * \brief broadcast Tensor1D into a higher dimension Tensor
         * input: Tensor<Device,1>: ishape[0]
         * output: Tensor<Device,dimdst> : oshape[dimcast] = ishape[0]
         * \tparam Device which device it lies
         * \tparam dimdst  target tensor dimension
         * \tparam dimcast the dimension where the 1D tensor fills in by index
         */
        template<typename Device, int dimdst, int dimcast>
        struct Broadcast1DExp: public MakeTensorExp< Broadcast1DExp<Device,dimdst,dimcast>,Tensor<Device,1>,dimdst>{
            /*! \brief source operand */
            const Tensor<Device,1> src_;
            /*! \brief constructor */
            Broadcast1DExp( const Tensor<Device,1> &src, Shape<dimdst> shape ):src_(src){
                this->shape_ = shape;
            }
        };

        /*!
         * \brief unpack local (overlap) patches of image to column of mat, can be used to implement convolution
         *  after getting unpacked mat, we can use: output = dot( weight, mat ) to get covolved results, the relations:
         * \tparam SrcExp source expression
         */
        template<typename SrcExp>
        struct UnpackPatchToColExp: public MakeTensorExp< UnpackPatchToColExp<SrcExp>, SrcExp, 2>{
            /*! \brief source operand */
            const SrcExp& img_;
            /*! \brief patch size */
            index_t psize_;
            /*! \brief patch stride */
            index_t pstride_;
            /*! \brief height of img */
            index_t i_height_;
            /*! \brief width of img */
            index_t i_width_;
            /*! \brief constructor */
            UnpackPatchToColExp( const SrcExp &img, index_t psize, index_t pstride )
                :img_(img), psize_(psize), pstride_(pstride){
                Shape<3> imshape = ShapeCheck<3,SrcExp>::Check( img_ );
                utils::Assert( imshape[0] >= psize && imshape[1] >= psize, "UnpackPatchToCol:image shape smaller than patch size");
                this->i_height_ = imshape[1];
                this->i_width_  = imshape[0];
                const index_t o_height = ( i_height_ - psize ) / pstride + 1;
                const index_t o_width  = ( i_width_  - psize ) / pstride + 1;
                this->shape_[0] = o_height * o_width;
                this->shape_[1] = psize * psize * imshape[2];
            }
        };

        /*!
         * \brief reverse operation of UnpackPatchToCol, used to backprop gradient back
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct PackColToPatchExp: public MakeTensorExp< PackColToPatchExp<Device>, Tensor<Device,2>, 3>{
            /*! \brief source operand */
            const Tensor<Device,2>& mat_;
            /*! \brief patch size */
            index_t psize_;
            /*! \brief patch stride */
            index_t pstride_;
            /*! \brief constructor */
            PackColToPatchExp( const Tensor<Device,2> &mat, Shape<3> imshape, index_t psize, index_t pstride )
                :mat_(mat), psize_(psize), pstride_(pstride){
                this->shape_ = imshape;
                const index_t o_height = ( imshape[1]  - psize ) / pstride + 1;
                const index_t o_width  = ( imshape[0]  - psize ) / pstride + 1;
                utils::Assert( mat.shape[0] == o_height * o_width, "PackColToPatchExp: mat.shape[0] mismatch" );
                utils::Assert( mat.shape[1] == psize * psize * imshape[2], "PackColToPatchExp: mat.shape[1] mismatch" );
            }
        };

        /*!
         * \brief reshape the content to another shape
         * input: Tensor<Device,dimsrc>: ishape
         * output: Tensor<Device,dimdst> ishape.Size() == oshape.Size()
         *
         * \tparam Device where the data lies
         * \tparam dimdst target dimension
         * \tparam dimsrc source dimension
         */
        template<typename Device, int dimdst, int dimsrc>
        struct ReshapeExp: public MakeTensorExp< ReshapeExp<Device,dimdst,dimsrc>, Tensor<Device,dimsrc>, dimdst>{
            /*! \brief source expression */
            const Tensor<Device,dimsrc>& src_;
            /*! \brief constructor */
            ReshapeExp( const Tensor<Device,dimsrc> &src, Shape<dimdst> shape ):src_(src){
                utils::Assert( shape.Size() == src.shape.Size(), "reshape size must match" );
                this->shape_ = shape;
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
            ReduceTo1DExp( const EType& src, real_t scale ):src_(src),scale_(scale){}
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
            /*! \brief kernel size */
            index_t ksize_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief source height shape[1] */
            index_t src_height_;
            /*! \brief source width shape[0] */
            index_t src_width_;
            /*! \brief constructor */
            PoolingExp( const SrcExp &src, index_t ksize, index_t kstride )
                : src_(src), ksize_(ksize), kstride_(kstride) {
                Shape< srcdim > sshape = ShapeCheck< srcdim,SrcExp>::Check( src_ );
                utils::Assert( sshape[0] >= ksize && sshape[1] >= ksize, "pool: kernel must be smaller than image" );
                this->src_height_ = sshape[1];
                this->src_width_  = sshape[0];
                this->shape_ = sshape;
                this->shape_[1] =  (src_height_ - ksize) / kstride + 1;                
                this->shape_[0] =  (src_width_  - ksize) / kstride + 1;
            }
            /*! \brief constructor, specify shape */
            PoolingExp( const SrcExp &src, Shape<2> pshape, index_t ksize, index_t kstride )
                : src_(src), ksize_(ksize), kstride_(kstride) {
                Shape< srcdim > sshape = ShapeCheck< srcdim,SrcExp>::Check( src_ );
                utils::Assert( sshape[0] >= ksize && sshape[1] >= ksize, "pool: kernel must be smaller than image" );
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
            /*! \brief kernel size */
            index_t ksize_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief constructor */
            UnPoolingExp( const Tensor<Device,4> &data_src,  const Tensor<Device,4> &data_pooled,
                          const Tensor<Device,4> &grad_pooled, index_t ksize, index_t kstride )
                : data_src_(data_src), data_pooled_(data_pooled), grad_pooled_(grad_pooled),
                  ksize_(ksize), kstride_(kstride) {
                utils::Assert( grad_pooled.shape == data_pooled.shape, "UnPoolingExp: pooled shape mismatch" );
                utils::Assert( grad_pooled.shape[2] == data_src.shape[2], "UnPoolingExp: pool and src shape mismatch" );
                utils::Assert( grad_pooled.shape[3] == data_src.shape[3], "UnPoolingExp: pool and src shape mismatch" );
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
            /*! \brief pad size */
            index_t pad_;
            /*! \brief source tensor height */
            index_t src_height_;
            /*! \brief source tensor width */
            index_t src_width_;
            /*! \brief constructor */
            PaddingExp( const SrcExp &src, index_t pad )
                : src_(src), pad_(pad) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check( src_ );
                src_height_ = this->shape_[1];
                src_width_  = this->shape_[0];
                this->shape_[1] += pad * 2; // height
                this->shape_[0] += pad * 2; // width
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
            CroppingExp(const SrcExp &src, Shape<2> cshape ): src_(src) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check( src_ );
                utils::Assert(this->shape_[1] >= cshape[1], "CroppingExp: height requirement not met");
                utils::Assert(this->shape_[0] >= cshape[0], "CroppingExp: width requirement not met");
                pad_height_ = (this->shape_[1] - cshape[1]) / 2;
                pad_width_ = (this->shape_[0] - cshape[0]) / 2;
                src_height_ = this->shape_[1];
                this->shape_[1] = cshape[1]; // width
                this->shape_[0] = cshape[0]; // height
            }
            /*! \brief constructor */
            CroppingExp(const SrcExp &src, Shape<2> cshape, index_t start_height, index_t start_width  )
                : src_(src), pad_height_(start_height), pad_width_(start_width) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check( src_ );
                utils::Assert(this->shape_[1] >= cshape[1], "CroppingExp: height requirement not met");
                utils::Assert(this->shape_[0] >= cshape[0], "CroppingExp: width requirement not met");
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
            MirroringExp( const SrcExp &src ): src_(src) {
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check( src_ );
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
            ChannelPoolingExp( const SrcExp &src, index_t nsize ): src_(src), nsize_(nsize){
                utils::Assert( nsize % 2 == 1, "ChannelPoolingExp: local size must be odd, to make it symmetric" );
                this->shape_ = ShapeCheck<srcdim,SrcExp>::Check( src_ );
                utils::Assert( this->shape_[2] >= nsize_, "ChannelPoolingExp: local size need to be smaller than number of channels" );
            }
        };
    }; // namespace expr


    // Declaration of all functions go here
    namespace expr{
        /*! \brief operator overload */
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*( const ReduceTo1DExp<E,R,d> &e, real_t scale ){
            return ReduceTo1DExp<E,R,d>( e.src_, e.scale_*scale );
        }
        /*! \brief operator overload */
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*( real_t scale, const ReduceTo1DExp<E,R,d> &e ){
            return ReduceTo1DExp<E,R,d>( e.src_, e.scale_*scale );
        }

        /*!
         * \brief a expression that replicate a 1 dimension tensor in dimension dimcast
         * \param src Tensor<Device,1>: shape[0]
         * \param shape shape of output
         * \return a expresion with type Tensor<Device,dimdst>
         * \tparam dimcast target dimension where the 1D tensor will be broadcasted
         * \tparam Device which device it lies
         * \tparam dimdst dimension of destination tensor
         */
        template<int dimcast,typename Device,int dimdst>
        inline Broadcast1DExp<Device,dimdst,dimcast> broadcast( const Tensor<Device,1> &src, Shape<dimdst> shape ){
            TypeCheckPass< dimcast<dimdst >::Error_Expression_Does_Not_Meet_Dimension_Req();
            utils::Assert( src.shape[0] == shape[dimcast], "broadcast, shape mismatch" );
            return Broadcast1DExp<Device,dimdst,dimcast>( src, shape );
        }

        /*!
         * \brief  unpack local (overlap) patches of image to column of mat, can be used to implement convolution
         *  after getting unpacked mat, we can use: output = dot( weight, mat ) to get covolved results, the relations:
         *
         *  weight; shape[1]: out_channel, shape[0]: ichannel*psize*psize
         *  output; shape[1]: out_channel, shape[0]: out_height*out_width
         *  out_height = ( in_height - psize ) / pstride + 1, this means we pad inperfect patch with 0
         *  out_width  = ( in_width - psize ) / pstride + 1
         *
         * \return mat target matrix; shape[1]: in_channel*psize*psize  shape[0]: out_height*out_width
         * \param img source image; shape[2]:  in_channels, shape[1]: in_height, shape[0]: in_width
         * \param psize height and width of each patch
         * \param pstride stride of each patch
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
        template<typename SrcExp, int etype>
        inline UnpackPatchToColExp<SrcExp> unpack_patch2col( const Exp<SrcExp,etype> &img, index_t psize, index_t pstride ){
            TypeCheckPass< ExpInfoXPU<SrcExp>::kDim == 3 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return UnpackPatchToColExp<SrcExp>( img.self(), psize, pstride );
        }

        /*!
         * \brief reverse operation of pack_col2patch, can be used to implement deconvolution
         * \return packed img expression
         * \param mat source matrix
         * \param imshape shape of target img
         * \param psize height and width of each patch
         * \param pstride stride of each patch
         */
        template<typename Device>
        inline PackColToPatchExp<Device> pack_col2patch( const Tensor<Device,2> &mat, Shape<3> imshape, index_t psize, index_t pstride ){
            utils::Assert( imshape[0] >= psize && imshape[1] >= psize, "PackColToPatch:image shape smaller than patch size");
            return PackColToPatchExp<Device>( mat, imshape, psize, pstride );
        }
        /*!
         * \brief a expression that reshapes a tensor to another shape
         * \param src Tensor<Device,dimsrc>:
         * \param oshape target shape
         * \return a expresion with type Tensor<Device,dimdst>
         * \tparam Device which device it lies
         * \tparam dimdst target dimension
         * \tparam dimsrc source dimension
         */
        template<typename Device, int dimdst, int dimsrc>
        inline ReshapeExp<Device,dimdst,dimsrc> reshape( const Tensor<Device,dimsrc> &src, Shape<dimdst> oshape ){
            return ReshapeExp<Device,dimdst,dimsrc>( src, oshape );
        }

        /*!
         * \brief a sum over all dimensions, except dimkeep
         * \param exp input expression that must be a matrix Tensor<?,2>
         * \return a expresion with type Tensor<Device,1>
         * \tparam dimkeep the dimension that will be kept
         * \tparam E expression
         * \tparam etype type of expression
         */
        template<int dimkeep,  typename E, int etype>
        inline ReduceTo1DExp<E, red::sum, dimkeep > sumall_except_dim( const Exp<E,etype> &exp ){
            return ReduceTo1DExp<E,red::sum,dimkeep>( exp.self(), 1.0f );
        }

        /*!
         * \brief pooling subregion results together
         * \param src source image, shape[3]: batch, shape[2]: channel shape[1]: height shape[0]:width
         * \param ksize kernel size
         * \param kstride stride for each kernel
         * \return expression of pooled result
         * \tparam Reducer reducer type
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
        template<typename Reducer, typename SrcExp, int etype>
        inline PoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim > pool( const Exp<SrcExp,etype> &src, index_t ksize, index_t kstride ) {
            TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return PoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim >(src.self(), ksize, kstride);
        }
        /*! \brief same as pool, except the output shape is specified by pshape */
        template<typename Reducer, typename SrcExp, int etype>
        inline PoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim > pool( const Exp<SrcExp,etype> &src, Shape<2> pshape, index_t ksize, index_t kstride ) {
            TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return PoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim >(src.self(), pshape, ksize, kstride);
        }
        /*!
         * \brief unpooling gradient for 4D, backprop gradient value back, revserse operation of pooling
         * \param data_src  source input, corresponds to src in pooling
         * \param data_pooled result of pooled data, corresponds to result of pooling
         * \param grad_pooled gradient data of pooled part, to be propgate down
         * \param ksize kernel size
         * \param kstride stride for each kernel
         * \return expression corresponding to unpooled 4D Tensor, storing backproped gradient
         * \tparam Reducer reducer type
         * \tparam Device device where data lies
         */
         template<typename Reducer, typename Device>
         inline UnPoolingExp<Reducer, Device> unpool( const Tensor<Device,4>&data_src, const Tensor<Device,4> &data_pooled,
                                                      const Tensor<Device,4> &grad_pooled, index_t ksize, index_t kstride ) {
             return UnPoolingExp<Reducer, Device>(data_src, data_pooled, grad_pooled,ksize, kstride);
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
         inline PaddingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim> pad(const Exp<SrcExp, etype> &src, index_t pad) {
             TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return PaddingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim>(src.self(), pad);
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
         inline CroppingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim> crop( const Exp<SrcExp, etype> &src, Shape<2> oshape ) {
             TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return CroppingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim>(src.self(), oshape);
         }
        /*!
         * \brief same as crop, but can specify starting position to do cropping
         * \param src original image batches
         * \param oshape output shape to be cropped
         * \return expression corresponding to padded result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline CroppingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim> crop( const Exp<SrcExp, etype> &src, Shape<2> oshape, index_t start_height, index_t start_width ) {
             TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return CroppingExp<SrcExp, ExpInfoXPU<SrcExp>::kDim>(src.self(), oshape, start_height, start_width);
         }

        /*!
         * \brief mirroring expression, mirror images in width
         * \param src original image batches
         * \return expression corresponding to mirrored result
         * \tparam SrcExp source expression
         * \tparam etype type of expression
         */
         template<typename SrcExp, int etype>
         inline MirroringExp<SrcExp, ExpInfoXPU<SrcExp>::kDim> mirror(const Exp<SrcExp, etype> &src) {
             TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 2 >::Error_Expression_Does_Not_Meet_Dimension_Req();
             return MirroringExp<SrcExp, ExpInfoXPU<SrcExp>::kDim>(src.self());
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
        inline ChannelPoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim > chpool( const Exp<SrcExp,etype> &src, index_t nsize ) {
            TypeCheckPass< ExpInfoXPU<SrcExp>::kDim >= 3 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return ChannelPoolingExp<Reducer,SrcExp, ExpInfoXPU<SrcExp>::kDim >(src.self(),nsize);
        }
        // short cut functions
        /*!
         * \brief a expression that replicate a 1 dimension tensor for nrow times
         * \param src Tensor<Device,1>: shape[0]
         * \param nrow number of rows to replicate
         * \return a expresion with type Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        template<typename Device>
        inline Broadcast1DExp<Device,2,0> repmat( const Tensor<Device,1> &src, index_t nrow ){
            return broadcast<0>( src, Shape2( nrow, src.shape[0] ) );
        }
        /*!
         * \brief a expression that sum over rows of a matrix
         * \param exp input expression that must be a matrix Tensor<?,2>
         * \return a expresion with type Tensor<Device,1>
         * \tparam E expression
         * \tparam etype type of expression
         */
        template<typename E, int etype>
        inline ReduceTo1DExp<E, red::sum, 0 > sum_rows( const Exp<E,etype> &exp ){
            return sumall_except_dim<0>( exp );
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
            inline static void Eval( Tensor<Device,1> &dst, const ReduceTo1DExp<EType,Reducer,dimkeep> &exp ){
                TypeCheckPass< dimkeep!=0 >::Error_Expression_Does_Not_Meet_Dimension_Req();
                MapReduceKeepHighDim<SV,Reducer,dimkeep>( dst, exp.src_, exp.scale_ );
            }
        };

        template<typename SV, typename Device, typename EType, typename Reducer>
        struct ExpComplexEngine< SV, Device, 1, ReduceTo1DExp<EType,Reducer,0> >{
            inline static void Eval( Tensor<Device,1> &dst, const ReduceTo1DExp<EType,Reducer,0> &exp ){
                MapReduceKeepLowest<SV,Reducer>( dst, exp.src_, exp.scale_ );
            }
        };
    }; // namespace expr

    namespace expr{
        /*! \brief execution plan of Broadcast1DExp */
        template<typename Device, int dimdst, int dimcast>
        struct Plan< Broadcast1DExp<Device,dimdst,dimcast> >{
        public:
            Plan( const Broadcast1DExp<Device,dimdst,dimcast> &e ): dptr_( e.src_.dptr ){
                TypeCheckPass< dimcast!=0 >::Error_Expression_Does_Not_Meet_Dimension_Req();
                ystride_ = 1;
                #pragma unroll
                for( int i = 1; i < dimcast; ++ i ){
                    ystride_ *= e.shape_[i];
                }
                length_ = e.shape_[ dimcast ];
            }
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ (y / ystride_) % length_ ];
            }
        private:
            const real_t  *dptr_;
            index_t  ystride_, length_;
        };

        /*! \brief execution plan of Broadcast1DExp */
        template<typename Device, int dimdst>
        struct Plan< Broadcast1DExp<Device,dimdst,0> >{
        public:
            Plan( const Broadcast1DExp<Device,dimdst,0> &e ): dptr_( e.src_.dptr ){}
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ x ];
            }
        private:
            const real_t *dptr_;
        };
    }; // namespace expr

    namespace expr{
        template<typename SrcExp>
        struct Plan< UnpackPatchToColExp<SrcExp> >{
        public:
            Plan( const UnpackPatchToColExp<SrcExp> &e )
                :src_(MakePlan(e.img_)),psize_(e.psize_), pstride_(e.pstride_),
                 i_height_(e.i_height_), i_width_(e.i_width_){
                o_width_  = ( i_width_  - psize_ ) / pstride_ + 1;
            }
            MSHADOW_XINLINE real_t Eval( index_t i, index_t j ) const{
                const index_t x_offset = i % psize_;
                const index_t idivp    = i / psize_;
                const index_t y_offset = idivp % psize_;
                const index_t channel  = idivp / psize_;
                const index_t y = (j / o_width_) * pstride_ + y_offset;
                const index_t x = (j % o_width_) * pstride_ + x_offset;
                if( x < i_width_ && y < i_height_ ){
                    return src_.Eval( channel * i_height_ + y, x );
                }else{
                    return 0.0f;
                }
            }
        private:
            Plan<SrcExp> src_;
            index_t psize_, pstride_, i_height_, i_width_, o_width_;
        };

        template<typename Device>
        struct Plan< PackColToPatchExp<Device> >{
        public:
            Plan( const PackColToPatchExp<Device> &e )
                :mat_(e.mat_), psize_(e.psize_), pstride_(e.pstride_){
                i_height_  = e.shape_[1];
                o_width_   = ( e.shape_[0]  - psize_ ) / pstride_ + 1;
                o_height_  = ( e.shape_[1]  - psize_ ) / pstride_ + 1;
            }
            MSHADOW_XINLINE real_t Eval( index_t i, index_t j ) const{
                using namespace std;
                const index_t c = i / i_height_;
                const index_t y = i % i_height_;
                const index_t x = j;
                const index_t py_min = y < psize_ ? 0 : (y-psize_+pstride_)/pstride_;
                const index_t px_min = x < psize_ ? 0 : (x-psize_+pstride_)/pstride_;
                const index_t py_max = min( (y+pstride_)/pstride_, o_height_);
                const index_t px_max = min( (x+pstride_)/pstride_, o_width_ );
                real_t res = 0.0f;
                for( index_t py = py_min; py < py_max; ++py ){
                    for( index_t px = px_min; px < px_max; ++px ){
                        res += mat_[ (c * psize_ + y - py*pstride_) * psize_ + x - px*pstride_ ][ py*o_width_+px ];
                    }
                }
                return res;
            }
        private:
            Tensor<Device,2> mat_;
            index_t psize_, pstride_, i_height_, o_width_, o_height_;
        };
    };

    namespace expr{
        template<typename Device, int dimdst, int dimsrc>
        struct Plan< ReshapeExp<Device,dimdst,dimsrc> >{
        public:
            Plan( const ReshapeExp<Device,dimdst,dimsrc> &e ): dptr_( e.src_.dptr ){
                oshape0_ = e.shape_[0];
                ishape0_ = e.src_.shape[0];
                istride_ = e.src_.shape.stride_;
            }
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                const index_t idx = y * oshape0_ + x;
                return dptr_[ ( idx / ishape0_ ) * istride_ + ( idx % ishape0_ ) ];
            }
        private:
            const real_t *dptr_;
            index_t oshape0_, ishape0_, istride_;
        };

        template<typename Device,int dimdst>
        struct Plan< ReshapeExp<Device,dimdst,1> >{
        public:
            Plan( const ReshapeExp<Device,dimdst,1> &e ): dptr_( e.src_.dptr ){
                oshape0_ = e.shape_[0];
            }
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                    return dptr_[ y * oshape0_ + x ];
            }
        private:
            const real_t *dptr_;
            index_t oshape0_;
        };
    };

    namespace expr{
        template<typename Reducer, typename SrcExp, int srcdim>
        struct Plan< PoolingExp< Reducer, SrcExp, srcdim> > {
        public:
            Plan( const PoolingExp<Reducer, SrcExp, srcdim> &e )
                : src_( MakePlan( e.src_ ) ), ksize_(e.ksize_), kstride_(e.kstride_),
                  src_height_(e.src_height_),src_width_(e.src_width_), new_height_(e.shape_[1]) {
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                using namespace std;
                const index_t py = i % new_height_;
                const index_t y_start = py * kstride_;
                const index_t y_end = min( y_start + ksize_, src_height_ );
                const index_t px = j;
                const index_t x_start = px * kstride_;
                const index_t x_end = min( x_start + ksize_, src_width_ );
                const index_t c = i / new_height_;

                real_t res = Reducer::kInitV;
                for (index_t y = y_start; y < y_end; ++y) {
                    for (index_t x = x_start; x < x_end; ++x) {
                        Reducer::Reduce( res, src_.Eval( c*src_height_+y, x ) );
                    }
                }
                return res;
            }
        private:
            Plan<SrcExp> src_;
            index_t ksize_, kstride_;
            index_t src_height_, src_width_;
            index_t new_height_;
        };

        template<typename Reducer, typename Device>
        struct Plan<UnPoolingExp<Reducer, Device> > {
        public:
            Plan(const UnPoolingExp<Reducer, Device> &e)
                : data_src_(e.data_src_), data_pooled_(e.data_pooled_), grad_pooled_(e.grad_pooled_),
                  ksize_(e.ksize_), kstride_(e.kstride_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                using namespace std;
                const index_t x = j;
                const index_t y = i % data_src_.shape[1];
                const index_t c = i / data_src_.shape[1];
                const real_t vsrc = data_src_[0][c][y][x];

                const index_t py_min = y < ksize_ ? 0 : (y-ksize_+kstride_)/kstride_;
                const index_t px_min = x < ksize_ ? 0 : (x-ksize_+kstride_)/kstride_;
                const index_t py_max = min( (y+kstride_)/kstride_, data_pooled_.shape[1]);
                const index_t px_max = min( (x+kstride_)/kstride_, data_pooled_.shape[0]);

                real_t val = 0;
                for( index_t py = py_min; py < py_max; ++py ){
                    for( index_t px = px_min; px < px_max; ++px ){
                        val += Reducer::PartialGrad(vsrc, data_pooled_[0][c][py][px]) * grad_pooled_[0][c][py][px];
                    }
                }
                return val;
            }
        private:
            Tensor<Device, 4> data_src_, data_pooled_, grad_pooled_;
            index_t ksize_;
            index_t kstride_;
        };
    }; // namespace expr

    namespace expr{
        template<typename SrcExp, int srcdim>
        struct Plan< PaddingExp<SrcExp, srcdim> > {
        public:
            Plan(const PaddingExp<SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), pad_(e.pad_), new_height_(e.shape_[1]),
                  src_height_(e.src_height_), src_width_(e.src_width_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                if (y < pad_ || x < pad_) return 0.0f;
                const index_t h = y - pad_;
                const index_t w = x - pad_;
                if (h < src_height_ && w < src_width_) {
                    return src_.Eval(c * src_height_ + h, w);
                } else {
                    return 0.0f;
                }
            }
        private:
            Plan<SrcExp> src_;
            index_t pad_;
            index_t new_height_;
            index_t src_height_;
            index_t src_width_;
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
            index_t pad_height_, pad_width_;
            index_t new_height_;
            index_t src_height_;
        };

        template<typename SrcExp, int srcdim>
        struct Plan< MirroringExp<SrcExp, srcdim> > {
        public:
            Plan(const MirroringExp<SrcExp, srcdim> &e)
                : src_(MakePlan(e.src_)), width_(e.shape_[0]){}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                return src_.Eval( i, width_ - j - 1 );
            }
        private:
            Plan<SrcExp> src_;
            index_t width_;
        };
    }; // namespace expr

    namespace expr{
        template<typename Reducer, typename SrcExp, int srcdim>
        struct Plan< ChannelPoolingExp< Reducer, SrcExp, srcdim> > {
        public:
            Plan( const ChannelPoolingExp<Reducer, SrcExp, srcdim> &e )
                : src_( MakePlan( e.src_ ) ), channel_(e.shape_[2]),
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
                const index_t cend   = min( c + hnsize_ + 1, channel_ );
                real_t res = Reducer::kInitV;
                for( index_t cc = cstart; cc < cend; ++ cc ){
                    Reducer::Reduce( res, src_.Eval( (n*channel_+cc)*height_ + y, x ) );
                }
                return res;
            }
        private:
            Plan<SrcExp> src_;
            index_t channel_, height_, width_, hnsize_;
        };
    };
}; // namespace mshadow

#if MSHADOW_USE_SSE
// implementations of SSE support, if possible
#include "tensor_sse-inl.hpp"
namespace mshadow{
    namespace expr{
        template<int dimdst>
        struct SSECheck< Broadcast1DExp<cpu,dimdst,0> >{
            const static bool kPass = true;
        };
        template<int dimdst>
        struct SSEAlignCheck<2, Broadcast1DExp<cpu,dimdst,0> >{
            inline static bool Check( const Broadcast1DExp<cpu,dimdst,0> &exp ){
                return sse2::CheckAlign( exp.src_.dptr );
            }
        };
        template<int dimdst>
        class SSEPlan< Broadcast1DExp<cpu,dimdst,0> >{
        public:
            SSEPlan( const Broadcast1DExp<cpu,dimdst,0> &t )
                :dptr_(t.src_.dptr){}
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const{
                return sse2::FVec<real_t>( &dptr_[ x ] );
            }
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ x ];
            }
        private:
            const real_t  *dptr_;
        };
    };
};
#endif

#endif

