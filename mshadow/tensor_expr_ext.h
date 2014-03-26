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
            const Tensor<Device,1> &src_;
            Broadcast1DExp( const Tensor<Device,1> &src, Shape<dimdst> shape ):src_(src){
                this->shape_ = shape;
            }
        };

        /*!
         * \brief unpack local (overlap) patches of image to column of mat, can be used to implement convolution
         *  after getting unpacked mat, we can use: output = dot( weight, mat ) to get covolved results, the relations:
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct UnpackPatchToColExp: public MakeTensorExp< UnpackPatchToColExp<Device>, Tensor<Device,3>, 2>{
            /*! \brief source operand */
            const Tensor<Device,3> &img_;
            /*! \brief patch size */
            index_t psize_;
            /*! \brief patch stride */
            index_t pstride_;
            UnpackPatchToColExp( const Tensor<Device,3> &img, index_t psize, index_t pstride )
                :img_(img), psize_(psize), pstride_(pstride){
                const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
                const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
                this->shape_[0] = o_height * o_width;
                this->shape_[1] = psize * psize * img.shape[2];
            }
        };

        /*!
         * \brief reverse operation of
         *  after getting unpacked mat, we can use: output = dot( weight, mat ) to get covolved results, the relations:
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct PackColToPatchExp: public MakeTensorExp< PackColToPatchExp<Device>, Tensor<Device,2>, 3>{
            /*! \brief source operand */
            const Tensor<Device,2> &mat_;
            /*! \brief patch size */
            index_t psize_;
            /*! \brief patch stride */
            index_t pstride_;
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
            Tensor<Device,dimsrc> src_;
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
            EType src_;
            /*! \brief source operand, scale of the  */
            real_t scale_;
            /*! \brief construct a repmat expression from src and nrow */
            ReduceTo1DExp( EType src, real_t scale ):src_(src),scale_(scale){}
        };

        /*!
         * \brief pooling expression, do reduction over local patches of a image
         * \tparam Reducer reduction method during pooling
         * \tparam SrcExp source expression to be pooled from
         */
        template<typename Reducer, typename SrcExp>
        struct PoolingExp: public MakeTensorExp< PoolingExp<Reducer, SrcExp>, SrcExp, 4> {
            /*! \brief source operand */
            const SrcExp src_;
            /*! \brief kernel size */
            index_t ksize_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief source height shape[1] */
            index_t src_height_;
            /*! \brief source width shape[0] */
            index_t src_width_;
            PoolingExp( const SrcExp &src, index_t ksize, index_t kstride )
                : src_(src), ksize_(ksize), kstride_(kstride) {
                Shape<4> srcshape = ShapeCheck<4,SrcExp>::Check( src_ );
                utils::Assert( srcshape[1] >= ksize && srcshape[0] >= ksize, "PoolingExp: source smaller than kernel" );
                this->src_height_ = srcshape[1];
                this->src_width_  = srcshape[0];
                const index_t p_height = (src_height_ - ksize) / kstride + 1;
                const index_t p_width  = (src_width_  - ksize) / kstride + 1;
                this->shape_ = Shape4( srcshape[3], srcshape[2], p_height, p_width );
            }
        };

        /*! \brief unpooling expr
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct UnPoolingExp: public MakeTensorExp<UnPoolingExp<Device>, Tensor<Device,3>, 3> {
            /*! \brief source operand */
            const Tensor<Device, 3> &img_;
            /*! \brief source pooled operand */
            const Tensor<Device, 3> &pooled_;
            /*! \brief kernel size */
            index_t ksize_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief pooling type */
            int type_;
            UnPoolingExp(const Tensor<Device,3> &img, const Tensor<Device,3> &pooled, index_t ksize, index_t kstride, int type) : img_(img), pooled_(pooled), ksize_(ksize), kstride_(kstride), type_(type) {
                this->shape_ = img_.shape;
            }
        };


        /*! \brief padding expr
         *  \tparam Device which device it lies
         */
        template<typename Device>
        struct PaddingExp : public MakeTensorExp<PaddingExp<Device>, Tensor<Device, 3>, 3> {
            /*! \brief source operand */
            const Tensor<Device, 3> &img_;
            /*! \brief pad size */
            index_t pad_;
            /*! \brief new height */
            index_t new_height_;

            PaddingExp(const Tensor<Device,3> &img, index_t pad)
                : img_(img), pad_(pad) {
                utils::Assert(pad > 0, "PaddingExp: Incorrect padding size");
                this->shape_[0] = img.shape[0] + pad * 2; // width
                this->shape_[1] = img.shape[1] + pad * 2; // height
                this->shape_[2] = img.shape[2]; // channel
                new_height_ = img.shape[1] + pad * 2;
            }
        };

        /*! \brief unpadding expr
         *  \tparam Device which device it lies
         */
        template<typename Device>
        struct UnPaddingExp : public MakeTensorExp<UnPaddingExp<Device>, Tensor<Device, 3>, 3> {
            /*! \brief source operand */
            const Tensor<Device, 3> &padded_img_;
            /*! \brief pad size */
            index_t pad_;
            /*! \brief new height */
            index_t new_height_;
            UnPaddingExp(const Tensor<Device,3> &padded_img, index_t pad)
                : padded_img_(padded_img), pad_(pad) {
                utils::Assert(pad > 0, "PaddingExp: Incorrect padding size");
                utils::Assert(padded_img.shape[0] > 2 * pad, "PaddingExp: padding size should be smaller than img width");
                utils::Assert(padded_img.shape[1] > 2 * pad, "PaddingExp: padding size should be smaller than img height");
                this->shape_[0] = padded_img.shape[0] - 2 * pad; // width
                this->shape_[1] = padded_img.shape[1] - 2 * pad; // height
                this->shape_[2] = padded_img.shape[2]; // channel
                new_height_ = padded_img.shape[1] - 2 * pad;
            }
        }; // struct UnPaddingExp
    }; // namespace expr


    // Declaration of all functions go here
    namespace expr{
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*( const ReduceTo1DExp<E,R,d> &e, real_t scale ){
            return ReduceTo1DExp<E,R,d>( e.src_, e.scale_*scale );
        }
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
         */
        template<typename Device>
        inline UnpackPatchToColExp<Device> unpack_patch2col( const Tensor<Device,3> &img, index_t psize, index_t pstride ){
            utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "UnpackPatchToCol:image shape smaller than patch size");
            return UnpackPatchToColExp<Device>( img, psize, pstride );
        }

        /*!
         * \brief reverse operation of pack_col2patch
         * \return packed img expression
         * \param mat, source matrix
         * \param imshape, shape of target img
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
         * \brief pooling for 4D tensor
         * \param src source image, shape[3]: batch, shape[2]: channel shape[1]: height shape[0]:width
         * \param ksize kernel size
         * \param kstride stride for each kernel
         * \return expression of pooled result
         */
        template<typename Reducer, typename SrcExp, int etype>
        inline PoolingExp<Reducer,SrcExp> pooling( const Exp<SrcExp,etype> &src, index_t ksize, index_t kstride ) {
            TypeCheckPass< ExpInfo<cpu,SrcExp>::kDim == 4|| ExpInfo<gpu,SrcExp>::kDim == 4 >::Error_Expression_Does_Not_Meet_Dimension_Req();
            return PoolingExp<Reducer,SrcExp>(src.self(), ksize, kstride);
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

        /*!
         * \brief unpooling gradient for 3D
         * \return unpooling 3D Tensor, shape[2]: channel, shape[1]:height, shape[0]: weight
         * \param pooled 3D Tensor Pooled
         * \param img original image
         * \param kstride stride for each kernel
         * \param type pooling type
         */
         template<typename Device>
         inline UnPoolingExp<Device> unpooling(const Tensor<Device, 3>&img, const Tensor<Device, 3> pooled, index_t ksize, index_t kstride, int type) {
            return UnPoolingExp<Device>(img, pooled, ksize, kstride, type);
         }

        /*!
         * \brief padding for 3D tensor
         * \return padded 3D tensor
         * \param img original image
         * \param pad padding size
         */
         template<typename Device>
         inline PaddingExp<Device> padding(const Tensor<Device, 3>&img, index_t pad) {
             return PaddingExp<Device>(img, pad);
         }

         /*!
          * \brief unpadding for 3d tensor
          * \return unpadding 3d tensor
          * \param padded_img padded img
          * \param pad
          */
         template<typename Device>
         inline UnPaddingExp<Device> unpadding(const Tensor<Device, 3> &padded_img, index_t pad) {
             return UnPaddingExp<Device>(padded_img, pad);
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
        template<typename Device>
        struct Plan< UnpackPatchToColExp<Device> >{
        public:
            Plan( const UnpackPatchToColExp<Device> &e )
                :img_(e.img_),psize_(e.psize_), pstride_(e.pstride_){
                o_width_  = ( img_.shape[0]  - psize_ ) / pstride_ + 1;
            }
            MSHADOW_XINLINE real_t Eval( index_t i, index_t j ) const{
                const index_t x_offset = i % psize_;
                const index_t idivp    = i / psize_;
                const index_t y_offset = idivp % psize_;
                const index_t channel  = idivp / psize_;
                const index_t y = (j / o_width_) * pstride_ + y_offset;
                const index_t x = (j % o_width_) * pstride_ + x_offset;
                if( x < img_.shape[0] && y < img_.shape[1] ){
                    return img_[channel][y][x];
                }else{
                    return 0.0f;
                }
            }
        private:
            Tensor<Device,3> img_;
            index_t psize_, pstride_, o_width_;
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
        template<typename Device, int dimsrc, int dimdst>
        struct Plan< ReshapeExp<Device,dimsrc,dimdst> >{
        public:
            Plan( const ReshapeExp<Device,dimsrc,dimdst> &e ): dptr_( e.src_.dptr ){
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

        template<typename Reducer, typename SrcExp>
        struct Plan< PoolingExp<Reducer, SrcExp> > {
        public:
            Plan( const PoolingExp<Reducer, SrcExp> &e )
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

        template<typename Device>
        struct Plan<UnPoolingExp<Device> > {
        public:
            Plan(const UnPoolingExp<Device> &e)
                : img_(e.img_), pooled_(e.pooled_), ksize_(e.ksize_), kstride_(e.kstride_), type_(e.type_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                real_t val = 0.0f;
                const index_t x = j;
                const index_t y = i % img_.shape[1];
                const index_t c = i / img_.shape[1];
                const index_t x_start = (x - ksize_) / kstride_ + 1;
                const index_t x_end = x / kstride_ + 1;
                const index_t y_start = (y -ksize_) / kstride_ + 1;
                const index_t y_end = y / kstride_ + 1;
                if (type_ == kMaxPooling) {
                    for (index_t h = y_start; h < y_end; ++h) {
                        for (index_t w = x_start; w < x_end; ++w) {
                            if (img_[c][y][x] == pooled_[c][h][w]) val++;
                        }
                    }
                } else {
                    utils::Error("Not implement");
                }
                return val;
            }
        private:
            Tensor<Device, 3> img_, pooled_;
            index_t ksize_;
            index_t kstride_;
            int type_;
        };

        template<typename Device>
        struct Plan< PaddingExp<Device> > {
        public:
            Plan(const PaddingExp<Device> &e)
                : img_(e.img_), pad_(e.pad_), new_height_(e.new_height_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                if (y < pad_ || x < pad_) return 0.0f;
                const index_t h = y - pad_;
                const index_t w = x - pad_;
                if (h >= img_.shape[1] || w >= img_.shape[0]) {
                    return 0;
                } else {
                    return img_[c][h][w];
                }
            }
        private:
            Tensor<Device, 3> img_;
            index_t pad_;
            index_t new_height_;
        };

        template<typename Device>
        struct Plan<UnPaddingExp<Device> > {
        public:
            Plan(const UnPaddingExp<Device> &e)
                : padded_img_(e.padded_img_), pad_(e.pad_), new_height_(e.new_height_) {}
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                const index_t h = y + pad_;
                const index_t w = x + pad_;
                return padded_img_[c][h][w];
            }
        private:
            Tensor<Device, 3> padded_img_;
            index_t pad_;
            index_t new_height_;
        };

    }; // namespace expr
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

