#ifndef MSHADOW_TENSOR_EXPR_EXT_H
#define MSHADOW_TENSOR_EXPR_EXT_H
/*!
 * \file tensor_expr_ext.h
 * \brief some extension of expressions, used to support something beyond elementwise op
 * \author Tianqi Chen
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
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*( const ReduceTo1DExp<E,R,d> &e, real_t scale ){
            return ReduceTo1DExp<E,R,d>( e.src_, e.scale_*scale );
        }
        template<typename E, typename R,int d>
        inline ReduceTo1DExp<E,R,d> operator*( real_t scale, const ReduceTo1DExp<E,R,d> &e ){
            return ReduceTo1DExp<E,R,d>( e.src_, e.scale_*scale );
        }
    }; // namespace expr


    // Declaration of all functions go here
    namespace expr{
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
    };

    namespace expr{
        /*! \brief execution plan of repmat */
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
    }; // namespace expr
}; // namespace mshadow

namespace mshadow {
    namespace expr {
        /*!
         * \brief max pooling expr.
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct PoolingExp: public MakeTensorExp<PoolingExp<Device>, Tensor<Device, 3>, 3> {
            /*! \brief source operand */
            const Tensor<Device, 3> &img_;
            /*! \brief kernel size */
            index_t ksize_;
            /*! \brief kernel stride */
            index_t kstride_;
            /*! \brief pooling type */
            int type_;
            /*! \brief new heigh */
            index_t new_height_;
            PoolingExp(const Tensor<Device, 3> &img, index_t ksize, index_t kstride, int type)
                : img_(img), ksize_(ksize), kstride_(kstride), type_(type) {
                  const index_t p_height = (img.shape[1] - ksize) / kstride + 1;
                  const index_t p_width = (img.shape[0] - ksize) / kstride + 1;
                  this->shape_[0] = p_width;
                  this->shape_[1] = p_height;
                  this->shape_[2] = img.shape[2];
                  new_height_ = p_height;
            }
        };

        /*!
         * \brief pooling for 3D tensor
         * \return mat pooling result, shape[2]: channel shape[1]: height shape[0]:weight
         * \param img source image, shape[2]: channel shape[1]: height shape[0]:weight
         * \param ksize kernel size
         * \param kstride stride for each kernel
         */
        template<typename Device>
        inline PoolingExp<Device> pooling(const Tensor<Device, 3> &img, index_t ksize, index_t kstride, int type) {
            return PoolingExp<Device>(img, ksize, kstride, type);
        }

        template<typename Device>
        struct Plan<PoolingExp<Device> > {
        public:
            Plan(const PoolingExp<Device> &e)
                :img_(e.img_), ksize_(e.ksize_), kstride_(e.kstride_), type_(e.type_), new_height_(e.new_height_) {
            }
            MSHADOW_XINLINE real_t Eval(index_t i, index_t j) const {
                real_t val = 0;
                const index_t x = j;
                const index_t y = i % new_height_;
                const index_t c = i / new_height_;
                const index_t x_start = x * kstride_;
                const index_t x_end = x_start + ksize_< img_.shape[0] ? x_start + ksize_ : img_.shape[0];
                const index_t y_start = y * kstride_;
                const index_t y_end = y_start + ksize_< img_.shape[1] ? y_start + ksize_ : img_.shape[1];
                // TODO: Better reduction
                switch(type_) {
                    case kMaxPooling:
                        for (index_t h = y_start; h < y_end; ++h) {
                            for (index_t w = x_start; w < x_end; ++w) {
                                if (img_[c][h][w] > val) val = img_[c][h][w];
                            }
                        }
                        return val;
                    case kSumPooling:
                        for (index_t h = y_start; h < y_end; ++h) {
                            for (index_t w = x_start; w < x_end; ++w) {
                                val += img_[c][h][w];
                            }
                        }
                        return val;
                    //case kAvgPooling: return val / ksize_ / ksize_;
                    default: utils::Error("Unknown Pooling type");
                }
                return 0;
            }
        private:
            Tensor<Device, 3> img_;
            index_t ksize_, kstride_;
            int type_;
            index_t new_height_;
        };

    }; //namespace expr
};
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

