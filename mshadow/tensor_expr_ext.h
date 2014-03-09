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
         * \brief replicate a 1 dimension tensor for nrow times 
         * input: Tensor<Device,1>: shape[0]
         * output: Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct RepmatExp: public MakeTensorExp< RepmatExp<Device>,Device,2>{
            /*! \brief source operand */
            const Tensor<Device,1> &src;
            /*! \brief construct a repmat expression from src and nrow */
            RepmatExp( const Tensor<Device,1> &src, index_t nrow ):src(src){
                this->shape[0] = src.shape[0];
                this->shape[1] = nrow;
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
         * \brief a expression that replicate a 1 dimension tensor for nrow times 
         * \param src Tensor<Device,1>: shape[0]
         * \param nrow number of rows to replicate
         * \return a expresion with type Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        template<typename Device>
        inline RepmatExp<Device> repmat( const Tensor<Device,1> &src, index_t nrow ){
            return RepmatExp<Device>( src, nrow );
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
            return ReduceTo1DExp<E,red::sum,0>( exp.self(), 1.0f );
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
                MapReduceTo1D<SV,Reducer,dimkeep>( dst, exp.src_, exp.scale_ );
            }
        };
    }; // namespace expr

    namespace expr{
        /*! \brief execution plan of repmat */
        template<typename Device>
        struct Plan< RepmatExp<Device> >{
        public:
            Plan( const RepmatExp<Device> &e ): dptr_( e.src.dptr ){}
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ x ];
            }
        private:
            const real_t *dptr_;          
        };        
    }; // namespace expr
}; // namespace mshadow

#if MSHADOW_USE_SSE 
// implementations of SSE support, if possible
#include "tensor_sse-inl.hpp"
namespace mshadow{
    namespace expr{
        template<>
        struct SSECheck< RepmatExp<cpu> >{
            const static bool kPass = true;
        };
        template<>
        struct SSEAlignCheck<2, RepmatExp<cpu> >{
            inline static bool Check( const RepmatExp<cpu> &exp ){
                return sse2::CheckAlign( exp.src.dptr );
            }
        };
        template<>
        class SSEPlan< RepmatExp<cpu> >{
        public:
            SSEPlan( const RepmatExp<cpu> &t )
                :dptr_(t.src.dptr){}
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

