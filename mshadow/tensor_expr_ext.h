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
         * \tparam Device which device it lies
         * \tparam Reducer which reducer to use
         * \tparam srcdim dimension of source 
         * \tparam dimkeep which dimension to be kept, 
         */
        //template<typename Device, typename Reducer, int srcdim, int dimkeep>
        //struct ReduceTo1DExp: public Exp< ReduceTo1DExp<Device,Reducer, dimkeep>, type::kComplex >{
            /*! \brief source operand */
        //const Tensor<Device,srcdim> &src;
        /*! \brief construct a repmat expression from src and nrow */
        //    ReductionExp( const Tensor<Device,srcdim> &src ):src(src){
        //this->shape[0] = src.shape[ dimkeep ];
        //}
        //};        
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
         * \brief a expression that replicate a 1 dimension tensor for nrow times 
         * \param src Tensor<Device,1>: shape[0]
         * \param nrow number of rows to replicate
         * \return a expresion with type Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        //template<typename Device>
        //inline ReduceTo1DExp<Device,red::sum,2,0> sum( const Tensor<Device,2> &src ){
        //return ReduceTo1DExp<Device,red::sum,2,0>( src );
        //}
        
    }; // namespace expr
}; // namespace mshadow

// ==================================================
//  implementations afterwards, 
//  no need to read if only use the functions
// --------------------------------------------------
namespace mshadow{
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

