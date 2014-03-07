#ifndef MSHADOW_TENSOR_EXPR_EXT_INL_HPP
#define MSHADOW_TENSOR_EXPR_EXT_INL_HPP
/*!
 * \file tensor_expr_ext-inl.hpp
 * \brief some extension of expressionsx
 * \author Tianqi Chen
 */
namespace mshadow{
    namespace expr{
        // declaration of expressions
        /*! 
         * \brief replicate a 1 dimension tensor for nrow times 
         * input: Tensor<Device,1>: shape[0]
         * output: Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        template<typename Device>
        struct RepmatExp: public MakeTensorExp<RepmatExp<Device>,Device,2>{
            /*! \brief source operand */
            const Tensor<Device,1> &src;
            /*! \brief construct a repmat expression from src and nrow */
            RepmatExp( const Tensor<Device,1> &src, index_t nrow ):src(src){
                this->shape[0] = src.shape[0];
                this->shape[1] = nrow;
            }
        };
    };
    
    namespace expr{
        /*! 
         * \brief replicate a 1 dimension tensor for nrow times 
         * \param src Tensor<Device,1>: shape[0]
         * \param nrow number of rows to replicate
         * \return a expresion with type Tensor<Device,2> shape[0], shape[1] = nrow
         * \tparam Device which device it lies
         */
        template<typename Device>
        inline RepmatExp<Device> repmat( const Tensor<Device,1> &src, index_t nrow ){
            return RepmatExp<Device>( src, nrow );
        }
    };
    
    namespace expr{
        // actuall implementation of expression and plan
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
    };
};

#if MSHADOW_USE_SSE    
#include "tensor_sse-inl.hpp"
namespace mshadow{
    namespace expr{
        //template<typename Device>
        //struct SSECheck< RepmatExp<Device> >{
        //const static bool kPass = true;
        //};
        // TODO: add SSPlan, change SSEAlignCheck
    };
};
#endif
#endif

