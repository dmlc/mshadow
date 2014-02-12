#ifndef CXXNET_TENSOR_EXP_H
#define CXXNET_TENSOR_EXP_H

#include "tensor.h"
#include "tensor_op.h"

/*!
 * \file tensor_exp.h
 * \brief definitions of tensor expression
 *
 * \author Tianqi Chen
 */
namespace cxxnet{
    /*! 
     * \brief namespace for algebra tree,
     *        abstract expressions in algebra 
     */
    namespace algebra{
        /*! \brief base class for expression */
        class Exp{
        public:
            /*! 
             * \brief evaluate the expression at index [y][x] 
             *        to be implemented by SubType
             */
            _XINLINE_ real_t eval( index_t y, index_t x ) const;            
        };

        /*! \brief tensor expression, evaluate tensor 2D */
        class TensorExp: public Exp{
        public:
            TensorExp( const real_t *dptr, index_t stride )
                :dptr_(dptr),stride_(stride){}

            template<typename Device, int dim>
            TensorExp( const Tensor<Device,dim> &t )
                :dptr_(t.dptr), stride_(t.shape.stride_){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return dptr_[ y * stride_ + x ];
            }            
        private:
            const real_t  *dptr_;
            index_t stride_;
        };
        
        /*! \brief scalar expression, evaluae */
        class ScalarExp: public Exp{
        public:
            ScalarExpExp( real_t scalar ):scalar_(scalar){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return scalar_;
            }
        private:
            real_t scalar_;        
        };

        /*! 
         * \brief binary map expression lhs [op] rhs
         * \tparam OP operator
         * \tparam TA type of lhs
         * \tparam TB type of rhs
         */
        template<typename OP, typename TA, typename TB>
        class BinaryMapExp: public Exp{
        public:
            BinaryMapExp( TA lhs, TB rhs )
                :lhs_(lhs), rhs_(rhs){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                OP::Map( lhs_.eval( y, x ), rhs_.eval( y, x ) );
            }
        private:
            TA lhs_;
            TB rhs_;
        }
    };
    
    namespace algebra{

    };
};
#endif
