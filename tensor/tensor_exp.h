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
        /*! 
         * \brief base class for expression 
         * \tparam SubType inheritated class must put their type into this 
         *         this is used to restrict the behavior of MakeExp
         */
        template<typename SubType>
        class Exp{
        public:
            /*! \brief get the subtype representation */
            _XINLINE_ const SubType& self( void ) const{
                return *static_cast<const SubType*>(this);
            }
            /*! 
             * \brief evaluate the expression at index [y][x] 
             *        to be implemented by SubType
             */
            _XINLINE_ real_t eval( index_t y, index_t x ) const;
        };

        /*! \brief tensor expression, evaluate tensor 2D */
        class TensorExp: public Exp<TensorExp>{
        public:
            TensorExp( const real_t *dptr, index_t stride )
                :dptr_(dptr),stride_(stride){}                        
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return dptr_[ y * stride_ + x ];
            }            
        private:
            const real_t  *dptr_;
            index_t stride_;
        };
        
        /*! \brief scalar expression, evaluae */
        class ScalarExp: public Exp<ScalarExp>{
        public:
            ScalarExp( real_t scalar ):scalar_(scalar){}
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
        class BinaryMapExp: public Exp< BinaryMapExp<OP,TA,TB> >{
        public:
            BinaryMapExp( const Exp<TA> &lhs, const Exp<TB> &rhs )
                :lhs_(lhs.self()), rhs_(rhs.self()){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return OP::Map( lhs_.eval( y, x ), rhs_.eval( y, x ) );
            }
        private:
            TA lhs_;
            TB rhs_;
        };

        /*! 
         * \brief binary map expression lhs [op] rhs
         * \tparam OP operator
         * \tparam TA type of src
         */
        template<typename OP, typename TA>
        class UnaryMapExp: public Exp< UnaryMapExp<OP,TA> >{
        public:
            UnaryMapExp( const Exp<TA> &src ):src_(src.self()){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return OP::Map( src_.eval( y, x ) );
            }
        private:
            TA src_;
        };
    }; // namespace algebra

    namespace algebra{
        // helper constructors

        template<typename Device, int dim>
        inline TensorExp MakeExp( const Tensor<Device,dim> &t ){
            return TensorExp( t.dptr, t.shape.stride_ );
        }
        
        inline ScalarExp MakeExp( index_t scalar ){
            return ScalarExp( scalar );
        }
        
        template<typename OP, typename TA,typename TB>
        inline BinaryMapExp<OP,TA,TB> MakeExp( const Exp<TA> &lhs, const Exp<TB> &rhs ){
            return BinaryMapExp<OP,TA,TB>( lhs, rhs );
        }
        
        template<typename OP, typename TA>
        inline UnaryMapExp<OP,TA> MakeExp( const Exp<TA> &src ){
            return UnaryMapExp<OP,TA>( src );
        }
    };
};
#endif
