#ifndef CXXNET_TENSOR_EXPR_ENGINE_INL_HPP
#define CXXNET_TENSOR_EXPR_ENGINE_INL_HPP
/*!
 * \file tensor_expr_engine-inl.hpp
 * \brief definitions of how expressions should be evaluated
 * \author Tianqi Chen
 */
#include "tensor_expr.h"

namespace cxxnet{    
    namespace expr{
        // plan that can be used to carry out execution
        template<typename ExpType>
        class Plan{
        public:   
            /*! 
             * \brief evaluate the expression at index [y][x] 
             *        to be implemented by SubType
             */
            _XINLINE_ real_t eval( index_t y, index_t x ) const;            
        };
        
        template <typename Device, int dim>
        class Plan< Tensor<Device,dim> >{
        public:
            Plan( const real_t *dptr, index_t stride )
                :dptr_(dptr),stride_(stride){}
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return dptr_[ y * stride_ + x ];
            }            
        private:
            const real_t  *dptr_;
            index_t stride_;
        };
        
        template<>
        class Plan<ScalarExp>{
        public:
            Plan( real_t scalar ):scalar_(scalar){}
            /*! \brief evaluate at [y][x] */
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                    return scalar_;
            }
        private:
            real_t scalar_;        
        };
        
        template<typename OP, typename TA, typename TB,int etype>
        class Plan< BinaryMapExp<OP,TA,TB,etype> >{
        public:
            Plan( const Plan<TA> &lhs, const Plan<TB> &rhs )
                :lhs_(lhs), rhs_(rhs){}
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return OP::Map( lhs_.eval( y, x ), rhs_.eval( y, x ) );
            }
        private:
            Plan<TA> lhs_;
            Plan<TB> rhs_;
        };
        
        template<typename OP, typename TA, int etype>
        class Plan< UnaryMapExp<OP,TA,etype> >{
        public:
            Plan( const Plan<TA> &src ):src_(src){}
            _XINLINE_ real_t eval( index_t y, index_t x ) const{
                return OP::Map( src_.eval( y, x ) );
            }
        private:
            Plan<TA> src_;
        };
    };

    namespace expr{
        /*! \brief namespace of execution engine */
        namespace engine{
            // translate from exp to execution plan
            inline Plan<ScalarExp> MakePlan( const ScalarExp &e ){
                return Plan<ScalarExp>( e.scalar_ );
            }
            
            template<typename Device, int dim>
            inline Plan< Tensor<Device,dim> > MakePlan( const Tensor<Device,dim> &e ){
                return Plan< Tensor<Device,dim> >( e.dptr, e.shape.stride_ );
            }

            template<typename OP, typename TA, typename TB, int etype>
            inline Plan< BinaryMapExp<OP,TA,TB,etype> > MakePlan( const BinaryMapExp<OP,TA,TB,etype> &e ){
                return Plan< BinaryMapExp<OP,TA,TB,etype> >( MakePlan(e.lhs_), MakePlan(e.rhs_) );
            }

            template<typename OP, typename TA, int etype>
            inline Plan< UnaryMapExp<OP,TA,etype> > MakePlan( const UnaryMapExp<OP,TA,etype> &e ){
                return Plan< UnaryMapExp<OP,TA,etype> >( MakePlan(e.src_) );
            }
        };


        template<typename SV, typename Device, int dim> 
        struct ExpEngine<SV, Tensor<Device,dim> >{
            template<typename E>
            inline static void eval( Tensor<Device,dim>& dst, const Exp<E,type::kMapper> &exp ){
                MapPlan<SV>( dst, engine::MakePlan( exp.self() ) );
            }
            template<typename E>
            inline static void eval( Tensor<Device,dim>& dst, const Exp<E,type::kContainer> &exp ){
                MapPlan<SV>( dst, engine::MakePlan( exp.self() ) );
            }
        };
    };

    // implementation of MapExp
    template<typename Saver, typename Device, int dim, typename E, int etype>
    inline void MapExp(Tensor<Device,dim> dst, const expr::Exp<E,etype> &exp ){
        expr::ExpEngine<Saver,Tensor<Device,dim> >::eval( dst, exp );
    }
};
#endif
