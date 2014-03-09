#ifndef MSHADOW_TENSOR_CPU_INL_HPP
#define MSHADOW_TENSOR_CPU_INL_HPP
/*!
 * \file tensor_cpu-inl.hpp
 * \brief implementation of CPU host code
 * \author Bing Hsu, Tianqi Chen
 */
#include <cstring>
#include "tensor_base.h"
#include "tensor_sse-inl.hpp"

namespace mshadow {
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj){
        size_t pitch;
        obj.dptr = (real_t*)sse2::AlignedMallocPitch
            ( pitch, obj.shape[0] * sizeof(real_t), obj.FlatTo2D().shape[1] ); 
        obj.shape.stride_ = static_cast<index_t>( pitch / sizeof(real_t) );
    }

    template<int dim>
    inline Tensor<cpu,dim> NewCTensor(const Shape<dim> &shape, real_t initv){
        Tensor<cpu, dim> obj( shape );
        AllocSpace( obj );
        MapExp<sv::saveto>( obj, expr::ScalarExp( initv ) );
        return obj;
    }

    template<int dim>
    inline void FreeSpace(Tensor<cpu,dim> &obj){
        sse2::AlignedFree( obj.dptr );
        obj.dptr = NULL;
    }

    template<int dim>
    inline void Copy(Tensor<cpu,dim> _dst, const Tensor<cpu,dim> &_src ){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        Tensor<cpu,2> src = _src.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            memcpy( dst[y].dptr, src[y].dptr, sizeof(real_t) * dst.shape[0] );
        }
    }

    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<cpu,dim> _dst, const expr::Plan<E> &plan){
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            for (index_t x = 0; x < dst.shape[0]; ++x ) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], plan.Eval( y, x ) );
            }
        }
    }
    
    // code to handle SSE optimization
    template<bool pass_check,typename Saver, int dim, typename E, int etype>
    struct MapExpCPUEngine;
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<false,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            MapPlan<SV>( dst, MakePlan( exp.self() ) );
        }        
    };

    #if MSHADOW_USE_SSE
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<true,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            using namespace expr;
            if( SSEAlignCheck<dim,E>::Check( exp.self() )){
                MapSSEPlan<SV>( dst, MakeSSEPlan( exp.self() ) );
            }else{
                MapPlan<SV>( dst, MakePlan( exp.self() ) );
            } 
        }
    };
    #endif

    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
        using namespace expr;        
        TypeCheckPass< TypeCheck<cpu,dim,E>::kMapPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        Shape<dim> eshape = ShapeCheck<dim,E>::Check( exp.self() );
        utils::Assert( eshape[0] == 0 || eshape == dst.shape, "shape of Tensors in expression is not consistent with target" );
        #if MSHADOW_USE_SSE
        MapExpCPUEngine< SSECheck<E>::kPass,Saver,dim,E,etype >::Map( dst, exp );
        #else
        MapExpCPUEngine< false,Saver,dim,E,etype >::Map( dst, exp );        
        #endif
    }
    
    // implementation of MapReduce to 1D
    template<int dimkeep, int dimsrc, typename Saver, typename Reducer, typename E>
    struct MapRedTo1DEngine{
        inline static void MapRed( Tensor<cpu,1> dst, const expr::Plan<E> &plan, Shape<dimsrc> eshape, real_t scale );
    };
    
    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceTo1D( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,1,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();        
        typedef Shape< ExpInfo<cpu,E>::kDim > SrcShape;
        SrcShape shape = ShapeCheck< ExpInfo<cpu,E>::kDim, E >::Check( exp.self() );

        utils::Assert( shape[dimkeep] == dst.shape[0], "reduction dimension do not match" );
        MapRedTo1DEngine< dimkeep, ExpInfo<cpu,E>::kDim, Saver, Reducer, E >::MapRed
            ( dst, MakePlan(exp.self()), shape, scale );
    }
    
    template<int dimsrc, typename Saver, typename Reducer, typename E>
    struct MapRedTo1DEngine<0,dimsrc,Saver,Reducer,E>{
        inline static void MapRed( Tensor<cpu,1> dst, const expr::Plan<E> &plan, Shape<dimsrc> eshape_, real_t scale ){
            Shape<2> eshape = eshape_.FlatTo2D();
            utils::Assert( eshape[1] != 0, "can not reduce over empty tensor" );                        
            for( index_t x = 0; x < eshape[0]; ++x ){
                real_t res = plan.Eval( 0, x );
                for( index_t y = 1; y < eshape[1]; ++y ){
                    Reducer::Reduce( res, plan.Eval( y, x ) );
                }
                Saver::Save( dst[x], res*scale );
            }
        }
    };

}; // namespace mshadow
#endif // TENSOR_CPU_INL_HPP
