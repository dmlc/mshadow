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
            if( SSEAlignCheck<dim>::Check( exp.self() )){
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
        TypeCheckPass< TypeCheck<cpu,dim,E>::kPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        utils::Assert( ShapeCheck<dim>::Check( exp.self(), dst.shape ), "shape of Tensors in expression is not consistent with target" );
        #if MSHADOW_USE_SSE
        MapExpCPUEngine< SSECheck<E>::kPass,Saver,dim,E,etype >::Map( dst, exp );
        #else
        MapExpCPUEngine< false,Saver,dim,E,etype >::Map( dst, exp );        
        #endif
    }    
}; // namespace mshadow
#endif // TENSOR_CPU_INL_HPP
