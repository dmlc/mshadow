#ifndef MSHADOW_TENSOR_CPU_INL_HPP
#define MSHADOW_TENSOR_CPU_INL_HPP
/*!
 * \file tensor_cpu-inl.hpp
 * \brief implementation of CPU host code
 * \author Bing Hsu, Tianqi Chen
 */

#include <cstring>
#include "tensor_base.h"

namespace mshadow {
    // cozy allocation, no alignment so far
    // TODO: aligned allocation for SSE
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj){
        obj.shape.stride_ = obj.shape[ 0 ];
        obj.dptr = new real_t[ obj.shape.MSize() ];
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
        delete [] obj.dptr; obj.dptr = NULL;
    }

    template<int dim>
    inline void Copy(Tensor<cpu,dim> _dst, const Tensor<cpu,dim> &_src ){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        CTensor2D dst = _dst.FlatTo2D();
        CTensor2D src = _src.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            memcpy( dst[y].dptr, src[y].dptr, sizeof(real_t) * dst.shape[0] );
        }
    }

    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<cpu,dim> _dst, const expr::Plan<E> &plan){
        CTensor2D dst = _dst.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], plan.eval( y, x ) );
            }
        }
    }
}; // namespace mshadow
#endif // TENSOR_CPU_INL_HPP
