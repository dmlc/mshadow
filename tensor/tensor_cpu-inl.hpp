#ifndef CXXNET_TENSOR_CPU_INL_HPP
#define CXXNET_TENSOR_CPU_INL_HPP

#include <algorithm>
#include <cstring>
#include "tensor_op.h"
#include "../utils/utils.h"

namespace cxxnet {
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
        Store<sv::saveto>( obj.FlatTo2D(), initv );
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

    template<typename Saver,int dim>
    inline void Store( Tensor<cpu,dim> _dst, real_t src ) {
        CTensor2D dst = _dst.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                Saver::Save(dst[y][x], src);
            }        
        }
    }

    template<typename Saver, typename BinaryMapper,int dim>
    inline void Map(Tensor<cpu,dim> _dst, const Tensor<cpu,dim> &_lhs, const Tensor<cpu,dim> &_rhs){
        utils::Assert( _dst.shape == _lhs.shape, "Map:shape mismatch" );
        utils::Assert( _dst.shape == _rhs.shape, "Map:shape mismatch" );
        CTensor2D dst = _dst.FlatTo2D();
        CTensor2D lhs = _lhs.FlatTo2D();
        CTensor2D rhs = _rhs.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], BinaryMapper::Map(lhs[y][x], rhs[y][x]));
            }
        }
    }
}; // namespace cxxnet
#endif // TENSOR_CPU_INL_HPP
