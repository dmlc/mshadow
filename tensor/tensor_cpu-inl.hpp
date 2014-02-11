#ifndef TENSOR_CPU_INL_HPP
#define TENSOR_CPU_INL_HPP

#include <algorithm>
#include "tensor_op.h"

namespace cxxnet {
    // cozy allocation, no alignment so far
    // TODO: aligned allocation for SSE
    template<int dimension>
    inline void AllocSpace(Tensor<cpu,dimension> &obj){
        obj.shape.stride_ = obj.shape[ 0 ];
        obj.dptr = new real_t[ obj.shape.MSize() ];
    }

    template<int dimension>
    inline Tensor<cpu,dimension> NewCTensor(const Shape<dimension> &shape, real_t initv){
        Tensor<cpu, dimension> obj( shape );
        AllocSpace( obj );
        Store<sv::saveto>( obj.FlatTo2D(), initv );
        return obj;
    }
    // free the space
    template<int dimension>
    inline void FreeSpace(Tensor<cpu,dimension> &obj){
        delete [] obj.dptr;
    }        
    // implementation of store
    template<typename SV>
    inline void Store( CTensor2D dst, real_t src ) {
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                sv::Saver<SV>::Save(dst[y][x], src);
            }        
        }
    }
    // implementation of map
    template<typename SV, typename OP>
    inline void Map(CTensor2D dst, const CTensor2D &lhs, const CTensor2D &rhs) {
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                // trust your compiler! -_- they will optimize it
                sv::Saver<SV>::Save(dst[y][x], op::BinaryMapper<OP>::Map(lhs[y][x], rhs[y][x]));
            }
        }
    }
}; // namespace cxxnet
#endif // TENSOR_CPU_INL_HPP
