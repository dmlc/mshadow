#ifndef _CXXNET_TENSOR_GPU_INL_CUH_
#define _CXXNET_TENSOR_GPU_INL_CUH_

namespace cxxnet{
    // implementation of map
    template<typename Saver, typename BinaryMapper>
    inline void Map( GTensor2D dst, const GTensor2D &lhs, const GTensor2D &rhs ){
        // TODO, redirect to cuda/tensor_gpu-inl.cuh
    }    
};
#endif
