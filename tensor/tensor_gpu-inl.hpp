#ifndef TENSOR_GPU_INL_HPP
#define TENSOR_GPU_INL_HPP

namespace cxxnet {
    // implementation of map
    template<typename Saver, typename BinaryMapper>
    inline void Map(GTensor2D dst, const GTensor2D &lhs, const GTensor2D &rhs ) {
        // TODO, redirect to cuda/tensor_gpu-inl.cuh
    }
}; // namespace cxxnet
#endif // TENSOR_GPU_INL_HPP
