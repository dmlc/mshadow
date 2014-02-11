#ifndef TENSOR_GPU_INL_HPP
#define TENSOR_GPU_INL_HPP
// exclude this code if the compiler is not nvcc
#ifdef __CUDA_ARCH__
#include "cuda/tensor_gpu-inl.cuh"

namespace cxxnet {
    // implementation of map
    template<typename Saver, typename BinaryMapper>
    inline void Map(GTensor2D dst, const GTensor2D &lhs, const GTensor2D &rhs ) {
        // TODO, redirect to cuda/tensor_gpu-inl.cuh
    }
}; // namespace cxxnet
#endif
#endif // TENSOR_GPU_INL_HPP
