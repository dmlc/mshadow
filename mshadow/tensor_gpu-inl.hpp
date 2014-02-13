#ifndef MSHADOW_TENSOR_GPU_INL_HPP
#define MSHADOW_TENSOR_GPU_INL_HPP
#ifdef __CUDACC__
/*!
 * \file tensor_cpu-inl.hpp
 * \brief implementation of GPU host code
 * \author Bing Hsu, Tianqi Chen
 */

// include this file only if the compiler is nvcc
#include "tensor.h"
#include "cuda/tensor_gpu-inl.cuh"

namespace mshadow {
    template<int dim>
    inline void AllocSpace(Tensor<gpu,dim> &obj){
        size_t pitch;
        cudaError_t err = cudaMallocPitch( (void**)&obj.dptr, &pitch, \
                                           obj.shape[0] * sizeof(real_t), obj.FlatTo2D().shape[1] );        
        utils::Assert( err == cudaSuccess, cudaGetErrorString(err) );
        obj.shape.stride_ = static_cast<index_t>( pitch / sizeof(real_t) );
    }

    template<int dim>
    inline Tensor<gpu,dim> NewGTensor(const Shape<dim> &shape, real_t initv){
        Tensor<gpu, dim> obj( shape );
        AllocSpace( obj );
        MapExp<sv::saveto>( obj, expr::ScalarExp( initv ) );
        return obj;
    }

    template<int dim>
    inline void FreeSpace(Tensor<gpu,dim> &obj){
        cudaFree( obj.dptr ); obj.dptr = NULL;
    }

    template<typename A,typename B, int dim>
    inline void Copy( Tensor<A,dim> _dst, Tensor<B,dim> _src, cudaMemcpyKind kind ){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        Tensor<A,2> dst = _dst.FlatTo2D();
        Tensor<B,2> src = _src.FlatTo2D();
        cudaError_t err = cudaMemcpy2D( dst.dptr, dst.shape.stride_ * sizeof(real_t), 
                                        src.dptr, src.shape.stride_ * sizeof(real_t), 
                                        dst.shape[0] * sizeof(real_t), 
                                        dst.shape[1], kind );
        utils::Assert( err == cudaSuccess, cudaGetErrorString(err) );
    }    
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<gpu,dim> &src ){
        Copy( dst, src, cudaMemcpyDeviceToHost );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &src ){
        Copy( dst, src, cudaMemcpyDeviceToDevice );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<cpu,dim> &src ){
        Copy( dst, src, cudaMemcpyHostToDevice );
    }

    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<gpu,dim> _dst, const expr::Plan<E> &plan ){ 
        cuda::MapPlan<Saver>( _dst.FlatTo2D(), plan );
    }
}; // namespace mshadow

#endif
#endif // TENSOR_GPU_INL_HPP
