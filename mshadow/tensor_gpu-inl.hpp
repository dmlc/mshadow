#ifndef MSHADOW_TENSOR_GPU_INL_HPP
#define MSHADOW_TENSOR_GPU_INL_HPP
/*!
 * \file tensor_gpu-inl.hpp
 * \brief implementation of GPU host code
 * \author Bing Hsu, Tianqi Chen
 */
#include "tensor.h"

#if !(MSHADOW_USE_CUDA)
namespace mshadow {
    // do nothing if no GPU operation is involved
    inline void InitTensorEngine( void ){
    }
    inline void ShutdownTensorEngine( void ){
    }
};
#else
namespace mshadow {
    inline void InitTensorEngine( void ){
        cublasInit();
    }
    inline void ShutdownTensorEngine( void ){
        cublasShutdown();
    }

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
    inline void Copy(Tensor<A,dim> _dst, Tensor<B,dim> _src, cudaMemcpyKind kind){
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
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<gpu,dim> &src){
        Copy( dst, src, cudaMemcpyDeviceToHost );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &src){
        Copy( dst, src, cudaMemcpyDeviceToDevice );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<cpu,dim> &src){
        Copy( dst, src, cudaMemcpyHostToDevice );
    }
}; // namespace mshadow

#ifdef __CUDACC__
// the following part is included only if compiler is nvcc
#include "cuda/tensor_gpu-inl.cuh"

namespace mshadow{
    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<gpu,dim> _dst, const expr::Plan<E> &plan){ 
        cuda::MapPlan<Saver>( _dst.FlatTo2D(), plan );
    }

    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<gpu,dim> dst, const expr::Exp<E,etype> &exp ){
        using namespace expr;
        TypeCheckPass< TypeCheck<gpu,dim,E>::kMapPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        Shape<dim> eshape = ShapeCheck<dim,E>::Check( exp.self() );
        utils::Assert( eshape[0] == 0 || eshape == dst.shape, "shape of Tensors in expression is not consistent with target" );
        MapPlan<Saver>( dst, MakePlan( exp.self() ) );
    }

    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceToLowest( Tensor<gpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<gpu,1,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        Shape<2> eshape = ShapeCheck< ExpInfo<gpu,E>::kDim, E >::Check( exp.self() ).FlatTo2D();
        
        utils::Assert( eshape[0] == dst.shape[0], "reduction dimension do not match" );
        utils::Assert( eshape[1] != 0, "can not reduce over empty tensor" );
        cuda::MapReduceToLowest<Saver,Reducer>( dst, MakePlan( exp.self() ), scale, eshape );
    }

    inline void Softmax( Tensor<gpu,2> dst, Tensor<gpu,2> src ){
        cuda::Softmax( dst, src );
    }
}; // namespace mshadow

#endif // __CUDACC__

#endif // MSHADOW_USE_CUDA
#endif // TENSOR_GPU_INL_HPP
