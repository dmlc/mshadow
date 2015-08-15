/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_cpu-inl.h
 * \brief implementation of CPU host code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_TENSOR_CPU_INL_H_
#define MSHADOW_TENSOR_CPU_INL_H_
#include <cstring>
#include "./base.h"
#include "./tensor.h"
#include "./sse-inl.h"

namespace mshadow {
template<>
inline void InitTensorEngine<cpu>(int dev_id) {
}
template<>
inline void ShutdownTensorEngine<cpu>(void) {
}

template<>
inline void SetDevice<cpu>(int devid) {
}
template<>
inline Stream<cpu> *NewStream<cpu>(bool create_blas_handle,
                                   bool create_dnn_handle) {
  return new Stream<cpu>();
}
template<>
inline void DeleteStream<cpu>(Stream<cpu> *stream) {
  delete stream;
}

template<typename xpu>
inline void *AllocHost_(size_t size);
template<typename xpu>
inline void FreeHost_(void * dptr);

#ifdef __CUDACC__
template<>
inline void *AllocHost_<gpu>(size_t size) {
  void *dptr;
  utils::Check(cudaMallocHost(&dptr, size,
                 cudaHostAllocPortable) == cudaSuccess,
               "AllocHost");
  return dptr;
}
template<>
inline void FreeHost_<gpu>(void *dptr) {
  cudaFreeHost(dptr);
}
#endif

template<>
inline void *AllocHost_<cpu>(size_t size) {
  size_t pitch;
  return sse2::AlignedMallocPitch(&pitch, size, 1);
}
template<>
inline void FreeHost_<cpu>(void *dptr) {
  sse2::AlignedFree(dptr);
}

template<typename xpu, int dim, typename DType>
inline void AllocHost(Tensor<cpu, dim, DType> *obj) {
  obj->stride_ = obj->size(dim - 1);
  utils::Assert(obj->CheckContiguous(), "AllocHost");
  void *dptr = AllocHost_<xpu>(obj->MSize() * sizeof(DType));
  obj->dptr_ = reinterpret_cast<DType*>(dptr);
}
template<typename xpu, int dim, typename DType>
inline void FreeHost(Tensor<cpu, dim, DType> *obj) {
  utils::Assert(obj->dptr_ != NULL, "FreeHost:: double free");
  FreeHost_<xpu>(obj->dptr_);
  obj->dptr_ = NULL;
}

template<int dim, typename DType>
inline void AllocSpace(Tensor<cpu, dim, DType> *obj, bool pad) {
  size_t pitch;
  void *dptr;
  if (pad) {
    dptr = sse2::AlignedMallocPitch
        (&pitch, obj->size(dim - 1) * sizeof(DType), obj->shape_.FlatTo2D()[0]);
    obj->stride_ = static_cast<index_t>(pitch / sizeof(DType));
  } else {
    obj->stride_ = obj->size(dim - 1);
    dptr = sse2::AlignedMallocPitch
        (&pitch, obj->shape_.Size() * sizeof(DType), 1);
  }
  obj->dptr_ = reinterpret_cast<DType*>(dptr);
}
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType>
NewTensor(const Shape<dim> &shape, DType initv, bool pad, Stream<Device> *stream_) {
  Tensor<Device, dim, DType> obj(shape);
  obj.stream_ = stream_;
  AllocSpace(&obj, pad);
  MapExp<sv::saveto>(&obj, expr::ScalarExp<DType>(initv));
  return obj;
}
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> *obj) {
  sse2::AlignedFree(obj->dptr_);
  obj->dptr_ = NULL;
}
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> _dst,
                 const Tensor<cpu, dim, DType> &_src,
                 Stream<cpu> *stream) {
  utils::Check(_dst.shape_ == _src.shape_, "Copy:shape mismatch");
  Tensor<cpu, 2, DType> dst = _dst.FlatTo2D();
  Tensor<cpu, 2, DType> src = _src.FlatTo2D();
  for (index_t y = 0; y < dst.size(0); ++y) {
    memcpy(dst[y].dptr_, src[y].dptr_, sizeof(DType) * dst.size(1));
  }
}

template<typename Saver, typename R, int dim,
         typename DType, typename E>
inline void MapPlan(TRValue<R, cpu, dim, DType> *dst,
                    const expr::Plan<E, DType> &plan) {
  Shape<2> shape = expr::ShapeCheck<dim, R>::Check(dst->self()).FlatTo2D();
  expr::Plan<R, DType> dplan = expr::MakePlan(dst->self());
  for (index_t y = 0; y < shape[0]; ++y) {
    for (index_t x = 0; x < shape[1]; ++x) {
      // trust your compiler! -_- they will optimize it
      Saver::Save(dplan.REval(y, x), plan.Eval(y, x));
    }
  }
}
// code to handle SSE optimization
template<bool pass_check, typename Saver,
         typename R, int dim,
         typename DType, typename E, int etype>
struct MapExpCPUEngine {
  inline static void Map(TRValue<R, cpu, dim, DType> *dst,
                         const expr::Exp<E, DType, etype> &exp) {
    MapPlan<Saver>(dst, MakePlan(exp.self()));
  }
};

#if MSHADOW_USE_SSE
template<typename SV, int dim, typename DType, typename E, int etype>
struct MapExpCPUEngine<true, SV, Tensor<cpu, dim, DType>,
                       dim, DType, E, etype> {
  inline static void Map(Tensor<cpu, dim, DType> *dst,
                         const expr::Exp<E, DType, etype> &exp) {
    if (expr::SSEAlignCheck<dim, E>::Check(exp.self()) &&
        expr::SSEAlignCheck<dim, Tensor<cpu, dim, DType> >::Check(*dst)) {
      expr::MapSSEPlan<SV>(dst->self(), MakeSSEPlan(exp.self()));
    } else {
      MapPlan<SV>(dst, MakePlan(exp.self()));
    }
  }
};
#endif

template<typename Saver, typename R, int dim,
         typename DType, typename E, int etype>
inline void MapExp(TRValue<R, cpu, dim, DType> *dst,
                   const expr::Exp<E, DType, etype> &exp) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, dim, DType, E>::kMapPass>
      ::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
  Shape<dim> eshape = expr::ShapeCheck<dim, E>::Check(exp.self());
  Shape<dim> dshape = expr::ShapeCheck<dim, R>::Check(dst->self());
  utils::Check(eshape[0] == 0 || eshape == dshape,
               "Assignment: Shape of Tensors are not consistent with target");
#if MSHADOW_USE_SSE
  MapExpCPUEngine<expr::SSECheck<E>::kPass, Saver, R, dim, DType, E, etype>
      ::Map(dst->ptrself(), exp);
#else
  MapExpCPUEngine<false, Saver, R, dim, DType, E, etype>::Map(dst, exp);
#endif
}

template<typename Saver, typename Reducer,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepLowest(TRValue<R, cpu, 1, DType> *dst,
                                const expr::Exp<E, DType, etype> &exp,
                                DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, 1, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  Shape<2> eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self()).FlatTo2D();
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  utils::Check(eshape[1] == dshape[0],
               "MapReduceKeepLowest::reduction dimension do not match");
  utils::Check(eshape[0] != 0, "can not reduce over empty tensor");
  // execution
  expr::Plan<R, DType> dplan = MakePlan(dst->self());
  expr::Plan<E, DType> splan = MakePlan(exp.self());
  for (index_t x = 0; x < eshape[1]; ++x) {
    DType res = splan.Eval(0, x);
    for (index_t y = 1; y < eshape[0]; ++y) {
      Reducer::Reduce(res, splan.Eval(y, x));
    }
    Saver::Save(dplan.REval(0, x), res * scale);
  }
}

template<typename Saver, typename Reducer, int dimkeep,
         typename R, typename DType, typename E, int etype>
inline void MapReduceKeepHighDim(TRValue<R, cpu, 1, DType> *dst,
                                 const expr::Exp<E, DType, etype> &exp,
                                 DType scale) {
  expr::TypeCheckPass<expr::TypeCheck<cpu, dimkeep, DType, E>::kRedPass>
      ::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
  typedef Shape<expr::ExpInfo<E>::kDim> EShape;
  EShape eshape = expr::ShapeCheck<expr::ExpInfo<E>::kDim, E>
      ::Check(exp.self());
  Shape<1> dshape = expr::ShapeCheck<1, R>::Check(dst->self());
  utils::Check(eshape[dimkeep] == dshape[0],
               "MapReduceKeepHighDim::reduction dimension do not match");
  // use equvalent form
  Shape<4> pshape = Shape4(eshape.ProdShape(0, dimkeep),
                           eshape[dimkeep],
                           eshape.ProdShape(dimkeep + 1, EShape::kSubdim),
                           eshape[EShape::kSubdim]);
  // execution
  expr::Plan<R, DType> dplan = MakePlan(dst->self());
  expr::Plan<E, DType> splan = MakePlan(exp.self());
  for (index_t c = 0; c < pshape[1]; ++c) {
    DType res; Reducer::SetInitValue(res);
    for (index_t n = 0; n < pshape[0]; ++n) {
      DType tres; Reducer::SetInitValue(tres);
      for (index_t y = 0; y < pshape[2]; ++y) {
        for (index_t x = 0; x < pshape[3]; ++x) {
          Reducer::Reduce(tres,
                          splan.Eval((n * pshape[1] + c) * pshape[2] + y, x));
        }
      }
      Reducer::Reduce(res, tres);
    }
    Saver::Save(dplan.REval(0, c), res * scale);
  }
}

template<typename DType>
inline void Softmax(Tensor<cpu, 1, DType> dst,
                    const Tensor<cpu, 1, DType> &energy) {
  DType mmax = energy[0];
  for (index_t x = 1; x < dst.size(0); ++x) {
    if (mmax < energy[x]) mmax = energy[x];
  }
  DType sum = 0.0f;
  for (index_t x = 0; x < dst.size(0); ++x) {
    dst[x] = std::exp(energy[x] - mmax);
    sum += dst[x];
  }
  for (index_t x = 0; x < dst.size(0); ++x) {
    dst[x] /= sum;
  }
}
template<typename DType>
inline void Softmax(Tensor<cpu, 2, DType> dst,
                    const Tensor<cpu, 2, DType> &energy) {
  utils::Check(dst.shape_ == energy.shape_, "Softmax: shape mismatch");
  for (index_t y = 0; y < dst.size(0); ++y) {
    Softmax(dst[y], energy[y]);
  }
}

template<typename DType>
inline DType VDot(const Tensor<cpu, 1, DType> &lhs,
                  const Tensor<cpu, 1, DType> &rhs) {
  utils::Check(lhs.shape_ == rhs.shape_, "VDot: shape mismatch");
  DType sum = static_cast<DType>(0);
  for (index_t x = 0; x < lhs.size(0); ++x) {
    sum += lhs[x] * rhs[x];
  }
  return sum;
}
}  // namespace mshadow
#endif  // MSHADOW_TENSOR_CPU_INL_H_
