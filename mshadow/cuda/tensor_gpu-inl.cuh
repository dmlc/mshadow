/*!
 *  Copyright (c) 2014 by Contributors
 * \file tensor_gpu-inl.cuh
 * \brief implementation of GPU code using CUDA
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
#define MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include "../tensor.h"
#include "./reduce.cuh"
#define MSHADOW_CUDA_POST_KERNEL_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)
namespace mshadow {
namespace cuda {
/* load unit for memory access, if CUDAARCH not defined, this is advanced nvcc */
#if MSHADOW_OLD_CUDA
const int kMemUnitBits = 4;
const int kMaxThreadsPerBlock = 512;
#else
const int kMemUnitBits = 5;
const int kMaxThreadsPerBlock = 1024;
#endif
/*! \brief number of units that can do synchronized update, half warp size */
const int kMemUnit = 1 << kMemUnitBits;
/*! \brief mask that could be helpful sometime */
const int kMemUnitMask = kMemUnit - 1;
/*! \brief suggested thread number(logscale) for mapping kernel */
const int kBaseThreadBits = 8;
/*! \brief suggested thread number for mapping kernel */
const int kBaseThreadNum  = 1 << kBaseThreadBits;
/*! \brief maximum value of grid */
const int kMaxGridNum = 65535;
/*! \brief suggested grid number for mapping kernel */
const int kBaseGridNum = 1024;
/*! \brief number of bytes memory arrays are expected to be aligned to */
const int kMemAlignBytes = 256;
/*! \brief get align stride for given size in x dimension */
inline index_t GetAlignStride(index_t xsize) {
  if (xsize >= MSHADOW_MIN_PAD_RATIO * 32) {
    return ((xsize  + kMemUnit - 1) >> kMemUnitBits) << kMemUnitBits;
  } else {
    // if originally space is not aligned, no necessary to to alligned thread allocation
    return xsize;
  }
}
/*! \brief align memory array size */
inline size_t AlignMemArraySize(const size_t size) {
  return ((size + kMemAlignBytes - 1)/kMemAlignBytes)*kMemAlignBytes;
}
inline void CheckLaunchParam(dim3 dimGrid, dim3 dimBlock, const char *estr = "") {
  if (dimBlock.x * dimBlock.y * dimBlock.z > static_cast<unsigned>(kMaxThreadsPerBlock) ||
      dimGrid.x > 65535 || dimGrid.y > 65535) {
    LOG(FATAL) << "too large launch parameter: "
      << estr << "["
      << dimBlock.x << ","
      << dimBlock.y << ","
      << dimBlock.z << "]";
  }
}
template<typename Saver, typename DstPlan,
         typename Plan, int block_dim_bits>
__device__ void MapPlanProc(DstPlan dst, index_t xstride,
                            Shape<2> dshape, const Plan exp, int block_idx) {
  const index_t tid = (block_idx << block_dim_bits) + threadIdx.x;
  const int y = tid / xstride;
  const int x = tid % xstride;
  if (y < dshape[0] && x < dshape[1]) {
    Saver::Save(dst.REval(y, x), exp.Eval(y,x));
  }
}
template<typename Saver,int block_dim_bits,
         typename DstPlan, typename Plan>
__global__ void MapPlanKernel(DstPlan dst, index_t xstride,
                              Shape<2> dshape, const Plan exp) {
  MapPlanProc<Saver, DstPlan, Plan, block_dim_bits>
      (dst, xstride, dshape, exp, blockIdx.x);
}
template<typename Saver, int block_dim_bits, int grid_size,
         typename DstPlan, typename Plan>
__global__ void MapPlanLargeKernel(DstPlan dst, index_t xstride,
                                   Shape<2> dshape, const Plan exp, int repeat) {
  for (int i = 0; i < repeat; ++i) {
  MapPlanProc<Saver, DstPlan, Plan, block_dim_bits>
      (dst, xstride, dshape, exp, blockIdx.x + i * grid_size);
  }
}

template<typename Saver, typename DstExp, typename E, typename DType>
inline void MapPlan(expr::Plan<DstExp, DType> dst,
                    const expr::Plan<E, DType> &plan,
                    Shape<2> dshape,
                    cudaStream_t stream) {
  const index_t xstride = GetAlignStride(dshape[1]);
  const int num_block = (dshape[0] * xstride + kBaseThreadNum-1) / kBaseThreadNum;
  dim3 dimBlock(kBaseThreadNum, 1, 1);

  if (num_block < kMaxGridNum) {
    dim3 dimGrid(num_block, 1, 1);
    MapPlanKernel<Saver, kBaseThreadBits,
                  expr::Plan<DstExp, DType>,
                  expr::Plan<E, DType> >
        <<<dimGrid, dimBlock, 0, stream>>>(dst, xstride, dshape, plan);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MapPlanKernel);
  } else {
    int repeat = (num_block + kBaseGridNum-1) / kBaseGridNum;
    dim3 dimGrid(kBaseGridNum, 1 , 1);
    MapPlanLargeKernel<Saver, kBaseThreadBits, kBaseGridNum,
                       expr::Plan<DstExp, DType>,
                       expr::Plan<E, DType> >
        <<<dimGrid, dimBlock, 0, stream>>>(dst, xstride, dshape, plan, repeat);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MapPlanLargeKernel);
  }
}
template<typename Saver,typename Reducer, int warp_bits,
         typename DType, typename DstPlan, typename Plan>
__global__ void MapRedKeepLowestKernel(DstPlan dst, Plan plan,
                                       DType scale, Shape<2> eshape) {
  const unsigned warp_size = 1 << warp_bits;
  const unsigned x = (blockIdx.x << warp_bits) + threadIdx.x;
  // to avoid bank conflict
  __shared__ DType s_res[warp_size][warp_size + 1];
  // note: reverse store [y][x], so that we can reduce over threadIdx.x, use warp optimization
  if (threadIdx.y < eshape[0] && x < eshape[1]) {
    s_res[threadIdx.x][threadIdx.y] = plan.Eval(threadIdx.y, x);
  }
  for (unsigned y = warp_size; y < eshape[0]; y += warp_size) {
    if (threadIdx.y + y < eshape[0] && x < eshape[1]) {
      Reducer::Reduce(s_res[threadIdx.x][threadIdx.y], plan.Eval(threadIdx.y + y, x));
    }
  }
  __syncthreads();
  if (eshape[0] >= warp_size) {
    Reduce1D<Reducer, warp_bits>(s_res[threadIdx.y]);
  } else {
    Reduce1DNotAlign<Reducer, warp_bits>(s_res[threadIdx.y], eshape[0]);
  }
  __syncthreads();

  if (threadIdx.y == 0 && x < eshape[1]) {
    Saver::Save(dst.REval(0, x),  DType(s_res[threadIdx.x][0] * scale));
  }
}

template<typename Saver, typename Reducer,
         typename DstExp, typename E, typename DType>
inline void MapReduceKeepLowest(expr::Plan<DstExp, DType> dst,
                                const expr::Plan<E, DType> &plan,
                                DType scale, Shape<2> eshape,
                                cudaStream_t stream) {
  dim3 dimBlock(kMemUnit, kMemUnit);
  dim3 dimGrid((eshape[1] + kMemUnit - 1) >> kMemUnitBits);
  CheckLaunchParam(dimGrid, dimBlock, "MapRedKeepLowestKernel");
  MapRedKeepLowestKernel<Saver, Reducer, kMemUnitBits, DType,
                         expr::Plan<DstExp, DType>,
                         expr::Plan<E, DType> >
      <<<dimGrid, dimBlock, 0, stream>>>(dst, plan, scale, eshape);
  MSHADOW_CUDA_POST_KERNEL_CHECK(MapRedKeepLowestKernel);
}

template<typename Saver, typename Reducer, int block_dim_bits,
         typename DType, typename DstPlan, typename Plan>
__global__ void MapReduceKeepDim1Kernel(DstPlan dst, Plan plan, DType scale, Shape<4> pshape) {
  const int block_size = 1 << block_dim_bits;
  __shared__ DType s_rec[block_size];
  const int c = blockIdx.x;
  const index_t tot = pshape[3] * pshape[2] * pshape[0];

  DType res; Reducer::SetInitValue(res);
  for (index_t i_offset = 0; i_offset < tot; i_offset += block_size) {
    index_t i = i_offset + threadIdx.x;
    if (i< tot) {
      const index_t x = i % pshape[3];
      i /= pshape[3];
      const index_t y = i % pshape[2];
      const index_t n = i / pshape[2];
      Reducer::Reduce(res, plan.Eval((n * pshape[1] + c) * pshape[2] + y, x));
    }
  }
  s_rec[threadIdx.x] = res;
  __syncthreads();
  Reduce1D<Reducer, block_dim_bits>(s_rec);
  if (threadIdx.x == 0) {
    Saver::Save(dst.REval(0, c), DType(s_rec[0] * scale));
  }
}

template<typename Saver, typename Reducer, typename DstExp, typename E, typename DType>
inline void MapReduceKeepDim1(expr::Plan<DstExp, DType> dst,
                              const expr::Plan<E, DType> &plan,
                              DType scale, Shape<4> pshape,
                              cudaStream_t stream) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid (pshape[1]);
  CheckLaunchParam(dimGrid, dimBlock, "MapReduceKeepDim1");
  MapReduceKeepDim1Kernel<Saver,Reducer,kBaseThreadBits, DType,
                          expr::Plan<DstExp, DType>,
                          expr::Plan<E, DType> >
      <<<dimGrid, dimBlock, 0, stream>>>(dst, plan, scale, pshape);
  MSHADOW_CUDA_POST_KERNEL_CHECK(MapReduceKeepDim1Kernel);
}

template<int x_bits, typename DType>
__global__ void GetBatchedViewKernel(DType **dst, DType *src, int num, int stride) {
  const int x_size = 1 << x_bits;
  const int start = threadIdx.x;
  // Copy the addresses of src to dst every stride steps
  for (int i = start; i < num; i += x_size) {
    dst[i] = src + i * stride;
  }
}

template<typename DType>
inline void GetBatchedView(DType **dst, DType *src, int num, int stride,
                           Stream<gpu> *stream) {
  cudaStream_t stream_ = Stream<gpu>::GetStream(stream);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(1);
  CheckLaunchParam(dimGrid, dimBlock, "GetBatchedView");
  GetBatchedViewKernel<kBaseThreadBits, DType>
    <<<dimGrid, dimBlock, 0, stream_>>> (dst, src, num, stride);
  MSHADOW_CUDA_POST_KERNEL_CHECK(GetBatchedViewKernel);
}

template<int x_bits, typename DType, typename DstPlan, typename SrcPlan1, typename SrcPlan2>
__global__ void SoftmaxGradKernel(DstPlan dst, SrcPlan1 src, SrcPlan2 label, index_t xmax) {
  const unsigned x_size = 1 << x_bits;
  const int y = blockIdx.x;
  const int k = static_cast<int>(label.Eval(0, y));

  // calculate normalizer, with writeback
  for (unsigned x = 0; x < xmax; x += x_size) {
    const unsigned xindex = x + threadIdx.x;
    if (xindex < xmax) {
      if (xindex == k) {
        dst.REval(y, xindex) = src.Eval(y, xindex) - 1.0f;
      } else {
        dst.REval(y, xindex) = src.Eval(y, xindex);
      }
    }
  }
}

template<int x_bits, typename DType, typename DstPlan, typename SrcPlan1, typename SrcPlan2>
__global__ void SoftmaxGradKernel(DstPlan dst, SrcPlan1 src, SrcPlan2 label, index_t xmax,
                                  DType ignore_label) {
  const unsigned x_size = 1 << x_bits;
  const int y = blockIdx.x;
  const int k = static_cast<int>(label.Eval(0, y));

  // calculate normalizer, with writeback
  for (unsigned x = 0; x < xmax; x += x_size) {
    const unsigned xindex = x + threadIdx.x;
    if (xindex < xmax) {
      if (static_cast<int>(ignore_label) == k) {
        dst.REval(y, xindex) = 0.0f;
      } else {
        if (xindex == k) {
          dst.REval(y, xindex) = src.Eval(y, xindex) - 1.0f;
        } else {
          dst.REval(y, xindex) = src.Eval(y, xindex);
        }
      }
    }
  }
}

template<int x_bits, typename DType,  typename DstPlan, typename SrcPlan>
__global__ void SoftmaxKernel(DstPlan dst, SrcPlan src, index_t xmax) {
  const unsigned x_size = 1 << x_bits;
  const int y = blockIdx.x;
  __shared__ DType s_rec[x_size];
  // step 1: get max
  if (threadIdx.x < xmax) {
    s_rec[threadIdx.x] = src.Eval(y, threadIdx.x);
  }
  for (unsigned x = x_size; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      DType a = src.Eval(y, x + threadIdx.x);
      s_rec[threadIdx.x] = max(a, s_rec[threadIdx.x]);
    }
  }
  __syncthreads();
  if (threadIdx.x >= xmax) {
    s_rec[threadIdx.x] = s_rec[0];
  }
  __syncthreads();
  Reduce1D<red::maximum, x_bits>(s_rec);
  __syncthreads();
  DType smax = s_rec[0];
  __syncthreads();
  s_rec[threadIdx.x] = 0.0f;
  __syncthreads();

  // calculate normalizer, with writeback
  for (unsigned x = 0; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      DType p = expf(src.Eval(y, x + threadIdx.x) - smax);
      s_rec[threadIdx.x] += p;
      // write back first, will fetch later
      dst.REval(y, x + threadIdx.x) = p;
    }
  }
  // calculate normalizer
  __syncthreads();
  Reduce1D<red::sum, x_bits>(s_rec);
  __syncthreads();
  DType ssum = s_rec[0];

  for (unsigned x = 0; x < xmax; x += x_size) {
    if (x + threadIdx.x < xmax) {
      dst.REval(y, x + threadIdx.x) /= ssum;
    }
  }
}

template<typename DType>
inline void Softmax(Tensor<gpu, 2, DType> &dst,
                    const Tensor<gpu, 2, DType> &src) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "Softmax: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "Softmax");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SoftmaxKernel<kBaseThreadBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(src),
       dst.size(1));
  MSHADOW_CUDA_POST_KERNEL_CHECK(SoftmaxKernel);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SoftmaxGradKernel<kBaseThreadBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(src),
       expr::MakePlan(label),
       dst.size(1));
  MSHADOW_CUDA_POST_KERNEL_CHECK(SoftmaxGradKernel);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 2, DType> &dst,
                        const Tensor<gpu, 2, DType> &src,
                        const Tensor<gpu, 1, DType> &label,
                        const DType &ignore_label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SoftmaxGradKernel<kBaseThreadBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(src),
       expr::MakePlan(label),
       dst.size(1),
       ignore_label);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SoftmaxGradKernel);
}

template<int n_bits, typename DType>
__global__ void Softmax3DGradKernel(Tensor<gpu, 3, DType> dst,
                                    const Tensor<gpu, 3, DType> src,
                                    const Tensor<gpu, 2, DType> label) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;

  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    const int k = static_cast<int>(label[y][n_index]);
    for (index_t i = 0; i < xmax; ++i) {
      if (i == k) {
        dst[y][i][n_index] = src[y][i][n_index] - 1.0f;
      } else {
        dst[y][i][n_index] = src[y][i][n_index];
      }
    }
  }
}

template<int n_bits, typename DType>
__global__ void Softmax3DGradKernel(Tensor<gpu, 3, DType> dst,
                                    const Tensor<gpu, 3, DType> src,
                                    const Tensor<gpu, 2, DType> label,
                                    DType ignore_label) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;
  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    int k = static_cast<int>(label[y][n_index]);
    if (k == static_cast<int>(ignore_label)) {
      for (index_t i = 0; i < xmax; ++i) {
        dst[y][i][n_index] = 0.0f;
      }
    } else {
      for (index_t i = 0; i < xmax; ++i) {
        if (i == k) {
          dst[y][i][n_index] = src[y][i][n_index] - 1.0f;
        } else {
          dst[y][i][n_index] = src[y][i][n_index];
        }
      }
    }
  }
}

template<int n_bits, typename DType>
__global__ void Softmax3DKernel(Tensor<gpu, 3, DType> dst,
                    const Tensor<gpu, 3, DType> src) {
  const index_t xmax = dst.size(1);
  const index_t nmax = dst.size(2);
  const unsigned n_size = 1 << n_bits;
  const int y = blockIdx.x;
  const int n = threadIdx.x;

  for (index_t n_index = n; n_index < nmax; n_index += n_size) {
    DType smax = src[y][0][n_index];
    for (index_t i = 1; i < xmax; ++i) {
      smax = max(smax, src[y][i][n_index]);
    }
    DType ssum = 0.0f;
    for (index_t i = 0; i < xmax; ++i) {
      DType p = expf(src[y][i][n_index] - smax);
      ssum += p;
      dst[y][i][n_index] = p;
    }
    for (index_t i = 0; i < xmax; ++i) {
      dst[y][i][n_index] /= ssum;
    }
  }
}

template<typename DType>
inline void Softmax(Tensor<gpu, 3, DType> &dst,
                    const Tensor<gpu, 3, DType> &src) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "Softmax: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "Softmax");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src);
  MSHADOW_CUDA_POST_KERNEL_CHECK(Softmax3DKernel);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CHECK_EQ(dst.size(2), label.size(1)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DGradKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src, label);
  MSHADOW_CUDA_POST_KERNEL_CHECK(Softmax3DGradKernel);
}

template<typename DType>
inline void SoftmaxGrad(Tensor<gpu, 3, DType> &dst,
                        const Tensor<gpu, 3, DType> &src,
                        const Tensor<gpu, 2, DType> &label,
                        const DType &ignore_label) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(0));
  CHECK_EQ(dst.shape_, src.shape_) << "SoftmaxGrad: shape mismatch";
  CHECK_EQ(dst.size(0), label.size(0)) << "SoftmaxGrad: label shape mismatch";
  CHECK_EQ(dst.size(2), label.size(1)) << "SoftmaxGrad: label shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  Softmax3DGradKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, src, label, ignore_label);
  MSHADOW_CUDA_POST_KERNEL_CHECK(Softmax3DGradKernel);
}

template<int x_bits, typename DType, typename DstPlan, typename SrcPlan1, typename SrcPlan2>
__global__ void AddTakeGradKernel(DstPlan dst,
                                  SrcPlan1 index, SrcPlan2 src,
                                  index_t ymax, index_t xmax) {
  const unsigned x_size = 1 << x_bits;
  const int xindex = blockIdx.x * x_size + threadIdx.x;
  __shared__ int ptr;
  for (unsigned y = 0; y < ymax; ++y) {
    if (threadIdx.x == 0)  ptr = index.Eval(0, y);
    __syncthreads();
    if (xindex < xmax) {
      dst.REval(ptr, xindex) += src.Eval(y, xindex);
    }
  }
}

template<int SZ, typename DType, typename IdxType>
__global__ void AddTakeGradLargeBatchKernel(DType* dst,
                                           // If idx_start == NULL, then in-kernel edge
                                           // detection is used
                                           const IdxType *idx_start,
                                           // idx_start_size_ptr ignored if idx_start == NULL
                                           const int* idx_start_size_ptr,
                                           const IdxType *sorted, const IdxType *index,
                                           const DType *src,
                                           int ymax, int xmax) {
  // Size of the shared memory is [blockDim.x*SZ*blockDim.y]*sizeof(DType)
  extern __shared__ char sh_grad_weight_char[];
  DType* sh_grad_weight = (DType*)sh_grad_weight_char;

  int iidx_end = (idx_start == NULL) ? ymax : *idx_start_size_ptr;

  for (int iidx = blockIdx.y;iidx < iidx_end;iidx += gridDim.y) {

    // Thread block sums up elements in the range [idx_begin, idx_end-1]
    int idx_begin, idx_end;
    int sorted_value;
    if (idx_start == NULL) {
      idx_begin = iidx;
      sorted_value = static_cast<int>(sorted[idx_begin]);
      if (idx_begin > 0 && sorted_value == static_cast<int>(sorted[idx_begin - 1])) continue;
      // Algorithm is explained using an example:
      //   blockDim.x = 32
      //   blockDim.y = 4
      //   sorted[idx_begin:] = [4 4 4 9]
      //   (3,4) denotes threadIdx.x=3, threadIdx.y=4, ":" is used for ranges 
      //   (0:31,0:3) sorted_value = 4
      idx_end = idx_begin + 1;
      unsigned int* sh_ballot = (unsigned int*)sh_grad_weight_char;
      int no_edge = 0;
      do {
        int idx = idx_end + threadIdx.x + threadIdx.y*blockDim.x;
        // Example:
        //   (0:1,0) sorted_idx = 4
        //   (rest)  sorted_idx = -1
        int sorted_idx = (idx < ymax) ? static_cast<int>(sorted[idx]) : -1;
        // Example:
        //   (0:31,0) sh_ballot[0]     = 0b100
        //   (rest)   sh_ballot[1...3] = 0
        // sh_ballot[] tells us which thread within the warp found the edge
        sh_ballot[threadIdx.y] = __ballot(sorted_value != sorted_idx);
        __syncthreads();
        // No edge if sh_ballot[threadIdx.x] == 0
        // NOTE: All warps have the same value for no_edge
        // Example:
        //   (0,:)  no_edge = 0
        //   (rest) no_edge = 1
        no_edge = (threadIdx.x < blockDim.y) ? (sh_ballot[threadIdx.x] == 0) : 1;
        idx_end += blockDim.x*blockDim.y;
        // Example:
        //   __all(no_edge) = 0 since no_edge = 0 for threadIdx.x = 0, hence we leave the loop
      } while (__all(no_edge));
      idx_end -= blockDim.x*blockDim.y;
      // Find the first edge
      // Example:
      //   (0,:)  val = 1 << 0 = 1
      //   (rest) val = 0
      unsigned int val = (threadIdx.x < blockDim.y && sh_ballot[threadIdx.x] != 0) ?
        (1 << threadIdx.x) : 0;
      // NOTE: We do butterfly reduction on the entire warp width
      //       so that all threads have the same result
      // Example:
      //   (all) val = 1
      #pragma unroll
      for (int i=warpSize/2;i>=1;i/=2) val |= __shfl_xor(val, i);
      // __ffs() returns the position of first set bit, 1...32. __ffs(1) = 1
      // j will be the warp index where edge was found
      // Example:
      //   (all) j = 1 - 1 = 0
      int j = __ffs(val) - 1;
      // j = warp index where the edge was found
      // __ffs(sh_ballot[j]) - 1 = warp lane where the edge was found
      // idx_end points to the one over the last value.
      // Example:
      //  idx_end += 0*blockDim.x + _ffs(0b100) - 1 = 0 + 3 - 1 = 2
      //  sorted[idx_end] = 9
      idx_end += j*blockDim.x + __ffs(sh_ballot[j]) - 1;
      __syncthreads();
    } else {
      idx_begin = idx_start[iidx];
      idx_end   = ((iidx + 1) < iidx_end) ? idx_start[iidx + 1] : ymax;
      sorted_value = static_cast<int>(sorted[idx_begin]);
    }

    const int start_feature = threadIdx.x + blockIdx.x * blockDim.x * SZ;
    const int dst_row = sorted_value * xmax;

    int num_idx = idx_end - idx_begin;
    int idx0 = idx_begin + threadIdx.y*num_idx/blockDim.y;
    int idx1 = idx_begin + (threadIdx.y + 1)*num_idx/blockDim.y;

    // Read and sum data into grad_weight[]
    DType grad_weight[SZ];
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++) {
      grad_weight[ii] = (DType)0;
    }
    for (int idx=idx0; idx < idx1;idx++) {
      const int src_row = static_cast<int>(index[idx]) * xmax;
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int feature_dim = start_feature + ii * blockDim.x;
        if (feature_dim < xmax)
        {
          grad_weight[ii] += src[src_row + feature_dim];
        }
      }
    }
    #pragma unroll
    for (int ii = 0; ii < SZ; ii++) {
      sh_grad_weight[threadIdx.x + ii*blockDim.x + threadIdx.y*blockDim.x*SZ] = grad_weight[ii];
    }
    __syncthreads();
    // We now have grad_weight[] values, reduce within thread block
    for (int t=1;t < blockDim.y;t <<= 1) {
      DType tmp[SZ];
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        tmp[ii] = (threadIdx.y + t < blockDim.y) ?
          sh_grad_weight[threadIdx.x + ii*blockDim.x + (threadIdx.y + t)*blockDim.x*SZ] : (DType)0;
      }
      __syncthreads();
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        sh_grad_weight[threadIdx.x + ii*blockDim.x + threadIdx.y*blockDim.x*SZ] += tmp[ii];
      }
      __syncthreads();
    }
    // Result is in sh_grad_weight[threadIdx.x + ii*blockDim.x]
    if (threadIdx.y == 0) {
      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * blockDim.x;
        if (feature_dim < xmax) {
          dst[dst_row + feature_dim] += sh_grad_weight[threadIdx.x + ii*blockDim.x];
        }
      }
    }
  
  }
}

template<typename IndexType, typename DType>
inline void AddTakeGrad(Tensor<gpu, 2, DType> dst,
                        const Tensor<gpu, 1, IndexType>& index,
                        const Tensor<gpu, 2, DType> &src) {
  CHECK_EQ(dst.CheckContiguous(), true);
  CHECK_EQ(index.CheckContiguous(), true);
  CHECK_EQ(src.CheckContiguous(), true);
  const int kUnitBits = kMemUnitBits + 1;
  dim3 dimBlock(1 << kUnitBits);
  dim3 dimGrid((dst.size(1) + (1 << kUnitBits) - 1) >> kUnitBits);

  CHECK_EQ(dst.size(1), src.size(1)) << "AddTakeGrad: shape mismatch";
  CHECK_EQ(index.size(0), src.size(0)) << "AddTakeGrad: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "AddTakeGrad");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);

  AddTakeGradKernel<kUnitBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(index),
       expr::MakePlan(src),
       src.size(0),
       src.size(1));
  MSHADOW_CUDA_POST_KERNEL_CHECK(AddTakeGradKernel);
}

template<typename IndexType>
inline size_t AddTakeGradLargeBatchWorkspaceSize(size_t num_items) {
  size_t encode_bytes = 0;
  cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
    (NULL, encode_bytes, NULL, NULL, NULL, NULL, num_items);
  size_t exclusivesum_bytes = 0;
  cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>(NULL, exclusivesum_bytes,
    NULL, NULL, num_items);
  size_t temporary_bytes = max(encode_bytes, exclusivesum_bytes);
  size_t unique_align_bytes = AlignMemArraySize(num_items*sizeof(IndexType));
  size_t counts_align_bytes = AlignMemArraySize(num_items*sizeof(IndexType));
  size_t num_runs_align_bytes = AlignMemArraySize(1*sizeof(int));
  size_t temporary_align_bytes = AlignMemArraySize(temporary_bytes);
  return (unique_align_bytes + counts_align_bytes + num_runs_align_bytes + temporary_align_bytes);
}

template<typename IndexType, typename DType>
inline void AddTakeGradLargeBatch(Tensor<gpu, 2, DType> dst,
                                  const Tensor<gpu, 1, IndexType>& sorted,
                                  const Tensor<gpu, 1, IndexType>& index,
                                  const Tensor<gpu, 2, DType> &src,
                                  Tensor<gpu, 1, char>* workspace) {
  CHECK_EQ(dst.CheckContiguous(), true);
  CHECK_EQ(sorted.CheckContiguous(), true);
  CHECK_EQ(index.CheckContiguous(), true);
  CHECK_EQ(src.CheckContiguous(), true);
  const int kWarpBits = kMemUnitBits;
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  IndexType* sum_counts_ptr = NULL;
  int* num_runs_ptr = NULL;
  if (dst.size(0)*4 < src.size(0) && workspace != NULL) {
    // Workspace given and potentially loops at least 4 times, use CUB to create sum_counts
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [unique_out, counts_out, temporary_storage]
    size_t unique_align_bytes = AlignMemArraySize(sorted.size(0)*sizeof(IndexType));
    size_t counts_align_bytes = AlignMemArraySize(sorted.size(0)*sizeof(IndexType));
    size_t num_runs_align_bytes = AlignMemArraySize(1*sizeof(int));

    size_t encode_bytes = 0;
    cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
      (NULL, encode_bytes, NULL, NULL, NULL, NULL, sorted.size(0), stream);
    size_t exclusivesum_bytes = 0;
    cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>
      (NULL, exclusivesum_bytes, NULL, NULL, sorted.size(0), stream);
    size_t temporary_bytes = max(encode_bytes, exclusivesum_bytes);

    // Check that we have enough storage
    CHECK_GE(workspace->size(0), unique_align_bytes + counts_align_bytes + 
      num_runs_align_bytes + temporary_bytes);

    IndexType* unique_out_ptr = reinterpret_cast<IndexType*>(workspace->dptr_);
    IndexType* counts_out_ptr = reinterpret_cast<IndexType*>(workspace->dptr_ + unique_align_bytes);
    num_runs_ptr = reinterpret_cast<int*>(workspace->dptr_ + unique_align_bytes +
      counts_align_bytes);
    void* temporary_storage = reinterpret_cast<void *>(workspace->dptr_ + unique_align_bytes + 
      counts_align_bytes + num_runs_align_bytes);

    cub::DeviceRunLengthEncode::Encode<IndexType*, IndexType*, IndexType*, int*>
    (temporary_storage, temporary_bytes, sorted.dptr_, unique_out_ptr, counts_out_ptr,
      num_runs_ptr, sorted.size(0), stream);

    sum_counts_ptr = unique_out_ptr;
    cub::DeviceScan::ExclusiveSum<IndexType*, IndexType*>
    (temporary_storage, temporary_bytes, counts_out_ptr, sum_counts_ptr,
      sorted.size(0), stream);
  }

  const int num_unique_est = min(dst.size(0), src.size(0));
  const int max_nthread = 128;
  const int num_y = max(src.size(0)/num_unique_est, 1);
  const int kWarpSize = (1 << kWarpBits);
  const int block_dim_x = kWarpSize;
  const int block_dim_y = min(num_y, max_nthread/block_dim_x);
  const int SZ = min((src.size(1) + block_dim_x - 1) / block_dim_x, 4);
  const int grid_dim_x = (src.size(1) + block_dim_x * SZ - 1) / (block_dim_x * SZ);
  const int grid_dim_y = min(num_unique_est, kBaseGridNum);
  dim3 dimBlock(block_dim_x, block_dim_y);
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  // Maximum shared memory usage: 128*4*sizeof(DType), which is 4K for 64bit DType elements
  int shmem_size = dimBlock.x*SZ*dimBlock.y*sizeof(DType);

  CHECK_EQ(dst.size(1), src.size(1)) << "AddTakeGradLargeBatch: shape mismatch";
  CHECK_EQ(index.size(0), src.size(0)) << "AddTakeGradLargeBatch: shape mismatch";
  CheckLaunchParam(dimGrid, dimBlock, "AddTakeGradLargeBatch");

  switch (SZ) {
    case 1:
    AddTakeGradLargeBatchKernel<1, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 2:
    AddTakeGradLargeBatchKernel<2, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 3:
    AddTakeGradLargeBatchKernel<3, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    case 4:
    AddTakeGradLargeBatchKernel<4, DType>
        <<<dimGrid, dimBlock, shmem_size, stream>>>
        (dst.dptr_, sum_counts_ptr, num_runs_ptr,
         sorted.dptr_, index.dptr_, src.dptr_,
         static_cast<int>(src.size(0)),
         static_cast<int>(src.size(1)));
    break;
    default:
    LOG(FATAL) << "AddTakeGradLargeBatch, incorrect value SZ " << SZ;
    break;
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(AddTakeGradLargeBatchKernel);
}

template<int warp_bits, typename DType, typename DstPlan, typename IndexPlan, typename SrcPlan>
__global__ void IndexFillKernel(DstPlan dst,
                                IndexPlan index, SrcPlan src,
                                index_t ymax, int xmax) {
  int src_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (src_idx < ymax) {
    int dst_idx = static_cast<int>(index.Eval(0, src_idx));
    for (int i = threadIdx.x; i < xmax; i += blockDim.x) {
      dst.REval(dst_idx, i) = src.Eval(src_idx, i);
    }
  }
}

template<typename IndexType, typename DType>
inline void IndexFill(Tensor<gpu, 2, DType> dst,
                      const Tensor<gpu, 1, IndexType>& index,
                      const Tensor<gpu, 2, DType> &src) {
  CHECK_EQ(dst.CheckContiguous(), true);
  CHECK_EQ(index.CheckContiguous(), true);
  CHECK_EQ(src.CheckContiguous(), true);
  CHECK_EQ(dst.size(1), src.size(1)) << "IndexFill: shape mismatch";
  CHECK_EQ(index.size(0), src.size(0)) << "IndexFill: shape mismatch";
  const int block_dim_x = 1 << kMemUnitBits;
  const int block_dim_y = 4;
  const int grid_dim_x = (src.size(0) + block_dim_y - 1) / block_dim_y;
  dim3 dimBlock(block_dim_x, block_dim_y);
  dim3 dimGrid(grid_dim_x);
  CheckLaunchParam(dimGrid, dimBlock, "IndexFill");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);

  IndexFillKernel<kMemUnitBits, DType>
      <<<dimGrid, dimBlock, 0, stream>>>
      (expr::MakePlan(dst),
       expr::MakePlan(index),
       expr::MakePlan(src),
       src.size(0),
       src.size(1));
  MSHADOW_CUDA_POST_KERNEL_CHECK(IndexFillKernel);
}

template<typename KDType, typename VDType>
inline size_t SortByKeyWorkspaceSize(size_t num_items) {
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
      NULL, NULL, NULL, NULL, num_items);
  size_t keys_align_bytes = AlignMemArraySize(num_items*sizeof(KDType));
  size_t values_align_bytes = AlignMemArraySize(num_items*sizeof(VDType));
  size_t sortpairs_align_bytes = AlignMemArraySize(sortpairs_bytes);
  return (keys_align_bytes + values_align_bytes + sortpairs_align_bytes);
}

template<typename KDType, typename VDType>
inline void SortByKey(Tensor<gpu, 1, KDType> keys, Tensor<gpu, 1, VDType> values,
                      bool is_ascend, Tensor<gpu, 1, char>* workspace,
                      const int begin_bit, const int end_bit) {
  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
#if CUDA_VERSION >= 7000
  cudaStream_t stream = Stream<gpu>::GetStream(keys.stream_);
  if (workspace != NULL) {
    // Workspace given, sort using CUB
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [keys_out, values_out, temporary_storage]
    size_t keys_align_bytes = AlignMemArraySize(keys.size(0)*sizeof(KDType));
    size_t values_align_bytes = AlignMemArraySize(keys.size(0)*sizeof(VDType));
    // Get the size of internal storage (for checking purposes only)
    size_t sortpairs_bytes = 0;
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
          NULL, NULL, NULL, NULL,
          keys.size(0), begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending<KDType, VDType>(NULL, sortpairs_bytes,
          NULL, NULL, NULL, NULL,
          keys.size(0), begin_bit, end_bit, stream);
    }
    // Check that we have enough storage
    CHECK_GE(workspace->size(0), keys_align_bytes + values_align_bytes + sortpairs_bytes);
    //
    KDType* keys_out_ptr = reinterpret_cast<KDType *>(workspace->dptr_);
    VDType* values_out_ptr = reinterpret_cast<VDType *>(workspace->dptr_ + keys_align_bytes);
    void* temp_storage = reinterpret_cast<void *>(workspace->dptr_ + keys_align_bytes + values_align_bytes);
    // Sort
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        keys.size(0), begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        keys.size(0), begin_bit, end_bit, stream);
    }
    // Copy result back to [keys, values]
    Tensor<gpu, 1, KDType> keys_out(keys_out_ptr, Shape1(keys.size(0)), keys.stream_);
    Tensor<gpu, 1, VDType> values_out(values_out_ptr, Shape1(keys.size(0)), keys.stream_);
    Copy(keys, keys_out, keys.stream_);
    Copy(values, values_out, values.stream_);
  } else {
    // No workspace, sort using thrust
    thrust::device_ptr<KDType> key_iter = thrust::device_pointer_cast(keys.dptr_);
    thrust::device_ptr<VDType> value_iter = thrust::device_pointer_cast(values.dptr_);
    if (is_ascend) {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter, thrust::less<KDType>());
    } else {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + keys.size(0), value_iter, thrust::greater<KDType>());
    }
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
#else
  LOG(FATAL) << "SortByKey is only supported for CUDA version >=7.0!";
#endif
}

template<typename DType>
inline void SortByKey(Tensor<gpu, 1, mshadow::half::half_t> keys, Tensor<gpu, 1, DType> values,
                      bool is_ascend) {
  LOG(FATAL) << "SortByKey for half_t is not implemented!";
}

template<typename DType>
inline void SortByKey(Tensor<gpu, 1, DType> keys, Tensor<gpu, 1, mshadow::half::half_t> values,
  bool is_ascend) {
  LOG(FATAL) << "SortByKey for half_t is not implemented!";
}
}  // namespace cuda
}  // namespace mshadow
#endif  // MSHADOW_CUDA_TENSOR_GPU_INL_CUH_
