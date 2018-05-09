/*!
 * Copyright (c) 2018 by Contributors
 * \file random_gpu.cuh
 * \brief random number generator implementation using CUDA
 * \author Deokjae Lee
 */
#ifndef MSHADOW_CUDA_RANDOM_GPU_CUH_
#define MSHADOW_CUDA_RANDOM_GPU_CUH_

#include <limits>
#include <type_traits>
#include "cumath.cuh"

namespace mshadow {
namespace cuda {

template<typename Rng, typename DType>
struct GPURandomImpl {
  /*!
   * \brief allocate internal parallel random number generators of `impl`.
   * \param stream cuda stream
   * \param seed seed for `impl`
   * \param sequence_id sequence id for `impl`.
   *        Generators with different sequence ids are statistically independent.
   * \param n_rngs number of internal parallel random generators.
   */
  static void AllocState(
    cudaStream_t stream,
    GPURandomImpl<Rng, DType>* impl,
    uint64_t seed,
    uint32_t sequence_id,
    unsigned n_rngs
  );

  static void FreeState(GPURandomImpl<Rng, DType>* impl);

  /*!
   * \brief seed this generator
   * \param stream cuda stream
   * \param seed seed for the generator
   * \param sequence_id sequence id for the generator
   *        Generators with different sequence ids are statistically independent.
   */
  inline void Seed(cudaStream_t stream, uint64_t seed, uint32_t sequence_id);

  /*!
   * \brief get a set of random integers
   * \param stream cuda stream
   * \param dst destination
   * \param size number of random numbers to draw
   * \tparam IntType any integer type
   */
  template<typename IntType>
  inline void GenerateInt(cudaStream_t stream, IntType* dst, size_t size);

  /*!
   * \brief generate uniformly distributed random numbers
   * \param stream cuda stream
   * \param dst destination
   * \param size number of random numbers to draw
   * \param a lower bound of the distribution. inclusive
   * \param b upper bound of the distribution. inclusive if `DType` is integer and exclusive otherwise.
   */
  inline void GenerateUniform(
    cudaStream_t stream,
    DType* dst,
    size_t size,
    DType a = 0,
    DType b = 1
  );

  /*!
   * \brief generate data from Gaussian distribution
   * \param stream cuda stream
   * \param dst destination
   * \param size number of random numbers to draw
   * \param mean mean
   * \param stddev standard deviation
   */
  inline void GenerateGaussian(
    cudaStream_t stream,
    DType* dst,
    size_t size,
    DType mean = 0,
    DType stddev = 1
  );

 private:
  /*! \brief random generators in parallel use */
  Rng* rngs;
  /*! \brief the number of parallel random generators */
  unsigned n_rngs;
  /*! \brief the number of cuda threads per block */
  unsigned n_threads;
  /*! \brief the number of cuda blocks */
  unsigned n_blocks;
};

#ifdef __CUDACC__

template<typename Rng>
__global__ void RandomInitKernel(
  Rng* const rngs,
  const uint64_t seed,
  const uint64_t stream_offset
) {
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  rngs[rng_id] = Rng(seed, stream_offset + rng_id);
}

template<typename Rng>
__global__ void RandomSeedKernel(
  Rng* const rngs,
  const uint64_t seed,
  const uint64_t stream_offset
) {
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  rngs[rng_id].seed(seed, stream_offset + rng_id);
}

template<typename IntType, typename Rng>
__global__ void RandomUIntKernel(
  Rng* const rngs,
  const unsigned n_rngs,
  const size_t size,
  IntType* const ret
) {
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  Rng rng = rngs[rng_id];
  IntType* const bound = ret + n_rngs * (size / n_rngs);
  for (IntType* offset = ret; offset < bound; offset += n_rngs) {
    *(offset + rng_id) = static_cast<IntType>(rng());
  }
  if (rng_id < size % n_rngs) {
    *(bound + rng_id) = static_cast<IntType>(rng());
  }
  rngs[rng_id] = rng;
}

template<typename From, typename To>
MSHADOW_FORCE_INLINE __device__ To CastWithRounginDown(From f) {
  return f;
}

template<>
MSHADOW_FORCE_INLINE __device__ mshadow::half::half_t CastWithRounginDown(float f) {
  return mshadow::half::half_t(__float2half_rd(f));
}

template<typename Rng, typename DType>
__global__ void RandomUniformRealKernel(
  Rng* const rngs,
  const unsigned n_rngs,
  const DType lo,
  const DType hi,
  const size_t size,
  DType* const ret
) {
  using FType = typename std::conditional<sizeof(DType) <= 4, float, double>::type;
  using CM = CuMath<FType>;
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  Rng rng = rngs[rng_id];
  const FType w = hi - lo;
  DType* const bound = ret + n_rngs * (size / n_rngs);
  for (DType* offset = ret; offset < bound; offset += n_rngs) {
    // Rounding down is used to ensure the upper bound exclusive.
    *(offset + rng_id) = CastWithRounginDown<FType, DType>(
      CM::FmaRD(FType(RandomReal<Rng, FType>::Generate(rng)), w, lo)
    );
  }
  if (rng_id < size % n_rngs) {
    // Rounding down is used to ensure the upper bound exclusive.
    *(bound + rng_id) = CastWithRounginDown<FType, DType>(
      CM::FmaRD(FType(RandomReal<Rng, FType>::Generate(rng)), w, lo)
    );
  }
  rngs[rng_id] = rng;
}

template<typename Rng, typename DType>
__global__ void RandomUniformIntKernel(
  Rng* const rngs,
  const unsigned n_rngs,
  const DType lo,
  const DType hi,
  const size_t size,
  DType* const ret
) {
  using UIType = typename Rng::result_type;
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  Rng rng = rngs[rng_id];
  DType* const bound = ret + n_rngs * (size / n_rngs);
  if (static_cast<UIType>(hi - lo) == rng.max()) {
    for (DType* offset = ret; offset < bound; offset += n_rngs) {
      *(offset + rng_id) = rng();
    }
    if (rng_id < size % n_rngs) {
      *(bound + rng_id) = rng();
    }
  } else {
    const UIType w = static_cast<UIType>(hi - lo) + 1;
    const UIType v = Rng::max() - Rng::max() % w;
    for (DType* offset = ret; offset < bound; offset += n_rngs) {
      UIType r = rng();
      while (r >= v) {
        r = rng();
      }
      *(offset + rng_id) = lo + static_cast<DType>(r % w);
    }
    if (rng_id < size % n_rngs) {
      UIType r = rng();
      while (r >= v) {
        r = rng();
      }
      *(bound + rng_id) = lo + static_cast<DType>(r % w);
    }
  }
  rngs[rng_id] = rng;
}

template<typename Rng, typename DType>
__global__ void RandomGaussianKernel(
  Rng* const rngs,
  const unsigned n_rngs,
  const DType mean,
  const DType stddev,
  const size_t size,
  DType* const ret
) {
  // Box-Muller method
  using FType = typename std::conditional<sizeof(DType) <= 4, float, double>::type;
  using CM = CuMath<FType>;
  const FType two_pi = 6.283185307179586;
  const size_t rng_id = blockIdx.x * blockDim.x + threadIdx.x;
  Rng rng = rngs[rng_id];
  DType* const bound1 = ret + 2 * n_rngs * (size / (2 * n_rngs));
  DType* const bound2 = ret + size;
  for (DType* offset = ret; offset < bound1; offset += 2 * n_rngs) {
    FType u1 = RandomReal<Rng, FType>::Generate(rng);
    while (u1 == 0) {
      u1 = RandomReal<Rng, FType>::Generate(rng);
    }
    const FType u2 = RandomReal<Rng, FType>::Generate(rng);
    const FType v1 = stddev * CM::Sqrt(-FType(2) * CM::Log(u1));
    const FType v2 = two_pi * u2;
    *(offset + rng_id) = static_cast<DType>(CM::Fma(v1, CM::Cos(v2), mean));
    DType* i = offset + n_rngs + rng_id;
    *i = static_cast<DType>(CM::Fma(v1, CM::Sin(v2), mean));
  }
  if (2 * rng_id < size % (2 * n_rngs)) {
    FType u1 = RandomReal<Rng, FType>::Generate(rng);
    while (u1 == 0) {
      u1 = RandomReal<Rng, FType>::Generate(rng);
    }
    const FType u2 = RandomReal<Rng, FType>::Generate(rng);
    const FType v1 = stddev * CM::Sqrt(-FType(2) * CM::Log(u1));
    const FType v2 = two_pi * u2;
    DType* i = bound1 + 2 * rng_id;
    *i = static_cast<DType>(CM::Fma(v1, CM::Cos(v2), mean));
    if (i + 1 < bound2) {
      *(i + 1) = static_cast<DType>(CM::Fma(v1, CM::Sin(v2), mean));
    }
  }
  rngs[rng_id] = rng;
}

template<typename DType>
struct RandomUniformRealKernelSelector {
  template<typename Rng>
  MSHADOW_FORCE_INLINE static void Launch(
    Rng* const rngs,
    const unsigned n_rngs,
    const DType lo,
    const DType hi,
    const size_t size,
    DType* const ret,
    const unsigned n_blocks,
    const unsigned n_threads,
    cudaStream_t s
  ) {
    RandomUniformRealKernel<<<dim3(n_blocks, 1, 1), n_threads, 0, s>>>(
      rngs, n_rngs, lo, hi, size, ret
    );
  }
};

template<typename DType>
struct RandomUniformKernelSelector {
  template<typename Rng>
  MSHADOW_FORCE_INLINE static void Launch(
    Rng* const rngs,
    const unsigned n_rngs,
    const DType lo,
    const DType hi,
    const size_t size,
    DType* const ret,
    const unsigned n_blocks,
    const unsigned n_threads,
    cudaStream_t s
  ) {
    RandomUniformIntKernel<<<dim3(n_blocks, 1, 1), n_threads, 0, s>>>(
      rngs, n_rngs, lo, hi, size, ret
    );
  }
};

template<>
struct RandomUniformKernelSelector<float>
: public RandomUniformRealKernelSelector<float> {};

template<>
struct RandomUniformKernelSelector<double>
: public RandomUniformRealKernelSelector<double> {};

template<>
struct RandomUniformKernelSelector<half::half_t>
: public RandomUniformRealKernelSelector<half::half_t> {};


template<typename Rng, typename DType>
void GPURandomImpl<Rng, DType>::AllocState(
  cudaStream_t s,
  GPURandomImpl<Rng, DType>* impl,
  const uint64_t seed,
  const uint32_t sequence_id,
  const unsigned n_rngs
) {
  CHECK_GT(n_rngs, 0) << "The number of internal random number generators must be positive";
  const unsigned max_threads_per_block = 64;
  impl->n_rngs = n_rngs;
  if (n_rngs <= max_threads_per_block) {
    impl->n_blocks = 1;
    impl->n_threads = n_rngs;
  } else {
    impl->n_blocks = 1 + (n_rngs - 1) / max_threads_per_block;
    impl->n_threads = max_threads_per_block;
  }
  MSHADOW_CUDA_CALL(cudaMalloc(&impl->rngs, sizeof(Rng) * n_rngs));
  const uint64_t stream_offset = uint64_t(sequence_id) * std::numeric_limits<uint32_t>::max();
  cuda::RandomInitKernel<<<dim3(impl->n_blocks, 1, 1), impl->n_threads, 0, s>>>(
    impl->rngs, seed, stream_offset
  );
  MSHADOW_CUDA_POST_KERNEL_CHECK(RandomInitKernel);
}

template<typename Rng, typename DType>
void GPURandomImpl<Rng, DType>::FreeState(GPURandomImpl<Rng, DType>* impl) {
  MSHADOW_CUDA_CALL(cudaFree(impl->rngs));
}

template<typename Rng, typename DType>
inline void GPURandomImpl<Rng, DType>::Seed(
  cudaStream_t s,
  const uint64_t seed,
  const uint32_t sequence_id
) {
  const uint64_t stream_offset = uint64_t(sequence_id) * std::numeric_limits<uint32_t>::max();
  RandomSeedKernel<<<dim3(n_blocks, 1, 1), n_threads, 0, s>>>(rngs, seed, stream_offset);
  MSHADOW_CUDA_POST_KERNEL_CHECK(RandomSeedKernel);
}

template<typename Rng, typename DType>
template<typename IntType>
inline void GPURandomImpl<Rng, DType>::GenerateInt(
  cudaStream_t s,
  IntType* const dst,
  const size_t size
) {
  RandomUIntKernel<<<dim3(n_blocks, 1, 1), n_threads, 0, s>>>(
    rngs, n_rngs, size, dst
  );
  MSHADOW_CUDA_POST_KERNEL_CHECK(RandomUniformKernel);
}

template<typename Rng, typename DType>
inline void GPURandomImpl<Rng, DType>::GenerateUniform(
  cudaStream_t s,
  DType* const dst,
  const size_t size,
  const DType a,
  const DType b
) {
  RandomUniformKernelSelector<DType>::Launch(
    rngs, n_rngs, a, b, size, dst, n_blocks, n_threads, s
  );
  MSHADOW_CUDA_POST_KERNEL_CHECK(RandomUniformKernel);
}

template<typename Rng, typename DType>
inline void GPURandomImpl<Rng, DType>::GenerateGaussian(
  cudaStream_t s,
  DType* const dst,
  const size_t size,
  const DType mean,
  const DType stddev
) {
  RandomGaussianKernel<<<dim3(n_blocks, 1, 1), n_threads, 0, s>>>(
    rngs, n_rngs, mean, stddev, size, dst
  );
  MSHADOW_CUDA_POST_KERNEL_CHECK(RandomGaussianKernel);
}

// Explicit instantiations.
// Sometimes needed to link object files separately compiled with nvcc and a host native compiler.
// For example, mxnet requires this.
template struct GPURandomImpl<PCGRandom64, double>;
template struct GPURandomImpl<PCGRandom64, uint64_t>;
template struct GPURandomImpl<PCGRandom64, int64_t>;
template struct GPURandomImpl<PCGRandom32, float>;
template struct GPURandomImpl<PCGRandom32, mshadow::half::half_t>;
template struct GPURandomImpl<PCGRandom32, uint32_t>;
template struct GPURandomImpl<PCGRandom32, int32_t>;
template struct GPURandomImpl<PCGRandom32, uint16_t>;
template struct GPURandomImpl<PCGRandom32, int16_t>;
template struct GPURandomImpl<PCGRandom32, uint8_t>;
template struct GPURandomImpl<PCGRandom32, int8_t>;

#endif  // __CUDACC__

}  // namespace cuda

#ifdef __CUDACC__

template<typename Rng, typename DType>
struct UniformRealDistribution<gpu, Rng, DType> {

  using result_type = DType;

  MSHADOW_FORCE_INLINE __device__
  explicit UniformRealDistribution(result_type a = 0, result_type b = 1)
  : a(a), b(b), w(FType(b) - FType(a)) {
  }

  /*!
   * \brief return a uniform random number assuming that `g` generates the full range of ingeters.
   */
  MSHADOW_FORCE_INLINE __device__ result_type operator()(Rng& g) const {
    using CM = cuda::CuMath<FType>;
    return cuda::CastWithRounginDown<FType, DType>(
      CM::FmaRD(FType(RandomReal<Rng, FType>::Generate(g)), w, a)
    );
  }

  private:
    using FType = typename std::conditional<sizeof(DType) <= 4, float, double>::type;

    FType a;
    DType b;
    FType w;
};

template<typename Rng, typename DType>
struct GaussianDistribution<gpu, Rng, DType> {
  using result_type = DType;

  MSHADOW_FORCE_INLINE __device__
  explicit GaussianDistribution(result_type mean = 0, result_type stddev = 1)
  : mean(mean), stddev(stddev), spare(0), has_spare(false) {
  }

  /*!
   * \brief return a Gaussian random number assuming that `g` generates the full range of ingeters.
   */
  MSHADOW_FORCE_INLINE __device__
  result_type operator()(Rng& g) {
    // Box-Muller method
    if (has_spare) {
      has_spare = false;
      return spare;
    }
    using CM = cuda::CuMath<FType>;
    const FType two_pi = 6.283185307179586;
    FType u1 = RandomReal<Rng, FType>::Generate(g);
    while (u1 == 0) {
      u1 = RandomReal<Rng, FType>::Generate(g);
    }
    const FType u2 = RandomReal<Rng, FType>::Generate(g);
    const FType v1 = stddev * CM::Sqrt(-FType(2) * CM::Log(u1));
    const FType v2 = two_pi * u2;
    spare = CM::Fma(v1, CM::Cos(v2), mean);
    has_spare = true;
    return static_cast<DType>(CM::Fma(v1, CM::Sin(v2), mean));
  }

 private:
  using FType = typename std::conditional<sizeof(DType) <= 4, float, double>::type;

  FType mean;
  FType stddev;
  FType spare;
  bool has_spare;
};

// Explicit instantiations.
// Sometimes needed to link object files separately compiled with nvcc and a host native compiler.
template struct UniformRealDistribution<gpu, PCGRandom32, float>;
template struct UniformRealDistribution<gpu, PCGRandom32, double>;
template struct UniformRealDistribution<gpu, PCGRandom32, mshadow::half::half_t>;
template struct UniformRealDistribution<gpu, PCGRandom64, float>;
template struct UniformRealDistribution<gpu, PCGRandom64, double>;
template struct UniformRealDistribution<gpu, PCGRandom64, mshadow::half::half_t>;
template struct GaussianDistribution<gpu, PCGRandom32, float>;
template struct GaussianDistribution<gpu, PCGRandom32, double>;
template struct GaussianDistribution<gpu, PCGRandom32, mshadow::half::half_t>;
template struct GaussianDistribution<gpu, PCGRandom64, float>;
template struct GaussianDistribution<gpu, PCGRandom64, double>;
template struct GaussianDistribution<gpu, PCGRandom64, mshadow::half::half_t>;

#endif  // __CUDACC__

}  // namespace mshadow

#endif  // MSHADOW_CUDA_RANDOM_GPU_CUH_