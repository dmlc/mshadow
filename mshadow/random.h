/*!
 *  Copyright (c) 2014 by Contributors
 *  \file random.h
 *  \brief Random inline functions for tensor.
 *  \author Bing Xu, Tianqi Chen
 *   Based on curand|MKL|stdlib
 */
#ifndef MSHADOW_RANDOM_H_
#define MSHADOW_RANDOM_H_

#include <cstdlib>
#include <algorithm>
#include <random>
#include <limits>
#include "./base.h"
#include "./tensor.h"
#include "./tensor_container.h"

#if MSHADOW_IN_CXX11
#include <random>  // use cxx11 random by default
#endif

#if _MSC_VER
#define rand_r(x) rand()
#endif

namespace mshadow {

#include <cstdint>
#include <limits>

#if MSHADOW_IN_CXX11

/*!
 * \brief PCG random number generator introduced by
 *        "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for
 *        Random Number Generation" (Melissa E. O'Neill, 2014).
 *        This generator provides 2^64 statistically independent "streams."
 *        Generators with different streams are statistically independent regardless of seeds.
 *        Here we use the term "sequence" instead of "stream" to avoid confusion with cuda stream.
 */
struct PCG32 {
  /*!
   * \brief seed this generator
   * \param seed seed
   * \param m sequence
   */
  MSHADOW_XINLINE void seed(uint64_t s, uint64_t m) {
    sequence = (m << 1u) | 1u;
    state = 0U;
    generate();
    state += s;
    generate();
  }

  /*!
   * \brief generate a 32-bit random number
   */
  MSHADOW_XINLINE uint32_t generate() {
    uint64_t s = state;
    state = s * 6364136223846793005ull + sequence;
    uint32_t x = ((s >> 18u) ^ s) >> 27u;
    uint32_t y = s >> 59u;
    return (x >> y) | (x << ((-y) & 31));
  }

  uint64_t state;
  uint64_t sequence;
};

/*!
 * C++11 style PCG random number generator for 32 random bits.
 */
class PCGRandom32 {
 public:
  typedef uint32_t result_type;

  /*!
    * \brief constructor.
    * \param seed random number seed. 2^64 different seeds are allowed.
    * \param m sequence id of the generator. 2^64 different sequence ids are allowed.
    */
  MSHADOW_XINLINE
  explicit PCGRandom32(uint64_t s = default_seed, uint64_t m = default_sequence) {
    seed(s, m);
  }

  /*!
    * \brief seed the generator.
    * \param seed random number seed. 2^64 different seeds are allowed.
    * \param m sequence id of the generator. 2^64 different sequence ids are allowed.
    */
  MSHADOW_XINLINE void seed(uint64_t s = default_seed, uint64_t m = default_sequence) {
    pcg.seed(s, m);
  }

  MSHADOW_XINLINE result_type operator()() {
    return pcg.generate();
  }

  MSHADOW_XINLINE static constexpr result_type min() {
    return 0;
  }

  MSHADOW_XINLINE static constexpr result_type max() {
    return 0xfffffffful;
  }

  /*!
    * \brief generate a random number of type T.
    */
  template<typename T>
  MSHADOW_XINLINE T generate();

 public:
  static constexpr uint32_t default_seed = 3917ull;
  static constexpr uint32_t default_sequence = 0ull;

 private:
  PCG32 pcg;
};

template<>
MSHADOW_XINLINE float PCGRandom32::generate<float>() {
  constexpr float e = 1.f / (uint32_t(1) << 24);
  return (operator()() >> 8) * e;
}

template<>
MSHADOW_XINLINE double PCGRandom32::generate<double>() {
  constexpr double e = 1. / (uint64_t(1) << 53);
  uint64_t u = (uint64_t(operator()()) << 32) | operator()();
  return (u >> 11) * e;
}

/*!
 * C++11 style PCG random number generator for 64 random bits.
 * This implementation uses two 32-bit generators to avoid the use of 128-bit arithmetic.
 */
class PCGRandom64 {
 public:
  typedef uint64_t result_type;

  /*!
    * \brief constructor.
    * \param seed random number seed. 2^64 different seeds are allowed.
    * \param m sequence id of the generator. 2^64 different sequence ids are allowed.
    */
  MSHADOW_XINLINE
  explicit PCGRandom64(uint64_t s = default_seed, uint64_t m = default_sequence) {
    seed(s, m);
  }

  /*!
    * \brief seed the generator
    * \param seed random number seed. 2^64 different seeds are allowed.
    * \param m sequence id of the generator. 2^64 different sequence ids are allowed.
    */
  MSHADOW_XINLINE void seed(uint64_t s = default_seed, uint64_t m = default_sequence) {
    pcg0.seed(s, m);
    pcg1.seed(splitmix64(s), ~m);
  }

  MSHADOW_XINLINE result_type operator()() {
    return static_cast<uint64_t>(pcg0.generate()) << 32 | pcg1.generate();
  }

  MSHADOW_XINLINE static constexpr result_type min() {
    return 0;
  }

  MSHADOW_XINLINE static constexpr result_type max() {
    return 0xffffffffffffffffull;
  }

  /*!
    * \brief generate a random number of type T.
    */
  template<typename T>
  MSHADOW_XINLINE T generate();

 public:
  static constexpr uint64_t default_seed = 81917ull;
  static constexpr uint64_t default_sequence = 10920ull;

 private:
  MSHADOW_XINLINE uint64_t splitmix64(uint64_t s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  PCG32 pcg0;
  PCG32 pcg1;
};

template<>
MSHADOW_XINLINE float PCGRandom64::generate<float>() {
  constexpr float e = 1.f / (uint32_t(1) << 24);
  return (uint32_t(operator()()) >> 8) * e;
}

template<>
MSHADOW_XINLINE double PCGRandom64::generate<double>() {
  constexpr double e = 1. / (uint64_t(1) << 53);
  return (operator()() >> 11) * e;
}

template<typename Device, typename Rng, typename DType>
struct UniformRealDistribution;

template<typename Rng, typename DType>
struct UniformRealDistribution<cpu, Rng, DType> {
  using result_type = DType;

  MSHADOW_FORCE_INLINE explicit UniformRealDistribution(result_type a = 0, result_type b = 1)
  : a(a), b(b), w(FType(b) - FType(a)) {
  }

  /*!
   * \brief return a uniform random number assuming that `g` generates the full range of ingeters.
   */
  MSHADOW_FORCE_INLINE result_type operator()(Rng& g) const {  // NOLINT(runtime/references)
    DType r;
    do {
      r = a + w * g.template generate<FType>();
    } while (b <= r);  // This can happen due to floating point rounding.
    return r;
  }

 private:
  using FType = typename std::conditional< sizeof(DType) <= 4, float, double>::type;

  FType a;
  DType b;
  FType w;
};

template<typename Device, typename Rng, typename DType>
struct GaussianDistribution;

template<typename Rng, typename DType>
struct GaussianDistribution<cpu, Rng, DType> {
  using result_type = DType;

  MSHADOW_FORCE_INLINE explicit GaussianDistribution(result_type mean = 0, result_type stddev = 1)
  : mean(mean), stddev(stddev), spare(0), has_spare(false) {
  }

  /*!
   * \brief return a Gaussian random number assuming that `g` generates the full range of ingeters.
   */
  MSHADOW_FORCE_INLINE result_type operator()(Rng& g) {  // NOLINT(runtime/references)
    // Marsaglia polar method
    if (has_spare) {
      has_spare = false;
      return mean + spare;
    }
    FType u, v, s;
    do {
      u = g.template generate<FType>() * 2 - 1;
      v = g.template generate<FType>() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);
    const FType t = stddev * sqrt(-FType(2) * log(s) / s);
    spare = v * t;
    has_spare = true;
    return static_cast<result_type>(mean + u * t);
  }

 private:
  using FType = typename std::conditional< sizeof(DType) <= 4, float, double>::type;

  FType mean;
  FType stddev;
  FType spare;
  bool has_spare;
};

#endif

/*!
 * \brief random number generator
 * \tparam Device the device of random number generator
 * \tparam DType the target data type of random number
 */
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class Random {};

/*! \brief CPU random number generator */
template<typename DType>
class Random<cpu, DType> {
 public:
#if MSHADOW_IN_CXX11
  using RandomBitGenerator =
    typename std::conditional< sizeof(DType) <= 4, PCGRandom32, PCGRandom64>::type;
#endif

  /*!
   * \brief constructor of random engine
   * \param seed random number seed
   * \param sequence_id generators with different sequence ids are statistically independent.
   */
  explicit Random(int seed, uint64_t sequence_id = 0) {
    this->Seed(seed, sequence_id);
    buffer_.Resize(Shape1(kRandBufferSize));
  }

  ~Random(void) {
  }

  /*!
   * \brief seed random number generator using this seed
   * \param seed seed of prng
   * \param sequence_id generators with different sequence ids are statistically independent.
   */
  inline void Seed(int seed, uint64_t sequence_id = 0) {
#if MSHADOW_IN_CXX11
    rnd_engine_.seed(seed, sequence_id);
#endif
    this->rseed_ = static_cast<unsigned>(seed);
  }
  /*!
   * \brief get random seed used in random generator
   * \return seed in unsigned
   */
  inline unsigned GetSeed() const {
    return rseed_;
  }
  /*!
   * \brief set the stream of computation
   * \param stream computation stream
   */
  inline void set_stream(Stream<cpu> *stream) {
  }

// These samplers are only avail in C++11.
#if MSHADOW_IN_CXX11

  /*!
   * \brief get a set of random integers in the range of [0, 2^32 - 1]
   *        if sizeof(DType) <= 4 or [0, 2^64 - 1] otherwise.
   * \tparam IntType integer type
   */
  template<typename IntType>
  inline void GetRandInt(Tensor<cpu, 1, IntType>& dst) {  // NOLINT(runtime/references)
    std::generate_n(dst.dptr_, dst.size(0), [&](){ return rnd_engine_(); });
  }

  /*!
   * \brief generate data from a distribution
   * \param dst destination
   * \tparam dim dimension of tensor
   * \param sampler sampler of the distribution
   */
  template<int dim, class Sampler>
  inline void SampleDistribution(Tensor<cpu, dim, DType> *dst, Sampler sampler) {
    if (dst->CheckContiguous()) {
      std::generate_n(dst->dptr_, dst->shape_.Size(), sampler);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        std::generate_n(mat[i].dptr_, mat.size(1), sampler);
      }
    }
  }

  /*!
   * \brief generate uniform random numbers. The lower bound is inclusive.
   *        The upper bound is inclusive for integers and exclusive for floating point numbers.
   * \param dst destination
   * \param a lower bound of the uniform distribution
   * \param b upper bound of the uniform distribution
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleUniform(Tensor<cpu, dim, DType> *dst,
                            PType a = 0.0f , PType b = 1.0f ) {
    typedef typename std::conditional<
      mshadow::IsIntegral<DType>::value,
      std::uniform_int_distribution<DType>,
      mshadow::UniformRealDistribution<cpu, RandomBitGenerator, DType>>::type GType;
    GType dist_uniform(a, b);
    SampleDistribution(dst, [&](){ return dist_uniform(rnd_engine_);});
  }

  /*!
   * \brief generate data from standard gaussian
   * \param dst destination
   * \param mu mean variable
   * \param sigma standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGaussian(Tensor<cpu, dim, DType> *dst,
                             PType mu = 0.0f, PType sigma = 1.0f ) {
    if (sigma <= 0) {
      *dst = mu; return;
    }
    typedef typename std::conditional<mshadow::IsFloatingPoint<DType>::value,
                                      DType, double>::type GType;
    mshadow::GaussianDistribution<cpu, RandomBitGenerator, GType> dist_normal(mu, sigma);
    SampleDistribution(dst, [&](){ return dist_normal(rnd_engine_);});
  }

  /*!
   * \brief generate data from a gamma distribution
   * \param dst destination
   * \param alpha (shape) parameter
   * \param beta (scale) parameter
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGamma(Tensor<cpu, dim, DType> *dst,
                          PType alpha, PType beta) {
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type GType;
    std::gamma_distribution<GType> dist_gamma(alpha, beta);
    SampleDistribution(dst, [&](){ return dist_gamma(rnd_engine_);});
  }

  /*!
   * \brief generate data from an exponential distribution
   * \param dst destination
   * \param lambda parameter (rate) of the distribution
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleExponential(Tensor<cpu, dim, DType> *dst, PType lambda ) {
    typedef typename std::conditional<std::is_floating_point<DType>::value,
                                      DType, double>::type GType;
    std::exponential_distribution<GType> dist_exp(lambda);
    SampleDistribution(dst, [&](){ return dist_exp(rnd_engine_);});
  }

  /*!
   * \brief generate data from a Poisson distribution
   * \param dst destination
   * \param lambda parameter (rate) of the distribution
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SamplePoisson(Tensor<cpu, dim, DType> *dst, PType lambda) {
    typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
    std::poisson_distribution<GType> dist_poisson(lambda);
    SampleDistribution(dst, [&](){ return static_cast<DType>(dist_poisson(rnd_engine_));});
  }

  /*!
   * \brief generate data from a negative binomial distribution
   * \param dst destination
   * \param k limit on number of failures
   * \param p success probability
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType1, typename PType2>
  inline void SampleNegativeBinomial(Tensor<cpu, dim, DType> *dst, PType1 k, PType2 p) {
    typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
    std::negative_binomial_distribution<GType> dist_negbinomial(k, p);
    SampleDistribution(dst, [&](){ return static_cast<DType>(dist_negbinomial(rnd_engine_));});
  }

  /*!
   * \brief generate data from a generalized negative binomial distribution
   * \param dst destination
   * \param mu parameter (mean) of the distribution
   * \param alpha parameter (over dispersion) of the distribution
   *   (for alpha=0 this gives a Poisson)
   * \tparam dim dimension of tensor
   */
  template<int dim, typename PType>
  inline void SampleGeneralizedNegativeBinomial(Tensor<cpu, dim, DType> *dst,
                                                PType mu, PType alpha) {
    if (alpha == PType(0)) {
      SamplePoisson(dst, mu);  // limit of Poisson
    } else {
      PType r(PType(1) / alpha);
      PType beta = mu * alpha;
      std::gamma_distribution<> dist_gamma(r, beta);
      typedef typename std::conditional<std::is_integral<DType>::value, DType, int>::type GType;
      SampleDistribution(dst,
        [&](){ std::poisson_distribution<GType> dist_poisson(dist_gamma(rnd_engine_));
               return static_cast<DType>(dist_poisson(rnd_engine_));});
    }
  }

  RandomBitGenerator &GetRndEngine() {
    return rnd_engine_;
  }

#endif

  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleGaussian(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }
  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = uniform(s1) * uniform(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<cpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape) {
    buffer_.Resize(Shape1(shape.Size()));
    this->SampleUniform(&buffer_, 0.0f, 1.0f);
    return expr::reshape(buffer_, shape);
  }

 private:
#if MSHADOW_IN_CXX11
  /*! \brief use c++11 random engine. */
  RandomBitGenerator rnd_engine_;
  /*! \brief random number seed used in random engine */
  unsigned rseed_;

#else

  /*! \brief random number seed used by PRNG */
  unsigned rseed_;
  // functions
  template<int dim>
  inline void SampleUniform(Tensor<cpu, dim, DType> *dst,
                            DType a = 0.0f, DType b = 1.0f) {
    if (dst->CheckContiguous()) {
      this->GenUniform(dst->dptr_, dst->shape_.Size(), a, b);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenUniform(mat[i].dptr_, mat.size(1), a, b);
      }
    }
  }
  template<int dim>
  inline void SampleGaussian(Tensor<cpu, dim, DType> *dst,
                             DType mu = 0.0f, DType sigma = 1.0f) {
    if (sigma <= 0.0f) {
      *dst = mu; return;
    }
    if (dst->CheckContiguous()) {
      this->GenGaussian(dst->dptr_, dst->shape_.Size(), mu, sigma);
    } else {
      Tensor<cpu, 2, DType> mat = dst->FlatTo2D();
      for (index_t i = 0; i < mat.size(0); ++i) {
        this->GenGaussian(mat[i].dptr_, mat.size(1), mu, sigma);
      }
    }
  }
  inline void GenUniform(float *dptr, index_t size, float a, float b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<float>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenUniform(double *dptr, index_t size, double a, double b) {
    for (index_t j = 0; j < size; ++j) {
      dptr[j] = static_cast<double>(RandNext()) * (b - a) + a;
    }
  }
  inline void GenGaussian(float *dptr, index_t size, float mu, float sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussian(double *dptr, index_t size, double mu, double sigma) {
    this->GenGaussianX(dptr, size, mu, sigma);
  }
  inline void GenGaussianX(DType *dptr, index_t size, DType mu, DType sigma) {
    DType g1 = 0.0f, g2 = 0.0f;
    for (index_t j = 0; j < size; ++j) {
      if ((j & 1) == 0) {
        this->SampleNormal2D(&g1, &g2);
        dptr[j] = mu + g1 * sigma;
      } else {
        dptr[j] = mu + g2 * sigma;
      }
    }
  }
  /*! \brief get next random number from rand */
  inline DType RandNext(void) {
    return static_cast<DType>(rand_r(&rseed_)) /
        (static_cast<DType>(RAND_MAX) + 1.0f);
  }
  /*! \brief return a real numer uniform in (0,1) */
  inline DType RandNext2(void) {
    return (static_cast<DType>(rand_r(&rseed_)) + 1.0f) /
        (static_cast<DType>(RAND_MAX) + 2.0f);
  }
  /*!
   * \brief sample iid xx,yy ~N(0,1)
   * \param xx first  gaussian output
   * \param yy second gaussian output
   */
  inline void SampleNormal2D(DType *xx_, DType *yy_) {
    DType &xx = *xx_, &yy = *yy_;
    DType x, y, s;
    do {
      x = 2.0f * RandNext2() - 1.0f;
      y = 2.0f * RandNext2() - 1.0f;
      s = x * x + y * y;
    } while (s >= 1.0f || s == 0.0f);
    DType t = std::sqrt(-2.0f * std::log(s) / s);
    xx = x * t; yy = y * t;
  }
#endif
  /*! \brief temporal space used to store random numbers */
  TensorContainer<cpu, 1, DType> buffer_;
};  // class Random<cpu, DType>

}  // namespace mshadow


#if MSHADOW_USE_CUDA

#include "./cuda/random_gpu.cuh"

namespace mshadow {

template<typename DType>
struct Random<gpu, DType> {
  /*! \brief default number of internal parallel random number generators */
  static const unsigned DEFAULT_NUMBER_OF_RNGS = 4096;

  /*!
   * \brief constructor of random number generator
   * \param seed random number seed
   * \param sequence_id generators with different sequence ids are statistically independent
   * \param n_rngs number of internal parallel generators
   * \param s stream
   */
  explicit Random(
    uint64_t seed,
    uint32_t sequence_id = 0,
    unsigned n_rngs = DEFAULT_NUMBER_OF_RNGS,
    Stream<gpu>* s = nullptr
  ) {
    stream = s ? Stream<gpu>::GetStream(s) : nullptr;
    Impl::AllocState(0, &impl, seed, sequence_id, n_rngs);
  }

  ~Random() {
    Impl::FreeState(&impl);
  }

  /*!
   * \brief seed this random number generator
   * \param seed random number seed
   * \param sequence_id generators with different sequence ids are statistically independent.
   */
  inline void Seed(uint64_t seed, uint32_t sequence_id = 0) {
    impl.Seed(0, seed, sequence_id);
  }

  /*!
   * \brief set the stream of computation
   * \param s computation stream
   */
  inline void set_stream(Stream<gpu> *s) {
    stream = Stream<gpu>::GetStream(s);
  }

  /*!
   * \brief get a set of random integers in the range of [0, 2^32 - 1]
   *        if sizeof(DType) <= 4 or [0, 2^64 - 1] otherwise.
   * \tparam IntType integer type
   */
  template<typename IntType>
  inline void GetRandInt(Tensor<gpu, 1, IntType>& dst) {  // NOLINT(runtime/references)
    impl.GenerateInt(stream, dst.dptr_, dst.shape_.Size());
  }

  /*!
   * \brief generate uniform random numbers. The lower bound is inclusive.
   *        The upper bound is inclusive for integers and exclusive for floating point numbers.
   * \param dst destination
   * \param a lower bound of uniform
   * \param b upper bound of uniform
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleUniform(Tensor<gpu, dim, DType>* dst, DType a = 0, DType b = 1) {
    if (dst->CheckContiguous()) {
      impl.GenerateUniform(stream, dst->dptr_, dst->shape_.Size(), a, b);
    } else {
      *dst = uniform(dst->shape_, a, b);
    }
  }

  /*!
   * \brief generate data from Gaussian distribution
   * \param dst destination
   * \param mean mean variable
   * \param stddev standard deviation
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline void SampleGaussian(Tensor<gpu, dim, DType>* dst, DType mean = 0, DType stddev = 1) {
    if (dst->CheckContiguous()) {
      impl.GenerateGaussian(stream, dst->dptr_, dst->shape_.Size(), mean, stddev);
    } else {
      *dst = gaussian(dst->shape_, mean, stddev);
    }
  }

  /*!
   * \brief return a temporal expression storing standard uniform [0,1)
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \return a temporal expression storing standard uniform [0,1)
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  uniform(Shape<dim> shape, DType a = 0, DType b = 1) {
    buffer_.Resize(Shape1(shape.Size()));
    impl.GenerateUniform(stream, buffer_.dptr_, buffer_.shape_.Size(), a, b);
    return expr::reshape(buffer_, shape);
  }

  /*!
   * \brief return a temporal expression storing standard gaussian random variables
   *        the temporal tensor is only valid before next call of gaussian or uniform
   *        can be used as part of expression
   *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
   *           since second call of gaussian(s2) makes gaussian(s1) invalid
   *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
   * \param shape shape of the tensor
   * \param mean mean
   * \param stddev standard deviation
   * \return a temporal expression storing standard gaussian random variables
   * \tparam dim dimension of tensor
   */
  template<int dim>
  inline expr::ReshapeExp<Tensor<gpu, 1, DType>, DType, dim, 1>
  gaussian(Shape<dim> shape, DType mean = 0, DType stddev = 1) {
    size_t aligned_sz = ((shape.Size() + 1UL) >> 1) << 1;
    // allocate alligned size
    buffer_.Resize(Shape1(aligned_sz));
    buffer_.Resize(Shape1(shape.Size()));
    impl.GenerateGaussian(stream, buffer_.dptr_, buffer_.shape_.Size(), mean, stddev);
    return expr::reshape(buffer_, shape);
  }

 private:
  using RandomBitGenerator =
    typename std::conditional< sizeof(DType) <= 4, PCGRandom32, PCGRandom64>::type;
  using Impl = cuda::GPURandomImpl<RandomBitGenerator, DType>;

  /*! \brief temporal buffer */
  TensorContainer<gpu, 1, DType> buffer_;
  /*! \brief implementation calling cuda kernels */
  Impl impl;
  /*! \brief cuda stream */
  cudaStream_t stream;
};

}  // namespace mshadow
#endif  // MSHADOW_USE_CUDA

#endif  // MSHADOW_RANDOM_H_
