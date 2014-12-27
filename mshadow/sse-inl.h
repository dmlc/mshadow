/*!
 *  Copyright (c) 2014 by Contributors
 * \file sse-inl.h
 * \brief support of sse2 optimization of some operations
 * \author Tianqi Chen
 */
#ifndef MSHADOW_SSE_INL_H_
#define MSHADOW_SSE_INL_H_
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include "./expression.h"
#include "./tensor.h"

namespace mshadow {
/*! \brief namespace to support sse2 vectorization */
namespace sse2 {
/*! 
 * \brief analog to cudaMallocPitch, allocate a aligned space with num_line * lspace cells
 * \param out_pitch output parameter, the actuall space allocated for each line
 * \param lspace number of cells required for each line
 * \param num_line number of lines to be allocated
 */
inline void* AlignedMallocPitch(size_t *out_pitch,
                                size_t lspace, size_t num_line) {
  size_t pitch = ((lspace+15) >> 4) << 4;
  *out_pitch = pitch;
#ifdef _MSC_VER
  void *res = _aligned_malloc(pitch * num_line, 16);
#else
#ifdef __APPLE__
  void *res = malloc(pitch * num_line);
#else
  void *res = memalign(16, pitch * num_line);
#endif
#endif
  utils::Check(res != NULL, "AlignedMallocPitch failed");
  return res;
}
/*! 
 * \brief free aligned space 
 * \param ptr pointer to space to be freed
 */
inline void AlignedFree(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
/*! \brief check if a pointer is aligned */
inline bool CheckAlign(size_t pitch) {
  return !(pitch & ((1 << 4) - 1));
}
/*! \brief check if a pointer is aligned */
inline bool CheckAlign(void *ptr) {
  return CheckAlign(reinterpret_cast<size_t>(ptr));
}
/*! 
 * \brief get upper bound of aligned index of size 
 * \param size size of the array
 * \param fsize size of float
 */
inline index_t UpperAlign(index_t size, size_t fsize) {
  return (((size * fsize + 15) >> 4) << 4) / fsize;
}
/*! 
 * \brief get lower bound of aligned index of size 
 * \param size size of the array
 * \param fsize size of float
 */
inline index_t LowerAlign(index_t size, size_t fsize) {
  return (((size * fsize) >> 4) << 4) / fsize;
}
}  // namespace sse2
}  // namespace  mshadow
#if MSHADOW_USE_SSE
// sse types are not compatible with nvcc, only use them in cpu mode
#include <emmintrin.h>

namespace mshadow {
namespace sse2 {
/*! 
 * \brief float vector real type, used for vectorization 
 * \tparam FloatType double or float
 */
template<typename FloatType>
struct FVec {
  // whether the vectorization is enabled
  static const bool kEnabled = false;
};
/*! \brief vector real type for float */
template<>
struct FVec<float> {
  // type
  typedef __m128 DType;
  // whether the vectorization is enabled
  static const bool kEnabled = true;
  /*! \brief number of float in vector */
  static const index_t kSize = 4;
  /*! \brief data content */
  DType data_;
  // functions
  /* constructors */
  FVec(void) {}
  explicit FVec(DType data) : data_(data) {}
  /* set the float */
  explicit FVec(const float &s) {
    data_ = _mm_set1_ps(s);
  }
  /*!\brief load from pointer src */
  explicit FVec(const float *src) {
    data_ = _mm_load_ps(src);
  }
  /*! \brief store data into dst space */
  inline void Store(float *dst) const {
    return _mm_store_ps(dst, data_);
  }
  /*! \brief sum of all content */
  inline float Sum(void) const {
    DType ans  = _mm_add_ps(data_, _mm_movehl_ps(data_, data_));
    DType rst  = _mm_add_ss(ans, _mm_shuffle_ps(ans, ans, 1));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return rst.m128_f32[0];
#else
    float rr = _mm_cvtss_f32(rst);
    return rr;
#endif
  }
};
/*! \brief vector real type for float */
template<>
struct FVec<double> {
  // data type
  typedef __m128d DType;
  // whether the vectorization is enabled
  static const bool kEnabled = true;
  /*! \brief number of float in vector */
  static const index_t kSize = 2;
  /*! \brief data content */
  DType data_;
  /* constructors */
  FVec(void) {}
  explicit FVec(DType data) : data_(data) {}
  /* set the float */
  explicit FVec(const double &s) {
    data_ = _mm_set1_pd(s);
  }
  /*!\brief load from pointer src */
  explicit FVec(const double *src) {
    data_ = _mm_load_pd(src);
  }
  /*! \brief store data into dst space */
  inline void Store(double *dst) const {
    return _mm_store_pd(dst, data_);
  }
  /*! \brief sum of all content */
  inline double Sum(void) const {
    DType tmp =  _mm_add_sd(data_, _mm_unpackhi_pd(data_, data_));
#if defined(_MSC_VER) && (_MSC_VER <= 1500) && defined(_WIN64)
    return tmp.m128d_f64[0];
#else
    double ans = _mm_cvtsd_f64(tmp);
    return ans;
#endif
  }
};
/*! \brief sse2 operator type of certain operator */
template<typename OP>
struct SSEOp{
  static const bool kEnabled = false;
};
template<>
struct SSEOp<op::plus> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static FVec<float>
  Map(const FVec<float> &lhs, const FVec<float> &rhs) {
    return FVec<float>(_mm_add_ps(lhs.data_, rhs.data_));
  }
  MSHADOW_CINLINE static FVec<double>
  Map(const FVec<double> &lhs, const FVec<double> &rhs) {
    return FVec<double>(_mm_add_pd(lhs.data_, rhs.data_));
  }
};
template<>
struct SSEOp<op::minus> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static FVec<float>
  Map(const FVec<float> &lhs, const FVec<float> &rhs) {
    return FVec<float>(_mm_sub_ps(lhs.data_, rhs.data_));
  }
  MSHADOW_CINLINE static FVec<double>
  Map(const FVec<double> &lhs, const FVec<double> &rhs) {
    return FVec<double>(_mm_sub_pd(lhs.data_, rhs.data_));
  }
};
template<>
struct SSEOp<op::mul> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static FVec<float>
  Map(const FVec<float> &lhs, const FVec<float> &rhs) {
    return FVec<float>(_mm_mul_ps(lhs.data_, rhs.data_));
  }
  MSHADOW_CINLINE static FVec<double>
  Map(const FVec<double> &lhs, const FVec<double> &rhs) {
    return FVec<double>(_mm_mul_pd(lhs.data_, rhs.data_));
  }
};
template<>
struct SSEOp<op::div> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static FVec<float>
  Map(const FVec<float> &lhs, const FVec<float> &rhs) {
    return FVec<float>(_mm_div_ps(lhs.data_, rhs.data_));
  }
  MSHADOW_CINLINE static FVec<double>
  Map(const FVec<double> &lhs, const FVec<double> &rhs) {
    return FVec<double>(_mm_div_pd(lhs.data_, rhs.data_));
  }
};
template<>
struct SSEOp<op::identity> {
  static const bool kEnabled = true;
  MSHADOW_CINLINE static FVec<float> Map(const FVec<float> &src) {
    return src;
  }
  MSHADOW_CINLINE static FVec<double> Map(const FVec<double> &src) {
    return src;
  }
};
// savers to do storage
template<typename SV, typename TFloat>
struct Saver{
  MSHADOW_CINLINE static void Save(TFloat *dst, const FVec<TFloat> &src) {
    FVec<TFloat> lhs(dst);
    FVec<TFloat> ans = SSEOp<typename SV::OPType>::Map(lhs, src);
    ans.Store(dst);
  }
};
template<typename TFloat>
struct Saver<sv::saveto, TFloat> {
  MSHADOW_CINLINE static void Save(TFloat *dst, const FVec<TFloat> &src) {
    src.Store(dst);
  }
};
}  // namespace sse2
namespace expr {
// same as plan, but use sse2
template<typename ExpType, typename DType>
class SSEPlan {
 public:
  /*!
   * \brief evaluate the expression at index [y][x], x will be aligned to 4
   *        to be implemented by SubType
   */
  MSHADOW_CINLINE sse2::FVec<DType> EvalSSE(index_t y, index_t x) const;
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const;
};
template <typename Device, int dim, typename DType>
class SSEPlan<Tensor<Device, dim, DType>, DType> {
 public:
  explicit SSEPlan(const Tensor<Device, dim, DType> &t)
      :dptr_(t.dptr_), stride_(t.stride_) {}
  MSHADOW_CINLINE sse2::FVec<DType> EvalSSE(index_t y, index_t x) const {
    return sse2::FVec<DType>(&dptr_[y * stride_ + x]);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return dptr_[y * stride_ + x];
  }

 private:
  const DType  *dptr_;
  index_t stride_;
};
template<typename DType>
class SSEPlan<ScalarExp<DType>, DType> {
 public:
  explicit SSEPlan(DType scalar) : scalar_(scalar) {}
  MSHADOW_CINLINE sse2::FVec<DType> EvalSSE(index_t y, index_t x) const {
    return sse2::FVec<DType>(scalar_);
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return scalar_;
  }

 private:
  DType scalar_;
};
template<typename OP, typename TA, typename TB, int etype, typename DType>
class SSEPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType> {
 public:
  SSEPlan(const SSEPlan<TA, DType> &lhs, const SSEPlan<TB, DType> &rhs)
      : lhs_(lhs), rhs_(rhs) {}
  MSHADOW_CINLINE sse2::FVec<DType> EvalSSE(index_t y, index_t x) const {
    return sse2::SSEOp<OP>::Map(lhs_.EvalSSE(y, x), rhs_.EvalSSE(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(lhs_.Eval(y, x), rhs_.Eval(y, x));
  }

 private:
  SSEPlan<TA, DType> lhs_;
  SSEPlan<TB, DType> rhs_;
};

template<typename OP, typename TA, int etype, typename DType>
class SSEPlan<UnaryMapExp<OP, TA, DType, etype>, DType> {
 public:
  SSEPlan(const SSEPlan<TA, DType> &src) : src_(src) {}
  MSHADOW_CINLINE sse2::FVec<DType> EvalSSE(index_t y, index_t x) const {
    return sse2::SSEOp<OP>::Map(src_.EvalSSE(y, x));
  }
  MSHADOW_CINLINE DType Eval(index_t y, index_t x) const {
    return OP::Map(src_.Eval(y, x));
  }

 private:
  SSEPlan<TA, DType> src_;
};

template<typename OP, typename TA, typename TB, typename DType, int etype>
inline SSEPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakeSSEPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e);

template<typename DType>
inline SSEPlan<ScalarExp<DType>, DType> MakeSSEPlan(const ScalarExp<DType> &e) {
  return SSEPlan<ScalarExp<DType>, DType>(e.scalar_);
}
template<typename T, typename DType>
inline SSEPlan<T, DType> MakeSSEPlan(const RValueExp<T, DType> &e) {
  return SSEPlan<T, DType>(e.self());
}
template<typename T, int dim, typename DType>
inline SSEPlan<T, DType>
MakeSSEPlan(const MakeTensorExp<T, cpu, dim, DType> &e) {
  return SSEPlan<T, DType>(e.real_self());
}
template<typename OP, typename TA, typename DType, int etype>
inline SSEPlan<UnaryMapExp<OP, TA, DType, etype>, DType>
MakeSSEPlan(const UnaryMapExp<OP, TA, DType, etype> &e) {
  return SSEPlan<UnaryMapExp<OP, TA, DType, etype>, DType>(MakeSSEPlan(e.src_));
}
template<typename OP, typename TA, typename TB, typename DType, int etype>
inline SSEPlan<BinaryMapExp<OP, TA, TB, DType, etype>, DType>
MakeSSEPlan(const BinaryMapExp<OP, TA, TB, DType, etype> &e) {
  return SSEPlan<BinaryMapExp<OP, TA, TB, DType, etype>,
                 DType>(MakeSSEPlan(e.lhs_), MakeSSEPlan(e.rhs_));
}
/*!
 * \brief static check sse enable
 *        if a expression E can not be evaluated using sse, then kPass = false
 * \tparam Device the type of Device
 * \tparam dim dimension of the tensor
 * \tparam E expression
 */
template<typename E>
struct SSECheck{
  static const bool kPass = false;
};
template<typename DType>
struct SSECheck<ScalarExp<DType> > {
  static const bool kPass = sse2::FVec<DType>::kEnabled;
};
template<int dim, typename DType>
struct SSECheck<Tensor<cpu, dim, DType> > {
  static const bool kPass = sse2::FVec<DType>::kEnabled;
};
template<typename OP, typename TA, typename DType, int etype>
struct SSECheck<UnaryMapExp<OP, TA, DType, etype> > {
  static const bool kPass = SSECheck<TA>::kPass && sse2::SSEOp<OP>::kEnabled;
};
template<typename OP, typename TA, typename TB, typename DType, int etype>
struct SSECheck< BinaryMapExp<OP, TA, TB, DType, etype> > {
  static const bool kPass = SSECheck<TA>::kPass &&
      SSECheck<TB>::kPass && sse2::SSEOp<OP>::kEnabled;
};
//-------------------------------------------------
// Check if data is aligned and allow sse operation
//-------------------------------------------------
template<int dim, typename E>
struct SSEAlignCheck {
  inline static bool Check(const E &exp) {
    return false;
  }
};
template<int dim, typename DType>
struct SSEAlignCheck<dim, ScalarExp<DType> > {
  inline static bool Check(const ScalarExp<DType> &exp) {
    return true;
  }
};
template<int dim, typename DType>
struct SSEAlignCheck<dim, Tensor<cpu, dim, DType> > {
  inline static bool Check(const Tensor<cpu, dim, DType> &t) {
    return sse2::CheckAlign(t.dptr_) &&
        sse2::CheckAlign(t.stride_ * sizeof(DType));
  }
};
template<int dim, typename OP, typename TA, typename DType, int etype>
struct SSEAlignCheck<dim, UnaryMapExp<OP, TA, DType, etype> > {
  inline static bool Check(const UnaryMapExp<OP, TA, DType, etype> &t) {
    return SSEAlignCheck<dim, TA>::Check(t.src_);
  }
};
template<int dim, typename OP, typename TA, typename TB,
         typename DType, int etype>
struct SSEAlignCheck<dim, BinaryMapExp<OP, TA, TB, DType, etype> > {
  inline static bool Check(const BinaryMapExp<OP, TA, TB, DType, etype> &t) {
    return SSEAlignCheck<dim, TA>::Check(t.lhs_) &&
        SSEAlignCheck<dim, TB>::Check(t.rhs_);
  }
};
/*!
 * \brief use SSEPlan to compute result
 */
template<typename SV, typename E, int dim, typename DType>
inline void MapSSEPlan(Tensor<cpu, dim, DType> _dst,
                       const expr::SSEPlan<E, DType> &plan) {
  Tensor<cpu, 2, DType> dst = _dst.FlatTo2D();
  const index_t xlen = sse2::LowerAlign(dst.size(1), sizeof(DType));
  for (index_t y = 0; y < dst.size(0); ++y) {
    for (index_t x = 0; x < xlen; x += sse2::FVec<DType>::kSize) {
      sse2::Saver<SV, DType>::Save(&dst[y][x], plan.EvalSSE(y, x));
    }
    for (index_t x = xlen; x < dst.size(1); ++x) {
      SV::Save(dst[y][x], plan.Eval(y, x));
    }
  }
}
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_USE_SSE
#endif  // MSHADOW_SSE_INL_H_
