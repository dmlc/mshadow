/*!
 *  Copyright (c) 2014 by Contributors
 * \file range.h
 * \brief support generating a range vector
 * \author Xingjian Shi
 */
#ifndef MSHADOW_EXTENSION_RANGE_H_
#define MSHADOW_EXTENSION_RANGE_H_

#include "../extension.h"

namespace mshadow {
namespace expr {
/*!
 * \brief Generate a range vector similar to python: range(start, stop[, step][, repeat]).
          If step is positive, the last element is the largest start + i * step less than stop
          If step is negative, the last element is the smallest start + i * step greater than stop.
          All elements are repeated for `repeat` times, e.g range(0, 4, 2, 3) --> 0, 0, 0, 2, 2, 2
 * \tparam SrcExp type of lhs expression
 * \tparam IndexExp type of index expression
 * \tparam DType the type of elements
 */
template<typename DType>
struct RangeExp:
      public Exp<RangeExp<DType>, DType, type::kMapper> {
  const float start_;
  const float stop_;
  const float step_;
  const int repeat_;
  /*! \brief constructor */
  RangeExp(DType start, DType stop, DType step, int repeat)
      : start_(start), stop_(stop), step_(step), repeat_(repeat) {}
};

template<typename DType>
inline RangeExp<DType>
range(DType start, DType stop, DType step = 1, int repeat = 1) {
  return RangeExp<DType>(start, stop, step, repeat);
}

//----------------------
// Execution plan
//----------------------
template<typename DType>
struct Plan<RangeExp<DType>, DType> {
 public:
  explicit Plan(const RangeExp<DType> &e)
      : start_(e.start_),
        stop_(e.stop_),
        step_(e.step_),
        repeat_(e.repeat_) {
  }
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    return start_ + static_cast<DType>((static_cast<int>(x) / repeat_)) * step_;
  }

 private:
  const DType start_;
  const DType stop_;
  const DType step_;
  const int repeat_;
};

template<typename DType>
inline Plan<RangeExp<DType>, DType>
MakePlan(const RangeExp<DType> &exp) {
  return Plan<RangeExp<DType>, DType>(exp);
}

inline int RangeOutSize(float start, float stop, float step, int repeat) {
  return repeat * static_cast<int>(ceil((stop - start) / step));
}

inline int RangeOutSize(double start, double stop, float step, int repeat) {
  return repeat * static_cast<int>(ceil((stop - start) / step));
}

inline int RangeOutSize(int start, int stop, int step, int repeat) {
  return repeat * ((stop - start - 1) / step + 1);
}

template<int dim, typename DType>
struct ShapeCheck<dim, RangeExp<DType> > {
  inline static Shape<dim>
  Check(const RangeExp<DType> &t) {
    CHECK(dim == 1)
        << "RangeExp only support 1 dimension output, received " << dim;
    CHECK(t.step_ != 0)
        << "RangeExp does not support step=0, received " << t.step_;
    CHECK(t.repeat_ > 0)
      << "RangeExp only supports repeat > 0, received " << t.repeat_;
    if (t.step_ > 0) {
      CHECK(t.start_ < t.stop_) << "RangeExp does not support (start, stop, step) = "
                                << "(" << t.start_ << "," << t.stop_ << "," << t.step_ << ")";
    } else {
      CHECK(t.start_ > t.stop_) << "RangeExp does not support (start, stop, step)= "
                                << "(" << t.start_ << "," << t.stop_ << "," << t.step_ << ")";
    }
    return Shape1(RangeOutSize(t.start_, t.stop_, t.step_, t.repeat_));
  }
};

template<typename DType>
struct ExpInfo<RangeExp<DType> > {
  static const int kDim = 1;
  static const int kDevMask = 0xffff;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_RANGE_H_
