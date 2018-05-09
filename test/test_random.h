#pragma once

#include <iostream>
#include <limits>
#include <type_traits>
#include "mshadow/tensor.h"
#include "assert.h"

using namespace mshadow;
using namespace std;

using hf_t = mshadow::half::half_t;

template<typename Device, int dim, typename DType, typename S>
Tensor<Device, dim, DType> tensor(S shape, Stream<Device>* stream) {
  Tensor<Device, dim, DType> ret(shape);
  ret.set_stream(stream);
  AllocSpace(&ret);
  ret = 123;
  return ret;
}

template<int dim, typename DType>
Tensor<cpu, dim, DType> copyAndFree(Tensor<cpu, dim, DType>& t) {
  return t;
}

template<typename DType>
DType inc(DType f, unsigned n) {
  for(unsigned i = 0; i < n; ++i) {
    f = nextafter(f, std::numeric_limits<DType>::infinity());
  }
  return f;
}

#if MSHADOW_USE_CUDA
template<int dim, typename DType>
Tensor<cpu, dim, DType> copyAndFree(Tensor<gpu, dim, DType>& t) {
  Tensor<cpu, dim, DType> ret(t.shape_);
  AllocSpace(&ret);
  Copy(ret, t, t.stream_);
  FreeSpace(&t);
  return ret;
}
#endif

template<>
hf_t inc<hf_t>(hf_t f, unsigned n) {
  // Valid if the significand is close to 1
  uint16_t ui = f.half_;
  for(unsigned i = 0; i < n; ++i) {
    ui = ((ui & 0x03ff) + 1) + (ui & 0xfc00);
  }
  return hf_t::Binary(ui);
}

template<typename T>
MSHADOW_XINLINE const char* typenameString();

template<>
MSHADOW_XINLINE const char* typenameString<uint16_t>() {
  return "uint16_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<int16_t>() {
  return "int16_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<uint32_t>() {
  return "uint32_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<int32_t>() {
  return "int32_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<uint64_t>() {
  return "uint64_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<int64_t>() {
  return "int64_t";
}

template<>
MSHADOW_XINLINE const char* typenameString<float>() {
  return "float";
}

template<>
MSHADOW_XINLINE const char* typenameString<double>() {
  return "double";
}

template<>
MSHADOW_XINLINE const char* typenameString<hf_t>() {
  return "half_t";
}

template<typename Device, typename DType>
bool CheckSeed(Stream<Device>* stream) {
  cout << "Check Seed <" << typenameString<DType>() << ">" << endl;
  {
    // Generators initialized with the same seed should produce the same random number sequence
    // The same default sequence id is used implicitly
    auto ts0 = tensor<Device, 1, DType>(Shape1(10), stream);
    auto ts1 = tensor<Device, 1, DType>(ts0.shape_, stream);
    Random<Device, DType> rand0(0);
    rand0.set_stream(stream);
    rand0.SampleUniform(&ts0, DType(0), DType(10000));
    Random<Device, DType> rand1(0);
    rand1.set_stream(stream);
    rand1.SampleUniform(&ts1, DType(0), DType(10000));
    auto tr0 = copyAndFree(ts0);
    auto tr1 = copyAndFree(ts1);
    for (unsigned i = 0; i < tr0.size(0); ++i) {
      assert(tr0[i] == tr1[i]);
    }
    FreeSpace(&tr0);
    FreeSpace(&tr1);
  }
  {
    // Generators initialized with different seeds should produce different random number sequences
    // The same default sequence id is used implicitly
    auto ts0 = tensor<Device, 1, DType>(Shape1(10), stream);
    auto ts1 = tensor<Device, 1, DType>(ts0.shape_, stream);
    Random<Device, DType> rand0(0);
    rand0.set_stream(stream);
    rand0.SampleUniform(&ts0, DType(0), DType(10000));
    Random<Device, DType> rand1(1);
    rand1.set_stream(stream);
    rand1.SampleUniform(&ts1, DType(0), DType(10000));
    auto tr0 = copyAndFree(ts0);
    auto tr1 = copyAndFree(ts1);
    for (unsigned i = 0; i < tr0.size(0); ++i) {
      // Extremely low probability of the same numbers
      assert(tr0[i] != tr1[i]);
    }
    FreeSpace(&tr0);
    FreeSpace(&tr1);
  }
  {
    // Check the effect of seeding after the initialization
    auto ts0 = tensor<Device, 1, DType>(Shape1(10), stream);
    auto ts1 = tensor<Device, 1, DType>(ts0.shape_, stream);
    Random<Device, DType> rand0(0);
    rand0.set_stream(stream);
    rand0.SampleUniform(&ts0, DType(0), DType(10000));
    Random<Device, DType> rand1(1);
    rand1.set_stream(stream);
    rand1.Seed(0);
    rand1.SampleUniform(&ts1, DType(0), DType(10000));
    auto tr0 = copyAndFree(ts0);
    auto tr1 = copyAndFree(ts1);
    for (unsigned i = 0; i < tr0.size(0); ++i) {
      // Extremely low probability of the same numbers
      assert(tr0[i] == tr1[i]);
    }
    FreeSpace(&tr0);
    FreeSpace(&tr1);
  }
  return true;
}

template<typename Device, typename DType>
bool CheckSequence(Stream<Device>* stream) {
  cout << "Check Sequence <" << typenameString<DType>() << ">" << endl;
  {
    // Generators with different sequence id should generate different random numbers
    auto ts0 = tensor<Device, 1, DType>(Shape1(10), stream);
    auto ts1 = tensor<Device, 1, DType>(ts0.shape_, stream);
    Random<Device, DType> rand0(0, 0);
    rand0.set_stream(stream);
    rand0.SampleUniform(&ts0, DType(0), DType(10000));
    Random<Device, DType> rand1(0, 1);
    rand0.set_stream(stream);
    rand1.SampleUniform(&ts1, DType(0), DType(10000));
    auto tr0 = copyAndFree(ts0);
    auto tr1 = copyAndFree(ts1);
    for (unsigned i = 0; i < tr0.size(0); ++i) {
      assert(tr0[i] != tr1[i]);
    }
    FreeSpace(&tr0);
    FreeSpace(&tr1);
  }
  {
    // Effect of setting the sequence id after initialization
    auto ts0 = tensor<Device, 1, DType>(Shape1(10), stream);
    auto ts1 = tensor<Device, 1, DType>(ts0.shape_, stream);
    Random<Device, DType> rand0(0, 0);
    rand0.set_stream(stream);
    rand0.SampleUniform(&ts0, DType(0), DType(10000));
    Random<Device, DType> rand1(0, 1);
    rand1.set_stream(stream);
    rand1.Seed(0, 0);
    rand1.SampleUniform(&ts1, DType(0), DType(10000));
    auto tr0 = copyAndFree(ts0);
    auto tr1 = copyAndFree(ts1);
    for (unsigned i = 0; i < tr0.size(0); ++i) {
      assert(tr0[i] == tr1[i]);
    }
    FreeSpace(&tr0);
    FreeSpace(&tr1);
  }
  return true;
}

template<typename Device, typename DType>
bool CheckGetRandInt(Stream<Device>* stream) {
  cout << "Check GetRandInt <" << typenameString<DType>() << ">" << endl;
  {
    using uint_type = typename conditional<sizeof(DType) <= 4, uint32_t, uint64_t>::type;
    auto ts = tensor<Device, 1, uint_type>(Shape1(10000), stream);
    Random<Device, DType> rand(0);
    rand.set_stream(stream);
    rand.GetRandInt(ts);
    auto tr = copyAndFree(ts);
    const uint64_t max = numeric_limits<uint_type>::max();
    for (unsigned i = 1; i < tr.size(0); ++i) {
      assert(tr[i] <= max);
      // Extremely low probability of the same integers in sequel
      assert(tr[i] != tr[i - 1]);
    }
    FreeSpace(&tr);
  }
  {
    // Tensor with any integer type should work
    auto ts = tensor<Device, 1, int>(Shape1(10000), stream);
    Random<Device, DType> rand(0);
    rand.set_stream(stream);
    rand.GetRandInt(ts);
    auto tr = copyAndFree(ts);
    for (unsigned i = 1; i < tr.size(0); ++i) {
      // Extremely low probability of the same integers in sequel
      assert(tr[i] != tr[i - 1]);
    }
    FreeSpace(&tr);
  }
  {
    // Tensor with any integer type should work
    auto ts = tensor<Device, 1, uint64_t>(Shape1(10000), stream);
    Random<Device, DType> rand(0);
    rand.set_stream(stream);
    rand.GetRandInt(ts);
    auto tr = copyAndFree(ts);
    for (unsigned i = 1; i < tr.size(0); ++i) {
      // Extremely low probability of same integers in sequel
      assert(tr[i] != tr[i - 1]);
    }
    FreeSpace(&tr);
  }
  return true;
}

template<typename Device, typename DType>
bool CheckSampleUniformInteger(Stream<Device>* stream) {
  cout << "Check SampleUniformInteger <" << typenameString<DType>() << ">" << endl;
  // Check the range and the first two moments of the generated random numbers.
  {
    for (DType lo = 0; lo < 10; ++lo) {
      for (DType hi = lo; hi < lo + 10; ++hi) {
        auto ts = tensor<Device, 2, DType>(Shape2(1000, 10000), stream);
        Random<Device, DType> rand(9191);
        rand.set_stream(stream);
        rand.SampleUniform(&ts, lo, hi);
        auto tr = copyAndFree(ts);
        double x1 = 0;
        double x2 = 0;
        unsigned lower_bound_count = 0;
        unsigned upper_bound_count = 0;
        for (unsigned i = 0; i < tr.shape_[0]; ++i) {
          for (unsigned j = 0; j < tr.shape_[1]; ++j) {
            const DType x = tr[i][j];
            assert(x != 1234); // Elements were initialized as 1234.
            assert(lo <= x && x <= hi);
            lower_bound_count += x == lo ? 1 : 0;
            upper_bound_count += x == hi ? 1 : 0;
            x1 += x;
            x2 += x * x;
          }
        }
        assert(lower_bound_count > 0);  // inclusive lower bound
        assert(upper_bound_count > 0);  // inclusive upper bound
        assert(abs(x1 / tr.shape_.Size() - (lo + hi) / 2.) < 0.1);
        assert(abs(x2 / tr.shape_.Size() - (2 * (hi * hi + lo * lo + hi * lo) + hi - lo) / 6.) < 0.2);
        FreeSpace(&tr);
      }
    }
  }
  // Runs only for signed type. Ignore the compiler warning.
  if (is_signed<DType>::value) {
    for (DType a = 1; a < 10; ++a) {
      auto ts = tensor<Device, 2, DType>(Shape2(1000, 10000), stream);
      Random<Device, DType> rand(9191);
      rand.set_stream(stream);
      rand.SampleUniform(&ts, DType(-a), a);
      auto tr = copyAndFree(ts);
      double x1 = 0;
      double x2 = 0;
      unsigned lower_bound_count = 0;
      unsigned upper_bound_count = 0;
      for (unsigned i = 0; i < tr.shape_[0]; ++i) {
        for (unsigned j = 0; j < tr.shape_[1]; ++j) {
          const DType x = tr[i][j];
          assert(x != 1234); // Elements were initialized as 1234.
          assert(-a <= x && x <= a);
          lower_bound_count += x == -a ? 1 : 0;
          upper_bound_count += x == a ? 1 : 0;
          x1 += x;
          x2 += x * x;
        }
      }
      assert(lower_bound_count > 0); // Boundaries must be inclusive.
      assert(upper_bound_count > 0); // Boundaries must be inclusive.
      assert(abs(x1 / tr.shape_.Size() - 0) < 0.01);
      assert(abs(x2 / tr.shape_.Size() - (1 + a) * a / 3.) < 0.1);
      FreeSpace(&tr);
    }
  }
  return true;
}

template<typename Device, typename DType>
bool CheckSampleUniformReal(Stream<Device>* stream) {
  cout << "Check SampleUniformReal <" << typenameString<DType>() << ">" << endl;
  {
    // Check the range and the first two moments of the generated random numbers.
    auto ts = tensor<Device, 2, DType>(Shape2(10000, 10000), stream);
    Random<Device, DType> rand(9191);
    rand.set_stream(stream);
    rand.SampleUniform(&ts, -5, 5);
    auto tr = copyAndFree(ts);
    double x1 = 0;
    double x2 = 0;
    for (unsigned i = 0; i < tr.shape_[0]; ++i) {
      for (unsigned j = 0; j < tr.shape_[1]; ++j) {
        const double x = tr[i][j];
        assert(x != 1234); // Elements were initialized as 1234.
        assert(-5 <= x && x < 5);
        x1 += x;
        x2 += x * x;
      }
    }
    assert(abs(x1 / tr.shape_.Size() - 0) < 0.01);
    assert(abs(x2 / tr.shape_.Size() - 100.0 / 12) < 1);
    FreeSpace(&tr);
  }
  {
    // The lower bound must be inclusive and the upper bound must be exclusive.
    // We test only the exclusive upper bound.
    // Testing inclusive lower bound is a nonsense.
    for (unsigned i = 0; i < 10; ++i) {
      for (unsigned j = 10; j < 20; ++j) {
        auto ts = tensor<Device, 1, DType>(Shape1(10000000), stream);
        Random<Device, DType> rand(9191);
        rand.set_stream(stream);
        const DType lo = inc(DType(1), i);
        const DType hi = inc(lo, j);
        rand.SampleUniform(&ts, lo, hi);
        auto tr = copyAndFree(ts);
        unsigned upper_bound_count = 0;
        for (unsigned k = 0; k < tr.size(0); ++k) {
          DType x = tr[k];
          assert(lo <= x && x < hi);
          upper_bound_count += x == hi ? 1 : 0;
        }
        assert(upper_bound_count == 0);
      }
    }
  }
  {
    // Here the lower and upper bounds of the distribution are negative and close to 0.
    for (unsigned i = 0; i < 10; ++i) {
      for (unsigned j = 10; j < 20; ++j) {
        auto ts = tensor<Device, 1, DType>(Shape1(10000000), stream);
        Random<Device, DType> rand(9876);
        rand.set_stream(stream);
        const DType a = inc(DType(1), i);
        const DType b = inc(a, j);
        const DType lo = 1 - b;
        const DType hi = 1 - a;
        rand.SampleUniform(&ts, lo, hi);
        auto tr = copyAndFree(ts);
        unsigned upper_bound_count = 0;
        for (unsigned k = 0; k < tr.size(0); ++k) {
          DType x = tr[k];
          assert(lo <= x && x < hi);
          upper_bound_count += x == hi ? 1 : 0;
        }
        assert(upper_bound_count == 0);
      }
    }
  }
  return true;
}

template<typename Device, typename DType>
bool CheckSampleGaussian(Stream<Device>* stream) {
  cout << "Check SampleGaussian <" << typenameString<DType>() << ">" << endl;
  {
    // Check the first three moments of the generated random numbers.
    for (DType m = 0; m < 1; m += 0.2) {
      for (DType s = 1; s > 0; s -= 0.2) {
        auto ts = tensor<Device, 2, DType>(Shape2(1000, 10000), stream);
        Random<Device, DType> rand(0);
        rand.set_stream(stream);
        rand.SampleGaussian(&ts, m, s);
        auto tr = copyAndFree(ts);
        double x1 = 0;
        double x2 = 0;
        double x3 = 0;
        double x4 = 0;
        for (unsigned i = 0; i < tr.shape_[0]; ++i) {
          for (unsigned j = 0; j < tr.shape_[1]; ++j) {
            const double x = tr[i][j];
            assert(x != 1234); // Elements were initialized as 1234.
            x1 += x;
            x2 += x * x;
            x3 += x * x * x;
            x4 += x * x * x * x;
          }
        }
        assert(abs(x1 / tr.shape_.Size() - m) < 0.02);
        assert(abs(x2 / tr.shape_.Size() - (m * m + s * s)) < 0.02);
        assert(abs(x3 / tr.shape_.Size() - (m * m * m + 3 * m * s * s)) < 0.02);
        FreeSpace(&tr);
      }
    }
  }
  {
    // Test for an odd size tensor.
    // The method to generate Gaussian random variables (Box-Muller or its variants)
    // needs a separate test for odd size tensors and even size tensors.
    for (DType m = 0; m < 1; m += 0.2) {
      const DType s = 1;
      auto ts = tensor<Device, 2, DType>(Shape2(999, 9999), stream);
      Random<Device, DType> rand(0);
      rand.set_stream(stream);
      rand.SampleGaussian(&ts, m, s);
      auto tr = copyAndFree(ts);
      double x1 = 0;
      double x2 = 0;
      double x3 = 0;
      double x4 = 0;
      for (unsigned i = 0; i < tr.shape_[0]; ++i) {
        for (unsigned j = 0; j < tr.shape_[1]; ++j) {
          const double x = tr[i][j];
          assert(x != 1234); // Elements were initialized as 1234.
          x1 += x;
          x2 += x * x;
          x3 += x * x * x;
          x4 += x * x * x * x;
        }
      }
      assert(abs(x1 / tr.shape_.Size() - m) < 0.02);
      assert(abs(x2 / tr.shape_.Size() - (m * m + s * s)) < 0.02);
      assert(abs(x3 / tr.shape_.Size() - (m * m * m + 3 * m * s * s)) < 0.02);
      FreeSpace(&tr);
    }
  }
  return true;
}