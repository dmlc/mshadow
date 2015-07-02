// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<gpu>();
  // create a 2 x 5 tensor, from existing space
  Tensor<gpu, 2, float> ts1 = NewTensor<gpu, float>(Shape2(2, 5), 0.0f);
  Tensor<gpu, 2, float> ts2 = NewTensor<gpu, float>(Shape2(2, 5), 0.0f);
  ts1.stream_ = NewStream<gpu>();
  ts2.stream_ = NewStream<gpu>();
  ts1 = 1; // Should use stream 0.
  ts2 = 2; // Should use stream 1. Can run in parallel with stream 0.
  Tensor<gpu, 2> res = NewTensor<gpu, float>(Shape2(2, 2), 0.0f);
  res.stream_ = NewStream<gpu>();
  res = dot(ts1, ts2.T()); //Should use stream 2.

  Tensor<cpu, 2> cpu_res = NewTensor<cpu, float>(Shape2(2, 2), 0.0f);
  Copy(cpu_res, res); // default stream, should be 0.
  for (index_t i = 0; i < cpu_res.size(0); ++i){
    for (index_t j = 0; j < cpu_res.size(1); ++j){
      printf("%.2f ", cpu_res[i][j]);
    }
    printf("\n");
  }
  // shutdown tensor enigne after usage
  ShutdownTensorEngine<gpu>();
  return 0;
}
