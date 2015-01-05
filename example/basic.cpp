// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main(void) {

  // intialize tensor engine before using tensor operation, needed for CuBLAS
  //InitTensorEngine();
  // assume we have a float space
  double data[20];
  // create a 2 x 5 x 2 tensor, from existing space
  Tensor<cpu, 3, double> ts(data, Shape3(2,5,2));
  Tensor<cpu, 4, double> ts4(data, Shape4(2,2,2,2));
  // take first subscript of the tensor
  Tensor<cpu, 2, double> mat = ts[0];
  // Tensor object is only a handle, assignment means they have same data content
  Tensor<cpu, 1, double> mat2= NewTensor<cpu, double>(Shape1(2), 0.0f);
  Tensor<cpu, 3, double> ts1= NewTensor<cpu, double>(ts.shape_, 0.0f);
  Random<cpu, double> rnd(0);
  ts.stream_ = NewStream<cpu>();
  DeleteStream(ts.stream_);

  mat2 = rnd.uniform(mat2.shape_);
  // shape of matrix, note shape order is different from numpy
  // shape[i] indicate the shape of i-th dimension
  printf("%u X %u matrix, stride=%u\n", mat.size(0), mat.size(1), mat.stride_);

  // assign some values
  mat[0][1] = 1.0f; mat[1][0] = 2.0f;
  // elementwise operations

  //ts = broadcast<0>(mat2, ts.shape_);  
  mat2 = sumall_except_dim<0>(mat);
  // print out matrix, note: mat2 and mat1 are handles(pointers)
  for (index_t c = 0; c < ts.size(0); ++c) {
  for (index_t i = 0; i < mat.size(0); ++i) {
    for (index_t j = 0; j < mat.size(1); ++j) {
      printf("%.2f ", ts[c][i][j]);
    }
    printf("\n");
  }
  }
  // create a tensor without explictly allocating spaces.
  Tensor<cpu, 2, float> mat3 = NewTensor<cpu, float>(Shape2(2, 5), 0.0f);
  Tensor<cpu, 2, float> mat4 = NewTensor<cpu, float>(Shape2(2, 5), 1.0f);
  // transpose, and then add mat4.
  mat3 = tcast<float>(mat.T()) + mat4;
  
  // index the shape using size(), this is more natural for MATLAB/numpy user.
  printf("%u X %u matrix\n", mat3.size(0), mat3.size(1));
  // print out matrix
  for (index_t i = 0; i < mat3.size(0); ++i) {
    for (index_t j = 0; j < mat3.size(1); ++j) {
      printf("%.2f ", mat3[i][j]);
    }
    printf("\n");
  }
  // shutdown tensor enigne after usage
  //ShutdownTensorEngine();
  return 0;
}
