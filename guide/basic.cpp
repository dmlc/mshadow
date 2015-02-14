// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main(void) {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<cpu>();
  // assume we have a float space
  float data[20];
  // create a 2 x 5 x 2 tensor, from existing space
  Tensor<cpu, 3> ts(data, Shape3(2,5,2));
    // take first subscript of the tensor
  Tensor<cpu, 2> mat = ts[0];
  // Tensor object is only a handle, assignment means they have same data content
  // we can specify content type of a Tensor, if not specified, it is float bydefault
  Tensor<cpu, 2, float> mat2 = mat;
  mat = Tensor<cpu, 1>(data, Shape1(10)).FlatTo2D();

  // shaape of matrix, note size order is same as numpy
  printf("%u X %u matrix\n", mat.size(0), mat.size(1));
  return 0;
  // initialize all element to zero
  mat = 0.0f;
  // assign some values
  mat[0][1] = 1.0f; mat[1][0] = 2.0f;
  // elementwise operations
  mat += (mat + 10.0f) / 10.0f + 2.0f;
  
  // print out matrix, note: mat2 and mat1 are handles(pointers)
  for (index_t i = 0; i < mat.size(0); ++i) {
    for (index_t j = 0; j < mat.size(1); ++j) {
      printf("%.2f ", mat2[i][j]);
    }
    printf("\n");
  }
  // shutdown tensor enigne after usage
  ShutdownTensorEngine<cpu>();
  return 0;
}
