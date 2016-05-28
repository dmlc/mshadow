#include <mshadow/tensor.h>
#include <vector>

// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

void test_tblob() {
  // intialize tensor engine before using tensor operation, needed for CuBLAS
  InitTensorEngine<cpu>();
  // assume we have a float space
  float data[20];
  // create a 2 x 5 x 2 tensor, from existing space
  Tensor<cpu, 3> ts(data, Shape3(2,5,2));
  TBlob tb = ts;
  TBlob xx(tb);
  // take first subscript of the tensor
  Tensor<cpu, 2> mat = tb.get<cpu, 3, float>()[0];
  // Tensor object is only a handle, assignment means they have same data content
  // we can specify content type of a Tensor, if not specified, it is float bydefault
  Tensor<cpu, 2, float> mat2 = mat;
  mat = Tensor<cpu, 1>(data, Shape1(10)).FlatTo2D();

  // shaape of matrix, note size order is same as numpy
  printf("%u X %u matrix\n", mat.size(0), mat.size(1));

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
}

void test_tshape() {
  std::vector<index_t> a = {1,2,3,4,5,6};
  TShape as; as = a;
  TShape b = std::move(as);
  for (index_t i = 0; i < b.ndim(); ++i) {
    printf("%u\n", b[i]);
  }
  TShape bb; bb = Shape3(1,2,3);
  for (index_t i = 0; i < bb.ndim(); ++i) {
    printf("%u\n", b[i]);
  }  
  b = bb;
  for (index_t i = 0; i < b.ndim(); ++i) {
    printf("%u\n", b[i]);
  }  
}

void test_broadcast_with_axis() {
  std::vector<mshadow::Shape<4> > test_shapes;
  std::vector<mshadow::Shape<4> > keepdim_input_shapes;

  test_shapes.push_back(mshadow::Shape4(5, 2, 3, 4));
  test_shapes.push_back(mshadow::Shape4(2, 5, 3, 4));
  test_shapes.push_back(mshadow::Shape4(2, 3, 5, 4));
  test_shapes.push_back(mshadow::Shape4(2, 3, 4, 5));

  keepdim_input_shapes.push_back(mshadow::Shape4(1, 2, 3, 4));
  keepdim_input_shapes.push_back(mshadow::Shape4(2, 1, 3, 4));
  keepdim_input_shapes.push_back(mshadow::Shape4(2, 3, 1, 4));
  keepdim_input_shapes.push_back(mshadow::Shape4(2, 3, 4, 1));

  for (int dim = -1; dim < 3; ++dim){
    mshadow::Tensor<mshadow::cpu, 3> input_tensor(NULL, mshadow::Shape3(2, 3, 4));
    mshadow::AllocSpace(&input_tensor);
    input_tensor = 11;
    mshadow::Tensor<mshadow::cpu, 4> n_tensor(NULL, test_shapes[dim + 1]);
    mshadow::AllocSpace(&n_tensor);
    n_tensor = broadcast_with_axis(input_tensor, dim, 5);
    printf("Test for keepdim = 0, dim = %d", dim);
    for (index_t i = 0; i < n_tensor.shape_[0]; i++) {
      for (index_t j = 0; j < n_tensor.shape_[1]; j++) {
        for (index_t k = 0; k < n_tensor.shape_[2]; k++) {
          for (index_t l = 0; l < n_tensor.shape_[3]; l++) {
            CHECK_EQ(n_tensor[i][j][k][l], 11);
          }
        }
      }
    }
    printf(" Pass!\n");
  }

  for (int dim = 0; dim < 4; ++dim){
    mshadow::Tensor<mshadow::cpu, 4> input_tensor(NULL, keepdim_input_shapes[dim]);
    mshadow::AllocSpace(&input_tensor);
    input_tensor = 11;
    mshadow::Tensor<mshadow::cpu, 4> n_tensor(NULL, test_shapes[dim]);
    mshadow::AllocSpace(&n_tensor);
    n_tensor = broadcast_keepdim(input_tensor, dim, 5);
    printf("Test for keepdim = 1, dim = %d", dim);
    for (index_t i = 0; i < n_tensor.shape_[0]; i++) {
      for (index_t j = 0; j < n_tensor.shape_[1]; j++) {
        for (index_t k = 0; k < n_tensor.shape_[2]; k++) {
          for (index_t l = 0; l < n_tensor.shape_[3]; l++) {
            CHECK_EQ(n_tensor[i][j][k][l], 11);
          }
        }
      }
    }
    printf(" Pass!\n");

  }
}

void test_reduce_with_axis() {
  std::vector<mshadow::Shape<4> > test_shapes;
  std::vector<mshadow::Shape<4> > keepdim_output_shapes;

  test_shapes.push_back(mshadow::Shape4(5, 2, 3, 4));
  test_shapes.push_back(mshadow::Shape4(2, 5, 3, 4));
  test_shapes.push_back(mshadow::Shape4(2, 3, 5, 4));
  test_shapes.push_back(mshadow::Shape4(2, 3, 4, 5));

  keepdim_output_shapes.push_back(mshadow::Shape4(1, 2, 3, 4));
  keepdim_output_shapes.push_back(mshadow::Shape4(2, 1, 3, 4));
  keepdim_output_shapes.push_back(mshadow::Shape4(2, 3, 1, 4));
  keepdim_output_shapes.push_back(mshadow::Shape4(2, 3, 4, 1));

  for (int dim = 0; dim < 4; ++dim){
    mshadow::Tensor<mshadow::cpu, 4> input_tensor(NULL, test_shapes[dim]);
    mshadow::AllocSpace(&input_tensor);
    input_tensor = 1;
    mshadow::Tensor<mshadow::cpu, 3> n_tensor(NULL, mshadow::Shape3(2, 3, 4));
    mshadow::AllocSpace(&n_tensor);
    n_tensor = reduce_with_axis<mshadow::red::sum, false>(input_tensor, dim);
    printf("Test for keepdim = 0, dim = %d", dim);
    for (index_t i = 0; i < n_tensor.shape_[0]; i++) {
      for (index_t j = 0; j < n_tensor.shape_[1]; j++) {
        for (index_t k = 0; k < n_tensor.shape_[2]; k++) {
          CHECK_EQ(n_tensor[i][j][k], 5);
        }
      }
    }
    printf(" Pass!\n");
  }

  for (int dim = 0; dim < 4; ++dim){
    mshadow::Tensor<mshadow::cpu, 4> input_tensor(NULL, test_shapes[dim]);
    mshadow::AllocSpace(&input_tensor);
    input_tensor = 1;
    mshadow::Tensor<mshadow::cpu, 4> n_tensor(NULL, keepdim_output_shapes[dim]);
    mshadow::AllocSpace(&n_tensor);
    n_tensor = reduce_keepdim<mshadow::red::sum, false>(input_tensor, dim);
    printf("Test for keepdim = 1, dim = %d", dim);
    for (index_t i = 0; i < n_tensor.shape_[0]; i++) {
      for (index_t j = 0; j < n_tensor.shape_[1]; j++) {
        for (index_t k = 0; k < n_tensor.shape_[2]; k++) {
          for (index_t l = 0; l < n_tensor.shape_[3]; l++) {
            CHECK_EQ(n_tensor[i][j][k][l], 5);
          }
        }
      }
    }
    printf(" Pass!\n");
  }
}

int main(void) {
  test_tshape();
  test_broadcast_with_axis();
  test_reduce_with_axis();
  return 0;
}



