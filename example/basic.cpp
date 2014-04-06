// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main( void ){
    // intialize tensor engine before using tensor operation, needed for CuBLAS
    InitTensorEngine();
    // assume we have a float space
    real_t data[ 20 ];
    // create a 2 x 5 x 2 tensor, from existing space
    Tensor<cpu,3> ts( data, Shape3(2,5,2) );
    // take first subscript of the tensor
    Tensor<cpu,2> mat = ts[0];
    // Tensor object is only a handle, assignment means they have same data content
    Tensor<cpu,2> mat2 = mat;

    // shape of matrix, note shape order is different from numpy
    // shape[i] indicate the shape of i-th dimension
    printf("%u X %u matrix\n", mat.shape[1], mat.shape[0] );

    // initialize all element to zero
    mat = 0.0f;
    // assign some values
    mat[0][1] = 1.0f; mat[1][0] = 2.0f;
    // elementwise operations
    mat += ( mat + 10.0f ) / 10.0f + 2.0f;

    // print out matrix, note: mat2 and mat1 are handles(pointers)
    for( index_t i = 0; i < mat.shape[1]; i ++ ){
        for( index_t j = 0; j < mat.shape[0]; j ++ ){
            printf("%.2f ", mat2[i][j]);
        }
        printf("\n");
    }
    // shutdown tensor enigne after usage
    ShutdownTensorEngine();
    return 0;
}
