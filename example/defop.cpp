#include <cmath>
// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads 
using namespace mshadow::expr;

// user defined unary operator addone
struct addone{
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return  a + 1.0f;
    }
};
// user defined binary operator max of two
struct maxoftwo{
    MSHADOW_XINLINE static real_t Map(real_t a,real_t b) {
        if( a > b ) return a;
        else return b;
    }
};

int main( void ){
    // intialize tensor engine before using tensor operation, needed for CuBLAS
    InitTensorEngine();
    // take first subscript of the tensor 
    Tensor<cpu,2> mat = NewTensor<cpu>( Shape2(2,3), 0.0f ); 
    Tensor<cpu,2> mat2= NewTensor<cpu>( Shape2(2,3), 0.0f );

    mat[0][0] = -2.0f;
    mat = F<maxoftwo>( F<addone>( mat ) + 1.0f, mat2 );
    
    for( index_t i = 0; i < mat.shape[1]; i ++ ){
        for( index_t j = 0; j < mat.shape[0]; j ++ ){
            printf("%.2f ", mat[i][j]);
        }
        printf("\n");
    }

    FreeSpace( mat ); FreeSpace( mat2 );
    // shutdown tensor enigne after usage
    ShutdownTensorEngine();
    return 0;
}
