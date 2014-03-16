#include "mshadow/tensor.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;

// implemented by testcuda.cu

int main( void ){
    Tensor<cpu,3> mat1 = NewTensor<cpu>( Shape3(2,4,10), 10 );
    Tensor<cpu,2> mat2 = NewTensor<cpu>( Shape2(1,80), 20 );
    Tensor<cpu,3> mat3 = NewTensor<cpu>( Shape3(2,4,10), 30 );

    mat1 += reshape( mat2, mat1.shape );
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            printf("%.2f ", mat3[i][j][0]);
        }
        printf("\n");
    }
    FreeSpace( mat1 );
    FreeSpace( mat2 );
    FreeSpace( mat3 );
    return 0;
}
