#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include <cstdio>
// simple file to test if it compiles
using namespace cxxnet;

int main( void ){    
    CTensor3D mat1 = NewCTensor( Shape3(2,4,10), 10 );
    CTensor3D mat2 = NewCTensor( Shape3(2,4,10), 20 );
    CTensor3D mat3 = NewCTensor( Shape3(2,4,10), 30 );
    Copy( mat1, mat2 );
    Map<sv::saveto, op::plus>(mat3, mat1, mat2);
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            printf("%.2f ", mat3[i][j][0]);
        }
        printf("\n");
    }

    // CTensor2D ms = mat.Slice( 2, 10 );

    // mat[0][0] = 10;
    // mat[1][0] = 100;
    // mat[2][0] = 200;
    // printf("%ld\n", ms.dptr - mat.dptr);
    // printf( "%f,%f\n", ms[0][0], ms[1][0] );
    // printf( "%f,%f\n", mat[0][0], mat[1][0] );
    
    return 0;
}
