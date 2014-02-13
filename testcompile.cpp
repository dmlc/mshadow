#include "mshadow/tensor.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;

// implemented by testcuda.cu
void testcuda( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );
void testcuda2( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );

int main( void ){    
    CTensor3D mat1 = NewCTensor( Shape3(2,4,10), 10 );
    CTensor3D mat2 = NewCTensor( Shape3(2,4,10), 20 );
    CTensor3D mat3 = NewCTensor( Shape3(2,4,10), 30 );
    testcuda2( mat1, mat2, mat3 );
    testcuda( mat1, mat2, mat3 );
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
