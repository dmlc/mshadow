#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include <cstdio>
// simple file to test if it compiles
using namespace cxxnet;

int main( void ){
    float a[20] = {0};
    float b[20] = {0};
    float c[20] = {0};
    Shape<2> s;
    s.stride_ = 5;
    s[0] = 4;
    s[1] = 5;
    #ifdef __CUDACC__
    printf("CUDA!\n");
    #endif
    CTensor2D mat1( &a[0], s );
    CTensor2D mat2( &b[0], s );
    CTensor2D mat3( &c[0], s );

    mat1[0][0] = 10;
    mat1[0][1] = 20;

    mat1[1][0] = 5;
    mat2[2][0] = 6;

    Map<sv::saveto, op::plus>(mat3, mat1, mat2);
    for (int i = 0; i < s[0]; ++i) {
        for (int j = 0; j < s[1]; ++j) {
            printf("%.2f ", mat3[i][j]);
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
