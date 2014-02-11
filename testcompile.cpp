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
    s[0] = 5;
    s[1] = 4;
    #ifdef __CUDACC__
    printf("CUDA!\n");
    #endif
    CTensor2D mat1( &a[0], s );
    CTensor2D mat2( &b[0], s );
    CTensor2D mat3( &c[0], s );

    CTensor2D mat( &a[0], s );
    CTensor2D matb( &b[0], s );
    CTensor2D ms = mat.Slice( 1, 10 );
    
    mat[0][0] = 10;
    mat[1][0] = 100;
    mat[2][0] = 200;
    Map<sv::saveto,op::plus>( matb, mat, mat );
    printf( "%f,%f\n", ms[0][0], ms[1][0] );
    printf( "%f,%f\n", mat[0][0], mat[1][0] );
    printf( "%f,%f\n", matb[0][0], matb[1][0] );

    mat1[0][0] = 10;
    mat1[0][1] = 20;

    mat1[1][0] = 5;
    mat2[2][0] = 6;

    Map<sv::saveto, op::plus>(mat3, mat1, mat2);
    for (unsigned i = 0; i < s[1]; ++i) {
        for (unsigned j = 0; j < s[0]; ++j) {
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
