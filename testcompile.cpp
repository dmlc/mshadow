#include "tensor/tensor.h"
#include <cstdio>
// simple file to test if it compiles
using namespace cxxnet;

int main( void ){
    float a[100]={10}, b[100] = {1};
    Shape<2> s;
    s.stride_ = 10;
    s[0] = 10; s[1] = 10;


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
    return 0;
}
