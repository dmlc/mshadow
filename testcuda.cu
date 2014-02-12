#include "tensor/tensor.h"
// must include this file to get gpu part implementation
#include "tensor/tensor_gpu-inl.hpp"

using namespace cxxnet;
int main( void ){
    Shape<3> s = Shape3(2,4,10); 
    CTensor3D mat1 = NewCTensor( s, 10 );
    CTensor3D mat2 = NewCTensor( s, 20 );
    CTensor3D mat3 = NewCTensor( s, 30 );
    GTensor3D gmat1(s), gmat2(s), gmat3(s);
    printf("alloc space\n");
    AllocSpace(gmat1); 
    AllocSpace(gmat2); 
    AllocSpace(gmat3);
    printf("alloc space finish\n");
    Copy( gmat1, mat1 );
    Copy( gmat2, mat2 );
    printf("alloc space finish\n");
    //Map<sv::saveto, op::plus>(gmat3, gmat1, gmat2);
    //Copy( mat3, gmat3 );
    for (unsigned i = 0; i < 2; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            printf("%.2f ", mat3[i][j][0]);
        }
        printf("\n");
    }
    return 0;
}
