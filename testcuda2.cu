#include "tensor/tensor.h"
#include "tensor/tensor_container.h"
// must include this file to get gpu part implementation

using namespace cxxnet;
extern void testcuda2( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );

void testcuda2( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 ){
    Shape<3> s = mat1.shape;
    TensorContainer<gpu,3>  gmat1(s), gmat2(s), gmat3(s);
    GTensor3D gm1 = gmat1;
    Copy( gmat1(), mat1 );
    Copy( gmat2(), mat2 );
    printf("alloc space finish\n");
    Map<sv::saveto, op::plus>(gmat3(), gmat1(), gmat2());
    Copy( mat3, gmat3() );    
}
