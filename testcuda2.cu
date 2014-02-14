#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
// must include this file to get gpu part implementation

using namespace mshadow;
using namespace mshadow::expr;
extern void testcuda2( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );

void testcuda2( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 ){
    Shape<3> s = mat1.shape;
    TensorContainer<cpu,3>  gmat1(s), gmat2(s), gmat3(s);
    TensorContainer<cpu,2>  gmat4;
    Copy( gmat1, mat1 );
    Copy( gmat2, mat2 );
    printf("alloc space finish\n");
    MapExp<sv::saveto>( gmat3, 
                        F<op::mul>( F<op::plus>( gmat3, ScalarExp(100.0f) ),
                                    ScalarExp(3.0f) ) );
    gmat3 += gmat1;
    Copy( mat3, gmat3 );
}
