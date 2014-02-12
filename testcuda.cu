#include "tensor/tensor.h"
// must include this file to get gpu part implementation

using namespace cxxnet;
using namespace cxxnet::algebra;
extern void testcuda( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );

void testcuda( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 ){
    Shape<3> s = mat1.shape;
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
    MapExp<sv::saveto>( gmat3, 
                        MakeExp<op::mul>
                        (
                         MakeExp<op::plus> ( MakeExp(gmat1), MakeExp(100.0f) ),
                         MakeExp(3.0f)
                         ) );
    
    Copy( mat3, gmat3 );    
    FreeSpace(gmat1);
    FreeSpace(gmat2);
    FreeSpace(gmat3);
}
