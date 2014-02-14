#include "mshadow/tensor.h"
// must include this file to get gpu part implementation

using namespace mshadow;
using namespace mshadow::expr;
extern void testcuda( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 );

void testcuda( CTensor3D mat1, CTensor3D mat2, CTensor3D mat3 ){
    Shape<3> s = mat1.shape;
    GTensor3D gmat1(s), gmat2(s), gmat3(s), gmat4;    
    printf("alloc space\n");
    AllocSpace(gmat1); 
    AllocSpace(gmat2); 
    AllocSpace(gmat3);
    printf("alloc space finish\n");
    Copy( gmat1, mat1 );
    Copy( gmat2, mat2 );
    // this dot expression is invalid
    //    gmat1 = 0.1 * dot( gmat2.T(), gmat3 );
    printf("alloc space finish\n");
    gmat3  = F<op::identity>(gmat1 + gmat2 + 3.0);
    gmat3 += gmat1;
    
    Copy( mat3, gmat3 );    
    gmat4 = gmat1;
    printf("%d==1\n", gmat4.dptr == gmat1.dptr);
    FreeSpace(gmat1);
    FreeSpace(gmat2);
    FreeSpace(gmat3);
}
