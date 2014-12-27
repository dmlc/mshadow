// header file to use mshadow
#include "mshadow/tensor.h"
// this namespace contains all data structures, functions
using namespace mshadow;
// this namespace contains all operator overloads
using namespace mshadow::expr;

int main( void ){
    // intialize tensor engine before using tensor operation, needed for CuBLAS
    InitTensorEngine();

    Tensor<cpu,2> mat = NewTensor<cpu>( Shape2(1000,1000), 1.0 ); 
	for (int i=0;i<100;i++)
		mat = dot(mat, mat);
	FreeSpace(mat);
    // shutdown tensor enigne after usage
	
    ShutdownTensorEngine();
    return 0;
}
