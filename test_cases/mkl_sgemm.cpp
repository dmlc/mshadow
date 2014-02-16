// only need tensor.h
#include "../mshadow/tensor.h"
#include "../mshadow/tensor_container.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;
using namespace mshadow::expr;

void print(const Tensor<cpu,2> &t) {
    index_t row = t.shape[1];
    index_t col = t.shape[0];
    printf("%2d X %2d\n", row, col);
    for (index_t r = 0; r < row; ++r) {
        for (index_t c = 0; c < col; ++c) {
            printf("%.2f ", t[r][c]);
        }
        printf("\n");
    }
}
// implemented by testcuda.cu
void testmkl( Tensor<cpu,2> mat1, Tensor<cpu,2> mat2, Tensor<cpu,2> mat3 );

int main( void ){
    InitTensorEngine();

    TensorContainer<cpu,2> lhs( Shape2(4,3), 0 );
    TensorContainer<cpu,2> rhs( Shape2(4,3), 0 );
    TensorContainer<cpu,2> dst( Shape2(4,4), 0.0 );
    TensorContainer<cpu,2> dst2( Shape2(3,3), 0.0 );
    lhs = 1.0f;
    print(lhs);
    printf("-\n");
    rhs[0] = 2.0f;
    rhs[1] = 0.0f;
    rhs[2] = 3.0f;
    rhs[0][1] = 1.0f;
    print(rhs);
    // A += 0.1*dot(B.T(),C)
    //dst += 0.1 * dot(lhs.T(), rhs);
    dst -= 0.1 *dot( lhs, rhs.T() );
    print(dst);
    
    dst[0] = dot( lhs[0], rhs.T() );
    print(dst);
    dst2 += dot( lhs[0].T(), rhs[0] );
    print(dst2);

    ShutdownTensorEngine();
    return 0;
}
