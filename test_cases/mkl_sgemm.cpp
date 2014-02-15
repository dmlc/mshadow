#include "../mshadow/tensor_base.h"
#include "../mshadow/tensor.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;
using namespace mshadow::expr;

void print(const CTensor2D &t) {
    index_t row = t.shape[0];
    index_t col = t.shape[1];
    printf("%2d X %2d\n", row, col);
    for (index_t r = 0; r < row; ++r) {
        for (index_t c = 0; c < col; ++c) {
            printf("%.2f ", t[c][r]);
        }
        printf("\n");
    }
}
// implemented by testcuda.cu
void testmkl( CTensor2D mat1, CTensor2D mat2, CTensor2D mat3 );

int main( void ){
    CTensor2D lhs = NewCTensor( Shape2(5, 3), 0 );
    CTensor2D rhs = NewCTensor( Shape2(4, 3), 0 );
    CTensor2D dst = NewCTensor( Shape2(4, 5), 0 );
    for (index_t i = 0; i < 15; ++i) {
        lhs.dptr[i] = i;
    }

    print(lhs);
    printf("-\n");
    for (index_t i = 0; i < 12; ++i) {
        rhs.dptr[i] = 0.1 * i;
    }

    print(rhs);
    // A += 0.1*dot(B.T(),C)
    MapExp<sv::plusto>(dst, \
                       F<op::mul>(DotExp<CTensor2D, CTensor2D, 1, 0>(lhs, rhs, 1), ScalarExp(0.1)));
    print(dst);
    FreeSpace( lhs );
    FreeSpace( rhs );
    FreeSpace( dst );
    return 0;
}
