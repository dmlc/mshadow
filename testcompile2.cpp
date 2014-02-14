#include "mshadow/tensor.h"
#include <cstdio>
// simple file to test if it compiles
using namespace mshadow;


void print(const CTensor2D &t) {
    index_t row = t.shape[0];
    index_t col = t.shape[1];
    printf("%2d X %2d\n", row, col);
    for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
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
    bool ltrans = true;
    bool rtrans = false;
    float scale = 1.0f;
    CBLAS_TRANSPOSE op_lhs = ltrans ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE op_rhs = rtrans ? CblasTrans : CblasNoTrans;
    index_t M = ltrans ? lhs.shape[1] : lhs.shape[0];
    index_t K = ltrans ? lhs.shape[0] : lhs.shape[1];
    index_t N = rtrans ? rhs.shape[0] : rhs.shape[1];
    index_t LDA = ltrans ? lhs.shape[1] : lhs.shape[0];
    index_t LDB = rtrans ? rhs.shape[1] : rhs.shape[0];
    // utils::Assert(HDA == LDB, "Matrix Dimension Mismatch\n");
    printf("M:%d K:%d N:%d LDA:%d LDB:%d\n", M, K, N, LDA, LDB);
    cblas_sgemm(CblasColMajor, \
                            op_lhs, op_rhs, \
                            M, \
                            N, \
                            K, \
                            scale, \
                            lhs.dptr, LDA, \
                            rhs.dptr, LDB, \
                            0, \
                            dst.dptr, LDA);
    printf("\n");
    for (int i = 0; i < 20; ++ i) {
        printf("%.2f ", dst.dptr[i]);
    }
    print(dst);
    FreeSpace( lhs );
    FreeSpace( rhs );
    FreeSpace( dst );
    return 0;
}
