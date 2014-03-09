#include "mshadow/tensor.h"
// must include this file to get gpu part implementation
#include "../mshadow/tensor_container.h"

using namespace mshadow;
using namespace mshadow::expr;

inline void print(const Tensor<cpu,2> &t) {
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


void testcuda( void ){
    TensorContainer<gpu,2> gmat1(Shape2(10,16),1.0f);
    TensorContainer<gpu,2> gmat2(Shape2(10,16),2.0f);
    TensorContainer<cpu,2> cmat1(Shape2(10,16),2.0f);

    gmat1[0] = sum_rows( gmat2 );
    Copy( cmat1, gmat1 );    
    print( cmat1 );    
}
