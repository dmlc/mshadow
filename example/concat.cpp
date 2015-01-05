#include "mshadow/tensor.h"

using namespace mshadow;
using namespace mshadow::expr;


void Print2D(const Tensor<cpu, 2, float>&t) {
    for (int i = 0; i < t.size(0); ++i) {
        for (int j = 0; j < t.size(1); ++j) {
            printf("%.2f ", t[i][j]);
        }
        printf("\n");
    }
}

int main() {
    Tensor<cpu, 4, float> t1 = NewTensor<cpu, float>(Shape4(2, 2, 3,2), 0.1f);
    Tensor<cpu, 4, float> t2 = NewTensor<cpu, float>(Shape4(2, 3, 3,2), 0.2f);
    Tensor<cpu, 4, float> t3 = NewTensor<cpu, float>(Shape4(2, 1, 3,2), 0.3f);
    Tensor<cpu, 4, float> t = NewTensor<cpu, float>(Shape4(2,6,3,2), 0.0f);
    t = concat(t1, concat(t2, t3));
    Print2D(t[0][2]);
    t += 1.0f;
    concat(t1, concat(t2, t3)) = t;
    Print2D(t3[1][0]);
    FreeSpace(&t1);
    FreeSpace(&t2);
    FreeSpace(&t3);
    FreeSpace(&t);
}
