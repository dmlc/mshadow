#ifndef TENSOR_CPU_INL_HPP
#define TENSOR_CPU_INL_HPP

#include "tensor_op.h"

namespace cxxnet {
    // implementation of map
    template<typename SV, typename OP>
    inline void Map(CTensor2D dst, const CTensor2D &lhs, const CTensor2D &rhs) {
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                // trust your compiler! -_- they will optimize it
                sv::Saver<SV>::Save(dst[y][x], op::BinaryMapper<OP>::Map(lhs[y][x], rhs[y][x]));
            }
        }
    }
}; // namespace cxxnet
#endif // TENSOR_CPU_INL_HPP
