#ifndef TENSOR_CPU_INL_HPP
#define TENSOR_CPU_INL_HPP

namespace cxxnet {
    // implementation of map
    template<typename Saver, typename BinaryMapper>
    inline void Map(CTensor2D dst, const CTensor2D &lhs, const CTensor2D &rhs) {
        for (index_t y = 0; y < dst.shape[1]; y ++) {
            for (index_t x = 0; x < dst.shape[0]; x ++) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], BinaryMapper::Map(lhs[y][x], rhs[y][x]));
            }
        }
    }
}; // namespace cxxnet
#endif // TENSOR_CPU_INL_HPP
