#ifndef TENSOR3D_H
#define TENSOR3D_H
#pragma once
#include "tensor.h"
#include "tensor2d.h"
#include <vector>
namespace cxxnet {
class Tensor3D {
//protected:
public:
    std::vector<Tensor2D> data;
    Shape shape;
    inline size_t index_to_assignment(size_t x, size_t y, size_t z) const {
        return data[z].index_to_assignment(x, y);
    }
// public:
    Tensor3D (size_t x, size_t y, size_t z) {
        shape[0] = x;
        shape[1] = y;
        shape[2] = z;

        for (auto i = 0; i < z; ++i) {
            data.push_back(Tensor2D(x, y));
        }
    }

    ~Tensor3D () {

    }
};

template<typename Saver, typename BinaryMapper>
inline void map_binary(Tensor3D &dst, const Tensor3D &a, const Tensor3D &b) {
    assert(dst.shape == a.shape && dst.shape == b.shape);
    // Not test yet
    for (auto c = 0; c < dst.shape[2]; ++c ) {
        map_binary<Saver, BinaryMapper>(dst.data[c], a.data[c], b.data[c]);
    }
}

void conv3d2d(Tensor3D &dst, Tensor3D &src, Tensor3D &kernel) {
    assert(dst.shape == src.shape);

    // OpenMP, Multi-GPU here
    for (auto c = 0; c < dst.shape[2]; ++c) {
        conv_op(dst.data[c], src.data[c], kernel.data[c]);
    }
}




};


#endif // TENSOR3D_H
