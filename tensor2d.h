#ifndef TENSOR2D_H
#define TENSOR2D_H
#pragma once
#include "tensor.h"

namespace cxxnet {
class Tensor2D : public Tensor {
//protected:
public:
    inline size_t index_to_assignment(size_t r, size_t c) const {
        // row_major storage
        return r * shape[1] + c;
    }
// public:
    Tensor2D (size_t r, size_t c) {
        shape[0] = r;
        shape[1] = c;
        host_ptr = new real[r * c];
    }

    ~Tensor2D () {
        if (host_ptr) {
            delete [] host_ptr;
        }
    }

    void print() {
        for (auto i = 0; i < shape[0]; ++i) {
            for (auto j = 0; j < shape[1]; ++j) {
                std::cout<<host_ptr[index_to_assignment(i, j)]<<",";
            }
            std::cout<<std::endl;
        }
    }

    void init() {
        for (auto i = 0; i < shape[0] * shape[1]; ++i) {
            host_ptr[i] = i;
        }
    }
};

template<typename Saver, typename BinaryMapper>
inline void map_binary(Tensor2D &dst, const Tensor2D &a, const Tensor2D &b) {
    assert(dst.shape == a.shape && dst.shape == b.shape);

    for (auto i = 0; i < dst.shape[0]; ++i) {
        for (auto j = 0; j < dst.shape[1]; ++j) {
            Saver::save(dst.host_ptr[dst.index_to_assignment(i, j)], \
                        BinaryMapper::map(a.host_ptr[a.index_to_assignment(i, j)], \
                                        b.host_ptr[b.index_to_assignment(i, j)]));
        }
    }
}

inline void conv_op(Tensor2D &dst, const Tensor2D &src, const Tensor2D &kernel) {
    // Not test yet
    assert(dst.shape == src.shape);
    assert(kernel.shape[0] == kernel.shape[1]);

    auto width = kernel.shape[0];
    auto radis = kernel.shape[0] / 2;
    for (auto i = 0; i < dst.shape[0]; ++i) {
        for (auto j = 0; j < dst.shape[1]; ++j) {
            real accu = 0;
            long in_x = i - radis;
            long in_y = j - radis;
            for (auto m = 0; m < width; ++m) {
                for (auto n = 0; n < width; ++n) {
                    if (in_x >=0 && in_x < dst.shape[0] && \
                        in_y >= 0 && in_y <dst.shape[1]) {
                        accu += \
                            kernel.host_ptr[kernel.index_to_assignment(m, n)] \
                            * \
                            src.host_ptr[src.index_to_assignment(in_x, in_y)];
                    }
                }
            }
            dst.host_ptr[dst.index_to_assignment(i, j)] = accu;
        }
    }
}


};


#endif // TENSOR2D_H
