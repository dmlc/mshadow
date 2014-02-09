#ifndef TENSOR1D_H
#define TENSOR1D_H
#pragma once
#include "tensor.h"

namespace cxxnet {
class Tensor1D : public Tensor {
//protected:
public:
    inline size_t index_to_assignment(size_t index) const {
        return index;
    }
// public:
    Tensor1D (size_t size) {
        shape[0] = size;
        host_ptr = new real[size];
    }

    ~Tensor1D () {
        if (host_ptr) {
            delete [] host_ptr;
        }
    }
};

template<typename Saver, typename BinaryMapper>
inline void map_binary(Tensor1D &dst, const Tensor1D &a, const Tensor1D &b) {
    assert(dst.shape == a.shape && dst.shape == b.shape);

    for (auto i = 0; i < dst.shape[0]; ++i) {
        Saver::save(dst.host_ptr[dst.index_to_assignment(i)], \
                    BinaryMapper::map(a.host_ptr[a.index_to_assignment(i)], \
                                        b.host_ptr[b.index_to_assignment(i)]));
    }
}


};


#endif // TENSOR1D_H
