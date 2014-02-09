#ifndef TENSOR_H
#define TENSOR_H
#pragma once

#include "base.h"
namespace cxxnet {

class Tensor {
//protected:
public:
    real *device_ptr;
    real *host_ptr;
    int gpu_number;
    Shape shape;
    // virtual inline size_t index_to_assignment() = 0;
// public:
    Tensor() {
        device_ptr = NULL;
        host_ptr = NULL;
        // judge gpu number
        //
    }

    // virtual int load(char*) = 0;
    // virtual int save(char*) = 0;

};

struct opmul {
    inline static real map(real a, real b) {
        return a * b;
    }
};

struct opdiv {
    inline static real map(real a, real b) {
        return a / b;
    }
};

struct opadd {
    inline static real map(real a, real b) {
        return a + b;
    }
};

struct opmin {
    inline static real map(real a, real b) {
        return a - b;
    }
};

struct addto {
    inline static void save(real& a, real b) {
        a += b;
    }
};

struct minto {
    inline static void save(real& a, real b) {
        a -= b;
    }
};

struct multo {
    inline static void save(real& a, real b) {
        a *= b;
    }
};

struct divto {
    inline static void save(real& a, real b) {
        a /= b;
    }
};
};
#endif // TENSOR_H
