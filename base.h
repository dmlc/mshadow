#ifndef BASE_H
#define BASE_H

#include <cstdlib>
#include <iostream>
#include <assert.h>

#pragma once
namespace cxxnet {
typedef float real;
const int MAX_SHAPE = 4;
class Shape {
public:
    size_t value[MAX_SHAPE];

    Shape() {
        for (auto i = 0; i < MAX_SHAPE; ++i) {
            value[i] = 0;
        }
    }

    bool operator==(const Shape& b) {
        for (auto i = 0; i < MAX_SHAPE; ++i) {
            if (this->value[i] != b.value[i]) {
                return false;
            }
        }
        return true;
    }

    size_t& operator[](const int idx) const {
        return const_cast<size_t&>(this->value[idx]);
    }

};

};
#endif // BASE_H
