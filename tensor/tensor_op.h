#ifndef TENSOR_OP_H
#define TENSOR_OP_H
/*!
 * \file tensor_op.h
 * \brief definitions of tensor operators
 *
 * \author Bing Hsu, Tianqi Chen
 */
#include "tensor/tensor.h"
namespace cxxnet {
    /*! \brief namespace for operators */
    namespace op {
        /*! \brief mul operator */
        struct mul {
            inline static real_t Map(real_t a, real_t b) {
                return a * b;
            }
            // another function for device
            __device__ inline static real_t DMap(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief plus operator */
        struct plus {
            inline static real_t Map(real_t a, real_t b) {
                return a + b;
            }

            __device__ inline static real_t DMap(real_t a, real_t b) {
                return a + b;
            }
        };
        /*! \brief minus operator */
        struct minus {
            inline static real_t Map(real_t a, real_t b) {
                return a - b;
            }

            __device__ inline static real_t DMap(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief divide operator */
        struct div {
            inline static real_t Map(real_t a, real_t b) {
                return a / b;
            }

            __device__ inline static real_t DMap(real_t a, real_t b) {
                return a / b;
            }
        };
    }; // namespace op
    /*! \brief namespace for savers */
    namespace sv {
        /*! \brief save to saver */
        struct saveto {
            inline static void Save(real_t& a, real_t b) {
                a = b;
            }
            __device__ inline static void DSave(real_t& a, real_t b) {
                a = b;
            }
        };
        /*! \brief add to saver */
        struct addto {
            inline static void Save(real_t& a, real_t b) {
                a += b;
            }
            __device__ inline static void DSave(real_t& a, real_t b) {
                a += b;
            }
        };
        /*! \brief minus to saver */
        struct minusto {
            inline static void Save(real_t& a, real_t b) {
                a -= b;
            }
            __device__ inline static void DSave(real_t& a, real_t b) {
                a -= b;
            }
        };
        /*! \brief mul to saver */
        struct multo {
            inline static void Save(real_t& a, real_t b) {
                a *= b;
            }
            __device__ inline static void DSave(real_t& a, real_t b) {
                a *= b;
            }
        };
        /*! \brief div to daver */
        struct divto {
            inline static void Save(real_t& a, real_t b) {
                a /= b;
            }
            __device__ inline static void DSave(real_t& a, real_t b) {
                a /= b;
            }
        };

    }; // namespace sv
}; // namespace cxxnet
#endif // TENSOR_OP_H
