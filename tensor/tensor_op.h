#ifndef _CXXNET_TENSOR_OP_H_
#define _CXXNET_TENSOR_OP_H_
/*!
 * \file tensor_op.h
 * \brief definitions of tensor operators
 * 
 * \author Bing Hsu, Tianqi Chen
 */
#include "tensor.h"
namespace cxxnet{
    /*! \brief namespace for operators */
    namespace op{
        struct mul {
            inline static real_t Map( real_t a, real_t b ){
                return a * b;
            }
            // another function for device
            __device__ inline static real_t DMap( real_t a, real_t b ){
                return a * b;
            }
        };
    };
    /*! \brief namespace for savers */
    namespace sv{
        struct saveto {
            inline static void Save( real_t& a, real_t b ) {
                a = b;
            }
            __device__ inline static void DSave( real_t& a, real_t b ) {
                a = b;
            }
        };
    };
};
#endif
