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
    namespace op{
        struct mul {
            inline static real_t Map( real_t a, real_t b ){
                return a * b;
            }
        };
    };
};
#endif
