#ifndef CXXNET_TENSOR_OP_H
#define CXXNET_TENSOR_OP_H
/*!
 * \file tensor_op.h
 * \brief definitions of tensor operators
 *
 * \author Bing Hsu, Tianqi Chen
 */
#include <cmath>

#ifdef _XINLINE_
  #error "_XINLINE_ must not be defined"
#endif
#ifdef __CUDACC__
  #define _XINLINE_ inline __device__ __host__
#else
  #define _XINLINE_ inline
#endif

/*! \brief namespace for cxxnet */
namespace cxxnet {
    /*! \brief type that will be used for content */
    typedef float real_t;
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
}; // namespace cxxnet

namespace cxxnet {
    /*! \brief namespace for savers */
    namespace sv { 
        /*! \brief save to saver: = */
        struct saveto {
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a  = b;
            }            
        };
        /*! \brief save to saver: += */
        struct addto {
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a += b;
            }        
        };
        /*! \brief minus to saver: -= */
        struct minusto {
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a -= b;
            }        
        };
        /*! \brief multiply to saver: *= */
        struct multo {
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a *= b;
            }        
        };
        /*! \brief divide to saver: /= */
        struct divto {
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a /= b;
            }  
        };
    }; // namespace sv

    /*! \brief namespace for operators */
    namespace op {
        // binary operator
        /*! \brief mul operator */
        struct mul{
            _XINLINE_ static real_t Map(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief plus operator */
        struct plus {
            _XINLINE_ static real_t Map(real_t a, real_t b) {
                return a + b;
            }        
        };
        /*! \brief minus operator */
        struct minus {
            _XINLINE_ static real_t Map(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief divide operator */
        struct div {
            _XINLINE_ static real_t Map(real_t a, real_t b) {
                return a / b;
            }        
        };
    }; // namespace op

    namespace op {
        // unary operator/ function
        /*! \brief function */
        struct identity{
            _XINLINE_ static real_t Map(real_t a) {
                return a;
            }
        };
        /*! \brief sigmoid operator */
        struct sigmoid {
            _XINLINE_ static real_t Map(real_t a) {
                return 1.0f /(1.0f + expf(-a));
            }        
        };
    }; // namespace op
}; // namespace cxxnet
#endif // TENSOR_OP_H
