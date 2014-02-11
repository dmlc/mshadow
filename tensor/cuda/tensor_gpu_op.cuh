#ifndef CXXNET_TENSOR_GPU_OP_CUH
#define CXXNET_TENSOR_GPU_OP_CUH
/*!
 * \file tensor_gpu_op.cuh
 * \brief definitions of tensor operators in GPU
 *
 * \author Bing Hsu, Tianqi Chen
 */
#include "../tensor.h"
#include "../tensor_op.h"

#ifndef _DINLINE_
#define _DINLINE_ inline __device__
#else
#error "_DINLINE_ must not be defined"
#endif

namespace cxxnet {
    /*! \brief namespace for operators */
    namespace op{
        // implementation of operators
        /*! 
         * \brief template base of binary mapper class,
         *        use template specification to implement customized mapper
         * \param OPType type of operations, see begining of this file
         */
        template<typename OPType>
        struct GBinaryMapper{
            _DINLINE_ static real_t Map(real_t a, real_t b);
        };
        /*! \brief implementation of mul */
        template<>
        struct GBinaryMapper<mul>{
            _DINLINE_ static real_t Map(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief implementation of plus */
        template<>
        struct GBinaryMapper<plus>{
            _DINLINE_ static real_t Map(real_t a, real_t b) {
                return a + b;
            }
        };
        /*! \brief implementation of minus */
        template<>
        struct GBinaryMapper<minus>{
            _DINLINE_ static real_t Map(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief implementation of minus */
        template<>
        struct GBinaryMapper<div>{
            _DINLINE_ static real_t Map(real_t a, real_t b) {
                return a / b;
            }
        };
    }; // namespace op


    /*! \brief namespace for savers */
    namespace sv {
        // implementation of savers
        /*! 
         * \brief template base of saver class,
         *        use template specification to implement customized saver
         * \param SVType type of operations, see begining of this file
         */
        template<typename SVType>
        struct GSaver{
            _DINLINE_ static void Save(real_t& a, real_t b);
        };
        /*! \brief implementation of save to saver */
        template<>
        struct GSaver<saveto> {
            _DINLINE_ static void Save(real_t& a, real_t b) {
                a = b;
            }
        };
        /*! \brief implementation of add to saver */
        template<>
        struct GSaver<addto> {
            _DINLINE_ static void Save(real_t& a, real_t b) {
                a += b;
            }
        };
        /*! \brief implementation of minus to saver */
        template<>
        struct GSaver<minusto> {
            _DINLINE_ static void Save(real_t& a, real_t b) {
                a -= b;
            }
        };
        /*! \brief implementation of mul to saver */
        template<>
        struct GSaver<multo> {
            _DINLINE_ static void Save(real_t& a, real_t b) {
                a *= b;
            }
        };
        /*! \brief implementation of div to saver */
        template<>
        struct GSaver<divto> {
            _DINLINE_ static void Save(real_t& a, real_t b) {
                a /= b;
            }
        };
    }; // namespace sv    
}; // namespace cxxnet

#endif

