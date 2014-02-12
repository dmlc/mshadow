#ifndef CXXNET_TENSOR_OP_H
#define CXXNET_TENSOR_OP_H
/*!
 * \file tensor_op.h
 * \brief definitions of tensor operators
 *
 * \author Bing Hsu, Tianqi Chen
 */
#include "tensor.h"

namespace cxxnet {
    /*! \brief namespace for operators */
    namespace op {
        /*! \brief mul operator */
        struct mul{};
        /*! \brief plus operator */
        struct plus {};
        /*! \brief minus operator */
        struct minus {};
        /*! \brief divide operator */
        struct div {};        
    }; // namespace op

    /*! \brief namespace for savers */
    namespace sv { 
        /*! \brief save to saver: = */
        struct saveto {};
        /*! \brief save to saver: += */
        struct addto {};
        /*! \brief minus to saver: -= */
        struct minusto {};
        /*! \brief multiply to saver: *= */
        struct multo {};
        /*! \brief divide to saver: /= */
        struct divto {};
    }; // namespace sv

    /*! \brief namespace for operators */
    namespace op {
        // implementation of operators
        /*! 
         * \brief template base of binary mapper class,
         *        use template specification to implement customized mapper
         * \param OPType type of operations, see begining of this file
         */
        template<typename OPType>
        struct BinaryMapper{
            inline static real_t Map(real_t a, real_t b);
        };
        /*! \brief implementation of mul */
        template<>
        struct BinaryMapper<mul>{
            inline static real_t Map(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief implementation of plus */
        template<>
        struct BinaryMapper<plus>{
            inline static real_t Map(real_t a, real_t b) {
                return a + b;
            }
        };
        /*! \brief implementation of minus */
        template<>
        struct BinaryMapper<minus>{
            inline static real_t Map(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief implementation of minus */
        template<>
        struct BinaryMapper<div>{
            inline static real_t Map(real_t a, real_t b) {
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
        struct Saver{
            inline static void Save(real_t& a, real_t b);
        };
        /*! \brief implementation of save to saver */
        template<>
        struct Saver<saveto> {
            inline static void Save(real_t& a, real_t b) {
                a = b;
            }
        };
        /*! \brief implementation of add to saver */
        template<>
        struct Saver<addto> {
            inline static void Save(real_t& a, real_t b) {
                a += b;
            }
        };
        /*! \brief implementation of minus to saver */
        template<>
        struct Saver<minusto> {
            inline static void Save(real_t& a, real_t b) {
                a -= b;
            }
        };
        /*! \brief implementation of mul to saver */
        template<>
        struct Saver<multo> {
            inline static void Save(real_t& a, real_t b) {
                a *= b;
            }
        };
        /*! \brief implementation of div to saver */
        template<>
        struct Saver<divto> {
            inline static void Save(real_t& a, real_t b) {
                a /= b;
            }
        };
    }; // namespace sv
}; // namespace cxxnet
#endif // TENSOR_OP_H
