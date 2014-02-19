#ifndef MSHADOW_TENSOR_RANDOM_H
#define MSHADOW_TENSOR_RANDOM_H
#pragma once

/*!
 *  \file tensor_random.h
 *  \brief Random inline functions for tensor.
 *   Based on curand|MKL|stdlib
 */
#include "tensor_base.h"

 namespace mshadow {
    /*! \brief namespace for random functions */
    namespace prng {
        /*! \brief Random engine for tensor container*/
        class RandomEngine {
        public:
        RandomEngine(index_t seed) {
            #ifdef MSHADOW_USE_MKL
            int status = vslNewStream(&vStream, VSL_BRNG_SFMT19937, seed);
            utils::Assert(status == VSL_STATUS_OK, "MKL VSL Random engine failed to be initialized.\n" );
            #endif
            #ifdef __CUDACC__

            #endif
        }
        RandomEngine (const RandomEngine& eng) {}
        ~RandomEngine() {
            #ifdef MSHADOW_USE_MKL
            vslDeleteStream(&vStream);
            #endif
        }
        /*! Generate uniform random number for cpu tensor|conteainer*/
        template<int dim>
        void uniform(Tensor<cpu, dim> &tensor, real_t a, real_t b) {
            #ifdef MSHADOW_USE_MKL
            #ifdef MSHADOW_SINGLE_PRECISION
            int status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream, tensor.shape.MSize(), tensor.dptr, a, b);
            #else
            int status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream, tensor.shape.MSize(), tensor.dptr, a, b);
            #endif
            utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
            #endif // MSHADOW_USE_MKL
        }
        template<int dim>
        void gaussian(Tensor<cpu, dim> &tensor, real_t a, real_t sigma) {
            #ifdef MSHADOW_USE_MKL
            #ifdef MSHADOW_SINGLE_PRECISION
            int status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vStream, tensor.shape.MSize(), tensor.dptr, a, sigma);
            #else
            int status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vStream, tensor.shape.MSize(), tensor.dptr, a, sigma);
            #endif
            utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
            #endif // MSHADOW_USE_MKL
        }
        private:
        #ifdef MSHADOW_USE_MKL
        VSLStreamStatePtr vStream;
        #endif
        #ifdef __CUDACC__

        #endif
        // left for later use for buffer
        real_t *buffer;
        index_t loc;
        }; // class RandomEngine


    }; // namespace prng
 }; // namespace mshadow


#endif // MSHADOW_TENSOR_RANDOM_H
