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
        // TODO: Use buffer
        class RandomEngine {
        private:
            /*! \brief gaussian rand (0, 1) by using stdlib */
            inline real_t _gaussian_rand() {
                real_t a = rand() / RAND_MAX;
                real_t b = rand() / RAND_MAX;
                return std::sqrt(-2 * std::log(a)) * std::cos(2 * PI * b);
            }
        public:
        RandomEngine(index_t seed) {
            #ifdef MSHADOW_USE_MKL
            int status = vslNewStream(&vStream, VSL_BRNG_SFMT19937, seed);
            utils::Assert(status == VSL_STATUS_OK, "MKL VSL Random engine failed to be initialized.\n" );
            #elif MSHADOW_USE_CUDA
            // TODO
            // curandStatus_t status = curandCreateGenerator(&cuGen, CURAND_RNG_QUASI_SOBOL32);
            // utils::Assert(status == cudaSuccess);
            // status = curandSetPseudoRandomGeneratorSeed(cuGen, seed);
            // utils::Assert(status == cudaSuccess);
            #else // use stdlib
            srand(seed);
            #endif
        }
        RandomEngine (const RandomEngine& eng) {}
        ~RandomEngine() {
            #ifdef MSHADOW_USE_MKL
            vslDeleteStream(&vStream);
            #endif
            #ifdef MSHADOW_USE_CUDA
            // TODO
            #endif
        }
        /*! \brief Generate uniform random number for cpu tensor|conteainer*/
        template<int dim>
        void uniform(Tensor<cpu, dim> &tensor, real_t a, real_t b) {
            utils::Assert(a < b, "Incorrect range\n");
            #ifdef MSHADOW_USE_MKL
            #ifdef MSHADOW_SINGLE_PRECISION
            int status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream, tensor.shape.MSize(), tensor.dptr, a, b);
            #else
            int status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream, tensor.shape.MSize(), tensor.dptr, a, b);
            #endif
            utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
            #else // use stdlib
            #pragma unroll
            for (index_t i = 0; i < tensor.shape.MSize(); ++i) {
                tensor.dptr[i] = (rand() / RAND_MAX) * (b - a) + a;
            }
            #endif // MSHADOW_USE_MKL
        }
        /*! \brief Generate uniform random number for gpu tensor|conteainer*/
        template<int dim>
        void uniform(Tensor<gpu, dim> &tensor, real_t a, real_t b) {
            // TODO: link to kernel
        }
        /*! \brief Generate gaussian random number for cpu tensor|container */
        template<int dim>
        void gaussian(Tensor<cpu, dim> &tensor, real_t a, real_t sigma) {
            #ifdef MSHADOW_USE_MKL
            #ifdef MSHADOW_SINGLE_PRECISION
            int status = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vStream, tensor.shape.MSize(), tensor.dptr, a, sigma);
            #else
            int status = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, vStream, tensor.shape.MSize(), tensor.dptr, a, sigma);
            #endif
            utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
            #else // use stdlib
            #pragma unroll
            for (index_t i = 0; i < tensor.shape.MSize(); ++i) {
                tensor.dptr[i] = a + sigma * _gaussian_rand();
            }
            #endif // MSHADOW_USE_MKL
        }
        /*! \brief Generate gaussian random number for gpu tensor|container */
        template<int dim>
        void gaussian(Tensor<gpu, dim> &tensor, real_t a, real_t sigma) {
            // TODO: link to kernel
        }
        public:
        /*! \brief Generate CTensor random binary mask with prob p to be 1 */
        template<int dim>
        void binary(Tensor<cpu, dim> &tensor, real_t p) {
            uniform(tensor, 0, 1);
            #pragma unroll
            for (int i = 0; i < tensor.shape.MSize(); ++i) {
                tensor.dptr[i] > p ? tensor.dptr[i] = 1 : tensor.dptr[i] = 0;
            }
        }
        /*! \brief Generate GTensor random binary mask with prob p to be 1 */
        template<int dim>
        void binary(Tensor<gpu, dim> &tensor, real_t p) {
            // TODO: link to kernel
        }
        private:
        #ifdef MSHADOW_USE_MKL
        VSLStreamStatePtr vStream;
        #endif
        #ifdef MSHADOW_USE_CUDA
        // curandStatus_t
        // curandGenerator_t cuGen;
        #endif
        // left for later use for buffer
        real_t *buffer;
        index_t loc;
        }; // class RandomEngine


    }; // namespace prng
 }; // namespace mshadow


#endif // MSHADOW_TENSOR_RANDOM_H
