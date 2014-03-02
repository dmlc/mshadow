#ifndef MSHADOW_TENSOR_RANDOM_H
#define MSHADOW_TENSOR_RANDOM_H

/*!
 *  \file tensor_random.h
 *  \brief Random inline functions for tensor.
 *  \author Bing Hsu
 *   Based on curand|MKL|stdlib
 */
#include <cstdlib>
#include "tensor_base.h"

namespace mshadow {
    /*! \brief random number generator */
    template<typename device>
    class Random {};
    
    template<>
    class Random<cpu> {        
    public:
        Random<cpu> (int seed) {
            #ifdef MSHADOW_USE_MKL
            status_ = vslNewStream(&vStream_, VSL_BRNG_SFMT19937, seed);
            utils::Assert(status_ == VSL_STATUS_OK, "MKL VSL Random engine failed to be initialized.\n" );
            #else
            srand(seed);
            #endif
            this->RegenerateUniform();
        }
        
        template<int dim>
        inline void Uniform(Tensor<cpu, dim> &dst_, real_t a, real_t b) {
            Tensor<cpu, 2> mat = dst_.FlatTo2D();
            for (index_t i = 0; i < mat.shape[1]; ++i) {
                for (index_t j = 0; j < mat.shape[0]; ++j) {
                    mat[i][j] = this->GetNextUniform() * (b - a) + a;
                }
            }
        }
        template<int dim>
        inline void Gaussian(Tensor<cpu, dim> &dst_, real_t mu, real_t sigma) {
            real_t g1 = 0.0f, g2 = 0.0f;
            Tensor<cpu, 2> mat = dst_.FlatTo2D();
            for (index_t i = 0; i < mat.shape[1]; ++i) {
                for (index_t j = 0; j < mat.shape[0]; ++j) {
                    if( (j&1) == 0 ){
                        // use std::sqrt so that it can check if it is double or float
                        // box muller transformation generate a pair of gaussian rng
                        const real_t a = this->GetNextUniform();
                        const real_t b = this->GetNextUniform();
                        const real_t u1 = std::sqrt(-2.0f * std::log(a));
                        g1 = std::cos(2.0f * kPi * b);
                        g2 = std::sin(2.0f * kPi * b);

                        mat[i][j] = mu + sigma * g1;
                    }else{
                        mat[i][j] = mu + sigma * g2;
                    }
                }
            }
        }
        
        ~Random<cpu>() {
            #ifdef MSHADOW_USE_MKL
            vslDeleteStream(&_vStream);
            #endif
        }
    private:
        #ifdef MSHADOW_USE_MKL
        int status_;
        VSLStreamStatePtr vStream_;
        #endif
        real_t buffer_[ kRandBufferSize ];
        index_t buf_loc_;

        inline real_t GetNextUniform( void ) {
            if (buf_loc_ < kRandBufferSize ) {
                return buffer_[buf_loc_++];
            } else {
                this->RegenerateUniform();
                return buffer_[buf_loc_++];
            }
        }
        
        inline void RegenerteUnifrom( void ){
            buf_loc_ = 0;
            #ifdef MSHADOW_USE_MKL
            #ifdef MSHADOW_SINGLE_PRECISION
            status_ = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream_, kRandBufferSize, buffer_, 0,1);
            #else
            status_ = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vStream_, kRandBufferSize, buffer_, 0,1);
            #endif
            utils::Assert(status_ == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
            #else
            for (index_t i = 0; i < kRandBufferSize; ++ i) { 
                buffer_[i] = static_cast<real_t>(rand()) / (static_cast<real_t>(RAND_MAX)+1.0f);
            }
            #endif
        }        
    }; // class Random<cpu>
    
    template<>
    class Random<gpu> {
        
    }; // class Random<gpu>
    
    
}; // namespace mshadow

#endif // MSHADOW_TENSOR_RANDOM_H
