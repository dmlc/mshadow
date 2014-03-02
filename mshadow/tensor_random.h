#ifndef MSHADOW_TENSOR_RANDOM_H
#define MSHADOW_TENSOR_RANDOM_H

/*!
 *  \file tensor_random.h
 *  \brief Random inline functions for tensor.
 *   Based on curand|MKL|stdlib
 */
#include "tensor_base.h"

namespace mshadow {


template<typename device>
class Random {};

template<>
class Random<cpu> {
private:
    #ifdef MSHADOW_USE_MKL
     int _status;
     VSLStreamStatePtr _vStream;
    #endif
    real_t _buffer[MSHADOW_BUFFER_MAX];
    index_t _buf_loc;
    MSHADOW_XINLINE real_t _get_next_uniform() {
        if (_buf_loc < MSHADOW_BUFFER_MAX) {
            return _buffer[_buf_loc++];
        } else {
            _regenerte_unifrom();
            return _buffer[_buf_loc++];
        }
    }

    MSHADOW_XINLINE void _regenerte_unifrom() {
        _buf_loc = 0;
        #ifdef MSHADOW_USE_MKL
         #ifdef MSHADOW_SINGLE_PRECISION
          _status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _vStream, MSHADOW_BUFFER_MAX, _buffer, 0,1);
         #else
          _status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _vStream, MSHADOW_BUFFER_MAX, _buffer, 0,1);
         #endif
         utils::Assert(_status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
        #else
          for (index_t i = 0; i < MSHADOW_BUFFER_MAX; ++i) { _buffer[i] = rand() / static_cast<float>(RAND_MAX);}
        #endif
    }
public:
    Random<cpu> (int seed) {
        #ifdef MSHADOW_USE_MKL
         _status = vslNewStream(&_vStream, VSL_BRNG_SFMT19937, seed);
         utils::Assert(_status == VSL_STATUS_OK, "MKL VSL Random engine failed to be initialized.\n" );
        #else
         srand(seed);
        #endif
        _regenerte_unifrom();
    }

    template<int dim>
    MSHADOW_XINLINE void Uniform(Tensor<cpu, dim> &t, real_t a, real_t b) {
        Tensor<cpu, 2> mat = t.FlatTo2D();
        for (index_t i = 0; i < mat.shape[1]; ++i) {
            for (index_t j = 0; j < mat.shape[0]; ++j) {
                mat[i][j] = _get_next_uniform() * (b - a) + a;
            }
        }
    }
    template<int dim>
    MSHADOW_XINLINE void Gaussian(Tensor<cpu, dim> &t, real_t mu, real_t sigma) {
        Tensor<cpu, 2> mat = t.FlatTo2D();
        for (index_t i = 0; i < mat.shape[1]; ++i) {
            for (index_t j = 0; j < mat.shape[0]; ++j) {
                real_t a = _get_next_uniform();
                real_t b = _get_next_uniform();
                mat[i][j] = mu + sigma * sqrt(-2 * log(a)) * cos(2 * MSHADOW_PI * b);
            }
        }
    }

    ~Random<cpu>() {
        #ifdef MSHADOW_USE_MKL
         vslDeleteStream(&_vStream);
        #endif
    }

}; // class Random<cpu>

template<>
class Random<gpu> {

}; // class Random<gpu>


}; // namespace mshadow


#endif // MSHADOW_TENSOR_RANDOM_H
