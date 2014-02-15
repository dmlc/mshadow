#ifndef MSHADOW_TENSOR_BASE_H
#define MSHADOW_TENSOR_BASE_H
/*!
 * \file tensor_base.h
 * \brief definitions of base types, macros functions
 *
 * \author Bing Hsu, Tianqi Chen
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
// defintiions 
/*! \brief use CBLAS for CBLAS */
#define MSHADOW_USE_CBLAS 0
/*! \brief use MKL for BLAS */
#define MSHADOW_USE_MKL   1

#if MSHADOW_USE_CBLAS
  #include <cblas.h>
#elif MSHADOW_USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#endif

#ifdef __CUDACC__
  #include <cublas.h>
#endif
// --------------------------------
// MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code.
#ifdef MSHADOW_XINLINE
  #error "MSHADOW_XINLINE must not be defined"
#endif
#ifdef __CUDACC__
  #define MSHADOW_XINLINE inline __attribute__((always_inline)) __device__ __host__
#else
  #define MSHADOW_XINLINE inline __attribute__((always_inline))
#endif

/*! \brief namespace for mshadow */
namespace mshadow {
    /*! \brief type that will be used for content */
    typedef float real_t;
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
}; // namespace mshadow

namespace mshadow {
    /*! \brief namespace for savers */
    namespace sv {
        /*! \brief save to saver: = */
        struct saveto {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a  = b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            const static real_t kAlphaBLAS = 1.0f;
            /*! \brief helper constant to use BLAS, beta */
            const static real_t kBetaBLAS  = 0.0f;
        };
        /*! \brief save to saver: += */
        struct plusto {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a += b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            const static real_t kAlphaBLAS = 1.0f;
            /*! \brief helper constant to use BLAS, beta */
            const static real_t kBetaBLAS  = 1.0f;
        };
        /*! \brief minus to saver: -= */
        struct minusto {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a -= b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            const static real_t kAlphaBLAS = -1.0f;
            /*! \brief helper constant to use BLAS, beta */
            const static real_t kBetaBLAS  = 1.0f;
        };
        /*! \brief multiply to saver: *= */
        struct multo {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a *= b;
            }
        };
        /*! \brief divide to saver: /= */
        struct divto {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a /= b;
            }
        };
    }; // namespace sv

    /*! \brief namespace for operators */
    namespace op {
        // binary operator
        /*! \brief mul operator */
        struct mul{
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief plus operator */
        struct plus {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a + b;
            }
        };
        /*! \brief minus operator */
        struct minus {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief divide operator */
        struct div {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a / b;
            }
        };
    }; // namespace op

    namespace op {
        // unary operator/ function: example
        // these operators can be defined by user, in the same style as binary and unary operator
        // to use, simply write F<op::identity>( src )
        /*! \brief identity function that maps a real number to it self */
        struct identity{
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a;
            }
        };
    }; // namespace op

    /*! \brief namespace for helper utils of the project */
    namespace utils{
        inline void Error( const char *msg ){
            fprintf( stderr, "Error:%s\n",msg );
            exit( -1 );
        }

        inline void Assert( bool exp ){
            if( !exp ) Error( "AssertError" );
        }

        inline void Assert( bool exp, const char *msg ){
            if( !exp ) Error( msg );
        }

        inline void Warning( const char *msg ){
            fprintf( stderr, "warning:%s\n",msg );
        }
    }; // namespace utils
}; // namespace mshadow
#endif // TENSOR_BASE_H
