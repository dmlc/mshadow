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
/*! \brief use CUDA support, must ensure that the cuda include path is correct, or directly compile using nvcc */
#ifndef MSHADOW_USE_CUDA
  #define MSHADOW_USE_CUDA   1
#endif
/*! \breif use single precition float */
#ifndef MSHADOW_SINGLE_PRECISION
  #define MSHADOW_SINGLE_PRECISION 0
#endif
/*! \breif whether use SSE */
#ifndef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 1
#endif

// SSE is conflict with cudacc
#ifdef __CUDACC__
  #undef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 0
#endif

#if MSHADOW_USE_CBLAS
  #include <cblas.h>
#elif MSHADOW_USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#endif

#if MSHADOW_USE_CUDA
  #include <cublas.h>
  #include <curand.h>
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
// cpu force inline
#define MSHADOW_CINLINE inline __attribute__((always_inline))

/*! \brief namespace for mshadow */
namespace mshadow {
#if MSHADOW_SINGLE_PRECISION
    /*! \brief type that will be used for content */
    typedef float real_t;
#else
    typedef double real_t;
#endif
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
}; // namespace mshadow

namespace mshadow {
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
        /*! \brief get rhs */
        struct right {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return b;
            }
        };
    }; // namespace op

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
            /*! \brief corresponding binary operator type */
            typedef op::right OPType;
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
            /*! \brief corresponding binary operator type */
            typedef op::plus OPType;
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
            /*! \brief corresponding binary operator type */
            typedef op::minus OPType;
        };
        /*! \brief multiply to saver: *= */
        struct multo {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a *= b;
            }
            /*! \brief corresponding binary operator type */
            typedef op::mul OPType;
        };
        /*! \brief divide to saver: /= */
        struct divto {
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a /= b;
            }
            /*! \brief corresponding binary operator type */
            typedef op::div OPType;
        };
    }; // namespace sv


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
