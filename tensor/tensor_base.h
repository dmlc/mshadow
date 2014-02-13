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

#ifdef _XINLINE_
  #error "_XINLINE_ must not be defined"
#endif
#ifdef __CUDACC__
  #define _XINLINE_ inline __device__ __host__
#else
  #define _XINLINE_ inline
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
            _XINLINE_ static void Save(real_t& a, real_t b) {
                a  = b;
            }            
        };
        /*! \brief save to saver: += */
        struct plusto {
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
