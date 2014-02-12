#ifndef CXXNET_UTILS_H
#define CXXNET_UTILS_H
#include <cstdio>
#include <cstdlib>
/*!
 * \file utils.h
 * \brief utilities that could be useful
 * \author Tianqi Chen
 */
namespace cxxnet{
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
    };    
};
#endif

