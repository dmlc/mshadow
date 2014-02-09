#ifndef _CXXNET_TENSOR_H_
#define _CXXNET_TENSOR_H_
/*!
 * \file tensor.h
 * \brief library definition of the tensor
 * 
 * \author Bing Hsu, Tianqi Chen
 */

#include <cstdio>
// add unroll loops for the shape
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

/*! \brief namespace for cxxnet */
namespace cxxnet{
    /*! \brief type that will be used for content */
    typedef float real_t;
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
};

namespace cxxnet{
    /*! 
     * \brief shape of a tensor 
     * \tparam dimension dimension of tensor
     */
    template<int dimension>
    struct Shape{
    public:
        /*! \brief maximum dimension of tensor */
        const static int zMaxShape = dimension;
        const static int zSubShape = dimension - 1;
    public:
        /*! \brief default constructor, do nothing */
        Shape( void ){}
        /*! 
         * \brief get corresponding index 
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        inline index_t& operator[]( size_t idx ){
            return shape_[ idx ];
        }
        /*! 
         * \brief get corresponding index 
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        inline const index_t& operator[]( size_t idx ) const{
            return shape_[ idx ];
        }
        /*! \return stride */
        inline const index_t& stride( void ) const{
            return stride_;
        }
        /*! \return whether two shape equals */
        inline bool operator==( const Shape<zMaxShape> &s ) const{
            #pragma unroll 
            for( int i = 0; i < zMaxShape; i ++ ){
                if( s.shape_[i] != this->shape_[i] ) return false;
            }
            return true;
        }
        /*! 
         * flatten the higher dimension to second dimension, return a 2D shape
         * \return the flat 2d shape 
         */
        inline Shape<2> FlatTo2D( void ) const{
            Shape<2> s; s.stride_ = this->stride_;
            s.shape_[ 0 ] = this->shape_[ 0 ];
            index_t ymax = 1;

            #pragma unroll 
            for( int i = 1; i < zMaxShape; ++ i ){
                ymax *= this->shape_[ i ];
            }
            s.shape_[1] = ymax;
            return s;
        }
        /*! \return number of valid elements */
        inline size_t Size( void ){
            size_t memsz = this->shape[ 0 ];
            #pragma unroll 
            for( int i = 1; i < zMaxShape; ++ i ){
                memsz *= this->shape_[ i ];
            }
            return memsz;            
        }
        /*! \return memory size, including the aligned x dimension */
        inline size_t MSize( void ) const{
            size_t memsz = this->stride_;
            #pragma unroll 
            for( int i = 1; i < zMaxShape; ++ i ){
                memsz *= this->shape_[ i ];
            }
            return memsz;
        }
        /*! 
         * \brief get subshape 
         * \return subshape 
         */
        inline Shape<zSubShape> SubShape( void ) const{
            Shape<zSubShape> s; s.stride_ = this->stride_;
            // for cuda 
            #pragma unroll 
            for( int i = 0; i < zSubShape; ++ i ){
                s.shape_[ i ] = this->shape_[ i ];
            }
            return s;
        }
    public:
        /*! \brief storing the dimension information */
        index_t shape_[ zMaxShape ];
        /*! 
         * \brief storing the stride information in x dimension 
         *    this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
         */
        index_t stride_;
    };
};

namespace cxxnet{
    // simple device name
    struct cpu{
        const static bool kDevCPU = true;
    };
    struct gpu{  
        const static bool kDevCPU = false;
    };

    // more compact template
    /*! 
     * \brief general tensor
     * \tparam Device which device the tensor is on
     * \tparam dimension dimension of the tensor
     */
    template<typename Device, int dimension>
    struct Tensor{
    public:
        /*! \brief whether current type lies in cpu */
        const static bool kDevCPU = Device::kDevCPU;
        /*! \brief dimension of subtype */
        const static int  kSubdim = dimension - 1;
    public:
        /*! \brief pointer to the data */
        real_t *dptr;
        /*! \brief shape of the tensor */
        Shape<dimension> shape;
    public:
        /*! \brief default constructor */
        Tensor( void ){}
        /*! \brief constructor from data pointer and shape  */
        Tensor( real_t *dptr, Shape<dimension> shape ): dptr(dptr),shape(shape){}
        /*! 
         * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together 
         * \return tensor after flatten
         */
        inline Tensor<Device,2> FlatTo2D( void ) const{
            return Tensor<Device,2>( (real_t*)dptr, shape.FlatTo2D() );
        }
        /*! 
         * \brief get a element of dimension - 1
         * \param idx index 
         * \return the result tensor
         */
        inline Tensor<Device,kSubdim> operator[]( index_t idx ) const {
            Shape<kSubdim> s = shape.SubShape();
            return Tensor<Device,kSubdim>( (real_t*)dptr + s.MSize() * idx, s );
        }
        /*! 
         * \brief slice the tensor
         * \return tensor after flatten
         */        
        inline Tensor<Device,dimension> Slice( index_t begin, index_t end ) const{
            Shape<dimension> s = this->shape;
            s[ dimension - 1 ] = end - begin;
            return Tensor<Device,dimension>( (real_t*)dptr + s.SubShape().MSize() * begin, s );
        }
    };

    template<typename Device>
    struct Tensor<Device,1>{
    public:
        real_t *dptr;
        Shape<1> shape;
    public:
        Tensor( void ){}
        Tensor( real_t *dptr, Shape<1> shape ) :dptr(dptr),shape(shape){}
        inline Tensor<Device,2> FlatTo2D( void ) const{
            return Tensor<Device,2>( (real_t*)dptr, shape.FlatTo2D() );
        }
        inline real_t &operator[]( index_t idx ){ return dptr[ idx ]; }
        inline const real_t &operator[]( index_t idx )const { return dptr[ idx ]; }
    };
    
};

namespace cxxnet{
    typedef Tensor<cpu,1> CTensor1D;
    typedef Tensor<cpu,2> CTensor2D;
    typedef Tensor<cpu,3> CTensor3D;
    typedef Tensor<cpu,4> CTensor4D;

    typedef Tensor<gpu,1> GTensor1D;
    typedef Tensor<gpu,2> GTensor2D;
    typedef Tensor<gpu,3> GTensor3D;
    typedef Tensor<gpu,4> GTensor4D;
};

namespace cxxnet{
    // function declarations   
};
#pragma GCC pop_options
#endif
