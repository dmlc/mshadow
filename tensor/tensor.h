#ifndef CXXNET_TENSOR_H
#define CXXNET_TENSOR_H
/*!
 * \file tensor.h
 * \brief header file of tensor data structure and functions
 * \author Bing Hsu, Tianqi Chen
 */
#include <cstdio>

#ifdef _XINLINE_
  #error "_XINLINE_ must not be defined"
#endif
#ifdef __CUDACC__
  #define _XINLINE_ inline __device__ __host__
#else
  #define _XINLINE_ inline
#endif

/*! \brief namespace for cxxnet */
namespace cxxnet {
    /*! \brief type that will be used for content */
    typedef float real_t;
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
}; // namespace cxxnet

namespace cxxnet {
    /*!
     * \brief shape of a tensor
     *       IMPORTANT NOTE: this shape is different from numpy.shape
     *       shape[0] gives the lowest dimension, shape[dimension-1] gives the highest dimension
     *       shape[k] corresponds to k-th dimension of tensor
     * \tparam dimension dimension of tensor
     */
    template<int dimension>
    struct Shape {
    public:
        /*! \brief maximum dimension of tensor */
        const static int zMaxShape = dimension;
        const static int zSubShape = dimension - 1;

    public:
        /*! \brief default constructor, do nothing */
        _XINLINE_ Shape(void) {}
        _XINLINE_ Shape( const Shape<dimension> &s ){
            #pragma unroll
            for( int i = 0; i < zMaxShape; i ++ ){
                this->shape_[i] = s[i];
            }
            this->stride_ = s.stride_;
        }
        /*!
         * \brief get corresponding index
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        _XINLINE_ index_t& operator[](index_t idx) {
            return shape_[ idx ];
        }
        /*!
         * \brief get corresponding index
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        _XINLINE_ const index_t& operator[](index_t idx) const {
            return shape_[ idx ];
        }
        /*! \return stride */
        _XINLINE_ const index_t& stride(void) const {
            return stride_;
        }
        /*! \return whether two shape equals */
        _XINLINE_ bool operator==(const Shape<zMaxShape> &s) const {
            #pragma unroll
            for (int i = 0; i < zMaxShape; i++) {
                if (s.shape_[i] != this->shape_[i]) return false;
            }
            return true;
        }
        /*!
         * flatten the higher dimension to second dimension, return a 2D shape
         * \return the flat 2d shape
         */
        _XINLINE_ Shape<2> FlatTo2D(void) const {
            Shape<2> s;
            s.stride_ = this->stride_;
            s.shape_[ 0 ] = this->shape_[ 0 ];
            index_t ymax = 1;

            #pragma unroll
            for (int i = 1; i < zMaxShape; ++i) {
                ymax *= this->shape_[ i ];
            }
            s.shape_[1] = ymax;
            return s;
        }
        /*! \return number of valid elements */
        _XINLINE_ size_t Size(void) {
            size_t memsz = this->shape_[ 0 ];
            #pragma unroll
            for (int i = 1; i < zMaxShape; ++i) {
                memsz *= this->shape_[ i ];
            }
            return memsz;
        }
        /*! \return memory size, including the aligned x dimension */
        _XINLINE_ size_t MSize(void) const {
            size_t memsz = this->stride_;
            #pragma unroll
            for (int i = 1; i < zMaxShape; ++i) {
                memsz *= this->shape_[ i ];
            }
            return memsz;
        }
        /*!
         * \brief get subshape
         * \return subshape
         */
        _XINLINE_ Shape<zSubShape> SubShape(void) const {
            Shape<zSubShape> s;
            s.stride_ = this->stride_;
            // for cuda
            #pragma unroll
            for (int i = 0; i < zSubShape; ++i) {
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
    // useful construction functions to generate shape    
    /*! 
     * \brief construct a one dimension shape 
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    _XINLINE_ Shape<1> Shape1( index_t s0 ){
        Shape<1> s; s[0] = s0;
        return s;
    }
    /*! 
     * \brief construct a two dimension shape 
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    _XINLINE_ Shape<2> Shape2( index_t s1, index_t s0 ){
        Shape<2> s; s[0] = s0; s[1] = s1;
        return s;
    }
    /*! 
     * \brief construct a three dimension shape 
     * \param s2 size of dimension 2
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    _XINLINE_ Shape<3> Shape3( index_t s2, index_t s1, index_t s0 ){
        Shape<3> s; s[0] = s0; s[1] = s1; s[2] = s2;
        return s;
    }
    /*! 
     * \brief construct a four dimension shape 
     * \param s3 size of dimension 3
     * \param s2 size of dimension 2
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    _XINLINE_ Shape<4> Shape4( index_t s3, index_t s2, index_t s1, index_t s0 ){
        Shape<4> s; s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        return s;
    }
}; // namespace cxxnet

namespace cxxnet {
    // simple device name
    struct cpu {
        const static bool kDevCPU = true;
    };
    struct gpu {
        const static bool kDevCPU = false;
    };

    // more compact template
    /*!
     * \brief general tensor
     * \tparam Device which device the tensor is on
     * \tparam dimension dimension of the tensor
     */
    template<typename Device, int dimension>
    struct Tensor {
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
        _XINLINE_ Tensor(void) {}
        /*! \brief constructor from shape  */
        _XINLINE_ Tensor(const Shape<dimension> &shape): shape(shape) {}
        /*! \brief constructor from data pointer and shape  */
        _XINLINE_ Tensor(real_t *dptr, const Shape<dimension> &shape): dptr(dptr), shape(shape) {}
        /*!
         * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
         * \return tensor after flatten
         */
        _XINLINE_ Tensor<Device, 2> FlatTo2D(void) const {
            return Tensor<Device, 2>(reinterpret_cast<real_t*> \
                                     (dptr), shape.FlatTo2D());
        }
        /*!
         * \brief get a element of dimension - 1
         * \param idx index
         * \return the result tensor
         */
        _XINLINE_ Tensor<Device, kSubdim> operator[](index_t idx) const {
            Shape<kSubdim> s = shape.SubShape();
            return Tensor<Device, kSubdim>(reinterpret_cast<real_t*> \
                                           (dptr) + s.MSize() * idx, s);
        }
        /*!
         * \brief slice the tensor
         * \return tensor after slice
         */
        _XINLINE_ Tensor<Device, dimension> Slice(index_t begin, index_t end) const {
            Shape<dimension> s = this->shape;
            s[ dimension - 1 ] = end - begin;
            return Tensor<Device, dimension>(reinterpret_cast<real_t*>\
                                             (dptr) + s.SubShape().MSize() * begin, s);
        }
    };

    /*!
     * \brief respecialized class Tensor1D,thei is due to different implementation in operator[]
     * \tparam Device device type
     */
    template<typename Device>
    struct Tensor<Device, 1> {
    public:
        real_t *dptr;
        Shape<1> shape;
    public:
        _XINLINE_ Tensor(void) {}
        _XINLINE_ Tensor(real_t *dptr, Shape<1> shape) :dptr(dptr), shape(shape) {}
        _XINLINE_ Tensor<Device, 2> FlatTo2D(void) const {
            return Tensor<Device, 2>(reinterpret_cast<real_t*> \
                                     (dptr), shape.FlatTo2D());
        }
        _XINLINE_ Tensor<Device, 1> Slice(index_t begin, index_t end) const {
            Shape<1> s;
            s[0] = s.stride_ = end  - begin;
            return Tensor<Device, 1>(reinterpret_cast<real_t*> \
                                     (dptr) + begin, s);
        }
        _XINLINE_ real_t &operator[](index_t idx) { return dptr[ idx ]; }
        _XINLINE_ const real_t &operator[](index_t idx)const { return dptr[ idx ]; }
    };
}; // namespace cxxnet

namespace cxxnet {
    typedef Tensor<cpu, 1> CTensor1D;
    typedef Tensor<cpu, 2> CTensor2D;
    typedef Tensor<cpu, 3> CTensor3D;
    typedef Tensor<cpu, 4> CTensor4D;

    typedef Tensor<gpu, 1> GTensor1D;
    typedef Tensor<gpu, 2> GTensor2D;
    typedef Tensor<gpu, 3> GTensor3D;
    typedef Tensor<gpu, 4> GTensor4D;
}; // namespace cxxnet

// add unroll loops for the shape
namespace cxxnet {
    // function declarations
    /*!
     * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
     *        this function is responsible to set the stride_ in each obj.shape
     * \tparam dim specify the dim of tensor
     * \param obj the tensor object, with shape specified
     */
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj);
    template<int dim>
    inline void AllocSpace(Tensor<gpu,dim> &obj);

    /*!
     * \brief CPU/GPU: free the space of tensor
     * \tparam dim specify the dim of tensor
     * \param obj the tensor object
     */
    template<int dim>
    inline void FreeSpace(Tensor<cpu,dim> &obj);
    template<int dim>
    inline void FreeSpace(Tensor<gpu,dim> &obj);

    /*!
     * \brief copy data from one tensor to another, with same shape
     * \tparam dim specify the dim of tensor
     * \param obj the tensor object, with shape specified
     */
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<cpu,dim> &src );
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<gpu,dim> &src );
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<cpu,dim> &src );
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &src );
    
    /*!
     * \brief CPU/GPU: storing function dst [st] src
     * \tparam Saver specify storage method [st]
     * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
     * \param dst destination
     * \param src the real data
     * \sa namespace cxxnet:sv
     */
    template<typename Saver,int dim>
    inline void Store(Tensor<cpu,dim> dst, real_t src);
    template<typename Saver,int dim>
    inline void Store(Tensor<gpu,dim> dst, real_t src);

    /*!
     * \brief CPU: binary mapping function dst [st] lhs [op] rhs
     * \tparam Saver specify storage method [st]
     * \tparam BinaryMapper specify binary operation [op]
     * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
     * \param dst destination
     * \param lhs left operand
     * \param rhs right operand
     * \sa namespace cxxnet:sv, cxxnet::op
     */
    template<typename Saver, typename BinaryMapper,int dim>
    inline void Map(Tensor<cpu,dim> dst, const Tensor<cpu,dim> &lhs, const Tensor<cpu,dim> &rhs);
    template<typename Saver, typename BinaryMapper,int dim>
    inline void Map(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &lhs, const Tensor<gpu,dim> &rhs);

}; // namespace cxxnet

namespace cxxnet{
    // the following function is name dependent, have different name for CPU and GPU
    /*!
     * \brief CPU: short cut to allocate and initialize a CTensor
     * \tparam dim specify the dim of tensor
     * \param shape shape of the tensor
     * \return the allocated tensor
     */
    template<int dim>
    inline Tensor<cpu,dim> NewCTensor(const Shape<dim> &shape, real_t initv);

    /*!
     * \brief GPU: short cut to allocate and initialize a GTensor
     * \tparam dim specify the dim of tensor
     * \param shape shape of the tensor
     * \return the allocated tensor
     */
    template<int dim>
    inline Tensor<gpu,dim> NewGTensor(const Shape<dim> &shape, real_t initv);
    
}; // namespace cxxnet
// implementation
#include "tensor_op.h"
#include "tensor_cpu-inl.hpp"
#include "tensor_gpu-inl.hpp"

#endif // TENSOR_H
