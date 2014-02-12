#ifndef CXXNET_TENSOR_GPU_CUH
#define CXXNET_TENSOR_GPU_CUH
/*!
 * \file tensor_gpu.h
 * \brief data structure for GPU, a replication of host, but runs in device
 *
 * \author Bing Hsu, Tianqi Chen
 */

#ifndef _DINLINE_
#define _DINLINE_ inline __device__
#else
#error "_DINLINE_ must not be defined"
#endif

#include "tensor.h"

namespace cxxnet{
    namespace cuda{
        /* load unit for memory access */
        #if __CUDA_ARCH__>=200
        const int MEM_UNIT_BITS = 5;
        const int MAX_THREADS_PER_BLOCK = 1024;
        #else
        const int MEM_UNIT_BITS = 4;
        const int MAX_THREADS_PER_BLOCK = 512;
        #endif
        
        const int MEM_UNIT      = 1 << MEM_UNIT_BITS;
        const int MEM_UNIT_MASK = MEM_UNIT - 1; 
        
        const int ALIGN_BITS       = MEM_UNIT_BITS;
        const int ALIGN_WIDTH      = 1 << ALIGN_BITS;
        const int BASE_THREAD_BITS = 8;
        const int BASE_THREAD_NUM  = 1 << BASE_THREAD_BITS;
        const int BASE_GRID_NUM    = 32;
        const int MAX_GRID_NUM     = 65535;
    };
};
namespace cxxnet{
    namespace cuda{
        /* replication of Shape, except all the functions are device */
        template<int dimension>
        struct GShape {
        public:
            /*! \brief maximum dimension of tensor */
            const static int zMaxShape = dimension;
            const static int zSubShape = dimension - 1;
        public:
            /*! \brief default constructor, do nothing */
            __device__ GShape(void) {}
            /*! \brief default constructor, do nothing */
            GShape(const Shape<dimension> &s) {
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
            _DINLINE_ index_t& operator[](index_t idx) {
                return shape_[ idx ];
            }
            /*!
             * \brief get corresponding index
             * \param idx dimension index
             * \return the corresponding dimension size
             */
            _DINLINE_ const index_t& operator[](index_t idx) const {
                return shape_[ idx ];
            }
            /*! \return stride */
            _DINLINE_ const index_t& stride(void) const {
                return stride_;
            }
            /*! \return whether two shape equals */
            _DINLINE_ bool operator==(const Shape<zMaxShape> &s) const {
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
            _DINLINE_ GShape<2> FlatTo2D(void) const {
                GShape<2> s;
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
            _DINLINE_ size_t Size(void) {
                size_t memsz = this->shape_[ 0 ];
                #pragma unroll
                for (int i = 1; i < zMaxShape; ++i) {
                    memsz *= this->shape_[ i ];
                }
                return memsz;
            }
            /*! \return memory size, including the aligned x dimension */
            _DINLINE_ size_t MSize(void) const {
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
            _DINLINE_ GShape<zSubShape> SubShape(void) const {
                GShape<zSubShape> s;
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
    };

    namespace cuda{
        /*!
         * \brief general tensor
         * \tparam Device which device the tensor is on
         * \tparam dimension dimension of the tensor
         */
        template<int dimension>
        struct GTensor {
        public:
            /*! \brief dimension of subtype */
            const static int  kSubdim = dimension - 1;  
        public:
            /*! \brief pointer to the data */
            real_t *dptr;
            /*! \brief shape of the tensor */
            GShape<dimension> shape;
        public:
            /*! \brief default constructor */
            GTensor(void) {}
            /*! \brief constructor from shape  */
            GTensor(Shape<dimension> shape): shape(shape) {}
            /*! \brief constructor from data pointer and shape  */
            __device__ GTensor(real_t *dptr, GShape<dimension> shape): dptr(dptr), shape(shape) {}
            /*! \brief constructor from data pointer and shape  */
            GTensor(const Tensor<gpu,dimension> &ts): dptr((real_t*)ts.dptr), shape(ts.shape) {}
            /*!
             * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
             * \return tensor after flatten
             */
            _DINLINE_ GTensor<2> FlatTo2D(void) const {
                return GTensor<2>(reinterpret_cast<real_t*>     \
                                  (dptr), shape.FlatTo2D());
            }
            /*!
             * \brief get a element of dimension - 1
             * \param idx index
             * \return the result tensor
             */
            _DINLINE_ GTensor<kSubdim> operator[](index_t idx) const {
                GShape<kSubdim> s = shape.SubShape();
                return GTensor<kSubdim>(reinterpret_cast<real_t*> \
                                        (dptr) + s.MSize() * idx, s);
            }
            /*!
             * \brief slice the tensor
             * \return tensor after slice
             */
            _DINLINE_ GTensor<dimension> Slice(index_t begin, index_t end) const {
                GShape<dimension> s = this->shape;
                s[ dimension - 1 ] = end - begin;
                return GTensor<dimension>(reinterpret_cast<real_t*> \
                                          (dptr) + s.SubShape().MSize() * begin, s);
            }
        };

        /*!
         * \brief respecialized class Tensor1D,thei is due to different implementation in operator[]
         * \tparam Device device type
         */
        template<>
        struct GTensor<1> {
        public:
            real_t *dptr;
            GShape<1> shape;
        public:
            GTensor(void) {}
            __device__ GTensor(real_t *dptr, GShape<1> shape) :dptr(dptr), shape(shape) {}
            GTensor(const Tensor<gpu,1> &ts): dptr((real_t*)ts.dptr), shape(ts.shape) {}
            _DINLINE_ GTensor<2> FlatTo2D(void) const {
                return GTensor<2>(reinterpret_cast<real_t*>         \
                                         (dptr), shape.FlatTo2D());
            }
            _DINLINE_ GTensor<1> Slice(index_t begin, index_t end) const {
                GShape<1> s;
                s[0] = s.stride_ = end  - begin;
                return GTensor<1>(reinterpret_cast<real_t*> \
                                  (dptr) + begin, s);
            }
            _DINLINE_ real_t &operator[](index_t idx) { return dptr[ idx ]; }
            _DINLINE_ const real_t &operator[](index_t idx)const { return dptr[ idx ]; }
        };
    };
};
#endif
