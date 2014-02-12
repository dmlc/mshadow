#ifndef CXXNET_TENSOR_CONTAINER_H
#define CXXNET_TENSOR_CONTAINER_H
/*!
 * \file tensor_container.h
 * \brief tensor container that does memory allocation and resize like STL
 * \author Tianqi Chen
 */

#include "tensor.h"

namespace cxxnet{
    /*!
     * \brief tensor container that does memory allocation and resize like STL,
     *        use it to save the lines of FreeSpace in class.
     *        Do not abuse it, efficiency can come from pre-allocation and no re-allocation
     *
     * \tparam Device which device the tensor is on
     * \tparam dimension dimension of the tensor
     */
    template<typename Device, int dimension>
    class TensorContainer: public Tensor<Device,dimension>{
    public:
        TensorContainer( void ){
            this->dptr = data_.dptr = NULL;
            this->shape[0] = 0; 
            this->shape.stride_ = 0;
        }
        TensorContainer( const Shape<dimension> &shape ){
            data_.dptr = NULL;
            this->AllocByShape( shape );
        }
        ~TensorContainer( void ){
            this->FreeSpace();
        }
        inline void Resize( const Shape<dimension> &shape ){
            Shape<2> s2 = shape.FlatTo2D();
            if( s2.shape_[0] > data_.shape.stride_ || s2.shape_[1] > data_.shape[1] ){
                this->AllocByShape( shape );
            }{
                this->shape = shape;
                this->shape.stride_ = data_.shape.stride_;
            }
        }
    private:
        /*! \brief the shape of data_ is actually current data space */
        Tensor<Device, 2> data_;
    private:
        inline void FreeSpace (void){
            if( data_.dptr != NULL ){
                cxxnet::FreeSpace( data_ );
                data_.dptr = this->dptr = NULL;
            }
        }
        inline void AllocByShape (const Shape<dimension>& shape){
            if( data_.dptr != NULL ){
                this->FreeSpace();
            }
            data_.shape = shape.FlatTo2D();
            cxxnet::AllocSpace( data_ );
            this->dptr  = data_.dptr;
            this->shape = shape;
            this->shape.stride_ = data_.shape.stride_;
        }
    };
};// namespace cxxnet

#endif
