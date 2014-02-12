#ifndef CXXNET_TENSOR_GPU_INL_CUH
#define CXXNET_TENSOR_GPU_INL_CUH

#include "../tensor.h"
#include "tensor_gpu.cuh"
#include "tensor_gpu_op.cuh"

namespace cxxnet {

    namespace cuda {                
        // implementation of map binary
        template<typename SV, typename OP, int block_dim_bits>
        __global__ void MapBinaryKernel(GTensor<2> dst, GTensor<2> lhs, GTensor<2> rhs) {
            const index_t tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const index_t x_mm = dst.shape.stride_;
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            if (y < dst.shape.shape_[1] && x < dst.shape.shape_[0]) {
                sv::GSaver<SV>::Save(dst[y][x], op::GBinaryMapper<OP>::Map(lhs[y][x], rhs[y][x]));
            }
        }
        template<typename Saver, typename BinaryMapper>
        inline void MapBinary(GTensor2D _dst, const GTensor2D &_lhs, const GTensor2D &_rhs) {
            const int num_block = (_dst.shape.MSize() + BASE_THREAD_NUM-1) / BASE_THREAD_NUM;
            dim3 dimBlock(BASE_THREAD_NUM, 1, 1);
            GTensor<2> dst(_dst), lhs(_lhs), rhs(_rhs );
            
            if (num_block < MAX_GRID_NUM) {
                dim3 dimGrid(num_block, 1, 1);
                MapBinaryKernel<Saver, BinaryMapper, BASE_THREAD_BITS>
                    <<<dimGrid,dimBlock>>>(dst, lhs, rhs);
            } else {
                //int repeat = (num_block + BASE_GRID_NUM-1) / BASE_GRID_NUM;
                //dim3 dimGrid(BASE_GRID_NUM, 1 , 1);
                // TODO                
            }
        }

    }; // namespace cuda
}; // namespace cxxnet
#endif // TENSOR_GPU_INL_H
