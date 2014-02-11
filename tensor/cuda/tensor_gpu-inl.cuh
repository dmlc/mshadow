#ifndef TENSOR_GPU_INL_CUH
#define TENSOR_GPU_INL_CUH

#include "../tensor.h"
#include "tensor_gpu_op.cuh"

namespace cxxnet {
    namespace cuda {
        #if __CUDA_ARCH__>=200
        const int MEM_UNIT_BITS = 5;
        const int MAX_THREADS_PER_BLOCK = 1024;
        #else
        const int MEM_UNIT_BITS = 4;
        const int MAX_THREADS_PER_BLOCK = 512;
        #endif

        const int BASE_THREAD_BITS = 8;
        const int BASE_THREAD_NUM  = 1 << BASE_THREAD_BITS;
        const int BASE_GRID_NUM    = 32;
        const int MAX_GRID_NUM     = 65535;
    }; // namespace cuda

    namespace cuda {
        template<typename SV, typename OP, int block_dim_bits>
        __global__ void MapBinaryKernel(GTensor2D dst, GTensor2D lhs, GTensor2D rhs) {
            const index_t tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const index_t x_mm = dst.shape.stride_;
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            if (y < dst.shape[1] && x < dst.shape[0]) {
                sv::GSaver<SV>::Save(dst[y][x], op::BinaryMapper<OP>::Map(lhs[y][x], rhs[y][x]));
            }
        }

        template<typename Saver, typename BinaryMapper>
        inline void MapBinary(GTensor2D dst, const GTensor2D &lhs, const GTensor2D &rhs) {
            const int num_block = (dst.shape.MSize() + BASE_THREAD_NUM-1) / BASE_THREAD_NUM;
            dim3 dimBlock(BASE_THREAD_NUM, 1, 1);

            if (num_block < MAX_GRID_NUM) {
                dim3 dimGrid(num_block, 1, 1);
                MapBinaryKernel<Saver, BinaryMapper, BASE_THREAD_BITS>
                    <<<dimGrid,dimBlock>>>(dst, lhs, rhs);
            } else {
                int repeat = (num_block + BASE_GRID_NUM-1) / BASE_GRID_NUM;
                dim3 dimGrid(BASE_GRID_NUM, 1 , 1);
                // TODO
                
            }
        }

    }; // namespace cuda
}; // namespace cxxnet
#endif // TENSOR_GPU_INL_H
