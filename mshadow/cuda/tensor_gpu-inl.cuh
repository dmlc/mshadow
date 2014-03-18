#ifndef MSHADOW_TENSOR_GPU_INL_CUH
#define MSHADOW_TENSOR_GPU_INL_CUH
/*!
 * \file tensor_gpu-inl.cuh
 * \brief implementation of GPU code using CUDA
 * \author Bing Hsu, Tianqi Chen
 */
#include "../tensor.h"
#include "cuda_reduce.cuh"

namespace mshadow{
    namespace cuda{
        /* load unit for memory access */
        #if __CUDA_ARCH__>=200
        const int kMemUnitBits = 5;
        const int kMaxThreadsPerBlock = 1024;
        #else
        const int kMemUnitBits = 4;
        const int kMaxThreadsPerBlock = 512;
        #endif
        /*! \brief number of units that can do synchronized update, half warp size */
        const int kMemUnit     = 1 << kMemUnitBits;
        /*! \brief mask that could be helpful sometime */
        const int kMemUnitMask = kMemUnit - 1;
        /*! \brief suggested thread number(logscale) for mapping kernel */
        const int kBaseThreadBits = 8;
        /*! \brief suggested thread number for mapping kernel */
        const int kBaseThreadNum  = 1 << kBaseThreadBits;
        /*! \brief maximum value of grid */
        const int kMaxGridNum     = 65535;
        
        /*! \brief get align stride for given size in x dimension */
        index_t GetAlignStride( index_t xsize ){
            return ( (xsize  + kMemUnit - 1) >> kMemUnitBits) << kMemUnitBits;
        }
        inline void CheckLaunchParam( dim3 dimGrid, dim3 dimBlock, const char *estr = "" ){
            if( dimBlock.x*dimBlock.y*dimBlock.z > (unsigned)kMaxThreadsPerBlock ||
                dimGrid.x > 65535 || dimGrid.y > 65535 ){
                fprintf( stderr, "%s[%u,%u,%u]:", estr, dimBlock.x, dimBlock.y, dimBlock.z );
                utils::Error( "too large launch parameter\n");
            } 
        }        
    };

    namespace cuda {
        template<typename Saver, typename Plan, int block_dim_bits>
        __global__ void MapPlanKernel( Tensor<gpu,2> dst, const index_t xstride, const Plan exp ){
            const index_t tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int y   = tid / xstride;
            const int x   = tid % xstride;
            if (y < dst.shape[1] && x < dst.shape[0]) {
                Saver::Save(dst[y][x], exp.Eval(y,x));
            }
        }
        template<typename Saver, typename E>
        inline void MapPlan( Tensor<gpu,2> dst, const expr::Plan<E> &plan ){
            const index_t xstride = GetAlignStride( dst.shape[0] );
            const int num_block = ( dst.shape[1]*xstride + kBaseThreadNum-1) / kBaseThreadNum;
            dim3 dimBlock(kBaseThreadNum, 1, 1);

            if (num_block < kMaxGridNum) {
                dim3 dimGrid(num_block, 1, 1);
                MapPlanKernel<Saver, expr::Plan<E>, kBaseThreadBits>   \
                    <<<dimGrid,dimBlock>>>(dst, xstride, plan);
            } else {
                utils::Error("not implemented");
            }
        }        
    }; // namespace cuda
    
    namespace cuda{
        template<typename Saver,typename Reducer, int warp_bits, typename Plan>
        __global__ void MapRedToLowestKernel( Tensor<gpu,1> dst, Plan plan, real_t scale, Shape<2> eshape ){
            const unsigned warp_size = 1 << warp_bits;
            const unsigned x = (blockIdx.x<<warp_bits) + threadIdx.x;
            // to avoid bank conflict
            __shared__ real_t s_res[ warp_size ][ warp_size + 1 ];

            // note: reverse store [y][x], so that we can reduce over threadIdx.x, use warp optimization
            if( threadIdx.y < eshape[1] && x < eshape[0] ){
                s_res[ threadIdx.x ][ threadIdx.y ] = plan.Eval( threadIdx.y, x );
            }
            for( unsigned y = warp_size; y < eshape[1]; y += warp_size ){
                if( threadIdx.y + y < eshape[1] && x < eshape[0] ){
                    Reducer::Reduce( s_res[ threadIdx.x ][ threadIdx.y ], plan.Eval( threadIdx.y + y, x ) );
                }
            } 
            __syncthreads();
            if( eshape[1] >= warp_size ){
                Reduce1D<Reducer,warp_bits>( s_res[ threadIdx.y ] );
            }else{
                Reduce1DNotAlign<Reducer,warp_bits>( s_res[ threadIdx.y ], eshape[1] );
            }
            __syncthreads();            
            
            if( threadIdx.y == 0 && x < eshape[0] ){
                Saver::Save( dst[x],  s_res[ threadIdx.x ][ 0 ] * scale );
            } 
        }        
        
        template<typename Saver, typename Reducer, typename E>
        inline void MapReduceToLowest( Tensor<gpu,1> dst, const expr::Plan<E> &plan, real_t scale, Shape<2> eshape ){
            dim3 dimBlock( kMemUnit, kMemUnit );
            dim3 dimGrid ( (eshape[0]+kMemUnit-1) >> kMemUnitBits );
            CheckLaunchParam( dimGrid, dimBlock, "MapRedToLowestKernel" );
            MapRedToLowestKernel<Saver,Reducer,kMemUnitBits><<<dimGrid,dimBlock>>>( dst, plan, scale, eshape );
        } 
    }; // namespace cuda
    
    namespace cuda{
        template<int x_bits>        
        __global__ void SoftmaxKernel( Tensor<gpu,2> dst, Tensor<gpu,2> src ){
            const unsigned x_size = 1 << x_bits;  
            const int y = blockIdx.x;
            __shared__ real_t s_rec[ x_size ];
            
            // step 1: get max
            if( threadIdx.x < dst.shape[ 0 ] ){
                s_rec[ threadIdx.x ] = src[ y ][ threadIdx.x ] ; 
            }
            for( unsigned x = x_size; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    real_t a = src[ y ][ x + threadIdx.x ];
                    s_rec[ threadIdx.x ] = max( a, s_rec[ threadIdx.x] );
                }
            }
            __syncthreads();
            if( threadIdx.x >= dst.shape[0] ){
                s_rec[ threadIdx.x ] = s_rec[0];
            }
            __syncthreads();
            Reduce1D<red::maximum,x_bits>( s_rec );
            __syncthreads();
            real_t smax = s_rec[0];            
            __syncthreads();
            s_rec[ threadIdx.x ] = 0.0f;
            __syncthreads();

            // calculate normalizer, with writeback
            for( unsigned x = 0; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    real_t p = expf( src[ y ][ x + threadIdx.x ] - smax );
                    s_rec[ threadIdx.x ] += p;
                    // write back first, will fetch later
                    dst[ y ][ x + threadIdx.x ] = p;
                }
            }
            // calculate normalizer
            __syncthreads();
            Reduce1D<red::sum,x_bits>( s_rec );
            __syncthreads();
            real_t ssum = s_rec[0];

            for( unsigned x = 0; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    dst[ y ][ x + threadIdx.x ] /= ssum;
                }
            }
        }
    
        inline void Softmax( Tensor<gpu,2> &dst, const Tensor<gpu,2> &src ){
            dim3 dimBlock( kBaseThreadNum );
            dim3 dimGrid ( dst.shape[1] );
            utils::Assert( dst.shape == src.shape, "Softmax: shape mismatch" );
            CheckLaunchParam( dimGrid, dimBlock, "Softmax" );
            SoftmaxKernel<kBaseThreadBits><<<dimGrid,dimBlock>>>( dst, src );
        }
    }; // namespace cuda

    namespace cuda{
        template<typename Saver, int block_dim_bits>        
        __global__ void UnpackPatchToColKernel( Tensor<gpu,2> mat, const Tensor<gpu,3> img, 
                                                const index_t mat_xstride,
                                                index_t psize, index_t pstride, index_t o_width ){
            const index_t tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const index_t i = tid / mat_xstride;
            const index_t j = tid % mat_xstride;
            // index caculations
            const index_t x_offset = i % psize; 
            const index_t idivp    = i / psize;
            const index_t y_offset = idivp % psize;
            const index_t channel  = idivp / psize;
            const index_t y = (j / o_width) * pstride + y_offset;  
            const index_t x = (j % o_width) * pstride + x_offset;
            // will be continuous write, but not continuous read
            if( i < mat.shape[1] && j < mat.shape[0] ){
                if( x < img.shape[0] && y < img.shape[1] ){
                    Saver::Save( mat[i][j], img[channel][y][x] );
                }else{
                    Saver::Save( mat[i][j], 0.0f );
                }
            }
        }

        inline void UnpackPatchToCol( Tensor<gpu,2> mat, const Tensor<gpu,3> &img, index_t psize, index_t pstride ){
            utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "UnpackPatchToCol:image shape smaller than patch size");
            const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
            const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
            utils::Assert( o_height*o_width == mat.shape[0], "UnpackPatchToCol: mat.shape[0] mismatch" );
            utils::Assert( psize*psize*img.shape[2] == mat.shape[1], "UnpackPatchToCol: mat.shape[1] mismatch" );
            const index_t xstride = GetAlignStride( mat.shape[0] );
            const int num_block = ( mat.shape[1]*xstride + kBaseThreadNum-1) / kBaseThreadNum;
            dim3 dimBlock(kBaseThreadNum, 1, 1);

            if (num_block < kMaxGridNum) {
                dim3 dimGrid(num_block, 1, 1);
                UnpackPatchToColKernel<sv::saveto, kBaseThreadBits>     \
                    <<<dimGrid,dimBlock>>>(mat, img, xstride, psize, pstride, o_width );
            } else {
                utils::Error("not implemented");
            }            
        }
    };

    namespace cuda{
        template<typename Saver, int block_dim_bits>        
        __global__ void PackPatchFromColKernel( Tensor<gpu,3> img, const Tensor<gpu,2> mat, 
                                                index_t psize, index_t pstride, index_t o_height, index_t o_width ){
            index_t tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const index_t x = tid % img.shape[0];
            tid /= img.shape[0];
            const index_t y = tid % img.shape[1];
            const index_t c = tid / img.shape[1];

            if( c < img.shape[2] ){
                // need ensure y - y_max >= 0    
                const index_t y_max = min( psize, y + 1 ); 
                const index_t x_max = min( psize, x + 1 );
                // need ensure (y - y_min) / pstride  < o_height
                const index_t y_min = (max( y/pstride, o_height-1 )+1-o_height-1) * pstride + ( y % pstride ); 
                const index_t x_min = (max( x/pstride, o_width-1 ) +1-o_width  ) * pstride  + ( x % pstride );                     
                
                real_t res = 0.0f;
                for( index_t y_offset = y_min; y_offset < y_max; y_offset += pstride ){
                    for( index_t x_offset = x_min; x_offset < x_max; x_offset += pstride ){
                        const index_t y_start = y - y_offset;
                        const index_t x_start = x - x_offset;
                        res += mat[ (c * psize + y_offset) * psize + x_offset ][ (y_start/pstride)*o_width+(x_start/pstride) ]; 
                    }
                }
                Saver::Save( img[c][y][x], res );
            }
        }
    
        inline void PackPatchFromCol( Tensor<gpu,3> img, const Tensor<gpu,2> &mat, index_t psize, index_t pstride ){
            utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "PackPatchFromCol:image shape smaller than patch size");
            const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
            const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
            utils::Assert( o_height*o_width == mat.shape[0], "PackPatchFromCol: mat.shape[0] mismatch" );
            utils::Assert( psize*psize*img.shape[2] == mat.shape[1], "PackPatchFromCol: mat.shape[1] mismatch" );
            
            const int num_block = ( img.shape.Size() + kBaseThreadNum-1) / kBaseThreadNum;
            dim3 dimBlock(kBaseThreadNum, 1, 1);
            if (num_block < kMaxGridNum) {
                dim3 dimGrid(num_block, 1, 1);
                PackPatchFromColKernel<sv::saveto, kBaseThreadBits>     \
                    <<<dimGrid,dimBlock>>>(img, mat, psize, pstride, o_height, o_width );
            } else {
                utils::Error("not implemented");
            }            
        }
    };    
}; // namespace mshadow
#endif // TENSOR_GPU_INL_H
