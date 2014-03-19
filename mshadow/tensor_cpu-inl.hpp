#ifndef MSHADOW_TENSOR_CPU_INL_HPP
#define MSHADOW_TENSOR_CPU_INL_HPP
/*!
 * \file tensor_cpu-inl.hpp
 * \brief implementation of CPU host code
 * \author Bing Hsu, Tianqi Chen
 */
#include <cstring>
#include "tensor_base.h"
#include "tensor_sse-inl.hpp"

namespace mshadow {
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj, bool pad ){
        size_t pitch;
        if( pad ){
            obj.dptr = (real_t*)sse2::AlignedMallocPitch
                ( pitch, obj.shape[0] * sizeof(real_t), obj.FlatTo2D().shape[1] );
            obj.shape.stride_ = static_cast<index_t>( pitch / sizeof(real_t) );
        }else{
            obj.shape.stride_ = obj.shape[0];
            obj.dptr = (real_t*)sse2::AlignedMallocPitch
                ( pitch, obj.shape.Size() * sizeof(real_t), 1 );
        }
    }

    template<typename Device, int dim>
    inline Tensor<Device,dim> NewTensor(const Shape<dim> &shape, real_t initv, bool pad ){
        Tensor<Device, dim> obj( shape );
        AllocSpace( obj, pad );
        MapExp<sv::saveto>( obj, expr::ScalarExp( initv ) );
        return obj;
    }

    template<int dim>
    inline void FreeSpace(Tensor<cpu,dim> &obj){
        sse2::AlignedFree( obj.dptr );
        obj.dptr = NULL;
    }

    template<int dim>
    inline void Copy(Tensor<cpu,dim> _dst, const Tensor<cpu,dim> &_src ){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        Tensor<cpu,2> src = _src.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            memcpy( dst[y].dptr, src[y].dptr, sizeof(real_t) * dst.shape[0] );
        }
    }

    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<cpu,dim> _dst, const expr::Plan<E> &plan){
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            for (index_t x = 0; x < dst.shape[0]; ++x ) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], plan.Eval( y, x ) );
            }
        }
    }

    // code to handle SSE optimization
    template<bool pass_check,typename Saver, int dim, typename E, int etype>
    struct MapExpCPUEngine;
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<false,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            MapPlan<SV>( dst, MakePlan( exp.self() ) );
        }
    };

    #if MSHADOW_USE_SSE
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<true,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            using namespace expr;
            if( SSEAlignCheck<dim,E>::Check( exp.self() )){
                MapSSEPlan<SV>( dst, MakeSSEPlan( exp.self() ) );
            }else{
                MapPlan<SV>( dst, MakePlan( exp.self() ) );
            }
        }
    };
    #endif

    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,dim,E>::kMapPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        Shape<dim> eshape = ShapeCheck<dim,E>::Check( exp.self() );
        utils::Assert( eshape[0] == 0 || eshape == dst.shape, "shape of Tensors in expression is not consistent with target" );
        #if MSHADOW_USE_SSE
        MapExpCPUEngine< SSECheck<E>::kPass,Saver,dim,E,etype >::Map( dst, exp );
        #else
        MapExpCPUEngine< false,Saver,dim,E,etype >::Map( dst, exp );
        #endif
    }

    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceKeepLowest( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,1,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        Shape<2> eshape = ShapeCheck< ExpInfo<cpu,E>::kDim, E >::Check( exp.self() ).FlatTo2D();

        utils::Assert( eshape[0] == dst.shape[0], "reduction dimension do not match" );
        utils::Assert( eshape[1] != 0, "can not reduce over empty tensor" );
        // execution
        expr::Plan<E> plan = MakePlan( exp.self() );
        for( index_t x = 0; x < eshape[0]; ++x ){
            real_t res = plan.Eval( 0, x );
            for( index_t y = 1; y < eshape[1]; ++y ){
                Reducer::Reduce( res, plan.Eval( y, x ) );
            }
            Saver::Save( dst[x], res*scale );
        }
    }

    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceKeepHighDim( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,dimkeep,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        typedef Shape< ExpInfo<cpu,E>::kDim > EShape;
        EShape eshape = ShapeCheck< ExpInfo<cpu,E>::kDim, E >::Check( exp.self() );
        utils::Assert( eshape[dimkeep] == dst.shape[0], "reduction dimension do not match" );
        // use equvalent form
        Shape<4> pshape = Shape4( 1, eshape[dimkeep], 1, eshape[0] );
        #pragma unroll
        for( int i = 1; i < dimkeep; ++ i ) pshape[1] *= eshape[i];
        #pragma unroll
        for( int i = dimkeep+1; i < EShape::kMaxShape; ++i ) pshape[3] *= eshape[i];

        // execution
        expr::Plan<E> plan = MakePlan( exp.self() );

        for( index_t c = 0; c < pshape[2]; ++c ){
            real_t res = Reducer::kInitV;
            for( index_t n = 0; n < pshape[3]; ++n ){
                for( index_t y = 0; y < pshape[1]; ++y ){
                    for( index_t x = 0; x < pshape[0]; ++x ){
                        Reducer::Reduce( res, plan.Eval( (n*pshape[2] + c) * pshape[1] + y, x ) );
                    }
                }
            }
            Saver::Save( dst[c], res*scale );
        }
    }

    inline void Softmax( Tensor<cpu,1> dst, const Tensor<cpu,1>& energy ){
        real_t mmax = energy[0];
        for( real_t x = 1; x < dst.shape[0]; ++x )
            if( mmax < energy[x] ) mmax = energy[x];
        real_t sum = 0.0f;
        for( index_t x = 0; x < dst.shape[0]; ++x ){
            dst[x] = std::exp( energy[x] - mmax );
            sum += dst[x];
        }
        for( index_t x = 0; x < dst.shape[0]; ++x ){
            dst[x] /= sum;
        }
    }
    inline void Softmax( Tensor<cpu,2> dst, const Tensor<cpu,2>& energy ){
        utils::Assert( dst.shape == energy.shape, "Softmax: shape mismatch" );
        for( index_t y = 0; y < dst.shape[1]; ++y ){
            Softmax( dst[y], energy[y] );
        }
    }

    inline void UnpackPatchToCol( Tensor<cpu,2> mat, const Tensor<cpu,3> &img, index_t psize, index_t pstride ){
        utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "UnpackPatchToCol:image shape smaller than patch size");
        const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
        const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
        utils::Assert( o_height*o_width == mat.shape[0], "UnpackPatchToCol: mat.shape[0] mismatch" );
        utils::Assert( psize*psize*img.shape[2] == mat.shape[1], "UnpackPatchToCol: mat.shape[1] mismatch" );

        for( index_t i = 0; i < mat.shape[1]; ++i ){
            const index_t x_offset = i % psize;
            const index_t y_offset = (i / psize) % psize;
            const index_t channel = i / (psize*psize);
            for( index_t j = 0; j < mat.shape[0]; ++j ){
                const index_t x = (j % o_width) * pstride + x_offset;
                const index_t y = (j / o_width) * pstride + y_offset;
                if( x < img.shape[0] && y < img.shape[1] ){
                    mat[i][j] = img[channel][y][x];
                }else{
                    mat[i][j] = 0.0f;
                }
            }
        }
    }

    // checked version, used for double check correctness
    inline void PackPatchFromColCHK( Tensor<cpu,3> &img, const Tensor<cpu,2> &mat, index_t psize, index_t pstride ){
        utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "PackPatchFromCol:image shape smaller than patch size");
        const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
        const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
        utils::Assert( o_height*o_width == mat.shape[0], "PackPatchFromCol: mat.shape[0] mismatch" );
        utils::Assert( psize*psize*img.shape[2] == mat.shape[1], "PackPatchFromCol: mat.shape[1] mismatch" );
        for( index_t c = 0; c < img.shape[2]; ++c ){
            for( index_t y = 0; y < img.shape[1]; ++y ){
                for( index_t x = 0; x < img.shape[0]; ++x ){
                    const index_t x_base = x % pstride;
                    const index_t y_base = y % pstride;
                    real_t res = 0.0f;
                    for( index_t y_offset = y_base; y_offset < psize; y_offset += pstride ){
                        for( index_t x_offset = x_base; x_offset < psize; x_offset += pstride ){
                            const int y_start = (int)y - (int)y_offset;
                            const int x_start = (int)x - (int)x_offset;
                            if( y_start >= 0 && x_start >= 0 && (y_start/pstride) < o_height && (x_start/pstride) < o_width ){
                                res += mat[ (c * psize + y_offset) * psize + x_offset ][ (y_start/pstride)*o_width + (x_start/pstride) ];
                            }
                        }
                    }
                    img[c][y][x] = res;
                }
            }
        }
    }

    inline void PackPatchFromCol( Tensor<cpu,3> img, const Tensor<cpu,2> &mat, index_t psize, index_t pstride ){
        using namespace std;
        utils::Assert( img.shape[0] >= psize && img.shape[1] >= psize, "PackPatchFromCol:image shape smaller than patch size");
        const index_t o_height = ( img.shape[1]  - psize ) / pstride + 1;
        const index_t o_width  = ( img.shape[0]  - psize ) / pstride + 1;
        utils::Assert( o_height*o_width == mat.shape[0], "PackPatchFromCol: mat.shape[0] mismatch" );
        utils::Assert( psize*psize*img.shape[2] == mat.shape[1], "PackPatchFromCol: mat.shape[1] mismatch" );

        for( index_t c = 0; c < img.shape[2]; ++c ){
            for( index_t y = 0; y < img.shape[1]; ++y ){
                for( index_t x = 0; x < img.shape[0]; ++x ){
                    // need ensure y - y_max >= 0
                    const index_t y_max = min( psize, y + 1 );
                    const index_t x_max = min( psize, x + 1 );
                    // need ensure (y - y_min) / pstride  < o_height
                    // equals y_min >= y - pstride * o_height + 1
                    // const int y_min = max( (y-pstride*o_height+pstride) /pstride, 0 ) * pstride + ( y % pstride );
                    // const int x_min = max( (x-pstride*o_width +pstride) /pstride, 0 ) * pstride + ( x % pstride );
                    // equvalent form: since we can not have negative value in unsign
                    const index_t y_min = (max( y/pstride, o_height-1 )+1-o_height) * pstride + ( y % pstride );
                    const index_t x_min = (max( x/pstride, o_width-1 ) +1-o_width ) * pstride + ( x % pstride );

                    real_t res = 0.0f;
                    for( index_t y_offset = y_min; y_offset < y_max; y_offset += pstride ){
                        for( index_t x_offset = x_min; x_offset < x_max; x_offset += pstride ){
                            const index_t y_start = y - y_offset;
                            const index_t x_start = x - x_offset;
                            res += mat[ (c * psize + y_offset) * psize + x_offset ][ (y_start/pstride)*o_width+(x_start/pstride) ];
                        }
                    }
                    img[c][y][x] = res;
                }
            }
        }
    }
}; // namespace mshadow

namespace mshadow {
    inline void MaxPooling(Tensor<cpu, 3> &pooled, Tensor<cpu, 3> &img, index_t ksize, index_t kstride) {
        utils::Assert(img.shape[0] >= ksize && img.shape[1] >= ksize, "Pooling: image shape smaller than kernel size");
        utils::Assert(pooled.shape[2] == img.shape[2], "Pooling: img and pooling target has different channel");
        utils::Assert(pooled.shape[1] == (img.shape[1] - ksize) / kstride + 1, "Pooling: image height error");
        utils::Assert(pooled.shape[0] == (img.shape[0] - ksize) / kstride + 1, "Pooling: image width error");

        for (index_t c = 0; c < pooled.shape[2]; ++c) {
            for (index_t ph = 0; ph < pooled.shape[1]; ++ph) {
                for (index_t pw = 0; pw < pooled.shape[0]; ++pw) {
                    index_t h_start = ph * kstride;
                    index_t h_end = std::min(h_start + ksize, img.shape[1]);
                    index_t w_start = pw * kstride;
                    index_t w_end = std::min(w_start + ksize, img.shape[0]);
                    for (index_t ih = h_start; ih < h_end; ++ih) {
                        for (index_t iw = w_start; iw < w_end; ++iw) {
                            pooled[c][ph][pw] = std::max(pooled[c][ph][pw], img[c][ih][iw]);
                        }
                    }
                }
            }
        }
    }


}; // namespace mshadow


#endif // TENSOR_CPU_INL_HPP
