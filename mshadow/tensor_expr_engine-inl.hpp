#ifndef MSHADOW_TENSOR_EXPR_ENGINE_INL_HPP
#define MSHADOW_TENSOR_EXPR_ENGINE_INL_HPP
/*!
 * \file tensor_expr_engine-inl.hpp
 * \brief definitions of how expressions should be evaluated
 * \author Tianqi Chen, Bing Xu
 */
#include "tensor_expr.h"

namespace mshadow{
    namespace expr{
        // This part of code gives plan that can be used to carry out execution
        template<typename ExpType>
        class Plan{
        public:
            /*!
             * \brief evaluate the expression at index [y][x]
             *        to be implemented by SubType
             */
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const;
        };

        template <typename Device, int dim>
        class Plan< Tensor<Device,dim> >{
        public:
            Plan( const Tensor<Device,dim> &t )
                :dptr_(t.dptr),stride_(t.shape.stride_){}
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ y * stride_ + x ];
            }
        private:
            const real_t  *dptr_;
            index_t stride_;
        };

        template<>
        class Plan<ScalarExp>{
        public:
            Plan( real_t scalar ):scalar_(scalar){}
            /*! \brief evaluate at [y][x] */
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                    return scalar_;
            }
        private:
            real_t scalar_;
        };

        template<typename OP, typename TA, typename TB,int etype>
        class Plan< BinaryMapExp<OP,TA,TB,etype> >{
        public:
            Plan( const Plan<TA> &lhs, const Plan<TB> &rhs )
                :lhs_(lhs), rhs_(rhs){}
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return OP::Map( lhs_.Eval( y, x ), rhs_.Eval( y, x ) );
            }
        private:
            Plan<TA> lhs_;
            Plan<TB> rhs_;
        };

        template<typename OP, typename TA, int etype>
        class Plan< UnaryMapExp<OP,TA,etype> >{
        public:
            Plan( const Plan<TA> &src ):src_(src){}
            MSHADOW_XINLINE real_t Eval( index_t y, index_t x ) const{
                return OP::Map( src_.Eval( y, x ) );
            }
        private:
            Plan<TA> src_;
        };

        // allow UnaryMap see the plan
        template<typename OP, typename TA, typename TB, int etype>
        inline Plan< BinaryMapExp<OP,TA,TB,etype> > MakePlan( const BinaryMapExp<OP,TA,TB,etype> &e );

        // translate from exp to execution plan
        inline Plan<ScalarExp> MakePlan( const ScalarExp &e ){
            return Plan<ScalarExp>( e.scalar_ );
        }

        template<typename T>
        inline Plan<T> MakePlan( const ContainerExp<T> &e ){
            return Plan<T>( e.self() );
        }

        template<typename OP, typename TA, int etype>
        inline Plan< UnaryMapExp<OP,TA,etype> > MakePlan( const UnaryMapExp<OP,TA,etype> &e ){
            return Plan< UnaryMapExp<OP,TA,etype> >( MakePlan(e.src_) );
        }

        template<typename OP, typename TA, typename TB, int etype>
        inline Plan< BinaryMapExp<OP,TA,TB,etype> > MakePlan( const BinaryMapExp<OP,TA,TB,etype> &e ){
                return Plan< BinaryMapExp<OP,TA,TB,etype> >( MakePlan(e.lhs_), MakePlan(e.rhs_) );
        }
    }; // namespace expr

    namespace expr{
        /*!
         * \brief static type check template,
         *        if a expression E does not match type Device, dim, then kPass = false
         * \tparam Device the type of Device
         * \tparam dim dimension of the tensor
         * \tparam E expression
         */
        template<typename Device, int dim, typename E>
        struct TypeCheck{
            const static bool kPass = false;
        };

        template<typename Device, int dim, typename OP, typename TA, typename TB, int etype>
        struct TypeCheck<Device,dim, BinaryMapExp<OP,TA,TB,etype> >;

        template<typename Device, int dim>
        struct TypeCheck<Device,dim,ScalarExp>{
            const static bool kPass = true;
        };
        template<typename Device, int dim>
        struct TypeCheck<Device,dim, Tensor<Device,dim> >{
            const static bool kPass = true;
        };
        template<typename Device, int dim, typename OP, typename TA, int etype>
        struct TypeCheck<Device,dim, UnaryMapExp<OP,TA,etype> >{
            const static bool kPass = TypeCheck<Device, dim, TA>::kPass;
        };
        template<typename Device, int dim, typename OP, typename TA, typename TB, int etype>
        struct TypeCheck<Device,dim, BinaryMapExp<OP,TA,TB,etype> >{
            const static bool kPass = TypeCheck<Device, dim, TA>::kPass && TypeCheck<Device,dim, TB>::kPass;
        };
        
        template<bool kPass>
        struct TypeCheckPass;
        template<>
        struct TypeCheckPass<false>{};
        template<>
        struct TypeCheckPass<true>{
            inline static void Error_All_Tensor_in_Exp_Must_Have_Same_Type( void ){}
        };
    }; // namespace expr

    namespace expr{
        // check shape consistency
        template<int dim>
        struct ShapeCheck{
            inline static bool Check( const ScalarExp &exp, const Shape<dim> &shape ){
                return true;
            }
            template<typename Device>
            inline static bool Check( const Tensor<Device,dim> &t, const Shape<dim> &shape ){
                return t.shape == shape;
            }
            template<typename OP, typename TA, int etype>
            inline static bool Check( const UnaryMapExp<OP,TA,etype> &t, const Shape<dim> &shape ){
                return Check( t.src_, shape );
            }
            template<typename OP, typename TA, typename TB, int etype>
            inline static bool Check( const BinaryMapExp<OP,TA,TB,etype> &t, const Shape<dim> &shape ){
                return Check( t.lhs_, shape ) && Check( t.rhs_, shape );
            }
        };
    }; // namespace expr

    // the matrix OP depends on BLAS
    #if (MSHADOW_USE_CBLAS||MSHADOW_USE_MKL)
    namespace expr{
        // handles the dot
        template<typename Device>
        struct BLASEngine;
        template<>
        struct BLASEngine<cpu>{
            inline static CBLAS_TRANSPOSE GetT( bool t ){
                return t ? CblasTrans : CblasNoTrans;
            }
            inline static void gemm( bool transa, bool transb, int m, int n, int k, float alpha, \
                                     const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc ){
                cblas_sgemm(CblasColMajor, GetT(transa), GetT(transb), m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
            }
            inline static void gemm( bool transa, bool transb, int m, int n, int k, double alpha, \
                                     const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc ){
                cblas_dgemm(CblasColMajor, GetT(transa), GetT(transb), m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
            }
            inline static void gemv( bool trans, int m, int n, float alpha, const float *A, int lda, \
                                     const float *X, int incX, float beta, float *Y, int incY ){
                cblas_sgemv(CblasColMajor, GetT(trans), m,n,alpha,A,lda,X,incX,beta,Y,incY);
            }
            inline static void gemv( bool trans, int m, int n, double alpha, const double *A, int lda, \
                                     const double *X, int incX, double beta, double *Y, int incY ){
                cblas_dgemv(CblasColMajor, GetT(trans), m,n,alpha,A,lda,X,incX,beta,Y,incY);
            }
            inline static void ger( int m, int n, float alpha, const float *X, int incX, const float *Y, int incY, float *A, int lda ){
                cblas_sger(CblasColMajor,m,n,alpha,X,incX,Y,incY,A,lda);
            }
            inline static void ger( int m, int n, double alpha, const double *X, int incX, const double *Y, int incY, double *A, int lda ){
                cblas_dger(CblasColMajor,m,n,alpha,X,incX,Y,incY,A,lda);
            }
        };

        #if MSHADOW_USE_CUDA
        // All CuBLAS goes to here, use legacy API: not threadsafe
        template<>
        struct BLASEngine<gpu>{
            inline static char GetT( bool t ){
                return t ? 'T' : 'N';
            }
            inline static void gemm( bool transa, bool transb, int m, int n, int k, float alpha, 
                                     const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc ){
                cublasSgemm(GetT(transa),GetT(transb),m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
            }
            inline static void gemm( bool transa, bool transb, int m, int n, int k, double alpha, 
                                     const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc ){
                cublasDgemm(GetT(transa),GetT(transb),m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);                
            }
            inline static void gemv( bool trans, int m, int n, float alpha, const float *A, int lda, \
                                     const float *X, int incX, float beta, float *Y, int incY ){
                cublasSgemv(GetT(trans), m,n,alpha,A,lda,X,incX,beta,Y,incY);
            }
            inline static void gemv( bool trans, int m, int n, double alpha, const double *A, int lda, \
                                     const double *X, int incX, double beta, double *Y, int incY ){
                cublasDgemv(GetT(trans), m,n,alpha,A,lda,X,incX,beta,Y,incY);
            }
            inline static void ger( int m, int n, float alpha, const float *X, int incX, const float *Y, int incY, float *A, int lda ){
                cublasSger(m,n,alpha,X,incX,Y,incY,A,lda);
            }
            inline static void ger( int m, int n, double alpha, const double *X, int incX, const double *Y, int incY, double *A, int lda ){
                cublasDger(m,n,alpha,X,incX,Y,incY,A,lda);
            }
        };
        #endif
        template<typename SV,typename Device, int ddim, int ldim, int rdim, bool ltrans, bool rtrans>
        struct DotEngine{
            inline static void Eval( Tensor<Device,ddim> &dst, const Tensor<Device,ldim> &lhs, const Tensor<Device,rdim> &rhs, real_t scale );
        };

        // helper function to decide which shape we are in 
        inline static Shape<2> GetShape( const Shape<2> &shape, bool transpose ){
            return transpose ? Shape2(shape[0],shape[1]) : shape;
        }
        // dst = dot( lhs[.T], rhs[.T] )
        template<typename SV, typename xpu, bool transpose_left, bool transpose_right>
        struct DotEngine<SV,xpu,2,2,2,transpose_left,transpose_right>{
            inline static void Eval( Tensor<xpu,2> &dst, const Tensor<xpu,2> &lhs, const Tensor<xpu,2> &rhs, real_t scale ) {
                Shape<2> sleft  = GetShape( lhs.shape, transpose_left );
                Shape<2> sright = GetShape( rhs.shape, transpose_right );
                utils::Assert( dst.shape[1] == sleft[1] && dst.shape[0] == sright[0] \
                               && sleft[0] == sright[1] , "dot-gemm: matrix shape mismatch" );
                // use column major argument to compatible with most BLAS
                BLASEngine<xpu>::gemm
                    ( transpose_right , transpose_left,
                      transpose_right ? rhs.shape[1] : rhs.shape[0],
                      transpose_left  ? lhs.shape[0] : lhs.shape[1],
                      transpose_right ? rhs.shape[0] : rhs.shape[1], 
                      scale * SV::kAlphaBLAS, 
                      rhs.dptr, rhs.shape.stride_,
                      lhs.dptr, lhs.shape.stride_,
                      SV::kBetaBLAS, 
                      dst.dptr, dst.shape.stride_ );
            }
        };
        template<typename SV, typename xpu, bool transpose_right>
        struct DotEngine<SV,xpu,1,1,2,false,transpose_right>{
            inline static void Eval( Tensor<xpu,1> &dst, const Tensor<xpu,1> &lhs, const Tensor<xpu,2> &rhs, real_t scale ) {
                Shape<2> sright = GetShape( rhs.shape, transpose_right );
                utils::Assert( dst.shape[0] == sright[0] && lhs.shape[0] == sright[1], "dot-gemv: matrix shape mismatch");
                BLASEngine<xpu>::gemv
                    ( transpose_right, 
                      rhs.shape[0], rhs.shape[1], scale * SV::kAlphaBLAS,
                      rhs.dptr, rhs.shape.stride_,
                      lhs.dptr, 1, SV::kBetaBLAS,
                      dst.dptr, 1 );
            }
        };        
        template<typename SV, typename xpu>
        struct DotEngine<SV,xpu,2,1,1,true,false>{
            inline static void Eval( Tensor<xpu,2> &dst, const Tensor<xpu,1> &lhs, const Tensor<xpu,1> &rhs, real_t scale ) {
                utils::Assert( dst.shape[1] == lhs.shape[0] && dst.shape[0] == rhs.shape[0], "dot-ger: matrix shape mismatch" );
                if( SV::kBetaBLAS < 1e-6f ){
                    BLASEngine<xpu>::ger
                        ( rhs.shape[0], lhs.shape[0], scale * SV::kAlphaBLAS,
                          rhs.dptr, 1, lhs.dptr, 1, dst.dptr, dst.shape.stride_ );
                }else{
                    DotEngine<SV,xpu,2,2,2,true,false>::Eval( dst, lhs.FlatTo2D(), rhs.FlatTo2D(), scale );
                }
            }
        };
    }; // namespace expr
    #endif

    namespace expr{
        template<typename SV, typename Device, int dim>
        struct ExpEngine<SV, Tensor<Device,dim> >{
            template<typename E>
            inline static void Eval( Tensor<Device,dim>& dst, const Exp<E,type::kMapper> &exp ){
                // static type check and shape check               
                MapExp<SV,dim,E>( dst, exp );

            }
            template<typename E>
            inline static void Eval( Tensor<Device,dim>& dst, const Exp<E,type::kContainer> &exp ){
                // static type check and shape check
                MapExp<SV,dim,E>( dst, exp );
            }

            template<int ldim,int rdim,bool ltrans,bool rtrans>
            inline static void Eval( Tensor<Device,dim> &dst, const DotExp< Tensor<Device,ldim>, Tensor<Device,rdim>, ltrans, rtrans > &exp ){
                DotEngine<SV,Device,dim,ldim,rdim,ltrans,rtrans>::Eval( dst, exp.lhs_, exp.rhs_, exp.scale_ );
            }
        };
    }; // namespace expr
};
#endif
