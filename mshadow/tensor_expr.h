#ifndef MSHADOW_TENSOR_EXPR_H
#define MSHADOW_TENSOR_EXPR_H
/*!
 * \file tensor_expr.h
 * \brief definitions of abstract expressions and expressions template
 * \author Tianqi Chen, Bing Hsu
 */
#include "tensor_base.h"

namespace mshadow{
    /*!
     * \brief namespace for abstract expressions and expressions template,
     *        have no dependecy on tensor.h,
     *        These data structure takes no charge in computations,
     *        they are only used to define operations and represent expression in a symbolic way
     */
    namespace expr{

        /*! \brief type of expressions */
        namespace type{
            /*! \brief this expression directly correspnds to a data class */
            const int kContainer = 0;
            /*! \brief this only contains element-wise vector operations */
            const int kMapper    = 1;
            /*! \brief othercase: e.g dot product */
            const int kComplex   = 3;
        };

        /*!
         * \brief expression engine that actually interprets these expressions
         *        this is a function template that needed to be implemented for specific expressions
         */
        template<typename Saver,typename Container>
        struct ExpEngine{
            template<typename EType>
            inline static void Eval( Container& dst, const EType &exp );
        };

        template<typename Container>
        class ContainerExp;
        class ScalarExp;

        /*!
         * \brief base class for expression
         * \tparam SubType inheritated class must put their type into this parameter
         * \tparam exp_type expression type, see namespace type
         */
        template<typename SubType, int exp_type>
        struct Exp{
        public:
            typedef SubType EType;
            const static int kType = exp_type;
        public:
            inline const SubType& self( void ) const{
                return *static_cast<const SubType*>(this);
            }
            inline SubType& refself( void ){
                return *static_cast<SubType*>(this);
            }
        };
        
        /*! \brief scalar expression */
        struct ScalarExp: public Exp<ScalarExp, type::kMapper>{
            real_t scalar_;
            ScalarExp( real_t scalar ):scalar_(scalar){}
            ScalarExp( double scalar ):scalar_(scalar){}
        };

        /*! \brief represent a transpose expression of a container */
        template<typename EType>
        struct TransposeExp: public Exp< TransposeExp<EType>, type::kComplex >{
        public:
            /*! \brief expression to be transposed */
            const EType &exp;
            /*! \brief constructor */
            TransposeExp( const EType &e ):exp(e){}
            /*! \brief transpose expression */
            inline const EType & T( void ) const{
                return exp;
            }
        };
        /*!
         * \brief base class of all variables, that can be assigned to values
         * \tparam Container the actually class of data container, e.g. CTensor1D
         */
        template<typename Container>
        class ContainerExp: public Exp< Container, type::kContainer >{
        public:
            /*! 
             *\brief transpose of a matrix
             *\return transpose of current expression
             */
            inline const TransposeExp<Container> T( void ) const{
                return TransposeExp<Container>( this->self() );
            }
        public:
            inline Container &operator+=( real_t s ){
                ExpEngine<sv::plusto,Container>::Eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator-=( real_t s ){
                ExpEngine<sv::minusto,Container>::Eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator*=( real_t s ){
                ExpEngine<sv::multo,Container>::Eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &operator/=( real_t s ){
                ExpEngine<sv::divto,Container>::Eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
            inline Container &__assign( real_t s ){
                ExpEngine<sv::saveto,Container>::Eval( this->refself(), ScalarExp(s) );
                return this->refself();
            }
        public:
            /*! \brief implementation of operator=, note that we can not define container = container */
            template<typename E>
            inline Container &__assign( const Exp<E,type::kMapper> &exp ){
                ExpEngine<sv::saveto,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
            template<typename E>
            inline Container &__assign( const Exp<E,type::kComplex> &exp ){
                ExpEngine<sv::saveto,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator+= */
            template<typename E,int etype>
            inline Container &operator+=( const Exp<E,etype> &exp ){
                ExpEngine<sv::plusto,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator-= */
            template<typename E,int etype>
            inline Container &operator-=( const Exp<E,etype> &exp ){
                ExpEngine<sv::minusto,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator*= */
            template<typename E,int etype>
            inline Container &operator*=( const Exp<E,etype> &exp ){
                ExpEngine<sv::multo,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
            /*! \brief implementation of operator/= */
            template<typename E,int etype>
            inline Container &operator/=( const Exp<E,etype> &exp ){
                ExpEngine<sv::divto,Container>::Eval( this->refself(), exp.self() );
                return this->refself();
            }
        };
    }; // namespace expr

    namespace expr{
        /*! 
         * \brief matrix multiplication expression dot( lhs[.T], rhs[.T] )
         * \tparam TA type of lhs
         * \tparam TB type of rhs
         * \tparam ltrans whether lhs is transposed
         * \tparam rtrans whether rhs is transposed
         */
        template<typename TA,typename TB,bool ltrans,bool rtrans>
        struct DotExp: public Exp< DotExp<TA,TB,ltrans,rtrans>, type::kComplex >{
            const TA &lhs_;
            const TB &rhs_;
            real_t scale_;
            DotExp( const TA &lhs, const TB &rhs, real_t scale )
                :lhs_(lhs), rhs_(rhs){}
        };
        
        template<typename TA, typename TB>
        inline DotExp<TA,TB,false,false> dot( const ContainerExp<TA> &lhs, const ContainerExp<TB> &rhs ){
            return DotExp<TA,TB,false,false>( lhs.self(), rhs.self(), 1.0f );
        }
        template<typename TA, typename TB>
        inline DotExp<TA,TB,true,false> dot( const TransposeExp<TA> &lhs, const ContainerExp<TB> &rhs ){
            return DotExp<TA,TB,true,false>( lhs.exp, rhs.self(), 1.0f );
        }
        template<typename TA, typename TB>
        inline DotExp<TA,TB,false,true> dot( const ContainerExp<TA> &lhs, const TransposeExp<TB> &rhs ){
            return DotExp<TA,TB,false,true>( lhs.self(), rhs.exp, 1.0f );
        }
        template<typename TA, typename TB>
        inline DotExp<TA,TB,true,true> dot( const TransposeExp<TA> &lhs, const TransposeExp<TB> &rhs ){
            return DotExp<TA,TB,true,true>( lhs.exp, rhs.exp, 1.0f );
        }
        template<typename TA, typename TB, bool ltrans, bool rtrans >
        inline DotExp<TA,TB,ltrans,rtrans> operator*( const DotExp<TA,TB,ltrans,rtrans> &lhs, real_t rhs ){
            return DotExp<TA,TB,ltrans,rtrans>( lhs.lhs_, lhs.rhs_, lhs.scale_ * rhs );
        }
        template<typename TA, typename TB, bool ltrans, bool rtrans >
        inline DotExp<TA,TB,ltrans,rtrans> operator*( real_t lhs, const DotExp<TA,TB,ltrans,rtrans> &rhs ){
            return DotExp<TA,TB,ltrans,rtrans>( rhs.lhs_, rhs.rhs_, rhs.scale_ * lhs );
        }
    }; // namespace expr

    namespace expr{
        /*!
         * \brief binary map expression lhs [op] rhs
         * \tparam OP operator
         * \tparam TA type of lhs
         * \tparam TB type of rhs
         * \tparam etype expression type, sa namespace::type
         */
        template<typename OP, typename TA, typename TB, int etype >
        struct BinaryMapExp: public Exp< BinaryMapExp<OP,TA,TB,etype>, etype >{
            const TA &lhs_;
            const TB &rhs_;
            BinaryMapExp( const TA &lhs, const TB &rhs )
                :lhs_(lhs), rhs_(rhs){}
        };

        // make expression
        template<typename OP,typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<OP,TA,TB, (ta|tb|type::kMapper) > MakeExp( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return BinaryMapExp<OP,TA,TB, (ta|tb|type::kMapper) >( lhs.self(), rhs.self() );
        }
        // short hand for MakeExp, F stands for function
        template<typename OP,typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<OP,TA,TB, (ta|tb|type::kMapper) > F( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<OP>( lhs, rhs );
        }

        // operator rules
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::plus,TA,TB, (ta|tb|type::kMapper) > operator+( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::plus>( lhs, rhs );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::minus,TA,TB, (ta|tb|type::kMapper) > operator-( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::minus>( lhs, rhs );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::mul,TA,TB, (ta|tb|type::kMapper) > operator*( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::mul>( lhs, rhs );
        }
        template<typename TA, typename TB, int ta, int tb>
        inline BinaryMapExp<op::div,TA,TB, (ta|tb|type::kMapper) > operator/( const Exp<TA,ta> &lhs, const Exp<TB,tb> &rhs ){
            return MakeExp<op::div>( lhs, rhs );
        }
        // constant operators
        template<typename TA, int ta>
        inline BinaryMapExp<op::plus, TA, ScalarExp, (ta|type::kMapper) > operator+( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::plus>( lhs, ScalarExp(rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::plus, TA, ScalarExp, (ta|type::kMapper) > operator-( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::plus>( lhs, ScalarExp(-rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::mul, TA, ScalarExp, (ta|type::kMapper) > operator*( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::mul>( lhs, ScalarExp(rhs) );
        }
        template<typename TA, int ta>
        inline BinaryMapExp<op::div, TA, ScalarExp, (ta|type::kMapper) > operator/( const Exp<TA,ta>& lhs,  real_t rhs ){
            return MakeExp<op::div>( lhs, ScalarExp(rhs) );
        }
        // constant operators 2
        template<typename TB, int tb>
        inline BinaryMapExp<op::plus, ScalarExp, TB, (tb|type::kMapper) > operator+( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::plus>( ScalarExp(lhs), rhs );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::minus, ScalarExp, TB, (tb|type::kMapper) > operator-( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::minus>( ScalarExp(lhs), rhs );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::mul, ScalarExp, TB, (tb|type::kMapper) > operator*( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::mul>( ScalarExp(lhs), rhs );
        }
        template<typename TB, int tb>
        inline BinaryMapExp<op::div, ScalarExp, TB, (tb|type::kMapper) > operator/( real_t lhs, const Exp<TB,tb>& rhs ){
            return MakeExp<op::div>( ScalarExp(lhs), rhs );
        }
    };

    namespace expr{
        /*!
         * \brief unary map expression op(src)
         * \tparam OP operator
         * \tparam TA type of src
         * \tparam etype expression type, sa namespace::type
         */
        template<typename OP, typename TA, int etype >
        struct UnaryMapExp: public Exp< UnaryMapExp<OP,TA,etype>, etype >{
            const TA &src_;
            UnaryMapExp( const TA &src )
                :src_(src){}
        };

        // make expression
        template<typename OP,typename TA, int ta>
        inline UnaryMapExp<OP,TA,(ta|type::kMapper) > MakeExp( const Exp<TA,ta> &src ){
            return UnaryMapExp<OP,TA, (ta|type::kMapper) >( src.self() );
        }

        // short hand for MakeExp
        template<typename OP,typename TA, int ta>
        inline UnaryMapExp<OP,TA,(ta|type::kMapper) > F( const Exp<TA,ta> &src ){
            return MakeExp<OP>(src);
        }
    };
};
#endif
